"""Cached batched Groq paraphrases with conservative entity/binding verification."""

import json
import re
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from cafe_agent.config import StrictModel, load_runtime
from cafe_agent.domain.models import load_domain
from cafe_agent.models.base import Message
from cafe_agent.models.groq import GroqClient
from cafe_agent.parsing.parser import ParseError, strict_json

from .config import DataConfig
from .io import append_jsonl, read_jsonl
from .schema import Entity
from .splitter import normalise_user_text


class ParaphraseRequest(StrictModel):
    id: str
    text: str
    entities: list[Entity]


class ParaphraseReply(StrictModel):
    id: str
    text: str


class ParaphraseResult(StrictModel):
    id: str
    text: str
    accepted: bool
    rejection_reason: str | None = None


def contains(text: str, phrase: str) -> bool:
    return bool(
        re.search(
            r"(?<!\w)" + re.escape(normalise_user_text(phrase)) + r"(?!\w)",
            normalise_user_text(text),
        )
    )


def verify_entities(
    original: str, candidate: str, entities: list[Entity], config: DataConfig
) -> str | None:
    if not candidate.strip() or "<tool_call" in candidate or "```" in candidate:
        return "invalid_wrapper"
    if normalise_user_text(original) == normalise_user_text(candidate):
        return "unchanged"
    for entity in entities:
        phrases = [entity.value, *entity.aliases, *config.synonyms.get(entity.value, [])]
        if not any(contains(candidate, phrase) for phrase in phrases):
            return "missing_" + entity.kind
    original_numbers = set(re.findall(r"\b\d+\b", original))
    if set(re.findall(r"\b\d+\b", candidate)) - original_numbers:
        return "new_number"
    for size in ["small", "medium", "large"]:
        if contains(candidate, size) and not contains(original, size):
            return "new_size"
    for modifier, synonyms in config.synonyms.items():
        if modifier.isdigit():
            continue
        forms = [modifier, *synonyms]
        if any(contains(candidate, form) for form in forms) and not any(
            contains(original, form) for form in forms
        ):
            return "new_modifier"

    # Longest-name matching prevents treating 'latte' inside 'iced latte' as another item.
    def menu_mentions(text: str) -> set[str]:
        remaining = normalise_user_text(text)
        found = set()
        for item in sorted(load_domain().items, key=lambda item: -len(item.name)):
            pattern = r"(?<!\w)" + re.escape(item.name) + r"s?(?!\w)"
            if re.search(pattern, remaining):
                found.add(item.name)
                remaining = re.sub(pattern, " ", remaining)
        return found

    if menu_mentions(candidate) - menu_mentions(original):
        return "new_item"
    negative = re.compile(r"\b(?:not|never|without|don't|cancel)\b", re.IGNORECASE)
    if set(negative.findall(candidate.casefold())) - set(negative.findall(original.casefold())):
        return "new_negation_or_cancellation"
    return None


class Paraphraser:
    def __init__(self, config: DataConfig, journal: Path, client: Any = None) -> None:
        self.config, self.journal = config, journal
        runtime = load_runtime()
        runtime.generation.max_tokens = config.paraphrase_max_tokens * config.paraphrase_batch_size
        self.client = client or GroqClient(runtime)

    def run(self, requests: list[ParaphraseRequest]) -> dict[str, ParaphraseResult]:
        results = {
            value["id"]: ParaphraseResult.model_validate(value)
            for value in read_jsonl(self.journal, repair_tail=True)
        }
        size = self.config.paraphrase_batch_size
        try:
            for start in range(0, len(requests), size):
                batch = requests[start : start + size]
                if all(request.id in results for request in batch):
                    continue
                prompt = (
                    "Paraphrase each café customer utterance. Return ONLY a JSON array of objects "
                    "with id and text. Keep every name, item, size, quantity, modifier, order ID, "
                    "intent and negation. Preserve every protected phrase verbatim, especially "
                    "order_line phrases so quantities/modifiers stay bound to their items. "
                    "Change only surrounding phrasing. Do not add requests, entities or tool calls."
                )
                payload = [
                    {
                        "id": request.id,
                        "text": request.text,
                        "protected_phrases": [entity.value for entity in request.entities],
                    }
                    for request in batch
                ]
                response = self.client.generate(
                    [
                        Message(role="system", content=prompt),
                        Message(role="user", content=json.dumps(payload, ensure_ascii=False)),
                    ],
                    [],
                )
                try:
                    replies = TypeAdapter(list[ParaphraseReply]).validate_python(
                        strict_json(response.content)
                    )
                    if len(replies) != len(batch) or {reply.id for reply in replies} != {
                        r.id for r in batch
                    }:
                        raise ValueError("Batch IDs did not match")
                    outputs = {reply.id: reply.text for reply in replies}
                except (ValidationError, ParseError, ValueError):
                    outputs = {}
                for request in batch:
                    if request.id in results:
                        continue
                    candidate = outputs.get(request.id, "")
                    reason = (
                        verify_entities(request.text, candidate, request.entities, self.config)
                        if outputs
                        else "invalid_batch_response"
                    )
                    result = ParaphraseResult(
                        id=request.id,
                        text=candidate if reason is None else request.text,
                        accepted=reason is None,
                        rejection_reason=reason,
                    )
                    append_jsonl(self.journal, result.model_dump())
                    results[result.id] = result
                print(
                    f"Paraphrases processed: {min(start + size, len(requests))}/{len(requests)}",
                    flush=True,
                )
        finally:
            close = getattr(self.client, "close", None)
            if close:
                close()
        return results
