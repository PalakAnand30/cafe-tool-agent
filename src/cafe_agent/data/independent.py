"""Convert independently LLM-authored customer text with reviewed gold labels."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import Field, model_validator

from cafe_agent.config import StrictModel, read_config
from cafe_agent.tools.schemas import ToolName, validate_arguments

from .builder import ConversationBuilder
from .config import load_data_config
from .io import read_jsonl
from .schema import AnnotationProvenance, Conversation, Intent
from .splitter import normalise_user_text
from .tokenization import load_tokenizer
from .validator import validate_conversation


def domain_digest() -> str:
    return hashlib.sha256(
        json.dumps(read_config("domain.yaml"), sort_keys=True).encode()
    ).hexdigest()


class ToolExpectation(StrictModel):
    action: Literal["tool_call"]
    tool: ToolName
    args: dict[str, Any]
    after_tool: Literal["inform_error"] | None = None

    @model_validator(mode="after")
    def check_arguments(self) -> "ToolExpectation":
        arguments = dict(self.args)
        if arguments.get("order_id") == "$LAST_ORDER_ID":
            arguments["order_id"] = "ORD-00001"
        validate_arguments(self.tool, arguments)
        return self


class ClarifyExpectation(StrictModel):
    action: Literal["clarify"]
    about: Literal["size", "item", "quantity", "order_id"]
    text: str | None = None


class DeclineExpectation(StrictModel):
    action: Literal["decline"]
    reason: Literal["off_menu", "unsupported_modifier", "out_of_scope"]
    text: str | None = None


Expectation = Annotated[
    ToolExpectation | ClarifyExpectation | DeclineExpectation, Field(discriminator="action")
]


class AuthoredTurn(StrictModel):
    user: str = Field(min_length=1)
    expect: Expectation


class AuthoredCase(StrictModel):
    id: str = Field(pattern=r"^h\d{3}$")
    intent: str
    author: Literal["claude"]
    tags: list[str] = Field(default_factory=list)
    turns: list[AuthoredTurn] = Field(min_length=1)


class AuthoredSet(StrictModel):
    cases: list[AuthoredCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> "AuthoredSet":
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Duplicate independent benchmark ID")
        return self


class CaseOverride(StrictModel):
    reason: str = Field(min_length=1)
    intent: Intent | None = None
    turns: dict[int, Expectation] = Field(default_factory=dict)


class GoldOverrides(StrictModel):
    source_sha256: str
    domain_sha256: str
    annotator: Literal["codex"]
    cases: dict[str, CaseOverride]


INTENTS = {
    "single_item": Intent.SINGLE_ORDER,
    "multi_item": Intent.MULTI_ORDER,
    "cancel": Intent.CANCEL_ORDER,
    "error_recovery": Intent.TOOL_ERROR,
    **{intent.value: intent for intent in Intent},
}


def load_independent_records(path: Path) -> list[Conversation]:
    records = [Conversation.model_validate(value) for value in read_jsonl(path)]
    if any(record.author != "claude" or record.split != "independent_test" for record in records):
        raise ValueError("Independent benchmark must retain Claude authorship")
    if len({record.id for record in records}) != len(records):
        raise ValueError("Duplicate independent benchmark ID")
    return records


def _reply(result: dict[str, Any], tool: str) -> str:
    if "error" in result:
        error = result["error"]
        alternatives = ", ".join(item["name"] for item in error.get("alternatives", []))
        return error["message"] + (f" Would you like {alternatives}?" if alternatives else "")
    if tool == "create_order":
        return f"Order {result['order_id']} received; total £{result['total']}."
    if tool == "cancel_order":
        return "Your order is cancelled." if result["cancelled"] else result.get("reason", "")
    if tool == "get_order_status":
        return f"Your order is {result['status']}; ETA {result['eta_minutes']} minutes."
    if tool == "check_availability":
        if result["available"]:
            return "That item is available."
        alternatives = ", ".join(item["name"] for item in result["alternatives"])
        return "That item is out of stock." + (
            f" Would you like {alternatives}?" if alternatives else ""
        )
    return "Menu items: " + ", ".join(item["name"] for item in result["items"]) + "."


def convert_authoring(
    source: Path, overrides_path: Path, tokenizer: Any
) -> tuple[list[Conversation], dict[str, Any]]:
    raw = source.read_bytes()
    authored = AuthoredSet.model_validate(yaml.safe_load(raw))
    overrides = GoldOverrides.model_validate(yaml.safe_load(overrides_path.read_text()))
    source_hash = hashlib.sha256(raw).hexdigest()
    if overrides.source_sha256 != source_hash or overrides.domain_sha256 != domain_digest():
        raise ValueError("Gold review is stale: source or domain changed")
    if overrides.cases.keys() - {case.id for case in authored.cases}:
        raise ValueError("Gold override refers to an unknown case")
    records, lengths, seen = [], [], set()
    revised_turns = 0
    for case in authored.cases:
        override = overrides.cases.get(case.id)
        if override and any(index < 1 or index > len(case.turns) for index in override.turns):
            raise ValueError(f"{case.id}: override turn is outside the conversation")
        intent = override.intent if override and override.intent else INTENTS[case.intent]
        builder = ConversationBuilder()
        last_order = None
        try:
            for index, turn in enumerate(case.turns, 1):
                key = normalise_user_text(turn.user)
                if key in seen:
                    raise ValueError("Duplicate independent customer text")
                seen.add(key)
                builder.user(turn.user)
                expected = override.turns.get(index, turn.expect) if override else turn.expect
                revised_turns += expected != turn.expect
                if isinstance(expected, ToolExpectation):
                    args = json.loads(json.dumps(expected.args))
                    if args.get("order_id") == "$LAST_ORDER_ID":
                        if last_order is None:
                            raise ValueError(f"{case.id}: no successful prior order")
                        args["order_id"] = last_order
                    result = builder.call(expected.tool, args)
                    if ("error" in result) != (expected.after_tool == "inform_error"):
                        raise ValueError(
                            f"{case.id}/turn {index}: unexpected tool outcome: {result}"
                        )
                    if expected.tool == "create_order" and "order_id" in result:
                        last_order = result["order_id"]
                    builder.say(_reply(result, expected.tool))
                elif isinstance(expected, ClarifyExpectation):
                    questions = {
                        "size": "Which size would you like: small, medium or large?",
                        "item": "Which exact menu item would you like?",
                        "quantity": "How many would you like?",
                        "order_id": "What is your order ID?",
                    }
                    builder.say(expected.text or questions[expected.about], "clarify")
                else:
                    replies = {
                        "off_menu": (
                            "We don't serve that item. "
                            "Would you like a latte or a blueberry muffin?"
                        ),
                        "unsupported_modifier": (
                            "That modifier isn't supported for this item. "
                            "Would you like it without the modifier?"
                        ),
                        "out_of_scope": (
                            "I can help with café orders and menu questions. "
                            "Would you like to see the menu?"
                        ),
                    }
                    builder.say(expected.text or replies[expected.reason], "decline")
            record = builder.finish(
                record_id=case.id,
                split="independent_test",
                intent=intent,
                template_id="independent.claude",
                author="claude",
                annotation_provenance=AnnotationProvenance(
                    annotator="codex",
                    source_file=str(source),
                    source_sha256=source_hash,
                    domain_sha256=domain_digest(),
                    original_intent=case.intent,
                    revision_reason=override.reason if override else None,
                ),
            )
            lengths.append(validate_conversation(record, tokenizer, load_data_config().max_tokens))
            records.append(record)
        finally:
            builder.close()
    report = {
        "conversations": len(records),
        "validated": len(records),
        "customer_message_author": "claude",
        "gold_annotator": "codex",
        "human_authored": 0,
        "source_sha256": source_hash,
        "domain_sha256": domain_digest(),
        "revised_cases": len(overrides.cases),
        "revised_turns": revised_turns,
        "intents": dict(Counter(record.intent.value for record in records)),
        "token_lengths": {"min": min(lengths), "max": max(lengths)},
        "limitations": (
            "LLM-authored customer messages with Codex gold labels; not human-written evidence. "
            "Original source expected actions targeted a different menu. "
            "Customer messages are unchanged."
        ),
    }
    return records, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    config = load_data_config()
    records, report = convert_authoring(
        config.independent_authoring_path, config.independent_overrides_path, load_tokenizer()
    )
    if (
        not config.independent_min_conversations
        <= len(records)
        <= config.independent_max_conversations
    ):
        raise ValueError("Independent benchmark count is outside configured bounds")
    output = "".join(record.model_dump_json() + "\n" for record in records)
    config.independent_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.independent_path.with_suffix(".jsonl.tmp")
    temporary.write_text(output, encoding="utf-8")
    temporary.replace(config.independent_path)
    Path("results/independent_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
