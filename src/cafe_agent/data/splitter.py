"""Deterministic intent allocation and explicit cross-split leakage checks."""

import random
import re
import unicodedata
from collections import defaultdict

from cafe_agent.config import StrictModel

from .config import DataConfig
from .schema import Conversation, Intent
from .templates import template_ids


def normalise_user_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


class Slot(StrictModel):
    id: str
    split: str
    intent: Intent
    multi_turn: bool


def make_plan(config: DataConfig) -> list[Slot]:
    result = []
    for split, size in config.splits.items():
        counts = {intent: int(size * weight) for intent, weight in config.intent_mix.items()}
        remaining = size - sum(counts.values())
        ordered = sorted(
            counts, key=lambda key: (-(size * config.intent_mix[key] - counts[key]), key)
        )
        for intent in ordered[:remaining]:
            counts[intent] += 1
        intents = [intent for intent, count in counts.items() for _ in range(count)]
        random.Random(f"{config.seed}:{split}:plan").shuffle(intents)
        mandatory = counts[Intent.ORDER_STATUS] + counts[Intent.CANCEL_ORDER]
        clarification_completions = round(size * config.multi_turn_rate) - mandatory
        if not 0 <= clarification_completions <= counts[Intent.CLARIFICATION]:
            raise ValueError(
                "Multi-turn rate must fit status/cancel plus clarification conversations"
            )
        for index, intent in enumerate(intents):
            multi_turn = intent in {Intent.ORDER_STATUS, Intent.CANCEL_ORDER}
            if intent == Intent.CLARIFICATION and clarification_completions:
                multi_turn = True
                clarification_completions -= 1
            result.append(
                Slot(id=f"{split}-{index:05d}", split=split, intent=intent, multi_turn=multi_turn)
            )
    return result


def item_modifier_pairs(record: Conversation) -> set[tuple[str, str]]:
    pairs = set()
    for message in record.messages:
        for call in message.tool_calls or []:
            if call.function.name == "create_order":
                for item in call.function.arguments["items"]:
                    pairs.update(
                        (item["item_name"], modifier) for modifier in item.get("modifiers", [])
                    )
    return pairs


def check_leakage(records: list[Conversation], config: DataConfig) -> dict[str, int]:
    owners: dict[str, str] = {}
    texts: dict[str, set[str]] = defaultdict(set)
    templates: dict[str, set[str]] = defaultdict(set)
    held = {(pair.item_name, pair.modifier) for pair in config.held_out_combinations}
    held_templates = set(template_ids(config, test=True))
    observed_held_pairs = set()
    for record in records:
        if record.id in owners:
            raise ValueError("Duplicate conversation ID")
        owners[record.id] = record.split
        templates[record.split].add(record.template_id)
        if record.split != "test" and record.template_id in held_templates:
            raise ValueError("Held-out template leaked into development data")
        pairs = item_modifier_pairs(record)
        if record.split != "test" and pairs & held:
            raise ValueError("Held-out item/modifier combination leaked")
        observed_held_pairs.update(pairs & held)
        for turn in record.scripted_user_turns:
            key = normalise_user_text(turn)
            if any(key in existing for existing in texts.values()):
                raise ValueError("Duplicate normalised user text across conversations")
            texts[record.split].add(key)
    return {
        "held_out_templates_observed": len(templates["test"] & held_templates),
        "held_out_combinations_observed": len(observed_held_pairs),
        "overlapping_user_texts": 0,
    }
