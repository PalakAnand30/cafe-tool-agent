"""Gold-label construction, strict replay, holdouts and resumable split generation."""

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from cafe_agent.data.config import load_data_config
from cafe_agent.data.generator import build_dataset
from cafe_agent.data.io import append_jsonl, read_jsonl
from cafe_agent.data.schema import Conversation, Intent
from cafe_agent.data.splitter import (
    check_leakage,
    item_modifier_pairs,
    make_plan,
    normalise_user_text,
)
from cafe_agent.data.templates import generate_example, template_ids
from cafe_agent.data.validator import validate_conversation


@pytest.mark.parametrize("intent", list(Intent))
@pytest.mark.parametrize("variant", range(5))
def test_templates_replay_for_every_intent_and_family(
    tokenizer, intent: Intent, variant: int
) -> None:
    config = load_data_config()
    record = generate_example(
        config,
        "example",
        "test" if variant == 4 else "train",
        intent,
        variant,
        True,
        random.Random(42 + variant),
    )
    assert validate_conversation(record, tokenizer, config.max_tokens) < 4096
    again = generate_example(
        config, "example", record.split, intent, variant, True, random.Random(42 + variant)
    )
    assert again == record
    assert record.scripted_user_turns == [m.content for m in record.messages if m.role == "user"]


def test_plan_intent_mix_and_multi_turn_rate() -> None:
    config = load_data_config()
    plan = make_plan(config)
    assert Counter(slot.split for slot in plan) == config.splits
    assert len({slot.id for slot in plan}) == 4200
    for split, size in config.splits.items():
        group = [slot for slot in plan if slot.split == split]
        assert sum(slot.multi_turn for slot in group) == round(size * config.multi_turn_rate)
        assert Counter(slot.intent for slot in group) == {
            intent: round(size * weight) for intent, weight in config.intent_mix.items()
        }
    train, test = set(template_ids(config, test=False)), set(template_ids(config, test=True))
    assert len(train) == 40 and len(test) == 10 and not train & test


@pytest.mark.parametrize(
    "mutation", ["tool_result", "final_state", "tool_id", "schema", "prompt", "scripted_turn"]
)
def test_validator_rejects_corruption(tokenizer, mutation: str) -> None:
    config = load_data_config()
    record = generate_example(
        config, "example", "train", Intent.SINGLE_ORDER, 0, False, random.Random(42)
    )
    if mutation == "tool_result":
        next(m for m in record.messages if m.role == "tool").content = '{"order_id":"ORD-99999"}'
    elif mutation == "final_state":
        record.expected_final_state.orders.clear()
    elif mutation == "tool_id":
        next(m for m in record.messages if m.role == "tool").tool_call_id = "wrong"
    elif mutation == "schema":
        record.tools.clear()
    elif mutation == "prompt":
        record.messages[0].content = "Different prompt"
    else:
        record.scripted_user_turns[0] = "Different user text"
    with pytest.raises(ValueError):
        validate_conversation(record, tokenizer, config.max_tokens)


def test_leakage_checks_text_template_and_combinations() -> None:
    config = load_data_config()
    first = generate_example(config, "a", "train", Intent.SINGLE_ORDER, 0, False, random.Random(42))
    second = first.model_copy(
        deep=True, update={"id": "b", "split": "test", "template_id": "single_order.v4"}
    )
    with pytest.raises(ValueError, match="Duplicate normalised"):
        check_leakage([first, second], config)
    first.template_id = "single_order.v4"
    with pytest.raises(ValueError, match="template"):
        check_leakage([first], config)
    first.template_id = "single_order.v0"
    call = next(m.tool_calls[0] for m in first.messages if m.tool_calls)
    call.function.arguments["items"][0].update(item_name="latte", modifiers=["vanilla_syrup"])
    assert ("latte", "vanilla_syrup") in item_modifier_pairs(first)
    with pytest.raises(ValueError, match="combination"):
        check_leakage([first], config)
    assert normalise_user_text("  SMALL latte, PLEASE! ") == normalise_user_text(
        "small latte please"
    )


def test_small_pipeline_resume_is_append_only_and_reproducible(tmp_path: Path, tokenizer) -> None:
    config = load_data_config()
    config.splits = {split: 20 for split in config.splits}
    config.paraphrase_rate = 0
    summary = build_dataset(config, tokenizer=tokenizer, output_dir=tmp_path)
    assert summary["conversations"] == summary["validated"] == 60
    before = {split: (tmp_path / f"{split}.jsonl").read_bytes() for split in config.splits}
    assert build_dataset(config, tokenizer=tokenizer, output_dir=tmp_path) == summary
    assert all(
        (tmp_path / f"{split}.jsonl").read_bytes() == value for split, value in before.items()
    )
    with (tmp_path / "train.jsonl").open("ab") as handle:
        handle.write(b'{"incomplete":')
    assert build_dataset(config, tokenizer=tokenizer, output_dir=tmp_path) == summary
    assert (tmp_path / "train.jsonl").read_bytes() == before["train"]
    assert (tmp_path / "train.jsonl.incomplete-tail").exists()
    rows = [
        Conversation.model_validate(row)
        for split in config.splits
        for row in read_jsonl(tmp_path / f"{split}.jsonl")
    ]
    assert check_leakage(rows, config)["overlapping_user_texts"] == 0
    config.noise_rate = 0
    with pytest.raises(ValueError, match="changed"):
        build_dataset(config, tokenizer=tokenizer, output_dir=tmp_path)


def test_jsonl_corruption_is_not_silently_discarded(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    append_jsonl(path, {"id": "a"})
    with path.open("a") as handle:
        handle.write("broken\n")
    with pytest.raises(ValueError, match="Invalid JSONL"):
        read_jsonl(path, repair_tail=True)
    assert path.read_text().endswith("broken\n")


def test_jsonl_valid_last_record_without_newline(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"id": "a"}))
    assert read_jsonl(path, repair_tail=True) == [{"id": "a"}]
    assert path.read_bytes().endswith(b"\n")
