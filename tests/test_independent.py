"""Independent authorship, reviewed gold, replay, and stale-review rejection."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cafe_agent.data.config import load_data_config
from cafe_agent.data.human import load_human_records
from cafe_agent.data.independent import convert_authoring, load_independent_records


def test_all_independent_cases_preserve_customer_text_and_replay(tokenizer, tmp_path) -> None:
    config = load_data_config()
    records, report = convert_authoring(
        config.independent_authoring_path, config.independent_overrides_path, tokenizer
    )
    source = yaml.safe_load(config.independent_authoring_path.read_text())["cases"]
    assert len(records) == report["validated"] == 84
    assert report["human_authored"] == 0
    assert report["token_lengths"]["max"] < config.max_tokens
    for original, record in zip(source, records, strict=True):
        assert record.id == original["id"]
        assert record.scripted_user_turns == [turn["user"] for turn in original["turns"]]
        assert record.author == "claude" and record.split == "independent_test"
        assert record.annotation_provenance.annotator == "codex"
    indexed = {record.id: record for record in records}
    assert indexed["h003"].actions[0].behaviour == "clarify"
    assert indexed["h011"].actions[0].behaviour == "decline"
    assert indexed["h028"].actions[-1].behaviour == "clarify"
    assert indexed["h048"].expected_final_state.orders == []
    assert indexed["h054"].expected_final_state.orders[0].status == "cancelled"
    assert len(indexed["h054"].expected_final_state.orders) == 2
    assert any(
        message.role == "tool" and "ORDER_NOT_FOUND" in message.content
        for message in indexed["h043"].messages
    )
    assert indexed["h071"].expected_final_state.orders == []
    path = tmp_path / "independent.jsonl"
    path.write_text("".join(record.model_dump_json() + "\n" for record in records))
    assert load_independent_records(path) == records
    with pytest.raises(ValueError, match="non-human"):
        load_human_records(path)
    tampered = records[0].model_dump(mode="json")
    tampered["author"] = "human"
    path.write_text(json.dumps(tampered) + "\n")
    with pytest.raises(ValidationError, match="Claude-authored"):
        load_independent_records(path)


@pytest.mark.parametrize("field", ["source_sha256", "domain_sha256"])
def test_converter_rejects_stale_gold_review(tokenizer, tmp_path: Path, field: str) -> None:
    config = load_data_config()
    overrides = yaml.safe_load(config.independent_overrides_path.read_text())
    overrides[field] = "stale"
    path = tmp_path / "overrides.yaml"
    path.write_text(yaml.safe_dump(overrides))
    with pytest.raises(ValueError, match="stale"):
        convert_authoring(config.independent_authoring_path, path, tokenizer)


@pytest.mark.parametrize("mutation", ["last_order", "unexpected_error", "unknown_case"])
def test_converter_rejects_invalid_gold(tokenizer, tmp_path: Path, mutation: str) -> None:
    config = load_data_config()
    source = yaml.safe_load(config.independent_authoring_path.read_text())
    source["cases"] = source["cases"][:1]
    if mutation == "last_order":
        source["cases"][0]["turns"][0]["expect"] = {
            "action": "tool_call",
            "tool": "cancel_order",
            "args": {"order_id": "$LAST_ORDER_ID"},
        }
    if mutation == "unexpected_error":
        source["cases"][0]["turns"][0]["expect"]["args"]["items"][0]["item_name"] = "pizza"
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(source))
    overrides = yaml.safe_load(config.independent_overrides_path.read_text())
    overrides["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    overrides["cases"] = {}
    if mutation == "unknown_case":
        overrides["cases"]["h999"] = {"reason": "Unknown case", "turns": {}}
    gold = tmp_path / "gold.yaml"
    gold.write_text(yaml.safe_dump(overrides))
    with pytest.raises(ValueError):
        convert_authoring(path, gold, tokenizer)
