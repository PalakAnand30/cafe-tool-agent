"""Upload preparation replays gold data and separates the human configuration."""

import random
from pathlib import Path

import pytest

from cafe_agent.data.config import load_data_config
from cafe_agent.data.human import seed_examples
from cafe_agent.data.independent import convert_authoring
from cafe_agent.data.io import append_jsonl
from cafe_agent.data.publishing import prepare_dataset
from cafe_agent.data.schema import Intent
from cafe_agent.data.stats import write_stats
from cafe_agent.data.templates import generate_example
from cafe_agent.data.validator import validate_conversation


def test_prepare_packet_and_reject_incomplete_split(tmp_path: Path, tokenizer) -> None:
    config = load_data_config()
    config.output_dir = tmp_path / "generated"
    config.output_dir.mkdir()
    config.human_path = tmp_path / "human.jsonl"
    config.independent_path = tmp_path / "independent.jsonl"
    config.splits = {split: 1 for split in config.splits}
    records, lengths = [], {}
    for index, split in enumerate(config.splits):
        record = generate_example(
            config,
            f"{split}-00000",
            split,
            Intent.SINGLE_ORDER,
            4 if split == "test" else index,
            False,
            random.Random(42 + index),
        )
        records.append(record)
        lengths[record.id] = validate_conversation(record, tokenizer, config.max_tokens)
        append_jsonl(config.output_dir / f"{split}.jsonl", record.model_dump(mode="json"))
    seed_examples(config.human_path, tokenizer)
    write_stats(config.output_dir, records, lengths, {}, [])
    (config.output_dir / "manifest.json").write_text("{}\n")
    destination = tmp_path / "upload"
    counts = prepare_dataset(config, destination, tokenizer)
    assert counts == {
        "synthetic": 3,
        "human_scored": 0,
        "independent_scored": 0,
        "examples_excluded": 5,
    }
    card = (destination / "README.md").read_text()
    assert "config_name: human_test" in card and "author=codex_example" in card
    assert (destination / "data/train.jsonl").exists()
    assert (destination / "data/human_test.jsonl").exists()
    independent, _ = convert_authoring(
        config.independent_authoring_path, config.independent_overrides_path, tokenizer
    )
    config.independent_path.write_text(
        "".join(record.model_dump_json() + "\n" for record in independent)
    )
    counts = prepare_dataset(config, destination, tokenizer)
    assert counts["independent_scored"] == 84 and counts["human_scored"] == 0
    assert (destination / "provenance/authoring.yaml").read_bytes() == (
        config.independent_authoring_path.read_bytes()
    )
    assert "not a human-written benchmark" in (destination / "README.md").read_text()
    independent[0].messages[1].content = "Changed customer text"
    independent[0].scripted_user_turns[0] = "Changed customer text"
    config.independent_path.write_text(
        "".join(record.model_dump_json() + "\n" for record in independent)
    )
    with pytest.raises(ValueError, match="reviewed source"):
        prepare_dataset(config, destination, tokenizer)
    config.splits["train"] = 2
    with pytest.raises(ValueError, match="incomplete"):
        prepare_dataset(config, tmp_path / "invalid", tokenizer)
