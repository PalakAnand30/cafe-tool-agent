"""Copy measured validation evidence and a small review sample after generation."""

import json
import shutil
from pathlib import Path

from cafe_agent.data.config import load_data_config
from cafe_agent.data.human import load_human_records
from cafe_agent.data.io import read_jsonl
from cafe_agent.data.schema import Conversation, Intent

config = load_data_config()
summary = json.loads((config.output_dir / "validation_summary.json").read_text())
assert summary["validated"] == summary["conversations"] == sum(config.splits.values())
assert summary["splits"] == config.splits
assert summary["token_lengths"]["max"] < config.max_tokens
assert summary["leakage"]["overlapping_user_texts"] == 0
samples = []
all_records = []
for split in config.splits:
    records = [
        Conversation.model_validate(value)
        for value in read_jsonl(config.output_dir / f"{split}.jsonl")
    ]
    assert len(records) == config.splits[split]
    all_records.extend(records)
    if split in {"train", "test"}:
        for intent in Intent:
            samples.append(next(record for record in records if record.intent == intent))
retained = sum(p.paraphrase_accepted for record in all_records for p in record.provenance)
summary["retained_paraphrase_turns"] = retained
human = load_human_records(config.human_path, scoreable_only=False)
summary["human_scoreable"] = sum(record.scoreable for record in human)
summary["codex_examples_excluded"] = sum(not record.scoreable for record in human)
Path("results").mkdir(exist_ok=True)
Path("results/data_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
destination = Path("data/samples")
destination.mkdir(exist_ok=True)
(destination / "conversations.jsonl").write_text(
    "".join(record.model_dump_json() + "\n" for record in samples)
)
shutil.copyfile(config.output_dir / "stats.md", destination / "stats.md")
(destination / "README.md").write_text(
    "# Dataset review sample\n\n"
    "Programmatically labeled records selected from the generated train and test splits, "
    "covering every intent. These are synthetic gold examples, not model predictions "
    "or human benchmark evidence.\n\n"
    "The full generated files and API cache are ignored by Git. "
    "See stats.md and results/data_validation.json for measured validation evidence.\n"
)
print(json.dumps(summary, sort_keys=True))
print(f"Review samples written: {len(samples)}")
