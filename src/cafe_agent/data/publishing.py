"""Validate and prepare private Hub dataset files with a separate human config."""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import HfApi

from .config import DataConfig, load_data_config
from .human import load_human_records
from .io import read_jsonl
from .schema import Conversation
from .splitter import check_leakage
from .tokenization import load_tokenizer
from .validator import validate_conversation


def prepare_dataset(config: DataConfig, destination: Path, tokenizer: Any) -> dict[str, int]:
    records = []
    for split, expected in config.splits.items():
        values = read_jsonl(config.output_dir / f"{split}.jsonl")
        group = [Conversation.model_validate(value) for value in values]
        if len(group) != expected or any(
            record.split != split or record.author != "synthetic" for record in group
        ):
            raise ValueError("Synthetic split is incomplete or mislabeled")
        records.extend(group)
    check_leakage(records, config)
    human = load_human_records(config.human_path, scoreable_only=False)
    for record in [*records, *human]:
        validate_conversation(record, tokenizer, config.max_tokens)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "data").mkdir(exist_ok=True)
    for split in config.splits:
        shutil.copyfile(
            config.output_dir / f"{split}.jsonl", destination / "data" / f"{split}.jsonl"
        )
    (destination / "data/human_test.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in human), encoding="utf-8"
    )
    scored = sum(record.scoreable for record in human)
    card = """---
language: [en]
task_categories: [text-generation]
tags: [tool-calling, synthetic, cafe, qwen]
configs:
  - config_name: default
    default: true
    data_files:
      - split: train
        path: data/train.jsonl
      - split: validation
        path: data/validation.jsonl
      - split: test
        path: data/test.jsonl
  - config_name: human_test
    data_files:
      - split: test
        path: data/human_test.jsonl
---
# Café tool calls

Synthetic café conversations with deterministic backend-derived tool calls,
results and final state. User text is template-based with verified Groq GPT-OSS
120B paraphrases and surface noise. Assistant labels and replies are programmatic.
The default config contains train/validation/test; human_test is a separate config.
All five function schemas and the full system prompt are included per conversation.
Use the raw JSONL files (hf_hub_download plus the shared JSONL reader) to preserve
omitted argument fields; Arrow table inference may fill absent struct keys with null.

## Validation and leakage

Records replay against a fresh SQLite backend, validate every tool call and stay
under the user-approved 4,096-token full-conversation budget at the pinned Qwen
revision. Reserved templates and item/modifier pairs occur only in synthetic test.
Normalised user text is deduplicated across all synthetic splits.

## Human benchmark

Only author=human is scored. Always exclude author=codex_example; the five worked
examples demonstrate the format and are not human-authored evidence. Human authors
must write the benchmark before seeing model evaluation results.

## Provenance, limitations and use

See the uploaded generation statistics and manifest for actual rates and versions.
Synthetic training data is not representative of all real café conversations.
No model performance claims accompany this dataset. The logical clock advances
only at recorded clock_advances; all prices are GBP decimal strings.
Groq paraphrases use the user-approved GPT-OSS 120B model, released under
Apache-2.0. Model outputs are not human-authored benchmark evidence.
This dataset starts private; the owner decides when to make it public.
"""
    (destination / "README.md").write_text(card, encoding="utf-8")
    for name in ["stats.md", "validation_summary.json", "manifest.json"]:
        shutil.copyfile(config.output_dir / name, destination / name)
    return {
        "synthetic": len(records),
        "human_scored": scored,
        "examples_excluded": len(human) - scored,
    }


def upload_private(destination: Path, *, username: str, token: str, api: Any = None) -> None:
    client = api or HfApi(token=token)
    repo_id = username + "/cafe-tool-calls"
    client.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    if not client.repo_info(repo_id=repo_id, repo_type="dataset").private:
        raise ValueError("The existing dataset is public; set it private before uploading")
    client.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(destination),
        commit_message="Add replay-validated cafe tool-call dataset",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--staging-dir", type=Path, default=Path(".cache/dataset-upload"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_data_config(args.config)
    counts = prepare_dataset(config, args.staging_dir, load_tokenizer())
    print(json.dumps(counts, sort_keys=True))
    if args.dry_run:
        print("Dry run complete; no Hub repository was created or changed.")
        return
    if (
        not config.human_min_conversations
        <= counts["human_scored"]
        <= config.human_max_conversations
    ):
        raise ValueError(
            "Write the required human-authored benchmark with the annotation CLI before publishing"
        )
    load_dotenv(".env")
    token, username = os.environ.get("HF_TOKEN"), os.environ.get("HF_USERNAME")
    if not token or not username:
        raise RuntimeError("Set HF_TOKEN and HF_USERNAME in .env before publishing")
    upload_private(args.staging_dir, username=username, token=token)
    print("Private dataset upload completed.")


if __name__ == "__main__":
    main()
