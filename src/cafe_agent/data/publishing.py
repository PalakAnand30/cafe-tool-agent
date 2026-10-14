"""Validate and privately publish synthetic data and the independent LLM benchmark."""

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
from .independent import convert_authoring, load_independent_records
from .io import read_jsonl
from .schema import Conversation
from .splitter import check_leakage, normalise_user_text
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
    independent = load_independent_records(config.independent_path)
    if independent:
        reviewed, report = convert_authoring(
            config.independent_authoring_path, config.independent_overrides_path, tokenizer
        )
        if independent != reviewed:
            raise ValueError("Independent JSONL differs from reviewed source; reconvert it")
        benchmark_texts = {
            normalise_user_text(text)
            for record in independent
            for text in record.scripted_user_turns
        }
        if any(
            normalise_user_text(text) in benchmark_texts
            for record in records
            for text in record.scripted_user_turns
        ):
            raise ValueError("Independent benchmark overlaps synthetic user text")
    else:
        report = {"conversations": 0, "validated": 0, "human_authored": 0}
    for record in [*records, *human, *independent]:
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
    (destination / "data/independent_test.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in independent), encoding="utf-8"
    )
    (destination / "independent_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if independent:
        (destination / "provenance").mkdir(exist_ok=True)
        shutil.copyfile(
            config.independent_authoring_path, destination / "provenance/authoring.yaml"
        )
        shutil.copyfile(
            config.independent_overrides_path, destination / "provenance/gold_overrides.yaml"
        )
    scored = sum(record.scoreable for record in human)
    card = f"""---
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
  - config_name: independent_test
    data_files:
      - split: test
        path: data/independent_test.jsonl
  - config_name: human_test
    data_files:
      - split: test
        path: data/human_test.jsonl
---
# Café tool calls

Synthetic café conversations with deterministic backend-derived tool calls,
results and final state. User text is template-based with verified Groq GPT-OSS
120B paraphrases and surface noise. Assistant labels and replies are programmatic.
The default config contains train/validation/test. independent_test is a separate
evaluation config: customer messages were written by Claude, with gold annotations
reviewed by Codex for the current menu. It is not a human-written benchmark.
The legacy human_test config contains {len(human) - scored} excluded Codex worked
examples and {scored} genuine human-authored cases. Do not score the worked examples.
All five function schemas and the full system prompt are included per conversation.
Use the raw JSONL files (hf_hub_download plus the shared JSONL reader) to preserve
omitted argument fields; Arrow table inference may fill absent struct keys with null.

## Validation and leakage

Records replay against a fresh SQLite backend, validate every tool call and stay
under the user-approved 4,096-token full-conversation budget at the pinned Qwen
revision. Reserved templates and item/modifier pairs occur only in synthetic test.
Normalised user text is deduplicated across all synthetic splits.
Independent customer text has zero normalized overlap with the synthetic splits.

## Independent LLM-authored benchmark

The user approved replacing the original human-written benchmark requirement with
this independent LLM-authored set before any model evaluation. Score author=claude
records in independent_test. Preserve author=claude and annotation_provenance:
annotator=codex; never describe these as human-written conversations or evidence.
The preserved source targeted another menu. Current-domain gold overrides retain
all customer messages unchanged and record revised actions/intents and reasons.
No initial orders are seeded; unknown order IDs return ORDER_NOT_FOUND.
See independent_validation.json for measured counts and the intent distribution.
Several original order cases become declines, so this set is weighted toward
unsupported requests and provides limited coverage of successful multi-item orders.
LLM-written wording and Codex labels can introduce shared model biases. Bootstrap
intervals on this set describe case-sampling variation, not real-customer performance.
Always exclude author=codex_example. Genuine human records, if later collected,
must remain separate and be reported separately.

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
        "independent_scored": len(independent),
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
        not config.independent_min_conversations
        <= counts["independent_scored"]
        <= config.independent_max_conversations
    ):
        raise ValueError(
            "Convert the required independent LLM-authored benchmark before publishing"
        )
    load_dotenv(".env")
    token, username = os.environ.get("HF_TOKEN"), os.environ.get("HF_USERNAME")
    if not token or not username:
        raise RuntimeError("Set HF_TOKEN and HF_USERNAME in .env before publishing")
    upload_private(args.staging_dir, username=username, token=token)
    print("Private dataset upload completed.")


if __name__ == "__main__":
    main()
