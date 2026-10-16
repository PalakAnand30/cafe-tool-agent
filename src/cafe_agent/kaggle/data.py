"""Pinned raw Hub data and assistant labels shared with the local preflight."""

import hashlib
import math
import random
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from cafe_agent.data.io import read_jsonl
from cafe_agent.data.schema import Conversation
from cafe_agent.data.tokenization import MaskedExample, mask_assistant_tokens
from cafe_agent.prompting.prompts import system_prompt
from cafe_agent.tools.schemas import tool_schemas


def read_records(path: Path, spec: dict[str, Any], split: str) -> list[Conversation]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"Frozen {split} dataset hash mismatch")
    records = [Conversation.model_validate(row) for row in read_jsonl(path)]
    author = "claude" if split == "independent_test" else "synthetic"
    if len(records) != spec["conversations"] or len({row.id for row in records}) != len(records):
        raise ValueError(f"Incomplete or duplicate {split} records")
    for row in records:
        if row.split != split or row.author != author:
            raise ValueError("Wrong split/authorship; worked examples must never be scored")
        if row.messages[0].content != system_prompt() or row.tools != tool_schemas():
            raise ValueError("Dataset prompt/tools differ from the installed package")
    return records


def hub_records(
    repo: str, revision: str, split: str, spec: dict, token: str
) -> tuple[Path, list[Conversation]]:
    path = Path(
        hf_hub_download(repo, spec["file"], repo_type="dataset", revision=revision, token=token)
    )
    return path, read_records(path, spec, split)


def select_train(records: list[Conversation], fraction: float, seed: int) -> list[Conversation]:
    if not 0 < fraction <= 1 or any(row.split != "train" for row in records):
        raise ValueError("Training requires only train records and a fraction in (0, 1]")
    values = sorted(records, key=lambda row: row.id)
    random.Random(seed).shuffle(values)
    return values[: max(1, math.ceil(len(values) * fraction))]


def encode_records(records: list[Conversation], tokenizer: Any, limit: int) -> list[dict]:
    if any(row.split not in {"train", "validation"} for row in records):
        raise ValueError("Only train/validation may enter the training data pipeline")
    return [
        mask_assistant_tokens(
            tokenizer, [message.to_message() for message in row.messages], row.tools, limit
        ).model_dump()
        for row in records
    ]


def pad_batch(examples: list[dict], pad_token_id: int) -> dict[str, list[list[int]]]:
    """Preserve precomputed assistant labels; never reconstruct labels from IDs."""
    rows = [MaskedExample.model_validate(row) for row in examples]
    length = max(len(row.input_ids) for row in rows)
    output: dict[str, list] = {"input_ids": [], "attention_mask": [], "labels": []}
    for row in rows:
        padding = length - len(row.input_ids)
        output["input_ids"].append(row.input_ids + [pad_token_id] * padding)
        output["attention_mask"].append(row.attention_mask + [0] * padding)
        output["labels"].append(row.labels + [-100] * padding)
    return output


class AssistantCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict]) -> dict:
        import torch  # Kaggle only; local checks call pad_batch directly.

        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in pad_batch(examples, self.pad_token_id).items()
        }
