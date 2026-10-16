"""CPU-only notebook checks: pinned raw data, rendering and assistant masks."""

import argparse
import json
import os
from pathlib import Path

from cafe_agent.config import read_config
from cafe_agent.data.tokenization import load_tokenizer
from cafe_agent.models.base import Message

from .clients import PromptClient
from .data import encode_records, hub_records, pad_batch, read_records
from .evaluate import few_shot_messages


def check_data(paths: dict[str, Path], *, check_eval: bool = True) -> dict:
    settings, train = read_config("kaggle.yaml"), read_config("train.yaml")
    tokenizer = load_tokenizer(local_files_only=True)
    groups = {
        split: read_records(path, settings["datasets"][split], split)
        for split, path in paths.items()
    }
    selected = groups["train"][:3]
    encoded = encode_records(selected, tokenizer, train["max_sequence_length"])
    batch = pad_batch(encoded, tokenizer.pad_token_id)
    for row, padded in zip(encoded, batch["labels"], strict=True):
        assert padded[: len(row["labels"])] == row["labels"]
        assert all(value == -100 for value in padded[len(row["labels"]) :])
    report = {
        "gpu_training_run": False,
        "gpu_inference_run": False,
        "examples": [
            {
                "id": row.id,
                "tokens": len(item["input_ids"]),
                "assistant_tokens": sum(value != -100 for value in item["labels"]),
            }
            for row, item in zip(selected, encoded, strict=True)
        ],
        "dataset_hashes_verified": sorted(groups),
    }
    if check_eval:
        prompt = PromptClient(
            tokenizer,
            settings["inference"]["context"],
            settings["inference"]["max_tokens"],
            few_shot_messages(),
        )
        longest = 0
        count = 0
        for split in ["test", "independent_test"]:
            for record in groups[split]:
                for action in record.actions:
                    _, ids = prompt.prompt(
                        [
                            message.to_message()
                            for message in record.messages[: action.message_index]
                        ],
                        record.tools,
                    )
                    longest = max(longest, len(ids))
                    count += 1
        report.update(
            s2_gold_prompts_checked=count,
            s2_max_prompt_tokens=longest,
            inference_context=settings["inference"]["context"],
        )
    assert all(isinstance(message, Message) for message in few_shot_messages())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = read_config("kaggle.yaml")
    # Training preflight reads no held-out examples. Local acceptance checks S2 as well.
    splits = ["train", "validation"] if args.dataset_repo else list(settings["datasets"])
    paths = {}
    for split in splits:
        if args.dataset_repo:
            path, _ = hub_records(
                args.dataset_repo,
                settings["dataset_revision"],
                split,
                settings["datasets"][split],
                os.environ["HF_TOKEN"],
            )
        else:
            path = (
                Path("data/human_test/independent_test.jsonl")
                if split == "independent_test"
                else Path("data/generated-independent-v1") / f"{split}.jsonl"
            )
        paths[split] = path
    load_tokenizer()  # Only tokenizer/config assets, including on Kaggle.
    result = check_data(paths, check_eval=not bool(args.dataset_repo))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
