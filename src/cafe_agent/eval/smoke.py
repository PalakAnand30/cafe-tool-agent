"""Run a fixed CPU regression sample using the frozen runner and stronger endpoint."""

import argparse
import fcntl
import hashlib
import json
import random
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from cafe_agent.config import StrictModel, load_runtime
from cafe_agent.cpu.client import CPUClient
from cafe_agent.cpu.download import file_hash
from cafe_agent.cpu.evaluate import cpu_manifest
from cafe_agent.cpu.settings import local_settings
from cafe_agent.data.io import read_jsonl
from cafe_agent.data.schema import Conversation
from cafe_agent.data.tokenization import load_tokenizer

from .config import EvalConfig, load_eval_config
from .metrics import score_case
from .runner import load_records, run_records
from .schema import Prediction


class SmokeSettings(StrictModel):
    conversations: int = Field(gt=0)
    split: Literal["test"]
    min_action_and_state_success: float = Field(ge=0, le=1)


def smoke_settings(path: Path = Path("configs/eval.yaml")) -> SmokeSettings:
    """Read one explicit YAML comment pragma without changing frozen config parsing."""
    prefix = "# smoke_eval: "
    entries = [
        line.removeprefix(prefix)
        for line in path.read_text().splitlines()
        if line.startswith(prefix)
    ]
    if len(entries) != 1:
        raise ValueError("Expected exactly one smoke_eval pragma in configs/eval.yaml")
    return SmokeSettings.model_validate(yaml.safe_load(entries[0]))


def select_cases(records: list[Conversation], count: int, seed: int) -> list[Conversation]:
    """Sample complete cases independently of file order, without inspecting predictions."""
    ordered = sorted(records, key=lambda record: record.id)
    if len(ordered) < count or len({record.id for record in ordered}) != len(ordered):
        raise ValueError("Smoke sample requires enough unique cases")
    return sorted(random.Random(seed).sample(ordered, count), key=lambda record: record.id)


def score_sample(
    records: list[Conversation],
    predictions: list[Prediction],
    config: EvalConfig,
    threshold: float,
    fingerprint: str,
) -> dict:
    """Require exactly one matching executed prediction per selected conversation."""
    by_id = {prediction.conversation_id: prediction for prediction in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {record.id for record in records}:
        raise ValueError("Smoke predictions have missing, duplicate or unexpected cases")
    successes = 0
    for record in records:
        prediction = by_id[record.id]
        if (
            prediction.system != "S5"
            or prediction.mode != "end_to_end"
            or prediction.fingerprint != fingerprint
            or prediction.split != record.split
            or prediction.author != record.author
            or prediction.intent != record.intent
        ):
            raise ValueError("Smoke prediction provenance differs from its case or manifest")
        successes += int(score_case(record, [prediction], config).counts["joint_success"])
    value = successes / len(records)
    return {
        "metric": "action_and_state_success",
        "conversations": len(records),
        "successful_conversations": successes,
        "value": value,
        "threshold": threshold,
        "passed": value >= threshold,
        "case_ids": [record.id for record in records],
        "definition": (
            "Frozen action sequence plus final backend state; full scripted conversations."
        ),
        "limitation": "Synthetic regression guard; not new independent benchmark evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/smoke-cpu"))
    parser.add_argument(
        "--saved-dir", type=Path, help="Check saved S5 predictions without inference"
    )
    args = parser.parse_args()
    config, runtime, local, smoke = (
        load_eval_config(),
        load_runtime(),
        local_settings(),
        smoke_settings(),
    )
    records = select_cases(load_records(config, smoke.split), smoke.conversations, config.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selection = {
            "settings": smoke.model_dump(),
            "seed": config.seed,
            "case_ids": [record.id for record in records],
            "source_sha256": file_hash(Path(__file__)),
            "scorer_sha256": file_hash(Path(__file__).with_name("metrics.py")),
            "saved_dir": str(args.saved_dir) if args.saved_dir else None,
        }
        if args.saved_dir:
            from cafe_agent.reporting.aggregate import checked_manifest

            manifest = checked_manifest(args.saved_dir, "S5")
            if manifest["datasets"][smoke.split] != file_hash(config.datasets[smoke.split].path):
                raise ValueError("Saved predictions use a different gold dataset")
            fingerprint = manifest["fingerprint"]
            path = args.saved_dir / f"S5__{smoke.split}.jsonl"
        else:
            config.output_dir = args.output_dir
            config.modes = ["end_to_end"]
            config.systems["S5"].enabled = True
            config.systems["S5"].model = local.model.repo
            runtime.generation.max_tokens = config.max_tokens
            receipt = json.loads(Path("results/model_download.json").read_text())
            model_path = local.model.directory / local.model.filename
            if (
                receipt["sha256"] != file_hash(model_path)
                or receipt["revision"] != local.model.revision
            ):
                raise ValueError("Model differs from verified receipt")
            fingerprint = cpu_manifest(config, runtime, local, receipt, args.output_dir)
            path = args.output_dir / f"S5__{smoke.split}.jsonl"
        selection["inference_fingerprint"] = fingerprint
        selection_path = args.output_dir / "selection.json"
        if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
            raise ValueError("Smoke inputs changed; choose a new output directory")
        selection_path.write_text(json.dumps(selection, indent=2) + "\n")
        if not args.saved_dir:
            client = CPUClient(
                local, load_tokenizer(), max_tokens=config.max_tokens, model_path=model_path
            )
            try:
                client.load()
                run_records(
                    records,
                    client,
                    runtime,
                    config,
                    system="S5",
                    fingerprint=fingerprint,
                    output=args.output_dir,
                )
            finally:
                client.close()
        selected = {record.id for record in records}
        predictions = [Prediction.model_validate(row) for row in read_jsonl(path)]
        if args.saved_dir:
            predictions = [
                row
                for row in predictions
                if row.mode == "end_to_end" and row.conversation_id in selected
            ]
        summary = score_sample(
            records, predictions, config, smoke.min_action_and_state_success, fingerprint
        )
        summary.update(
            execution="saved_predictions_check" if args.saved_dir else "cpu_inference",
            predictions_sha256=file_hash(path),
            selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
            inference_fingerprint=fingerprint,
        )
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        if not summary["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
