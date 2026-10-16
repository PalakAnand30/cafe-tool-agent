"""One GPU system/backend per process, using the frozen local evaluator."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
from pathlib import Path

from cafe_agent.config import load_runtime, read_config
from cafe_agent.data.tokenization import load_tokenizer
from cafe_agent.data.validator import validate_conversation
from cafe_agent.eval.config import load_eval_config
from cafe_agent.eval.runner import run_manifest, run_records
from cafe_agent.models.base import Message

from .clients import load_client
from .coverage import check_coverage
from .data import hub_records
from .provenance import installed_versions, package_provenance


def few_shot_messages() -> list[Message]:
    values = read_config("few_shot.yaml")["examples"]
    if len(values) != 4 or any(not row["source_id"].startswith("train-") for row in values):
        raise ValueError("S2 requires exactly four frozen training examples")
    return [Message.model_validate(message) for row in values for message in row["messages"]]


def gpu_manifest(config, runtime, output: Path, system: str, gpu: dict) -> str:
    # Reuse the same base contract without rewriting any historical S3 manifest.
    with tempfile.TemporaryDirectory() as directory:
        run_manifest(config, runtime, Path(directory), system)
        value = json.loads((Path(directory) / f"{system}__manifest.json").read_text())
    value.pop("fingerprint")
    value["gpu"] = gpu
    value["gpu_sources"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    value["few_shot"] = read_config("few_shot.yaml") if system == "S2" else None
    fingerprint = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    path = output / f"{system}__manifest.json"
    output.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("GPU run inputs/backend changed; use a new output directory")
    else:
        if list(output.glob(f"{system}__*.jsonl")):
            raise ValueError("Predictions without a matching immutable manifest")
        path.write_text(json.dumps({"fingerprint": fingerprint, **value}, indent=2) + "\n")
    return fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=["S1", "S2", "S4"], required=True)
    parser.add_argument("--backend", choices=["vllm", "transformers"], required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--merged-repo", required=True)
    parser.add_argument("--merged-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", args.merged_revision):
        parser.error("Use the exact merged model revision printed by the training notebook")
    os.environ["VLLM_USE_V1"] = "0"  # T4 uses the V0 engine.
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    config, train, kaggle = (
        load_eval_config(),
        read_config("train.yaml"),
        read_config("kaggle.yaml"),
    )
    config.output_dir = args.output
    runtime = load_runtime()
    token = os.environ["HF_TOKEN"]
    records = {}
    for split in config.datasets:
        path, group = hub_records(
            args.dataset_repo, kaggle["dataset_revision"], split, kaggle["datasets"][split], token
        )
        config.datasets[split].path = path
        records[split] = group
    tokenizer = load_tokenizer()
    for group in records.values():
        for record in group:
            validate_conversation(record, tokenizer, train["max_sequence_length"])
    model = args.merged_repo if args.system == "S4" else train["model_id"]
    revision = args.merged_revision if args.system == "S4" else train["revision"]
    config.systems[args.system].model = model
    config.systems[args.system].enabled = True
    config.systems[args.system].backend = args.backend
    examples = few_shot_messages() if args.system == "S2" else []
    settings = {
        **kaggle["inference"],
        **{name: train[name] for name in ["model_id", "revision", "seed"]},
    }
    if settings["max_tokens"] != config.max_tokens or settings["seed"] != config.seed:
        raise ValueError("GPU decoding must match the shared evaluation configuration")
    gpu = {
        "backend": args.backend,
        "model": model,
        "revision": revision,
        "settings": settings,
        "dataset_revision": kaggle["dataset_revision"],
        "package": package_provenance(),
        "context_failure_policy": (
            "No truncation/inference; record CONTEXT_LIMIT_EXCEEDED as a length failure."
        ),
        "versions": installed_versions(),
    }
    try:
        client = load_client(args.backend, model, revision, tokenizer, settings, examples)
        if args.backend == "vllm":
            gpu["versions"]["vllm"] = importlib.metadata.version("vllm")
    except Exception as exc:
        # Parent retries in a fresh process/environment only for initialization failures.
        print(json.dumps({"initialization_failed": args.backend, "error_type": type(exc).__name__}))
        raise SystemExit(42) from None
    import torch

    gpu["device"] = torch.cuda.get_device_name(0)
    gpu["compute_capability"] = list(torch.cuda.get_device_capability(0))
    gpu["cuda_runtime"] = torch.version.cuda
    fingerprint = gpu_manifest(config, runtime, args.output, args.system, gpu)
    for group in records.values():
        run_records(
            group,
            client,
            runtime,
            config,
            system=args.system,
            fingerprint=fingerprint,
            output=args.output,
        )
    results = check_coverage(config, args.output, args.system)
    if not all(group["complete"] for group in results["groups"]):
        raise RuntimeError("GPU run ended with incomplete prediction coverage")
    (args.output / f"{args.system}__coverage.json").write_text(
        json.dumps(
            {
                "system": args.system,
                "backend": args.backend,
                "complete": True,
                "model": model,
                "revision": revision,
                "groups": [
                    {
                        "split": group["split"],
                        "complete": group["complete"],
                        "predictions_sha256": group["predictions_sha256"],
                    }
                    for group in results["groups"]
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{args.system}: both modes and both splits complete ({args.backend}).")


if __name__ == "__main__":
    main()
