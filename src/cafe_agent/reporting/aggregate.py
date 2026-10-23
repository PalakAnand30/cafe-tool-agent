"""Apply the frozen scorer to every system, with hardware-aware cost provenance."""

import hashlib
import json
import random
from decimal import Decimal
from pathlib import Path

from cafe_agent.config import load_runtime, read_config
from cafe_agent.cpu.download import file_hash
from cafe_agent.cpu.settings import LocalSettings
from cafe_agent.eval.config import EvalConfig, load_eval_config
from cafe_agent.eval.metrics import quantile
from cafe_agent.eval.report import compute_results
from cafe_agent.eval.schema import Prediction
from cafe_agent.prompting.prompts import system_prompt
from cafe_agent.tools.schemas import tool_schemas


def prediction_directory(settings: LocalSettings, system: str) -> Path:
    return settings.output_directory if system == "S5" else load_eval_config().output_dir


def checked_manifest(directory: Path, system: str) -> dict:
    manifest = json.loads((directory / f"{system}__manifest.json").read_text())
    value = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if (
        hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        != manifest["fingerprint"]
    ):
        raise ValueError(f"Invalid {system} manifest fingerprint")
    if manifest["system"] != system:
        raise ValueError("Manifest system mismatch")
    expected = {
        "prompt": system_prompt(),
        "tools": tool_schemas(),
        "domain": read_config("domain.yaml"),
        "runtime": load_runtime().model_dump(mode="json"),
    }
    for key, value in expected.items():
        if key in manifest and manifest[key] != value:
            raise ValueError(f"Frozen {system} {key} differs from the shared evaluator")
    if system == "S2" and manifest.get("few_shot") != read_config("few_shot.yaml"):
        raise ValueError("S2 few-shot examples differ from the frozen training examples")
    root = Path(__file__).resolve().parents[1]
    for name, digest in manifest["sources"].items():
        if file_hash(root / name) != digest:
            raise ValueError(f"Frozen inference source changed: {name}")
    for key, subdirectory in [("gpu_sources", "kaggle"), ("cpu_sources", "cpu")]:
        for name, digest in manifest.get(key, {}).items():
            if file_hash(root / subdirectory / name) != digest:
                raise ValueError(f"Frozen {system} source changed: {name}")
    return manifest


def local_cost(predictions: list[Prediction], settings: LocalSettings, *, bootstrap: bool) -> dict:
    rows = [p for p in predictions if p.mode == "end_to_end"]
    values = [sum(o.latency_seconds for o in p.observations) for p in rows]
    factor = settings.cost.usd_per_cpu_hour * 1000 / 3600
    point = sum(values) / len(values) * factor if values else None
    samples = []
    if bootstrap and values:
        rng = random.Random(settings.seed)
        for _ in range(load_eval_config().bootstrap_samples):
            samples.append(sum(rng.choices(values, k=len(values))) / len(values) * factor)
    return {
        "usd_per_1000_conversations": point,
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)] if samples else None,
        "assumption": settings.cost.assumption,
        "basis": "Sum of observed generation request times per end-to-end conversation.",
    }


def compute_comparison(settings: LocalSettings, *, require_complete: bool = True) -> dict:
    local_config = load_eval_config()
    groups, manifests = [], {}
    definitions = None
    for system in settings.report.systems:
        directory = prediction_directory(settings, system)
        manifest = checked_manifest(directory, system)
        config = EvalConfig.model_validate(manifest["eval"])
        for split in config.datasets:
            config.datasets[split].path = local_config.datasets[split].path
        # The shared scorer's token pricing applies only to the Groq API.
        if system != "S3":
            config.pricing.input_per_million_usd = Decimal("0")
            config.pricing.output_per_million_usd = Decimal("0")
        result = compute_results(config, directory, system)
        definitions = result["definitions"]
        manifests[system] = {
            "path": str(directory / f"{system}__manifest.json"),
            "sha256": file_hash(directory / f"{system}__manifest.json"),
            "fingerprint": manifest["fingerprint"],
        }
        for group in result["groups"]:
            if require_complete and not group["complete"]:
                raise ValueError(f"{system}/{group['split']} is incomplete; retain and resume it")
            if system in {"S1", "S2", "S4"}:
                gpu = manifest["gpu"]
                group["execution"] = {
                    "backend": gpu["backend"],
                    "device": gpu["device"],
                    "revision": gpu["revision"],
                    "precision": "float16",
                    "latency_basis": "Kaggle GPU generation request wall time; no network API.",
                }
                group["cost"] = {
                    "usd_per_1000_conversations": None,
                    "api_fee_usd": 0,
                    "assumption": "Kaggle GPU quota; no billing or GPU-hour price supplied.",
                }
            elif system == "S3":
                group["execution"] = {
                    "backend": "groq",
                    "device": "remote API",
                    "latency_basis": "Includes network/retries; excludes pacing and cache hits.",
                }
                group["cost"] = {
                    **group["modes"]["end_to_end"]["metrics"][
                        "paid_equivalent_usd_per_1000_conversations"
                    ],
                    "assumption": config.pricing.model_dump(mode="json"),
                }
            else:
                group["execution"] = {
                    "backend": "llama.cpp",
                    "device": manifest["cpu"]["machine"],
                    "revision": manifest["cpu"]["model"]["revision"],
                    "n_gpu_layers": 0,
                    "precision": "GGUF Q4_K_M",
                    "latency_basis": (
                        "CPU request wall time; evaluation may reuse prompt prefixes. "
                        "The separate 100-request benchmark resets KV context each time."
                    ),
                }
                path = directory / f"{system}__{group['split']}.jsonl"
                rows = [
                    Prediction.model_validate_json(line) for line in path.read_text().splitlines()
                ]
                group["cost"] = local_cost(
                    rows,
                    settings,
                    bootstrap=(group["split"] == "independent_test" and group["complete"]),
                )
            if system != "S3":
                for mode in group["modes"].values():
                    for part in [mode, *mode["per_intent"].values()]:
                        part["metrics"]["paid_equivalent_usd_per_1000_conversations"] = {
                            "value": None,
                            "ci95": None,
                        }
            groups.append(group)
    return {
        "schema_version": 2,
        "groups": groups,
        "manifests": manifests,
        "bootstrap": result["bootstrap"],
        "scorer_sha256": result["scorer_sha256"],
        "definitions": {
            **definitions,
            "latency": "See each group's execution.latency_basis.",
            "cost": "Groq paid-equivalent token estimate only for S3; "
            "S5 has an explicit assumed CPU-hour cost; GPU cost is unknown.",
        },
        "all_complete": all(g["complete"] for g in groups),
    }
