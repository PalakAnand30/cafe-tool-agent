"""Regenerate the full Phase 5 report offline from complete saved predictions."""

import json
import math
from pathlib import Path

from cafe_agent.cpu.download import file_hash
from cafe_agent.cpu.settings import local_settings
from cafe_agent.eval.latency import summarize_latency

from .aggregate import compute_comparison
from .charts import make_charts
from .errors import error_analysis, write_error_analysis
from .text import write_text


def verified_latency(root: Path, s5_manifest: Path) -> dict:
    manifest_path = root / "cpu_latency_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["s5_manifest_sha256"] != file_hash(s5_manifest):
        raise ValueError("Latency benchmark belongs to a different S5 run")
    source = Path(__file__).resolve().parents[1] / "eval/latency.py"
    if manifest["source_sha256"] != file_hash(source):
        raise ValueError("Latency benchmark code changed")
    rows = [json.loads(line) for line in (root / "cpu_latency.jsonl").read_text().splitlines()]
    expected = len(manifest["cases"])
    if (
        expected < 100
        or {row["index"] for row in rows} != set(range(expected))
        or len(rows) != expected
    ):
        raise ValueError("CPU latency benchmark needs 100 complete, unique requests")
    for row in rows:
        if [row["conversation_id"], row["assistant_index"]] != manifest["cases"][row["index"]]:
            raise ValueError("Latency sample does not match its frozen prompt")
        if not math.isfinite(row["seconds"]) or row["seconds"] < 0 or row["prompt_tokens"] <= 0:
            raise ValueError("Invalid CPU latency observation")
    result = summarize_latency(rows, expected)
    result["manifest_sha256"] = file_hash(manifest_path)
    if result != json.loads((root / "cpu_latency.json").read_text()):
        raise ValueError("CPU latency summary differs from its request samples")
    return result


def training_metadata(root: Path) -> dict:
    receipt = json.loads((root / "model_download.json").read_text())
    destination = root / "training"
    destination.mkdir(exist_ok=True)
    values = {}
    for key in ["manifest", "summary"]:
        name = f"training_{key}.json"
        target = destination / name
        source = Path(receipt["originals"][name]["path"])
        chosen = target if target.exists() else source
        if file_hash(chosen) != receipt["originals"][name]["sha256"]:
            raise ValueError("Original training metadata changed")
        if not target.exists():
            target.write_bytes(chosen.read_bytes())
        values[key] = json.loads(target.read_text())
    return values


def main() -> None:
    settings = local_settings()
    root = settings.report.directory
    results = compute_comparison(settings)
    results["cpu_latency"] = verified_latency(root, settings.output_directory / "S5__manifest.json")
    results["training"] = training_metadata(root)
    hardware_path = root / "local_hardware.json"
    if hardware_path.exists():
        results["local_hardware"] = json.loads(hardware_path.read_text())
    inputs = {}
    for group in results["groups"]:
        key = f"{group['system']}__{group['split']}"
        inputs[key] = {
            "predictions_sha256": group["predictions_sha256"],
            "dataset_sha256": group["dataset_sha256"],
        }
    results["inputs"] = inputs
    metrics_path = root / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    # Every human-facing artifact reads the persisted JSON, never hand-entered scores.
    analysis = error_analysis(Path("results/predictions"), settings.report.representative_failures)
    analysis_path = root / "error_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    write_error_analysis(root / "error_analysis.md", json.loads(analysis_path.read_text()))
    write_text(metrics_path, root)
    figures = make_charts(metrics_path, settings.report.charts_directory)
    receipt = {
        "all_systems_complete": results["all_complete"],
        "metrics_sha256": file_hash(metrics_path),
        "prediction_inputs": inputs,
        "manifests": results["manifests"],
        "latency_manifest_sha256": file_hash(root / "cpu_latency_manifest.json"),
        "hardware_sha256": file_hash(hardware_path) if hardware_path.exists() else None,
        "artifacts": {
            str(path): file_hash(path)
            for path in [
                root / "report.md",
                root / "cv_bullets.md",
                root / "error_analysis.md",
                root / "error_analysis.json",
                *map(Path, figures),
            ]
        },
    }
    (root / "report_validation.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(f"Complete five-system report: {root / 'report.md'}; {len(figures)} chart files.")


if __name__ == "__main__":
    main()
