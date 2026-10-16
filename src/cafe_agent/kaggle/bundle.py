"""Collect complete GPU predictions without mixing backend directories."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from cafe_agent.eval.config import EvalConfig

from .coverage import check_coverage


def bundle_predictions(selection: dict[str, str], destination: Path) -> dict:
    if set(selection) != {"S1", "S2", "S4"}:
        raise ValueError("The GPU download must contain S1, S2 and S4")
    files, systems = {}, {}
    for system, directory in selection.items():
        root = Path(directory)
        manifest_path = root / f"{system}__manifest.json"
        manifest = json.loads(manifest_path.read_text())
        config = EvalConfig.model_validate(manifest["eval"])
        results = check_coverage(config, root, system)
        if not all(group["complete"] for group in results["groups"]):
            raise ValueError("Cannot label a partial GPU run complete")
        systems[system] = manifest["gpu"]
        for path in [
            manifest_path,
            *(root / f"{system}__{split}.jsonl" for split in config.datasets),
        ]:
            files[path.name] = path
    summary = {
        "complete": True,
        "systems": systems,
        "files": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(files.items())
        },
        "restore": "Extract files into the local results/predictions directory; retain S3 files.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr("gpu_evaluation_summary.json", json.dumps(summary, indent=2) + "\n")
    destination.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            bundle_predictions(json.loads(args.selection.read_text()), args.output), indent=2
        )
    )


if __name__ == "__main__":
    main()
