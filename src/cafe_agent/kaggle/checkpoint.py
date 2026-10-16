"""Guard training resumes and completed-run reuse before loading model weights."""

import json
import shutil
from pathlib import Path


def prepare_run(output: Path, manifest: dict, resume: Path | None) -> bool:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "training_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Training inputs changed; use a new output directory")
    if resume:
        previous = resume.parent.parent / "training_manifest.json"
        if (
            not resume.is_dir()
            or not previous.is_file()
            or json.loads(previous.read_text()) != manifest
        ):
            raise ValueError("Restore the full matching run, including training_manifest.json")
    if (output / "training_summary.json").exists():
        if not path.is_file():
            raise ValueError("Completed output has no immutable training manifest")
        if not (output / "adapter/adapter_model.safetensors").is_file():
            raise ValueError("Completed summary has no saved adapter; restore the full output")
        return True
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return False


def restore_predictions(source: Path, destination: Path) -> None:
    """Restore backend folders while preserving newer local checkpoint prefixes."""
    paths = [
        path
        for path in source.rglob("S*__*")
        if path.is_file()
        and path.parent.name in {"vllm", "transformers"}
        and path.name.startswith(("S1__", "S2__", "S4__"))
        and (path.suffix == ".jsonl" or path.name.endswith("__manifest.json"))
    ]
    if not paths:
        raise ValueError("Select the predictions folder containing backend subdirectories")
    actions = []
    for path in paths:
        target = destination / path.parent.name / path.name
        incoming = path.read_bytes()
        if target.exists():
            current = target.read_bytes()
            if current == incoming or (path.suffix == ".jsonl" and current.startswith(incoming)):
                continue
            if path.suffix == ".jsonl" and incoming.startswith(current):
                actions.append((path, target, incoming[len(current) :]))
                continue
            raise ValueError(
                "Restored predictions/manifest differ; use a separate output directory"
            )
        else:
            actions.append((path, target, None))
    # Validate the whole restore before writing any file.
    for path, target, suffix in actions:
        target.parent.mkdir(parents=True, exist_ok=True)
        if suffix is None:
            shutil.copyfile(path, target)
        else:
            with target.open("ab") as handle:
                handle.write(suffix)
