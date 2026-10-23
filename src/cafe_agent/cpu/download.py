"""Download an immutable private GGUF and verify its Hub LFS content hash."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

from .settings import LocalSettings, local_settings


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_model(settings: LocalSettings) -> dict:
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    model = settings.model
    api = HfApi(token=token)
    info = api.model_info(model.repo, revision=model.revision, files_metadata=True)
    if info.sha != model.revision:
        raise ValueError("Hub returned a different model revision")
    entry = next(file for file in info.siblings if file.rfilename == model.filename)
    expected = entry.lfs.sha256 if entry.lfs else None
    if not expected:
        raise ValueError("GGUF must have a verifiable Hub LFS SHA-256")
    model.directory.mkdir(parents=True, exist_ok=True)
    path = Path(
        hf_hub_download(
            model.repo,
            model.filename,
            revision=model.revision,
            token=token,
            local_dir=model.directory,
        )
    )
    digest = file_hash(path)
    if digest != expected or path.stat().st_size != entry.size:
        raise ValueError("Downloaded GGUF hash/size differs from immutable Hub metadata")
    originals = {}
    for name in ["training_manifest.json", "training_summary.json", "README.md"]:
        downloaded = Path(
            hf_hub_download(
                model.repo, name, revision=model.revision, token=token, local_dir=model.directory
            )
        )
        originals[name] = {"path": str(downloaded), "sha256": file_hash(downloaded)}
    manifest = json.loads((model.directory / "training_manifest.json").read_text())
    if manifest["train"]["revision"] != read_base_revision():
        raise ValueError("Exported training base revision differs from the frozen tokenizer")
    receipt = {
        "repo": model.repo,
        "revision": model.revision,
        "filename": model.filename,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "hub_lfs_sha256_verified": True,
        "private_at_download": info.private,
        "originals": originals,
    }
    output = Path("results/model_download.json")
    if output.exists() and json.loads(output.read_text()) != receipt:
        raise ValueError("Model receipt changed; preserve the previous run before changing models")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def read_base_revision() -> str:
    from cafe_agent.config import read_config

    return read_config("train.yaml")["revision"]


def download_evaluation_data(settings: LocalSettings) -> dict:
    """Restore ignored gold test files on a fresh checkout without regenerating data."""
    from cafe_agent.config import read_config
    from cafe_agent.eval.config import load_eval_config

    load_dotenv()
    kaggle, config = read_config("kaggle.yaml"), load_eval_config()
    repo = settings.model.repo.split("/")[0] + "/cafe-tool-calls"
    receipt = {"repo": repo, "revision": kaggle["dataset_revision"], "files": {}}
    for split, dataset in config.datasets.items():
        expected = kaggle["datasets"][split]["sha256"]
        if not dataset.path.exists():
            cached = Path(
                hf_hub_download(
                    repo,
                    kaggle["datasets"][split]["file"],
                    repo_type="dataset",
                    revision=kaggle["dataset_revision"],
                    token=os.environ.get("HF_TOKEN"),
                    cache_dir=".cache/huggingface",
                )
            )
            if file_hash(cached) != expected:
                raise ValueError("Hub evaluation data differs from the frozen dataset hashes")
            dataset.path.parent.mkdir(parents=True, exist_ok=True)
            with dataset.path.open("xb") as handle:
                handle.write(cached.read_bytes())
        if file_hash(dataset.path) != expected:
            raise ValueError("Local evaluation data changed; preserve it before restoring gold")
        receipt["files"][split] = {"path": str(dataset.path), "sha256": expected}
    Path("results/evaluation_data_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    settings = local_settings(args.config)
    receipt = download_model(settings)
    download_evaluation_data(settings)
    print(f"Verified GGUF: {receipt['path']} ({receipt['bytes']} bytes)")


if __name__ == "__main__":
    main()
