"""Reviewable allowlisted Docker Space bundle; publication requires --deploy."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

SPACE_README = """---
title: The café counter
emoji: ☕
colorFrom: amber
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
---

# The café counter

A simulated café assistant with an isolated order board for each session.
Runs the exported Qwen2.5-1.5B Q4_K_M model on CPU. This is a research demo;
orders are simulated and discarded on reset, expiry or restart.

Set the private model's read token as the HF_TOKEN Space secret before startup.
"""


def bundle_files(root: Path) -> dict[str, bytes]:
    paths = [
        root / name
        for name in ["pyproject.toml", "uv.lock", ".dockerignore", "docker/entrypoint.sh"]
    ]
    paths += sorted((root / "src/cafe_agent").rglob("*.py"))
    for suffix in ["*.css", "*.svg"]:
        paths += sorted((root / "src/cafe_agent/serve/assets").glob(suffix))
    paths += sorted((root / "configs").glob("*.yaml"))
    paths += sorted((root / "configs").glob("*.cmake"))
    files = {str(path.relative_to(root)): path.read_bytes() for path in paths}
    files["Dockerfile"] = (root / "docker/Dockerfile").read_bytes()
    files["README.md"] = SPACE_README.encode()
    return files


def prepare_bundle(root: Path, parent: Path) -> tuple[Path, dict]:
    files = bundle_files(root)
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in sorted(files.items())}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    directory = parent / digest[:12]
    if directory.exists():
        existing = {
            str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*")
            if p.is_file()
        }
        if existing != hashes:
            raise ValueError(
                "Prepared bundle changed; preserve it and choose another output parent"
            )
    else:
        directory.mkdir(parents=True)
        for name, value in files.items():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
    return directory, {"sha256": digest, "files": hashes, "published": False}


def publish_bundle(api: HfApi, repo: str, directory: Path, token: str) -> str:
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        info = api.space_info(repo)
        if info.sdk != "docker":
            raise ValueError("Existing Space does not use Docker; no files were uploaded")
    except RepositoryNotFoundError:
        api.create_repo(repo, repo_type="space", space_sdk="docker", private=True)
    # Tokens are transmitted through the SDK, never added to uploaded files or URLs.
    api.add_space_secret(repo, key="HF_TOKEN", value=token)
    result = api.upload_folder(
        repo_id=repo,
        repo_type="space",
        folder_path=directory,
        commit_message="Deploy CPU café demo",
    )
    return result.oid
