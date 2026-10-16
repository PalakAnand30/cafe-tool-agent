"""Kaggle CPU merge/conversion and private Hub artifact publication."""

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

from cafe_agent.config import read_config
from cafe_agent.data.tokenization import load_tokenizer

from .settings import train_settings


def model_card(kind: str, summary: dict, train: dict, dataset: str) -> str:
    return (
        "---\nlibrary_name: "
        + ("peft" if kind == "adapter" else "transformers" if kind == "merged" else "gguf")
        + f"\nlicense: apache-2.0\nbase_model: {train['model_id']}\n"
        + "tags: [cafe, tool-calling, qlora]\n---\n# Café ordering assistant\n\n"
        + f"Artifact: {kind}. Base revision: `{train['revision']}`. "
        + f"Training data: {dataset}, revision `{summary['dataset_revision']}`.\n\n"
        + f"W&B run: {summary['wandb_url']}. Training summary accompanies this card.\n\n"
        + "QLoRA uses NF4 double quantization, float16 compute, the seven configured "
        + "attention/MLP projections and assistant-only labels including closing tokens. "
        + "See training_manifest.json for the actual rank, optimizer, batch and seed. "
        + "The merged model is fp16; GGUF uses Q4_K_M from the recorded llama.cpp revision.\n\n"
        + "Synthetic training does not establish real-customer performance. The separate "
        + "Claude-authored benchmark has Codex gold labels, is weighted toward declines, "
        + "and is not human-written evidence. No evaluation claim is made by this card. "
        + "Use the shared prompt, schemas and strict parser from cafe-tool-agent.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--dataset-repo", required=True)
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    train, config = train_settings(), read_config("kaggle.yaml")
    summary = json.loads((args.output / "training_summary.json").read_text())
    token = os.environ["HF_TOKEN"]
    base = AutoModelForCausalLM.from_pretrained(
        train["model_id"],
        revision=train["revision"],
        token=token,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
        trust_remote_code=False,
    )
    merged = PeftModel.from_pretrained(base, args.output / "adapter").merge_and_unload()
    merged.config.use_cache = True
    merged_path = args.output / "merged"
    merged.save_pretrained(merged_path, safe_serialization=True, max_shard_size="2GB")
    load_tokenizer().save_pretrained(merged_path)
    del merged, base
    gc.collect()
    source = Path("/tmp/cafe-llama-cpp")
    if not source.exists():
        subprocess.run(["git", "init", str(source)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "remote",
                "add",
                "origin",
                "https://github.com/ggml-org/llama.cpp.git",
            ],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(source), "fetch", "--depth", "1", "origin", config["llama_cpp_revision"]],
        check=True,
    )
    subprocess.run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != config["llama_cpp_revision"]:
        raise ValueError("llama.cpp revision mismatch")
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(source / "build"),
            "-DGGML_CUDA=OFF",
            "-DLLAMA_CURL=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(source / "build"), "--target", "llama-quantize", "-j", "2"],
        check=True,
    )
    folder = args.output / "gguf"
    folder.mkdir(exist_ok=True)
    fp16 = args.output / "intermediate-f16.gguf"
    quantized = folder / "cafe-agent-qwen2.5-1.5b-Q4_K_M.gguf"
    cpu_env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run(
        [
            sys.executable,
            str(source / "convert_hf_to_gguf.py"),
            str(merged_path),
            "--outfile",
            str(fp16),
            "--outtype",
            "f16",
        ],
        env=cpu_env,
        check=True,
    )
    subprocess.run(
        [str(source / "build/bin/llama-quantize"), str(fp16), str(quantized), "Q4_K_M"],
        env=cpu_env,
        check=True,
    )
    if not quantized.is_file() or quantized.stat().st_size == 0:
        raise RuntimeError("GGUF conversion did not produce an artifact")
    fp16.unlink()  # This invocation's temporary conversion file, not a user artifact.
    api = HfApi(token=token)
    published = {}
    for kind in ["adapter", "merged", "gguf"]:
        path = args.output / kind
        (path / "README.md").write_text(model_card(kind, summary, train, args.dataset_repo))
        for name in ["training_summary.json", "training_manifest.json"]:
            (path / name).write_bytes((args.output / name).read_bytes())
        repo = args.namespace + "/" + config["hub_suffixes"][kind]
        api.create_repo(repo, private=True, exist_ok=True)
        if not api.repo_info(repo).private:
            raise ValueError("Keep model repositories private until final publication")
        commit = api.upload_folder(
            repo_id=repo, folder_path=str(path), commit_message=f"Add café {kind} artifact"
        )
        published[kind] = {
            "repo": repo,
            "revision": commit.oid,
            "url": "https://huggingface.co/" + repo,
        }
    summary["hub"] = published
    summary["llama_cpp_revision"] = actual
    (args.output / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
