"""Resumable S5 evaluation with immutable model/source/runtime provenance."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from cafe_agent.config import load_runtime
from cafe_agent.data.tokenization import load_tokenizer
from cafe_agent.eval.config import load_eval_config
from cafe_agent.eval.runner import load_records, run_manifest, run_records
from cafe_agent.kaggle.coverage import check_coverage

from .client import CPUClient
from .download import file_hash
from .settings import local_settings


def cpu_manifest(config, runtime, settings, model: dict, output: Path) -> str:
    import tempfile

    import llama_cpp

    if llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError("Install the CPU-only build using scripts/setup_cpu.py first")
    with tempfile.TemporaryDirectory() as directory:
        run_manifest(config, runtime, Path(directory), "S5")
        value = json.loads((Path(directory) / "S5__manifest.json").read_text())
    value.pop("fingerprint")
    value["cpu"] = {
        "settings": settings.model_dump(mode="json"),
        "n_gpu_layers": 0,
        "offload_kqv": False,
        "gpu_offload_supported_by_build": False,
        "native_libraries": {
            path.name: file_hash(path)
            for path in sorted((Path(llama_cpp.__file__).parent / "lib").glob("*"))
            if path.suffix in {".dylib", ".so", ".dll"}
        },
        "build_hook_sha256": file_hash(Path("configs/cpu_only.cmake")),
        "model": model,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["llama-cpp-python", "transformers", "huggingface-hub"]
        },
        "generation": {
            "temperature": 0,
            "max_tokens": config.max_tokens,
            "seed": settings.seed,
            "repeat_penalty": 1.0,
            "stop": ["<|im_end|>"],
        },
        "context_policy": "Reject overflow; verify GGUF/HF prompt token equality; never truncate.",
    }
    value["cpu_sources"] = {
        name: file_hash(Path(__file__).parent / name)
        for name in ["__init__.py", "client.py", "evaluate.py", "settings.py"]
    }
    fingerprint = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    path = output / "S5__manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != {"fingerprint": fingerprint, **value}:
            raise ValueError("S5 inference inputs changed; use a new output directory")
    else:
        if list(output.glob("S5__*.jsonl")):
            raise ValueError("S5 predictions have no immutable manifest")
        path.write_text(json.dumps({"fingerprint": fingerprint, **value}, indent=2) + "\n")
    return fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    settings, config, runtime = local_settings(), load_eval_config(), load_runtime()
    config.output_dir = settings.output_directory
    if args.output_dir:
        config.output_dir = args.output_dir
    if config.seed != settings.seed or config.temperature != 0:
        raise ValueError("CPU seed and greedy decoding must match the shared evaluator")
    receipt = json.loads(Path("results/model_download.json").read_text())
    model_path = settings.model.directory / settings.model.filename
    if receipt["sha256"] != file_hash(model_path) or receipt["revision"] != settings.model.revision:
        raise ValueError("GGUF differs from the verified download receipt")
    config.systems["S5"].enabled = True
    config.systems["S5"].model = settings.model.repo
    runtime.generation.max_tokens = config.max_tokens
    config.output_dir.mkdir(parents=True, exist_ok=True)
    records = {split: load_records(config, split) for split in config.datasets}
    with (config.output_dir / ".s5.run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fingerprint = cpu_manifest(config, runtime, settings, receipt, config.output_dir)
        client = CPUClient(
            settings, load_tokenizer(), max_tokens=config.max_tokens, model_path=model_path
        )
        status_path = config.output_dir / "S5__status.json"
        status = {"state": "running", "fingerprint": fingerprint, "n_gpu_layers": 0}
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        try:
            client.load()
            for group in records.values():
                run_records(
                    group,
                    client,
                    runtime,
                    config,
                    system="S5",
                    fingerprint=fingerprint,
                    output=config.output_dir,
                )
            coverage = check_coverage(config, config.output_dir, "S5")
            status.update(state="complete", coverage=coverage)
        except BaseException as exc:
            status.update(
                state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error_type=type(exc).__name__,
            )
            raise
        finally:
            client.close()
            status_path.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
