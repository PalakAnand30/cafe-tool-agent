"""Warm-up and 100 fixed CPU requests, with fresh KV context and append-only resume."""

import fcntl
import hashlib
import json
import random
import time
from pathlib import Path

from cafe_agent.cpu.client import CPUClient
from cafe_agent.cpu.download import file_hash
from cafe_agent.cpu.settings import local_settings
from cafe_agent.data.io import append_jsonl, read_jsonl
from cafe_agent.data.tokenization import load_tokenizer

from .config import load_eval_config
from .metrics import quantile
from .runner import load_records


def benchmark_cases(records, count: int, seed: int):
    candidates = [(r, a) for r in sorted(records, key=lambda r: r.id) for a in r.actions]
    if len(candidates) < count:
        raise ValueError("Not enough fixed benchmark prompts")
    return random.Random(seed).sample(candidates, count)


def summarize_latency(rows: list[dict], expected: int) -> dict:
    values = [row["seconds"] for row in rows]
    return {
        "complete": len(rows) == expected,
        "requests": len(rows),
        "expected_requests": expected,
        "p50_seconds": quantile(values, 0.5),
        "p95_seconds": quantile(values, 0.95),
        "definition": "CPU request wall time: rendering, token checks, prefill and generation; "
        "fresh KV context per request; excludes model load, warm-up and explicit reset.",
        "n_gpu_layers": 0,
    }


def main() -> None:
    root = local_settings().report.directory
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".cpu_latency.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_benchmark()


def run_benchmark() -> None:
    settings, config = local_settings(), load_eval_config()
    if not settings.latency.reset_context_each_request:
        raise ValueError("The CPU benchmark requires a fresh KV context for each request")
    receipt = json.loads(Path("results/model_download.json").read_text())
    if file_hash(settings.model.directory / settings.model.filename) != receipt["sha256"]:
        raise ValueError("Latency GGUF differs from the verified model receipt")
    import llama_cpp

    s5_manifest = json.loads((settings.output_directory / "S5__manifest.json").read_text())
    libraries = {
        p.name: file_hash(p)
        for p in sorted((Path(llama_cpp.__file__).parent / "lib").glob("*"))
        if p.suffix in {".dylib", ".so", ".dll"}
    }
    if (
        llama_cpp.llama_supports_gpu_offload()
        or libraries != s5_manifest["cpu"]["native_libraries"]
        or settings.model_dump(mode="json") != s5_manifest["cpu"]["settings"]
    ):
        raise ValueError("Latency runtime differs from the recorded CPU-only S5 run")
    records = load_records(config, settings.latency.split, validate=False)
    selected = benchmark_cases(records, settings.latency.measured_requests, settings.seed)
    root = settings.report.directory
    manifest = {
        "settings": settings.latency.model_dump(),
        "seed": settings.seed,
        "s5_manifest_sha256": file_hash(settings.output_directory / "S5__manifest.json"),
        "source_sha256": file_hash(Path(__file__)),
        "cases": [[r.id, a.message_index] for r, a in selected],
        "warmup": "Excluded; repeated at each resumed process start.",
    }
    manifest_path = root / "cpu_latency_manifest.json"
    path = root / "cpu_latency.jsonl"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Latency inputs/code changed; retain old output and choose a new run")
    else:
        if path.exists():
            raise ValueError("Latency samples have no matching immutable manifest")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    rows = read_jsonl(path, repair_tail=True)
    done = {row["index"] for row in rows}
    if len(done) != len(rows) or any(i not in range(len(selected)) for i in done):
        raise ValueError("Duplicate/unexpected latency sample")
    for row in rows:
        r, a = selected[row["index"]]
        if [row["conversation_id"], row["assistant_index"]] != [r.id, a.message_index]:
            raise ValueError("Latency sample differs from the fixed request list")
    client = CPUClient(
        settings,
        load_tokenizer(),
        max_tokens=config.max_tokens,
        model_path=settings.model.directory / settings.model.filename,
    )
    try:
        if len(done) < len(selected):
            client.load()
            for i in range(settings.latency.warmup_requests):
                r, a = selected[i % len(selected)]
                client.reset()
                client.generate([m.to_message() for m in r.messages[: a.message_index]], r.tools)
            for index, (record, action) in enumerate(selected):
                if index in done:
                    continue
                client.reset()
                started = time.perf_counter()
                response = client.generate(
                    [m.to_message() for m in record.messages[: action.message_index]], record.tools
                )
                seconds = time.perf_counter() - started
                row = {
                    "index": index,
                    "conversation_id": record.id,
                    "assistant_index": action.message_index,
                    "seconds": seconds,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "finish_reason": response.finish_reason,
                    "response_sha256": hashlib.sha256(
                        response.model_dump_json().encode()
                    ).hexdigest(),
                }
                append_jsonl(path, row)
                rows.append(row)
                if (index + 1) % config.progress_every == 0:
                    print(f"CPU latency: {index + 1}/{len(selected)} requests", flush=True)
    finally:
        client.close()
        result = summarize_latency(rows, len(selected))
        result["manifest_sha256"] = file_hash(manifest_path)
        (root / "cpu_latency.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
