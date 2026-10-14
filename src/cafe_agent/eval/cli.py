"""Run the S3 baseline or recompute all metrics without network/model access."""

import argparse
import fcntl
import json
from pathlib import Path

from groq import APIConnectionError, APIStatusError

from cafe_agent.config import load_runtime

from .config import load_eval_config
from .inference import EvaluationGroqClient, QuotaUnavailable
from .report import write_results
from .runner import load_records, run_manifest, run_records


def provider_status(exc: Exception) -> dict:
    cause = exc
    while cause is not None:
        if isinstance(cause, QuotaUnavailable):
            return {
                "state": "quota_limited",
                "retry_after_seconds": cause.seconds,
                "rate_headers": cause.headers,
            }
        cause = cause.__cause__
    return {
        "state": "provider_unavailable",
        "error_type": type(exc).__name__,
        "http_status": getattr(exc, "status_code", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--limit", type=int, help="Smoke run only; partial coverage is labelled")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    config = load_eval_config(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.metrics_only:
        write_results(config, config.output_dir, "S3")
        print("Metrics recomputed locally from prediction files.")
        return
    runtime = load_runtime()
    runtime.generation.temperature = config.temperature
    runtime.generation.max_tokens = config.max_tokens
    runtime.groq.model = config.systems["S3"].model
    if not config.systems["S3"].enabled:
        raise ValueError("S3 is disabled in the evaluation configuration")
    # Validate every gold record before requesting any model prediction.
    datasets = {split: load_records(config, split) for split in config.datasets}
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with (config.output_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another evaluator is writing this output directory") from None
        fingerprint = run_manifest(config, runtime, config.output_dir, "S3")
        client = EvaluationGroqClient(runtime, config.max_quota_wait_seconds)
        status = {"state": "running", "system": "S3", "fingerprint": fingerprint}
        Path("results/eval_status.json").write_text(json.dumps(status, indent=2) + "\n")
        failed = False
        try:
            for records in datasets.values():
                selected = records[: args.limit] if args.limit else records
                run_records(
                    selected,
                    client,
                    runtime,
                    config,
                    system="S3",
                    fingerprint=fingerprint,
                    output=config.output_dir,
                )
        except QuotaUnavailable as exc:
            failed = True
            status.update(
                state="quota_limited", retry_after_seconds=exc.seconds, rate_headers=exc.headers
            )
            print(
                "Groq quota requires a longer wait. Progress is saved; rerun to resume.", flush=True
            )
        except (APIConnectionError, APIStatusError) as exc:
            failed = True
            status.update(provider_status(exc))
            print(
                "Groq request could not complete. Progress is saved; rerun to resume.", flush=True
            )
        except KeyboardInterrupt:
            failed = True
            status.update(state="interrupted")
            print("Stopped; completed predictions and API cache are preserved.", flush=True)
        finally:
            client.close()
            results = write_results(config, config.output_dir, "S3")
            complete = all(group["complete"] for group in results["groups"])
            if not failed:
                status["state"] = "complete" if complete else "partial"
            status["complete"] = complete
            Path("results/eval_status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(f"Evaluation status: {status['state']}. Metrics: {config.metrics_path}", flush=True)
        if failed:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
