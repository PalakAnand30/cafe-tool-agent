"""Markdown reports and README/CV excerpts, all read back from results JSON."""

import json
from pathlib import Path

from cafe_agent.eval.report import baseline_table

LIMITATIONS = (
    "The independent benchmark contains 84 Claude-authored cases with Codex gold labels, "
    "not human or real-customer evidence. It is weighted toward unsupported requests and "
    "has limited successful multi-item-order coverage. Synthetic templates and LLM wording "
    "can bias results. Intervals reflect case-sampling uncertainty, not real-world generalization. "
    "Plain-text actions use frozen approximate lexical rules; reply factuality and which "
    "detail was clarified are not judged. State-only success can overstate clarification, "
    "decline and lookup performance; action + state is the stronger endpoint. Hardware, "
    "quantization, tokenizer/runtime numerics and inference backends differ."
)


def percent(metric: dict) -> str:
    if metric["value"] is None:
        return "—"
    value = f"{metric['value']:.1%}"
    if metric.get("ci95"):
        value += f" [{metric['ci95'][0]:.1%}, {metric['ci95'][1]:.1%}]"
    return value


def write_text(metrics_path: Path, directory: Path) -> None:
    data = json.loads(metrics_path.read_text())
    table = baseline_table(data)
    groups = {(g["system"], g["split"]): g for g in data["groups"]}
    lines = [
        "# Five-system evaluation report",
        "",
        table,
        "",
        "Action and argument metrics use gold histories; end metrics execute the full agent. "
        "Brackets are 95% conversation-cluster percentile bootstrap intervals on the complete "
        "independent set (1,000 resamples, seed 42). Undefined metrics are null, displayed as —.",
        "",
        "## Execution provenance",
        "",
        "| System | Model | Backend / device |",
        "| --- | --- | --- |",
    ]
    for system in data["manifests"]:
        group = groups[(system, "independent_test")]
        execution = group["execution"]
        lines.append(
            f"| {system} | {group['model']} | {execution['backend']} / {execution['device']} |"
        )
    lines += [
        "",
        "## Observed comparisons",
        "",
    ]
    for split in ["independent_test", "test"]:
        action = {
            s: groups[(s, split)]["modes"]["turn"]["metrics"]["action_accuracy"]["value"]
            for s in ["S2", "S3", "S4"]
        }
        joint = {
            s: groups[(s, split)]["modes"]["end_to_end"]["metrics"]["action_and_state_success"][
                "value"
            ]
            for s in ["S2", "S3", "S4"]
        }
        lines += [
            f"On {split}, S4 action accuracy is {action['S4']:.1%}, versus S2 "
            f"{action['S2']:.1%} and S3 {action['S3']:.1%}. Stronger action-and-state "
            f"success is S4 {joint['S4']:.1%}, S2 {joint['S2']:.1%}, S3 {joint['S3']:.1%}.",
            "",
        ]
    lines += [
        "These are observed point comparisons; individual bootstrap intervals do not "
        "establish significance of a paired difference. Training-loss reduction alone "
        "does not establish superiority over the larger prompted model.",
        "",
    ]
    lines += [
        "",
        "S1/S2/S4 used the recorded Transformers fallback, float16 on T4. "
        "The supplied bundle does not record why vLLM was unavailable. S5 uses Q4_K_M, "
        "n_gpu_layers=0, no K/Q/V offload and a CPU-only build. All systems use the same "
        "prompt/tools, temperature 0, seed 42 and 512 generated-token limit.",
        "",
        "## Detailed metrics",
        "",
        "| System / set | Valid tool calls | Clarify P / R | Decline P / R |",
        "| --- | --- | --- | --- |",
    ]
    for group in data["groups"]:
        m = group["modes"]["turn"]["metrics"]
        lines.append(
            f"| {group['system']} / {group['split']} | {percent(m['valid_tool_call_rate'])} | "
            f"{percent(m['clarification_precision'])} / {percent(m['clarification_recall'])} | "
            f"{percent(m['decline_precision'])} / {percent(m['decline_recall'])} |"
        )
    lines += [
        "",
        "All field-level F1 values, per-intent breakdowns, confusion matrices, "
        "coverage and source hashes are in metrics.json. Figures are in charts/; "
        "error_analysis.md contains deterministic S4/S2 failure comparisons.",
        "",
        "![Independent benchmark comparison](charts/independent_comparison.png)",
        "",
        "![Independent intent accuracy](charts/per_intent_independent_test.png)",
        "",
        "![Synthetic intent accuracy](charts/per_intent_test.png)",
        "",
        "![Independent confusion matrices](charts/confusion_matrices.png)",
        "",
        "![Latency versus accuracy](charts/latency_accuracy.png)",
        "",
        "## CPU latency and cost",
        "",
    ]
    latency = data.get("cpu_latency")
    if latency and latency["complete"]:
        lines += [
            f"After warm-up, {latency['requests']} fixed requests with fresh KV context: "
            f"p50 **{latency['p50_seconds']:.3f} s**, p95 **{latency['p95_seconds']:.3f} s**. "
            + latency["definition"],
            "",
        ]
        if data.get("local_hardware"):
            hardware = data["local_hardware"]
            lines += [
                f"Measured on {hardware['cpu_brand']}, "
                f"{hardware['ram_bytes'] / 1024**3:.0f} GiB RAM, "
                f"{hardware['configured_inference_threads']} inference threads; "
                "CPU only, with no concurrent model-evaluation process.",
                "",
            ]
    else:
        lines += ["CPU latency benchmark is incomplete; no full benchmark latency claim.", ""]
    for system in ["S3", "S5"]:
        cost = groups[(system, "independent_test")]["cost"]
        value = cost.get("value", cost.get("usd_per_1000_conversations"))
        assumption = cost["assumption"]
        if system == "S3":
            lines += [
                f"S3 paid-equivalent estimate: USD {value:.4f} per 1,000 independent "
                f"end-to-end conversations. Rates checked {assumption['verified_date']}: "
                f"[Groq pricing]({assumption['source']}). {assumption['assumption']} "
                "This is not an actual bill.",
                "Missing token usage from provider errors is excluded; see usage_coverage.",
                "",
            ]
        elif value is not None:
            lines += [
                f"S5 assumed local compute: USD {value:.4f} per 1,000 independent "
                f"end-to-end conversations. {assumption}",
                "",
            ]
    lines += [
        "S1/S2/S4 API fees are zero; GPU compute cost is unknown without a supplied "
        "GPU-hour price. Latency comparisons span different hardware. S3 includes network "
        "and retries; saved request metrics exclude cache hits and explicit rate pacing.",
        "",
        "## Limitations",
        "",
        LIMITATIONS,
        "",
    ]
    training = data.get("training")
    if training:
        lines += [
            "## Training provenance",
            "",
            f"Training validation loss: "
            f"{training['summary']['initial_validation_loss']:.5f} → "
            f"{training['summary']['validation_loss']:.5f}; "
            f"{training['manifest']['batch_size']} example per device × "
            f"{training['manifest']['accumulation']} accumulation. "
            "These losses are not test accuracy. Original metadata hashes are retained "
            "in model_download.json; all test results come from prediction files.",
            "",
        ]
    (directory / "report.md").write_text("\n".join(lines))
    readme = Path("README.md")
    if readme.exists():
        content = readme.read_text()
        start, stop = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        if content.count(start) != 1 or content.count(stop) != 1:
            raise ValueError("README must contain exactly one results marker pair")
        prefix, rest = content.split(start)
        _, suffix = rest.split(stop)
        readme.write_text(
            prefix
            + start
            + "\n"
            + table
            + "\n\n"
            + LIMITATIONS
            + "\n\nSee [the full report](results/report.md) for execution, latency "
            "and cost provenance.\n\n"
            "![Independent comparison](results/charts/independent_comparison.png)\n" + stop + suffix
        )
    s4, s2 = [
        groups[(s, "independent_test")]["modes"]["turn"]["metrics"]["action_accuracy"]
        for s in ["S4", "S2"]
    ]
    s5 = groups[("S5", "independent_test")]["modes"]["end_to_end"]["metrics"][
        "action_and_state_success"
    ]
    bullets = [
        f"- Built a shared tool-calling evaluation across five systems; fine-tuned S4 achieved "
        f"{percent(s4)} action accuracy versus prompted S2 {percent(s2)} on 84 independent "
        "Claude-authored cases with Codex gold (95% conversation bootstrap intervals).",
        f"- Ran Q4_K_M inference on local CPU with zero GPU layers; measured "
        f"{percent(s5)} action-and-state task success on the same LLM-authored benchmark.",
        "- Produced reproducible schema validation, append-only prediction checkpoints, "
        "backend replay and deterministic reports on 400 synthetic and 84 independent cases "
        "per system; excluded worked examples and reported benchmark provenance limitations.",
    ]
    (directory / "cv_bullets.md").write_text(
        "# Evidence-based CV bullets\n\n" + "\n".join(bullets) + "\n"
    )
