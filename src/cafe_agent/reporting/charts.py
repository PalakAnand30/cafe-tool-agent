"""Standalone PNG/SVG scientific figures generated exclusively from results artifacts."""

import json
import math
import os
from collections import Counter
from pathlib import Path


def action_label(label: str) -> str:
    if not label.startswith("tool:"):
        return label
    aliases = {
        "get_menu": "menu",
        "check_availability": "availability",
        "create_order": "create",
        "get_order_status": "status",
        "cancel_order": "cancel",
    }
    counts = Counter(label.removeprefix("tool:").split("+"))
    return "+".join(
        aliases.get(name, name) + (f"×{count}" if count > 1 else "")
        for name, count in sorted(counts.items())
    )


def make_charts(metrics_path: Path, directory: Path) -> list[str]:
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.hashsalt": "cafe-agent-seed42", "font.size": 9})
    data = json.loads(metrics_path.read_text())
    systems = list(data["manifests"])
    groups = {(g["system"], g["split"]): g for g in data["groups"]}
    directory.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig, name):
        for suffix in ["png", "svg"]:
            path = directory / f"{name}.{suffix}"
            metadata = {"Date": None} if suffix == "svg" else {"Software": "cafe-agent"}
            fig.savefig(path, dpi=160, bbox_inches="tight", metadata=metadata)
            written.append(str(path))
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, mode, metric, title in zip(
        axes,
        ["turn", "turn", "end_to_end"],
        ["action_accuracy", "argument_exact_match", "action_and_state_success"],
        ["Action accuracy", "Argument exact match", "Action + state success"],
        strict=True,
    ):
        values = [
            groups[(s, "independent_test")]["modes"][mode]["metrics"][metric] for s in systems
        ]
        heights = [v["value"] if v["value"] is not None else math.nan for v in values]
        lower = [max(0, v["value"] - v["ci95"][0]) if v["ci95"] else 0 for v in values]
        upper = [max(0, v["ci95"][1] - v["value"]) if v["ci95"] else 0 for v in values]
        ax.bar(
            systems,
            heights,
            yerr=[lower, upper],
            capsize=4,
            color=["#94a3b8", "#64748b", "#a78bfa", "#0f766e", "#38bdf8"],
        )
        ax.set(title=title, ylim=(0, 1), ylabel="Fraction correct")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Independent Claude-authored benchmark (84 cases, Codex gold)\n"
        "95% conversation-cluster bootstrap intervals; not human evidence",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, "independent_comparison")

    for split in ["independent_test", "test"]:
        intents = sorted(
            {
                intent
                for s in systems
                for intent in groups[(s, split)]["modes"]["turn"]["per_intent"]
            }
        )
        matrix = [
            [
                groups[(s, split)]["modes"]["turn"]["per_intent"]
                .get(intent, {})
                .get("metrics", {})
                .get("action_accuracy", {})
                .get("value")
                for s in systems
            ]
            for intent in intents
        ]
        matrix = [[math.nan if v is None else v for v in row] for row in matrix]
        fig, ax = plt.subplots(figsize=(8, 6))
        intent_groups = groups[(systems[0], split)]["modes"]["turn"]["per_intent"]
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set(
            xticks=range(len(systems)),
            xticklabels=systems,
            yticks=range(len(intents)),
            yticklabels=[
                f"{intent} (n={intent_groups[intent]['conversations']})" for intent in intents
            ],
            title=f"Gold-history action accuracy: {split} (n = conversations)",
        )
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(
                    j,
                    i,
                    "—" if math.isnan(value) else f"{value:.0%}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.6 else "black",
                )
        fig.colorbar(image, ax=ax)
        save(fig, f"per_intent_{split}")

    matrices = [
        groups[(s, "independent_test")]["modes"]["turn"]["confusion_matrix"] for s in systems
    ]
    labels = sorted(
        {k for m in matrices for k in m} | {k for m in matrices for row in m.values() for k in row}
    )
    vmax = max((v for m in matrices for row in m.values() for v in row.values()), default=1)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout="constrained")
    for ax, system, matrix in zip(axes.flat, systems, matrices, strict=False):
        values = [[matrix.get(gold, {}).get(actual, 0) for actual in labels] for gold in labels]
        image = ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax)
        ax.set(
            xticks=range(len(labels)),
            xticklabels=[action_label(label) for label in labels],
            yticks=range(len(labels)),
            yticklabels=[action_label(label) for label in labels],
            title=system,
            xlabel="Predicted",
            ylabel="Gold",
        )
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                if value:
                    ax.text(
                        j,
                        i,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if value > vmax / 2 else "black",
                    )
    axes.flat[-1].axis("off")
    fig.colorbar(image, ax=list(axes.flat[:-1]), shrink=0.7)
    fig.suptitle(
        "Independent benchmark action confusion (counts; shared scale)\n"
        "Tool labels shortened; ×N retains repeated calls. Full labels are in metrics.json."
    )
    save(fig, "confusion_matrices")

    fig, ax = plt.subplots(figsize=(8, 5))
    for system in systems:
        group = groups[(system, "independent_test")]
        latency = group["modes"]["end_to_end"]["metrics"]["latency_p50_seconds"]["value"]
        accuracy = group["modes"]["turn"]["metrics"]["action_accuracy"]["value"]
        if latency is not None and latency > 0 and accuracy is not None:
            ax.scatter(latency, accuracy, s=70)
            ax.annotate(system, (latency, accuracy), xytext=(5, 5), textcoords="offset points")
    ax.set(
        xscale="log",
        ylim=(0, 1),
        xlabel="Median generation request seconds (log scale)",
        ylabel="Gold-history action accuracy",
        title="Independent benchmark: latency versus accuracy",
    )
    ax.grid(alpha=0.2)
    fig.text(
        0.02,
        0.01,
        "Hardware differs: S1/S2/S4 Kaggle T4; S3 includes network; S5 local CPU.\n"
        "Evaluation permits prompt-prefix reuse; cold-context S5 benchmark is separate.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "latency_accuracy")
    return written
