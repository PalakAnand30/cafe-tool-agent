"""Reports generated only from validated records and measured token lengths."""

import json
from collections import Counter
from pathlib import Path
from statistics import mean

from .schema import Conversation


def write_stats(
    output: Path,
    records: list[Conversation],
    lengths: dict[str, int],
    leakage: dict[str, int],
    journal: list[dict],
) -> dict:
    summary = {
        "conversations": len(records),
        "validated": len(lengths),
        "splits": dict(Counter(record.split for record in records)),
        "multi_turn_conversations": sum(len(record.scripted_user_turns) > 1 for record in records),
        "user_turns": sum(len(record.scripted_user_turns) for record in records),
        "noisy_turns": sum(p.noise_applied for record in records for p in record.provenance),
        "token_lengths": {
            "min": min(lengths.values()),
            "max": max(lengths.values()),
            "mean": mean(lengths.values()),
        },
        "paraphrase_attempts": len(journal),
        "paraphrase_rejections": sum(not value["accepted"] for value in journal),
        "paraphrase_rejection_reasons": dict(
            Counter(value["rejection_reason"] for value in journal if not value["accepted"])
        ),
        "leakage": leakage,
    }
    attempts = summary["paraphrase_attempts"]
    summary["paraphrase_rejection_rate"] = (
        summary["paraphrase_rejections"] / attempts if attempts else 0
    )
    lines = [
        "# Generated dataset statistics",
        "",
        "Measured from validated JSONL; no model evaluation.",
        "",
        f"Validated: {len(lengths)}/{len(records)} conversations.",
        "",
        "| Split | Intent | Conversations | Mean messages | Mean tokens | Max tokens |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ["train", "validation", "test"]:
        for intent in sorted({record.intent for record in records}):
            group = [
                record for record in records if record.split == split and record.intent == intent
            ]
            if group:
                lines.append(
                    f"| {split} | {intent} | {len(group)} | "
                    f"{mean(len(record.messages) for record in group):.2f} | "
                    f"{mean(lengths[record.id] for record in group):.2f} | "
                    f"{max(lengths[record.id] for record in group)} |"
                )
    lines.extend(
        [
            "",
            f"Multi-turn conversations: {summary['multi_turn_conversations']}.",
            f"User turns: {summary['user_turns']}; noisy turns: {summary['noisy_turns']}.",
            f"Paraphrase attempts: {attempts}; rejections: {summary['paraphrase_rejections']} "
            f"({summary['paraphrase_rejection_rate']:.2%}).",
            "",
            "Rejection reasons: "
            + json.dumps(summary["paraphrase_rejection_reasons"], sort_keys=True),
            "",
            "Leakage checks: " + json.dumps(leakage, sort_keys=True),
            "",
        ]
    )
    (output / "stats.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
