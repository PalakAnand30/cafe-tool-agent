"""Deterministic failure categories and fixed examples, without tuning the evaluator."""

import json
from collections import Counter
from pathlib import Path

from cafe_agent.eval.actions import arguments_equal, recognize
from cafe_agent.eval.config import load_eval_config
from cafe_agent.eval.metrics import gold_action, score_case
from cafe_agent.eval.runner import load_records
from cafe_agent.eval.schema import Prediction


def action_failure(expected, actual) -> str | None:
    if not actual.parse_valid:
        return "invalid_output"
    if expected.label != actual.label:
        return "wrong_action"
    if expected.calls and not arguments_equal(actual.calls, expected.calls):
        return "wrong_arguments"
    return None


def error_analysis(directory: Path, count: int) -> dict:
    config = load_eval_config()
    failures, categories, end_errors = [], {}, {}
    for system in ["S2", "S4"]:
        categories[system] = Counter()
        end_errors[system] = Counter()
    for split in config.datasets:
        records = {r.id: r for r in load_records(config, split, validate=False)}
        rows = {
            s: [
                Prediction.model_validate_json(line)
                for line in (directory / f"{s}__{split}.jsonl").read_text().splitlines()
            ]
            for s in ["S2", "S4"]
        }
        by_key = {s: {r.key: r for r in values} for s, values in rows.items()}
        for key in sorted(by_key["S2"], key=lambda k: (k[0], k[1], k[2] or -1)):
            pair = {s: by_key[s][key] for s in ["S2", "S4"]}
            if pair["S2"].mode == "end_to_end":
                for system, row in pair.items():
                    sample = score_case(records[row.conversation_id], [row], config)
                    label = row.error or (
                        "wrong_final_state"
                        if not sample.counts["end_success"]
                        else "required_action_failure"
                        if not sample.counts["joint_success"]
                        else "success"
                    )
                    end_errors[system][label] += 1
                continue
            record = records[key[0]]
            expected = gold_action(record, key[2])
            labels = {}
            for system, row in pair.items():
                obs = row.observations[0]
                actual = recognize(obs.response, obs.context_role, obs.error)
                labels[system] = action_failure(expected, actual)
                categories[system][labels[system] or "success"] += 1
            if any(labels.values()):
                failures.append(
                    {
                        "split": split,
                        "conversation_id": key[0],
                        "assistant_index": key[2],
                        "intent": record.intent.value,
                        "author": record.author,
                        "customer_messages": [
                            m.content for m in record.messages[: key[2]] if m.role == "user"
                        ],
                        "expected_label": expected.label,
                        "expected_arguments": [c.model_dump(mode="json") for c in expected.calls],
                        "categories": labels,
                        "responses": {
                            s: row.observations[0].response.model_dump(mode="json")
                            for s, row in pair.items()
                        },
                    }
                )
    # One per category/intent first, then earliest remaining failures. All selection is
    # deterministic after evaluation; it changes neither prompts nor scoring rules.
    selected, seen, used = [], set(), set()
    for index, row in enumerate(failures):
        key = (row["categories"]["S2"], row["categories"]["S4"], row["intent"])
        if key not in seen and len(selected) < count:
            selected.append(row)
            seen.add(key)
            used.add(index)
    for index, row in enumerate(failures):
        if len(selected) == count:
            break
        if index not in used:
            selected.append(row)
    return {
        "scope": "Category counts pool both sets; per-set metrics are reported separately.",
        "categories": categories,
        "end_to_end_categories": end_errors,
        "selection": "First sorted distinct category/category/intent, then first remaining.",
        "failure_turns": len(failures),
        "examples": selected,
    }


def write_error_analysis(path: Path, data: dict) -> None:
    lines = [
        "# S4 versus S2 failure analysis",
        "",
        "Frozen parser/action rules; no reply-factuality judge. Independent text is "
        "Claude-authored with Codex gold. Examples are illustrative, not prevalence estimates.",
        "",
        "```json",
        json.dumps({k: v for k, v in data.items() if k != "examples"}, indent=2),
        "```",
        "",
    ]
    for index, example in enumerate(data["examples"], 1):
        lines += [
            f"## Example {index}: {example['split']}/{example['conversation_id']}",
            "",
            "```json",
            json.dumps(example, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    path.write_text("\n".join(lines))
