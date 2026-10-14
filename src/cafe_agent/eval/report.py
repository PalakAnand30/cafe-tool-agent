"""Recompute metrics, provenance and the first baseline table from saved files."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from cafe_agent.data.io import read_jsonl

from .config import EvalConfig, SystemID
from .metrics import score_case, summarize
from .runner import load_records
from .schema import Prediction


def compute_results(config: EvalConfig, output: Path, system: SystemID) -> dict[str, Any]:
    groups = []
    for split, dataset in config.datasets.items():
        records = load_records(config, split, validate=False)
        by_id = {record.id: record for record in records}
        path = output / f"{system}__{split}.jsonl"
        predictions = [Prediction.model_validate(value) for value in read_jsonl(path)]
        expected_keys = {
            (record.id, mode, action.message_index if action else None)
            for record in records
            for mode in config.modes
            for action in (record.actions if mode == "turn" else [None])
        }
        if len({prediction.key for prediction in predictions}) != len(predictions):
            raise ValueError("Duplicate predictions would bias metrics")
        if any(
            prediction.key not in expected_keys
            or prediction.system != system
            or prediction.split != split
            for prediction in predictions
        ):
            raise ValueError("Unexpected prediction key/system/split")
        manifest_path = output / f"{system}__manifest.json"
        if predictions:
            manifest = json.loads(manifest_path.read_text())
            if hashlib.sha256(dataset.path.read_bytes()).hexdigest() != manifest["datasets"][split]:
                raise ValueError("Dataset changed after prediction generation")
            if any(prediction.fingerprint != manifest["fingerprint"] for prediction in predictions):
                raise ValueError("Mixed prediction fingerprints")
            if any(
                prediction.intent != by_id[prediction.conversation_id].intent
                or prediction.author != by_id[prediction.conversation_id].author
                for prediction in predictions
            ):
                raise ValueError("Prediction intent/authorship differs from source")
        modes = {}
        for mode in config.modes:
            selected = [prediction for prediction in predictions if prediction.mode == mode]
            expected = {key for key in expected_keys if key[1] == mode}
            complete = {prediction.key for prediction in selected} == expected
            rows = defaultdict(list)
            for prediction in selected:
                rows[prediction.conversation_id].append(prediction)
            samples, per_intent = [], defaultdict(list)
            for record_id, values in sorted(rows.items()):
                sample = score_case(by_id[record_id], values, config)
                samples.append(sample)
                per_intent[by_id[record_id].intent.value].append(sample)
            modes[mode] = {
                "complete": complete,
                "prediction_records": len(selected),
                "expected_prediction_records": len(expected),
                **summarize(samples, config, bootstrap=split == "independent_test" and complete),
                "per_intent": {
                    intent: summarize(values, config, bootstrap=False)
                    for intent, values in sorted(per_intent.items())
                },
            }
        groups.append(
            {
                "system": system,
                "model": config.systems[system].model,
                "split": split,
                "author": dataset.author,
                "expected_conversations": dataset.conversations,
                "complete": all(mode["complete"] for mode in modes.values()),
                "modes": modes,
                "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                if path.exists()
                else None,
                "dataset_sha256": hashlib.sha256(dataset.path.read_bytes()).hexdigest(),
            }
        )
    return {
        "schema_version": 1,
        "groups": groups,
        "bootstrap": {
            "unit": "conversation",
            "resamples": config.bootstrap_samples,
            "confidence": config.confidence,
            "seed": config.seed,
        },
        "pricing": config.pricing.model_dump(mode="json"),
        "scorer_sha256": hashlib.sha256(
            (Path(__file__).parent / "metrics.py").read_bytes()
        ).hexdigest(),
        "definitions": {
            "action_accuracy": (
                "Exact tool-name/multiplicity or clarify/decline/reply label; no LLM judge."
            ),
            "text_actions": (
                "Frozen lexical rules; plain replies following tool results count as reply. "
                "Text correctness is not judged."
            ),
            "argument_exact_match": (
                "Joint exact normalized arguments on every gold tool turn; "
                "missing/wrong tool is incorrect."
            ),
            "normalization": (
                "Schema defaults/nulls normalized; modifier/item/call order ignored; "
                "duplicates, quantities and bindings preserved."
            ),
            "field_f1": (
                "Micro multiset F1, aligned by tool and item-name occurrence; includes "
                "item/size/quantity/modifiers plus IDs/category/customer name."
            ),
            "valid_tool_call_rate": (
                "Schema-valid parsed tool responses / all attempted tool responses, "
                "including malformed and provider-rejected attempts."
            ),
            "end_to_end_success": (
                "Final backend state equals gold and the agent terminates without "
                "an execution/protocol error."
            ),
            "action_and_state_success": (
                "Additionally requires correct first action/arguments at every user turn "
                "and no missing/extra tool calls."
            ),
            "state_equality": (
                "Line/modifier ordering ignored; all orders, IDs, prices, quantities, "
                "customer names, statuses and logical times retained."
            ),
            "latency": (
                "Uncached request wall time including network/retries; deliberate quota "
                "pacing and local cache hits excluded. Not S5 CPU latency."
            ),
            "cost": (
                "Cold paid-equivalent cost, not an actual bill. "
                "Missing provider-error usage is excluded; see usage_coverage."
            ),
            "confidence_intervals": (
                "Conversation-cluster percentile bootstrap on complete independent_test only. "
                "Undefined resample metrics omitted; undefined point metrics are null."
            ),
            "coverage": (
                "Independent cases are Claude-authored with Codex gold and weighted "
                "toward declines; not human-written evidence."
            ),
        },
    }


def _percent(mode: dict, name: str) -> str:
    metric = mode["metrics"][name]
    if metric["value"] is None:
        return "—"
    text = f"{metric['value']:.1%}"
    if metric["ci95"]:
        text += f" [{metric['ci95'][0]:.1%}, {metric['ci95'][1]:.1%}]"
    return text


def baseline_table(results: dict) -> str:
    lines = [
        "| System | Set | Coverage (turn / end) | Action accuracy | Argument exact | "
        "Field F1 | End success | Action + state |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for group in results["groups"]:
        turn, end = group["modes"]["turn"], group["modes"]["end_to_end"]
        coverage = (
            f"{turn['prediction_records']}/{turn['expected_prediction_records']} turns; "
            f"{end['conversations']}/{group['expected_conversations']} conversations"
        )
        state = "complete" if group["complete"] else "**partial**"
        lines.append(
            f"| {group['system']} | {group['split']} ({state}) | {coverage} | "
            f"{_percent(turn, 'action_accuracy')} | {_percent(turn, 'argument_exact_match')} | "
            f"{_percent(turn, 'argument_field_f1')} | {_percent(end, 'end_to_end_success')} | "
            f"{_percent(end, 'action_and_state_success')} |"
        )
    return "\n".join(lines)


def write_results(config: EvalConfig, output: Path, system: SystemID) -> dict:
    results = compute_results(config, output, system)
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_path.write_text(json.dumps(results, indent=2) + "\n")
    # Re-read the artifact: README/table values always originate in results/.
    results = json.loads(config.metrics_path.read_text())
    coverage = {
        group["split"]: {
            "source_conversations": group["expected_conversations"],
            "turn_predictions": group["modes"]["turn"]["prediction_records"],
            "expected_turn_predictions": group["modes"]["turn"]["expected_prediction_records"],
            "end_predictions": group["modes"]["end_to_end"]["prediction_records"],
            "complete": group["complete"],
        }
        for group in results["groups"]
    }
    config.metrics_path.with_name("eval_coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n"
    )
    table = baseline_table(results)
    config.report_path.write_text(
        "# S3 baseline\n\n" + table + "\n\n"
        "Bracketed intervals are 95% conversation-cluster bootstrap intervals on the "
        "complete independent LLM-authored set. Partial runs have no confidence intervals. "
        "Action/argument scores use gold-history turns; end scores use the real agent.\n\n"
        "Final-state success can overstate clarification/decline/lookup performance because "
        "these tasks may leave the backend empty. The stricter action + state column also "
        "checks user-turn actions and missing/extra tool calls. Text actions use frozen "
        "lexical rules and do not verify reply factuality, tone, or which detail was clarified.\n\n"
        "Customer text in independent_test was authored by Claude and labelled by Codex. "
        "The set is weighted toward unsupported requests and has limited successful "
        "multi-item-order coverage. These results are not human or real-customer evidence.\n\n"
        "metrics.json contains field scores, precision/recall, per-intent breakdowns, "
        "confusion matrices, uncached request latency (including network), cost assumptions, "
        "coverage and provenance. Cached requests are excluded from latency.\n\n"
        f"Paid-equivalent prices checked {results['pricing']['verified_date']}: "
        f"[Groq model pricing]({results['pricing']['source']}). "
        "This estimate excludes input-cache discounts and is not an actual bill. "
        "S1, S2, S4 and S5 have not been evaluated in Phase 3.\n"
    )
    readme = Path("README.md")
    if readme.exists():
        content = readme.read_text()
        begin, finish = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        if begin in content and finish in content:
            prefix, rest = content.split(begin, 1)
            _, suffix = rest.split(finish, 1)
            readme.write_text(
                prefix + begin + "\n" + table + "\n\n"
                "S3 uses GPT-OSS 120B through Groq. Brackets show 95% intervals on the "
                "complete independent LLM-authored benchmark; partial runs are labelled. "
                "End success uses final state; action + state also checks required actions. "
                "Other systems have not run. See results/S3_baseline.md for definitions "
                "and limitations.\n" + finish + suffix
            )
    return results
