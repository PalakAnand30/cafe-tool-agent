"""Local scoring and conversation-cluster bootstrap from shared prediction JSONL."""

import random
from collections import Counter, defaultdict
from typing import Any

from pydantic import Field

from cafe_agent.config import StrictModel
from cafe_agent.data.schema import Conversation
from cafe_agent.models.base import ModelResponse
from cafe_agent.parsing.parser import parse_response

from .actions import Action, arguments_equal, field_tokens, recognize, states_equal
from .config import EvalConfig
from .runner import user_segments
from .schema import Observation, Prediction


class CaseStatistics(StrictModel):
    counts: dict[str, float] = Field(default_factory=dict)
    latency: list[float] = Field(default_factory=list)
    cost_usd: float = 0
    confusion: dict[str, dict[str, int]] = Field(default_factory=dict)


class Metric(StrictModel):
    value: float | None
    ci95: list[float] | None = None


def gold_action(record: Conversation, index: int) -> Action:
    annotation = next(action for action in record.actions if action.message_index == index)
    message = record.messages[index].to_message()
    calls = (
        parse_response(ModelResponse(tool_calls=message.tool_calls or [])).calls
        if (message.tool_calls)
        else []
    )
    return Action(behaviour=annotation.behaviour, calls=calls, tool_attempt=bool(calls))


def _score_action(stats: CaseStatistics, actual: Action, expected: Action) -> bool:
    counts = stats.counts

    def add(key: str, value: float = 1) -> None:
        counts[key] = counts.get(key, 0) + value

    add("actions")
    correct = actual.label == expected.label
    add("correct_actions", int(correct))
    stats.confusion.setdefault(expected.label, {})[actual.label] = (
        stats.confusion.get(expected.label, {}).get(actual.label, 0) + 1
    )
    for behaviour in ["clarify", "decline"]:
        add(behaviour + "_gold", int(expected.behaviour == behaviour))
        add(behaviour + "_pred", int(actual.behaviour == behaviour))
        add(behaviour + "_tp", int(actual.behaviour == expected.behaviour == behaviour))
    if expected.calls:
        add("gold_tool_turns")
        add("correct_tool_turns", int(correct))
        exact = bool(actual.calls) and arguments_equal(actual.calls, expected.calls)
        add("arguments_exact", int(exact))
        correct = correct and exact
    elif actual.calls:
        correct = False
    actual_fields, gold_fields = field_tokens(actual.calls), field_tokens(expected.calls)
    for field in actual_fields.keys() | gold_fields.keys():
        prediction, gold = actual_fields.get(field, Counter()), gold_fields.get(field, Counter())
        tp = sum((prediction & gold).values())
        fp, fn = sum((prediction - gold).values()), sum((gold - prediction).values())
        for name in [field, "all"]:
            add(f"field:{name}:tp", tp)
            add(f"field:{name}:fp", fp)
            add(f"field:{name}:fn", fn)
    return correct


def _observe(stats: CaseStatistics, observation: Observation, config: EvalConfig) -> Action:
    action = recognize(observation.response, observation.context_role, observation.error)
    values = {
        "responses": 1,
        "parse_valid": int(action.parse_valid),
        "tool_attempts": int(action.tool_attempt),
        "valid_tool_responses": int(bool(action.calls)),
        "cached_responses": int(observation.response.cached),
        "usage_observed": int(observation.response.usage.total_tokens > 0),
    }
    for key, value in values.items():
        stats.counts[key] = stats.counts.get(key, 0) + value
    if not observation.response.cached:
        stats.latency.append(observation.latency_seconds)
    usage, pricing = observation.response.usage, config.pricing
    stats.cost_usd += float(
        (
            usage.prompt_tokens * pricing.input_per_million_usd
            + usage.completion_tokens * pricing.output_per_million_usd
        )
        / 1_000_000
    )
    return action


def score_case(
    record: Conversation,
    predictions: list[Prediction],
    config: EvalConfig,
) -> CaseStatistics:
    stats = CaseStatistics()
    for prediction in predictions:
        if prediction.mode == "turn":
            actual = _observe(stats, prediction.observations[0], config)
            _score_action(stats, actual, gold_action(record, prediction.assistant_index))
            continue
        by_user: dict[int, list[Action]] = defaultdict(list)
        for observation in prediction.observations:
            by_user[observation.user_index].append(_observe(stats, observation, config))
        all_correct = prediction.error is None
        for user_index, (start, stop) in enumerate(user_segments(record)):
            gold = [action for action in record.actions if start < action.message_index < stop]
            first = (
                by_user[user_index][0]
                if by_user[user_index]
                else Action(behaviour="invalid", parse_valid=False)
            )
            all_correct &= _score_action(stats, first, gold_action(record, gold[0].message_index))
            expected_calls = [
                call for action in gold for call in gold_action(record, action.message_index).calls
            ]
            actual_calls = [call for action in by_user[user_index] for call in action.calls]
            all_correct &= arguments_equal(actual_calls, expected_calls)
        equal = states_equal(prediction.final_state, record.expected_final_state)
        stats.counts["end_conversations"] = 1
        stats.counts["state_equal"] = int(equal)
        stats.counts["end_success"] = int(equal and prediction.error is None)
        stats.counts["joint_success"] = int(equal and all_correct)
    return stats


def combine(samples: list[CaseStatistics]) -> CaseStatistics:
    result = CaseStatistics()
    for sample in samples:
        for key, value in sample.counts.items():
            result.counts[key] = result.counts.get(key, 0) + value
        result.latency.extend(sample.latency)
        result.cost_usd += sample.cost_usd
        for expected, row in sample.confusion.items():
            for actual, count in row.items():
                result.confusion.setdefault(expected, {})[actual] = (
                    result.confusion.get(expected, {}).get(actual, 0) + count
                )
    return result


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def metric_values(stats: CaseStatistics, conversations: int) -> dict[str, float | None]:
    counts = stats.counts

    def ratio(numerator: str, denominator: str) -> float | None:
        return counts.get(numerator, 0) / counts[denominator] if counts.get(denominator) else None

    result = {
        "valid_tool_call_rate": ratio("valid_tool_responses", "tool_attempts"),
        "parse_valid_rate": ratio("parse_valid", "responses"),
        "action_accuracy": ratio("correct_actions", "actions"),
        "argument_exact_match": ratio("arguments_exact", "gold_tool_turns"),
        "argument_exact_match_given_correct_tool": ratio("arguments_exact", "correct_tool_turns"),
        "clarification_precision": ratio("clarify_tp", "clarify_pred"),
        "clarification_recall": ratio("clarify_tp", "clarify_gold"),
        "decline_precision": ratio("decline_tp", "decline_pred"),
        "decline_recall": ratio("decline_tp", "decline_gold"),
        "end_to_end_success": ratio("end_success", "end_conversations"),
        "state_equality": ratio("state_equal", "end_conversations"),
        "action_and_state_success": ratio("joint_success", "end_conversations"),
        "cache_hit_rate": ratio("cached_responses", "responses"),
        "usage_coverage": ratio("usage_observed", "responses"),
        "latency_p50_seconds": quantile(stats.latency, 0.5),
        "latency_p95_seconds": quantile(stats.latency, 0.95),
        "paid_equivalent_usd_per_1000_conversations": (
            stats.cost_usd / conversations * 1000 if conversations else None
        ),
    }
    for field in [
        "all",
        "item_name",
        "size",
        "quantity",
        "modifiers",
        "order_id",
        "category",
        "customer_name",
    ]:
        tp = counts.get(f"field:{field}:tp", 0)
        denominator = (
            2 * tp + counts.get(f"field:{field}:fp", 0) + counts.get(f"field:{field}:fn", 0)
        )
        result["argument_field_f1" if field == "all" else field + "_f1"] = (
            2 * tp / denominator if denominator else None
        )
    return result


def summarize(
    samples: list[CaseStatistics],
    config: EvalConfig,
    *,
    bootstrap: bool,
) -> dict[str, Any]:
    total = combine(samples)
    point = metric_values(total, len(samples))
    intervals: dict[str, list[float]] = defaultdict(list)
    if bootstrap and samples:
        rng = random.Random(config.seed)
        for _ in range(config.bootstrap_samples):
            selected = [rng.choice(samples) for _ in samples]
            values = metric_values(combine(selected), len(selected))
            for name, value in values.items():
                if value is not None:
                    intervals[name].append(value)
    tail = (1 - config.confidence) / 2
    return {
        "conversations": len(samples),
        "counts": total.counts,
        "uncached_latency_observations": len(total.latency),
        "metrics": {
            name: Metric(
                value=value,
                ci95=[quantile(intervals[name], tail), quantile(intervals[name], 1 - tail)]
                if intervals[name]
                else None,
            ).model_dump()
            for name, value in point.items()
        },
        "confusion_matrix": total.confusion,
    }
