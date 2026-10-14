"""Hand-built perfect, wrong, invalid and adversarial evaluation fixtures."""

import json
from pathlib import Path

import httpx
import pytest
from groq import Groq

from cafe_agent.config import load_runtime
from cafe_agent.data.builder import ConversationBuilder
from cafe_agent.data.io import read_jsonl
from cafe_agent.data.schema import Intent
from cafe_agent.eval.actions import arguments_equal, field_tokens, recognize
from cafe_agent.eval.config import load_eval_config
from cafe_agent.eval.inference import (
    CachedToolFailure,
    EvaluationGroqClient,
    QuotaUnavailable,
    canonical_history,
    request_key,
)
from cafe_agent.eval.metrics import CaseStatistics, metric_values, score_case, summarize
from cafe_agent.eval.report import write_results
from cafe_agent.eval.runner import end_prediction, run_manifest, run_records, turn_prediction
from cafe_agent.eval.schema import Prediction
from cafe_agent.models.base import Message, ModelResponse, TokenUsage
from cafe_agent.parsing.parser import ToolCall
from cafe_agent.tools.schemas import tool_schemas


class Scripted:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.histories = []

    def generate(self, messages, tools):
        assert tools == tool_schemas()
        self.histories.append(messages)
        return next(self.responses)


def gold_responses(record):
    return [
        ModelResponse(
            content=record.messages[action.message_index].content or "",
            tool_calls=record.messages[action.message_index].to_message().tool_calls or [],
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        for action in record.actions
    ]


def order_record(*, cancellation=False):
    builder = ConversationBuilder()
    try:
        builder.user("One medium latte, please.")
        builder.call(
            "create_order", {"items": [{"item_name": "latte", "size": "medium", "quantity": 1}]}
        )
        builder.say("Your order is received.")
        if cancellation:
            builder.advance(12)
            builder.user("Cancel ORD-00001, please.")
            builder.call("cancel_order", {"order_id": "ORD-00001"})
            builder.say("Your collected order cannot be cancelled.")
        return builder.finish(
            record_id="fixture", split="test", intent=Intent.SINGLE_ORDER, template_id="fixture"
        )
    finally:
        builder.close()


def test_perfect_turn_and_end_predictions_score_one():
    record, config = order_record(), load_eval_config()
    rows = [
        turn_prediction(
            record, action.message_index, Scripted([response]), system="S3", fingerprint="f"
        )
        for action, response in zip(record.actions, gold_responses(record), strict=True)
    ]
    values = metric_values(score_case(record, rows, config), 1)
    assert values["action_accuracy"] == values["argument_exact_match"] == 1
    assert values["argument_field_f1"] == values["valid_tool_call_rate"] == 1
    assert values["paid_equivalent_usd_per_1000_conversations"] == pytest.approx(0.009)
    end = end_prediction(
        record, Scripted(gold_responses(record)), load_runtime(), system="S3", fingerprint="f"
    )
    scores = metric_values(score_case(record, [end], config), 1)
    assert scores["end_to_end_success"] == scores["action_and_state_success"] == 1
    assert Prediction.model_validate_json(end.model_dump_json()) == end


def test_correct_tool_wrong_fields_and_invalid_json():
    record, config = order_record(), load_eval_config()
    wrong = ModelResponse(
        content=json.dumps(
            {
                "name": "create_order",
                "arguments": {
                    "items": [
                        {
                            "item_name": "latte",
                            "size": "large",
                            "quantity": 2,
                            "modifiers": ["oat_milk"],
                        }
                    ]
                },
            }
        )
    )
    row = turn_prediction(record, 2, Scripted([wrong]), system="S3", fingerprint="f")
    values = metric_values(score_case(record, [row], config), 1)
    assert values["action_accuracy"] == values["item_name_f1"] == 1
    assert values["argument_exact_match"] == values["size_f1"] == values["quantity_f1"] == 0
    assert values["modifiers_f1"] == 0
    invalid = ModelResponse(content='<tool_call>{"name":')
    row = turn_prediction(record, 2, Scripted([invalid]), system="S3", fingerprint="f")
    values = metric_values(score_case(record, [row], config), 1)
    assert values["valid_tool_call_rate"] == values["parse_valid_rate"] == 0
    assert values["action_accuracy"] == values["argument_exact_match"] == 0
    end = end_prediction(record, Scripted([invalid]), load_runtime(), system="S3", fingerprint="f")
    assert end.error == "PARSE_ERROR" and not end.final_state.orders


def test_empty_state_alone_does_not_prove_correct_decline():
    builder = ConversationBuilder()
    try:
        builder.user("Please help with my homework.")
        builder.say("I can only help with café orders.", "decline")
        record = builder.finish(
            record_id="decline", split="test", intent=Intent.OUT_OF_SCOPE, template_id="fixture"
        )
    finally:
        builder.close()
    prediction = end_prediction(
        record,
        Scripted([ModelResponse(content="Done.")]),
        load_runtime(),
        system="S3",
        fingerprint="f",
    )
    metrics = metric_values(score_case(record, [prediction], load_eval_config()), 1)
    assert metrics["end_to_end_success"] == 1
    assert metrics["action_and_state_success"] == metrics["action_accuracy"] == 0


def test_extra_orders_fail_even_when_one_order_is_correct():
    record = order_record()
    response = gold_responses(record)[0]
    second = response.tool_calls[0].model_copy(deep=True, update={"id": "extra"})
    response.tool_calls.append(second)
    prediction = end_prediction(
        record,
        Scripted([response, ModelResponse(content="Done.")]),
        load_runtime(),
        system="S3",
        fingerprint="f",
    )
    assert len(prediction.final_state.orders) == 2
    assert (
        metric_values(score_case(record, [prediction], load_eval_config()), 1)["end_to_end_success"]
        == 0
    )


def test_explicit_clock_and_cancellation_replay():
    record = order_record(cancellation=True)
    prediction = end_prediction(
        record, Scripted(gold_responses(record)), load_runtime(), system="S3", fingerprint="f"
    )
    assert prediction.final_state == record.expected_final_state
    assert prediction.final_state.minute == 12
    assert prediction.trace[-1].result["error"]["code"] == "CANNOT_CANCEL"


def test_arguments_ignore_defaults_and_order_but_preserve_bindings():
    first = ToolCall(
        name="create_order",
        arguments={
            "items": [
                {
                    "item_name": "latte",
                    "size": "large",
                    "quantity": 2,
                    "modifiers": ["oat_milk", "extra_shot"],
                },
                {"item_name": "americano", "size": "small", "quantity": 1},
            ]
        },
    )
    second = first.model_copy(deep=True)
    second.arguments["customer_name"] = None
    second.arguments["items"].reverse()
    second.arguments["items"][0]["modifiers"] = []
    second.arguments["items"][1]["modifiers"].reverse()
    assert arguments_equal([first], [second])
    assert field_tokens([first]) == field_tokens([second])
    second.arguments["items"][0]["quantity"] = 2
    second.arguments["items"][1]["quantity"] = 1
    assert not arguments_equal([first], [second])


@pytest.mark.parametrize(
    "text,role,expected",
    [
        ("Which size would you like?", "user", "clarify"),
        ("Please provide your order ID.", "user", "clarify"),
        ("We don't serve pizza. Would you like a sandwich?", "user", "decline"),
        ("I can only help with café orders.", "user", "decline"),
        ("That item is out of stock. Would you like an espresso?", "tool", "reply"),
        ("Thank you.", "user", "reply"),
        ("", "user", "invalid"),
    ],
)
def test_frozen_text_action_rules(text, role, expected):
    assert recognize(ModelResponse(content=text), role).behaviour == expected


def test_conversation_bootstrap_is_deterministic_and_handles_undefined_metrics():
    config = load_eval_config()
    samples = [CaseStatistics(counts={"actions": 1, "correct_actions": value}) for value in [0, 1]]
    first = summarize(samples, config, bootstrap=True)
    assert first == summarize(samples, config, bootstrap=True)
    assert first["metrics"]["action_accuracy"] == {"value": 0.5, "ci95": [0, 1]}
    assert first["metrics"]["valid_tool_call_rate"] == {"value": None, "ci95": None}
    perfect = summarize([samples[1]], config, bootstrap=True)
    assert perfect["metrics"]["action_accuracy"]["ci95"] == [1, 1]


def test_predictions_resume_without_requests_and_reject_stale_manifest(tmp_path):
    record, config, runtime = order_record(), load_eval_config(), load_runtime()
    config.output_dir = tmp_path
    fingerprint = run_manifest(config, runtime, tmp_path, "S3")
    responses = gold_responses(record)
    run_records(
        [record],
        Scripted([*responses, *responses]),
        runtime,
        config,
        system="S3",
        fingerprint=fingerprint,
        output=tmp_path,
    )
    path = tmp_path / "S3__test.jsonl"
    before = path.read_bytes()
    run_records(
        [record],
        Scripted([]),
        runtime,
        config,
        system="S3",
        fingerprint=fingerprint,
        output=tmp_path,
    )
    assert path.read_bytes() == before and len(read_jsonl(path)) == 3
    runtime.seed += 1
    with pytest.raises(ValueError, match="changed"):
        run_manifest(config, runtime, tmp_path, "S3")


def test_canonical_history_is_stable_and_preserves_input():
    record = order_record()
    history = [message.to_message() for message in record.messages[:-1]]
    before = [message.model_dump() for message in history]
    first = canonical_history(history)
    assert first == canonical_history(first)
    assert [message.model_dump() for message in history] == before
    assert first[2].tool_calls[0].id == first[3].tool_call_id


def test_eval_cache_key_matches_real_groq_client_and_provider_failures_cache(tmp_path):
    runtime = load_runtime()
    runtime.groq.cache_dir = tmp_path
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "bad tool",
                    "code": "tool_use_failed",
                    "failed_generation": "broken",
                }
            },
        )

    with Groq(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as sdk:
        client = EvaluationGroqClient(runtime, 60)
        client.sdk = sdk
        messages = [Message(role="user", content="Menu?")]
        with pytest.raises(CachedToolFailure) as caught:
            client.generate(messages, tool_schemas())
        assert not caught.value.cached
        with pytest.raises(CachedToolFailure) as caught:
            client.generate(messages, tool_schemas())
        assert caught.value.cached and len(calls) == 1
        assert (
            tmp_path / "eval_errors" / (request_key(runtime, messages, tool_schemas()) + ".json")
        ).exists()
        actual = json.loads(calls[0].content)
        import hashlib

        digest = hashlib.sha256(
            json.dumps(actual, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        assert digest == request_key(runtime, messages, tool_schemas())


def test_daily_quota_does_not_block_or_become_a_model_failure():
    client = EvaluationGroqClient(load_runtime(), 60)
    with pytest.raises(QuotaUnavailable) as caught:
        client._capture(httpx.Response(429, headers={"retry-after": "3600"}, content=b"quota"))
    assert caught.value.seconds == 3600


def test_sdk_wrapped_quota_is_reported_as_quota_not_connectivity(tmp_path):
    from groq import APIConnectionError

    from cafe_agent.eval.cli import provider_status

    runtime = load_runtime()
    runtime.groq.cache_dir = tmp_path
    client = EvaluationGroqClient(runtime, 60)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(429, headers={"retry-after": "3600"}, content=b"quota")
    )
    with Groq(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.Client(transport=transport, event_hooks={"response": [client._capture]}),
    ) as sdk:
        client.sdk = sdk
        with pytest.raises(APIConnectionError) as caught:
            client.generate([Message(role="user", content="Menu?")], tool_schemas())
        status = provider_status(caught.value)
        assert status["state"] == "quota_limited" and status["retry_after_seconds"] == 3600


def test_provider_failure_counts_as_invalid_tool_attempt():
    action = recognize(ModelResponse(content="broken"), "user", "PROVIDER_TOOL_USE_FAILED")
    assert action.tool_attempt and not action.parse_valid


def test_precision_recall_and_confusion_counts():
    from cafe_agent.eval.actions import Action
    from cafe_agent.eval.metrics import _score_action

    stats = CaseStatistics()
    _score_action(stats, Action(behaviour="clarify"), Action(behaviour="clarify"))
    _score_action(stats, Action(behaviour="clarify"), Action(behaviour="decline"))
    _score_action(stats, Action(behaviour="reply"), Action(behaviour="clarify"))
    values = metric_values(stats, 3)
    assert values["clarification_precision"] == values["clarification_recall"] == 0.5
    assert values["decline_recall"] == 0
    assert values["decline_precision"] is None
    assert stats.confusion["decline"]["clarify"] == 1


def test_saved_file_scoring_and_metrics_only_cli_are_offline(tmp_path, monkeypatch):
    import sys

    import yaml

    from cafe_agent.eval import cli
    from cafe_agent.eval.config import DatasetConfig

    monkeypatch.chdir(tmp_path)
    record, runtime, config = order_record(), load_runtime(), load_eval_config()
    source = tmp_path / "test.jsonl"
    source.write_text(record.model_dump_json() + "\n")
    config.datasets = {"test": DatasetConfig(path=source, author="synthetic", conversations=1)}
    config.output_dir = tmp_path / "predictions"
    config.metrics_path = tmp_path / "results/metrics.json"
    config.report_path = tmp_path / "results/baseline.md"
    Path("README.md").write_text("Intro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\n")
    fingerprint = run_manifest(config, runtime, config.output_dir, "S3")
    responses = gold_responses(record)
    run_records(
        [record],
        Scripted([*responses, *responses]),
        runtime,
        config,
        system="S3",
        fingerprint=fingerprint,
        output=config.output_dir,
    )
    result = write_results(config, config.output_dir, "S3")
    assert result["groups"][0]["complete"]
    assert "100.0%" in Path("README.md").read_text()
    assert result["groups"][0]["modes"]["turn"]["prediction_records"] == 2

    def forbid_network(*args, **kwargs):
        raise AssertionError("metrics-only must not construct a model client")

    monkeypatch.setattr(cli, "EvaluationGroqClient", forbid_network)
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--config", str(config_path), "--metrics-only"]
    )
    cli.main()
    path = config.output_dir / "S3__test.jsonl"
    before = path.read_bytes()
    path.write_bytes(before + before.splitlines(keepends=True)[0])
    with pytest.raises(ValueError, match="Duplicate"):
        write_results(config, config.output_dir, "S3")
    path.write_bytes(before)
    source.write_text(
        source.read_text().replace('"template_id":"fixture"', '"template_id":"changed"')
    )
    with pytest.raises(ValueError, match="Dataset changed"):
        write_results(config, config.output_dir, "S3")
