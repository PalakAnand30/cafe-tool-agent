"""The loop executes valid tools, feeds results back and stops predictably."""

import json

import pytest

from cafe_agent.agent.loop import Agent
from cafe_agent.backend.service import CafeService
from cafe_agent.config import load_runtime
from cafe_agent.models.base import Message, ModelResponse, WireFunction, WireToolCall
from cafe_agent.prompting.prompts import system_prompt
from cafe_agent.tools.executor import ToolExecutor


class ScriptedClient:
    def __init__(self, *outputs: str | ModelResponse) -> None:
        self.outputs = iter(outputs)
        self.histories = []

    def generate(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        self.histories.append([m.model_copy(deep=True) for m in messages])
        assert len(tools) == 5
        value = next(self.outputs)
        return value if isinstance(value, ModelResponse) else ModelResponse(content=value)


def history(text: str = "Two medium oat lattes with extra shots, please.") -> list[Message]:
    return [Message(role="system", content=system_prompt()), Message(role="user", content=text)]


def test_order_tool_result_and_final_reply(cafe: CafeService, latte_order: dict) -> None:
    client = ScriptedClient(
        json.dumps({"name": "create_order", "arguments": latte_order}),
        "Your order ORD-00001 totals £9.00.",
    )
    initial = history()
    result = Agent(client, ToolExecutor(cafe)).run(initial)
    assert result.error is None
    assert result.reply == "Your order ORD-00001 totals £9.00."
    assert len(result.trace) == len(cafe.snapshot().orders) == 1
    assistant, tool = client.histories[1][-2:]
    assert assistant.tool_calls[0].id == tool.tool_call_id
    assert json.loads(tool.content)["order_id"] == "ORD-00001"
    assert len(initial) == 2  # Caller history is not mutated.


@pytest.mark.parametrize(
    "reply", ["Which size would you like?", "We don't serve pizza; try a sandwich."]
)
def test_clarify_and_decline_have_no_side_effects(cafe: CafeService, reply: str) -> None:
    result = Agent(ScriptedClient(reply), ToolExecutor(cafe)).run(history())
    assert result.reply == reply
    assert not result.trace and not cafe.snapshot().orders


def test_error_recovery_receives_backend_alternatives(cafe: CafeService) -> None:
    client = ScriptedClient(
        json.dumps(
            {
                "name": "create_order",
                "arguments": {
                    "items": [{"item_name": "filter coffee", "quantity": 1, "size": "small"}]
                },
            }
        ),
        "Filter coffee is out of stock. Would you like an espresso?",
    )
    result = Agent(client, ToolExecutor(cafe)).run(history())
    assert result.trace[0].result["error"]["code"] == "OUT_OF_STOCK"
    assert json.loads(client.histories[1][-1].content)["error"]["alternatives"]
    assert not cafe.snapshot().orders


def test_multiple_calls_and_native_ids(cafe: CafeService) -> None:
    response = ModelResponse(
        content="Checking both.",
        tool_calls=[
            WireToolCall(id="one", function=WireFunction(name="get_menu", arguments="{}")),
            WireToolCall(
                id="two",
                function=WireFunction(name="check_availability", arguments='{"item_name":"latte"}'),
            ),
        ],
    )
    result = Agent(ScriptedClient(response, "Latte is available."), ToolExecutor(cafe)).run(
        history()
    )
    assert len(result.trace) == 2
    assert [m.tool_call_id for m in result.messages if m.role == "tool"] == ["one", "two"]


def test_all_calls_validate_before_any_execute(cafe: CafeService, latte_order: dict) -> None:
    output = json.dumps(
        [
            {"name": "create_order", "arguments": latte_order},
            {"name": "cancel_order", "arguments": {"order_id": "bad"}},
        ]
    )
    result = Agent(ScriptedClient(output), ToolExecutor(cafe)).run(history())
    assert result.error == "PARSE_ERROR"
    assert not cafe.snapshot().orders


def test_max_rounds(cafe: CafeService) -> None:
    config = load_runtime()
    config.agent.max_rounds = 2
    call = '{"name":"get_menu","arguments":{}}'
    result = Agent(ScriptedClient(call, call), ToolExecutor(cafe), config).run(history())
    assert result.error == "MAX_ROUNDS"
    assert len(result.trace) == 2


def test_reused_native_id_stops_before_repeat_execution(
    cafe: CafeService, latte_order: dict
) -> None:
    response = ModelResponse(
        tool_calls=[
            WireToolCall(
                id="one",
                function=WireFunction(name="create_order", arguments=json.dumps(latte_order)),
            )
        ]
    )
    result = Agent(ScriptedClient(response, response), ToolExecutor(cafe)).run(history())
    assert result.error == "DUPLICATE_CALL_ID"
    assert len(cafe.snapshot().orders) == 1
