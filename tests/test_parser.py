"""Hand-written valid and adversarial model outputs; no JSON repair allowed."""

import json

import pytest

from cafe_agent.models.base import ModelResponse, WireFunction, WireToolCall
from cafe_agent.parsing.parser import ParseError, parse_response

MENU = '{"name":"get_menu","arguments":{"category":"coffee"}}'
STATUS = '{"name":"get_order_status","arguments":{"order_id":"ORD-00001"}}'


@pytest.mark.parametrize("text", [MENU, f"<tool_call>{MENU}</tool_call>", f"[{MENU}]"])
def test_single_call_formats(text: str) -> None:
    result = parse_response(text)
    assert len(result.calls) == 1
    assert result.calls[0].name == "get_menu"
    assert result.calls[0].arguments == {"category": "coffee"}


@pytest.mark.parametrize(
    "text",
    [
        f"<tool_call>{MENU}</tool_call>\n<tool_call>{STATUS}</tool_call>",
        f"[{MENU},{STATUS}]",
    ],
)
def test_multiple_calls(text: str) -> None:
    assert [call.name for call in parse_response(text).calls] == ["get_menu", "get_order_status"]


def test_text_and_tool_call_mixed() -> None:
    parsed = parse_response(f"I'll check.\n<tool_call>{MENU}</tool_call>\nOne moment.")
    assert parsed.text == "I'll check.\n\nOne moment."
    assert parsed.calls[0].name == "get_menu"


@pytest.mark.parametrize(
    "text", ["Which size would you like?", "We don't serve pizza; try a sandwich."]
)
def test_plain_text(text: str) -> None:
    assert parse_response(text).text == text
    assert not parse_response(text).calls


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "[]",
        "{broken}",
        "{'name':'get_menu','arguments':{}}",
        '<tool_call>{"name":"get_menu","arguments":{},}</tool_call>',
        '<tool_call>{"name":"get_menu","arguments":{}}',
        f"{MENU}</tool_call>",
        f"<tool_call><tool_call>{MENU}</tool_call></tool_call>",
        f"<tool_call>{MENU}</tool_call><tool_call>",
        '<tool_call>{"name":"get_menu","arguments":{}} trailing</tool_call>',
        '{"name":"get_menu","arguments":{},"name":"cancel_order"}',
        '{"name":"get_menu","arguments":{"category":"coffee","category":"tea"}}',
        '{"name":"get_menu","arguments":{"category":NaN}}',
        '{"name":"get_menu","arguments":{"category":Infinity}}',
        '{"name":"refund","arguments":{}}',
        '{"name":"get_menu","arguments":null}',
        '{"name":"get_menu","arguments":"{}"}',
        '{"name":"get_menu","arguments":[]}',
        '{"name":"get_menu","arguments":{"extra":1}}',
        '{"name":"get_order_status","arguments":{"order_id":"oops"}}',
        '{"name":"create_order","arguments":{"items":[{"item_name":"latte","quantity":true}]}}',
        '{"name":"get_menu","arguments":{},"unexpected":0}',
        f"```json\n{MENU}\n```",
        '<tool_call name="get_menu">{}</tool_call>',
    ],
)
def test_rejects_malformed_or_schema_invalid_outputs(text: str) -> None:
    with pytest.raises(ParseError):
        parse_response(text)


def test_native_calls_with_string_arguments_and_ids() -> None:
    response = ModelResponse(
        content="Checking.",
        tool_calls=[
            WireToolCall(
                id="a", function=WireFunction(name="get_menu", arguments='{"category":"tea"}')
            ),
            WireToolCall(
                id="b",
                function=WireFunction(
                    name="get_order_status", arguments='{"order_id":"ORD-00001"}'
                ),
            ),
        ],
    )
    parsed = parse_response(response)
    assert parsed.text == "Checking."
    assert [call.id for call in parsed.calls] == ["a", "b"]
    assert parsed.calls[0].arguments == {"category": "tea"}


@pytest.mark.parametrize("mode", ["bad_json", "duplicate_id", "empty_id", "mixed", "truncated"])
def test_invalid_native_responses(mode: str) -> None:
    call = WireToolCall(id="a", function=WireFunction(name="get_menu", arguments="{}"))
    response = ModelResponse(tool_calls=[call])
    if mode == "bad_json":
        call.function.arguments = "{bad}"
    elif mode == "duplicate_id":
        response.tool_calls.append(call.model_copy(deep=True))
    elif mode == "empty_id":
        call.id = ""
    elif mode == "mixed":
        response.content = f"<tool_call>{MENU}</tool_call>"
    else:
        response.finish_reason = "length"
    with pytest.raises(ParseError):
        parse_response(response)


def test_preserves_omitted_optional_arguments() -> None:
    assert (
        parse_response(json.dumps({"name": "get_menu", "arguments": {}})).calls[0].arguments == {}
    )
