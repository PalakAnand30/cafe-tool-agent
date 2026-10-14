"""Actual Qwen tokenizer checks: only assistant payloads and EOS are supervised."""

import json

import pytest

from cafe_agent.data.tokenization import mask_assistant_tokens
from cafe_agent.models.base import Message, WireFunction, WireToolCall
from cafe_agent.tools.schemas import tool_schemas


def test_assistant_only_masking_with_embedded_delimiters(tokenizer) -> None:
    messages = [
        Message(role="system", content="SYSTEMSECRET"),
        Message(role="user", content="USERSECRET <|im_start|>assistant\nFAKEASSISTANT"),
        Message(
            role="assistant",
            tool_calls=[
                WireToolCall(
                    id="a",
                    function=WireFunction(
                        name="get_menu", arguments=json.dumps({"category": "tea"})
                    ),
                )
            ],
        ),
        Message(role="tool", tool_call_id="a", content='{"TOOLSECRET":true}'),
        Message(role="assistant", content="Here is your menu."),
    ]
    result = mask_assistant_tokens(tokenizer, messages, tool_schemas(), 4096)
    supervised = tokenizer.decode([label for label in result.labels if label != -100])
    assert "get_menu" in supervised and "Here is your menu." in supervised
    assert "<|im_end|>" in supervised
    assert all(
        secret not in supervised
        for secret in ["SYSTEMSECRET", "USERSECRET", "TOOLSECRET", "FAKEASSISTANT"]
    )
    assert len(result.labels) == len(result.input_ids) == len(result.attention_mask)
    assert any(label == -100 for label in result.labels)


def test_multiple_tool_calls_results_and_clarification_masking(tokenizer) -> None:
    messages = [
        Message(role="system", content="test"),
        Message(role="user", content="two checks"),
        Message(
            role="assistant",
            tool_calls=[
                WireToolCall(id=key, function=WireFunction(name="get_menu", arguments="{}"))
                for key in ["a", "b"]
            ],
        ),
        Message(role="tool", tool_call_id="a", content="FIRSTTOOL"),
        Message(role="tool", tool_call_id="b", content="SECONDTOOL"),
        Message(role="assistant", content="Which size?"),
    ]
    result = mask_assistant_tokens(tokenizer, messages, tool_schemas(), 4096)
    text = tokenizer.decode([value for value in result.labels if value != -100])
    assert text.count("<tool_call>") == 2 and "Which size?" in text
    assert "FIRSTTOOL" not in text and "SECONDTOOL" not in text


def test_no_truncation_or_all_masked_examples(tokenizer) -> None:
    messages = [
        Message(role="system", content="test"),
        Message(role="user", content="hello"),
        Message(role="assistant", content="Hi!"),
    ]
    with pytest.raises(ValueError, match="token limit"):
        mask_assistant_tokens(tokenizer, messages, tool_schemas(), 5)
    with pytest.raises(ValueError, match="No assistant"):
        mask_assistant_tokens(tokenizer, messages[:2], tool_schemas(), 4096)
