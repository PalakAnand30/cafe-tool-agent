"""Repair-free parsing of Qwen tags, bare JSON and native tool calls."""

import json
import re
from typing import Any

from pydantic import Field, ValidationError

from cafe_agent.config import StrictModel
from cafe_agent.models.base import ModelResponse
from cafe_agent.tools.schemas import validate_arguments


class ParseError(ValueError):
    """The output does not describe a valid next assistant action."""


class ToolCall(StrictModel):
    name: str
    arguments: dict[str, Any]
    id: str | None = None


class ParsedResponse(StrictModel):
    text: str = ""
    calls: list[ToolCall] = Field(default_factory=list)


def strict_json(value: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, item in items:
            if key in result:
                raise ParseError("Duplicate JSON key")
            result[key] = item
        return result

    def constant(value: str) -> None:
        raise ParseError(f"Invalid JSON constant: {value}")

    try:
        return json.loads(value, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, TypeError, RecursionError) as exc:
        raise ParseError("Malformed JSON") from exc


def _validated(value: Any, call_id: str | None = None) -> ToolCall:
    try:
        call = ToolCall.model_validate(value)
        if call.id is not None and call_id is not None and call.id != call_id:
            raise ParseError("Conflicting tool call IDs")
        validate_arguments(call.name, call.arguments)
        # Preserve omitted fields: argument exact-match should not silently add defaults.
        return call.model_copy(update={"id": call_id or call.id})
    except (ValidationError, ValueError, KeyError) as exc:
        raise ParseError("Unknown tool or schema-invalid arguments") from exc


def parse_response(response: str | ModelResponse) -> ParsedResponse:
    if isinstance(response, str):
        response = ModelResponse(content=response)
    if response.finish_reason == "length":
        raise ParseError("Generation was truncated")
    text = response.content.strip()
    if response.tool_calls:
        if "<tool_call>" in text or "</tool_call>" in text:
            raise ParseError("Native and textual tool calls cannot be combined")
        calls = [
            _validated(
                {"name": call.function.name, "arguments": strict_json(call.function.arguments)},
                call.id,
            )
            for call in response.tool_calls
        ]
        ids = [call.id for call in calls]
        if len(ids) != len(set(ids)) or any(not value for value in ids):
            raise ParseError("Native tool call IDs must be non-empty and unique")
        return ParsedResponse(text=text, calls=calls)
    if "<tool_call>" in text or "</tool_call>" in text:
        pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        matches = list(pattern.finditer(text))
        residual = pattern.sub("", text).strip()
        if not matches or "<tool_call>" in residual or "</tool_call>" in residual:
            raise ParseError("Unbalanced tool-call tags")
        calls = [_validated(strict_json(match.group(1))) for match in matches]
        return ParsedResponse(text=residual, calls=calls)
    if text.startswith(("{", "[")):
        values = strict_json(text)
        values = values if isinstance(values, list) else [values]
        if not values:
            raise ParseError("Empty tool-call array")
        return ParsedResponse(calls=[_validated(value) for value in values])
    if not text:
        raise ParseError("Empty assistant output")
    if "```" in text or "<tool_call" in text or "</tool_call" in text:
        raise ParseError("Unsupported or malformed tool-call wrapper")
    return ParsedResponse(text=text)
