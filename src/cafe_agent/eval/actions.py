"""Frozen repair-free action recognition and semantic argument normalization."""

import json
import re
from collections import Counter
from typing import Any

from pydantic import Field

from cafe_agent.config import StrictModel
from cafe_agent.domain.models import BackendState
from cafe_agent.models.base import ModelResponse
from cafe_agent.parsing.parser import ParseError, ToolCall, parse_response
from cafe_agent.tools.schemas import validate_arguments

# Fixed before the live run. No LLM judge or expected-label input is used.
DECLINE = re.compile(
    r"\b(?:off[ -]menu|out of scope|unsupported|not supported|not (?:on|in) (?:our|the) menu|"
    r"(?:do not|don't|don’t|does not|doesn't|doesn’t|cannot|can't|can’t|unable to) "
    r"(?:serve|offer|provide|make|sell|do|help|assist|process|handle|support|fulfil|fulfill)|"
    r"(?:is not|isn't|isn’t|aren't|are not) (?:available|offered|supported|on (?:our|the) menu)|"
    r"(?:can|am able to) only (?:help|assist)|only (?:handle|support|assist with)|"
    r"limited to (?:café|cafe|menu|order))\b",
    re.IGNORECASE,
)
QUESTION = re.compile(
    r"\b(?:which|what|how many|could you (?:specify|clarify|provide|tell)|"
    r"please (?:specify|clarify|provide|choose)|do you (?:mean|want|prefer)|"
    r"would you like|need (?:a|the|your) (?:size|quantity|order (?:id|number)))\b",
    re.IGNORECASE,
)


class Action(StrictModel):
    behaviour: str
    calls: list[ToolCall] = Field(default_factory=list)
    tool_attempt: bool = False
    parse_valid: bool = True

    @property
    def label(self) -> str:
        if self.calls:
            return "tool:" + "+".join(sorted(call.name for call in self.calls))
        return self.behaviour


def recognize(response: ModelResponse, context_role: str, error: str | None = None) -> Action:
    text = response.content.strip()
    attempt = bool(response.tool_calls) or text.startswith(("{", "[")) or "tool_call" in text
    if error:
        return Action(
            behaviour="invalid",
            tool_attempt=attempt or error == "PROVIDER_TOOL_USE_FAILED",
            parse_valid=False,
        )
    try:
        parsed = parse_response(response)
    except ParseError:
        return Action(behaviour="invalid", tool_attempt=attempt, parse_valid=False)
    if parsed.calls:
        return Action(behaviour="tool", calls=parsed.calls, tool_attempt=True)
    if context_role == "tool":
        return Action(behaviour="reply")
    if DECLINE.search(parsed.text):
        return Action(behaviour="decline")
    if "?" in parsed.text or QUESTION.search(parsed.text):
        return Action(behaviour="clarify")
    return Action(behaviour="reply")


def normalized_arguments(call: ToolCall) -> dict[str, Any]:
    args = validate_arguments(call.name, call.arguments).model_dump(mode="json", exclude_none=True)
    if "items" in args:
        for item in args["items"]:
            item["modifiers"] = sorted(item["modifiers"])
        args["items"].sort(
            key=lambda item: (
                item["item_name"],
                item.get("size", ""),
                item["quantity"],
                item["modifiers"],
            )
        )
    return args


def canonical_calls(calls: list[ToolCall]) -> list[str]:
    return sorted(
        json.dumps({"name": call.name, "arguments": normalized_arguments(call)}, sort_keys=True)
        for call in calls
    )


def arguments_equal(predicted: list[ToolCall], expected: list[ToolCall]) -> bool:
    return canonical_calls(predicted) == canonical_calls(expected)


def field_tokens(calls: list[ToolCall]) -> dict[str, Counter]:
    fields: dict[str, Counter] = {}
    occurrences = Counter()
    for call in sorted(
        calls, key=lambda call: (call.name, json.dumps(normalized_arguments(call), sort_keys=True))
    ):
        args = normalized_arguments(call)
        call_key = (call.name, occurrences[call.name])
        occurrences[call.name] += 1
        item_occurrences = Counter()
        for item in args.get("items", []):
            name = item["item_name"]
            item_key = (*call_key, name, item_occurrences[name])
            item_occurrences[name] += 1
            for field in ["item_name", "size", "quantity", "modifiers"]:
                if field not in item:
                    continue
                values = item[field] if field == "modifiers" else [item[field]]
                for value in values:
                    fields.setdefault(field, Counter())[(*item_key, value)] += 1
        for field in ["customer_name", "order_id", "category", "item_name", "size"]:
            if field in args:
                fields.setdefault(field, Counter())[(*call_key, args[field])] += 1
    return fields


def states_equal(actual: BackendState, expected: BackendState) -> bool:
    def canonical(state: BackendState) -> dict:
        value = state.model_dump(mode="json")
        for order in value["orders"]:
            for line in order["items"]:
                line["modifiers"].sort()
            order["items"].sort(key=lambda line: json.dumps(line, sort_keys=True))
        value["orders"].sort(key=lambda order: order["order_id"])
        return value

    return canonical(actual) == canonical(expected)
