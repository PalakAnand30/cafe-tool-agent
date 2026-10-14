"""Schema, protocol, real-backend replay and full-token-budget validation."""

from collections import defaultdict
from typing import Any

from cafe_agent.backend.service import CafeService
from cafe_agent.models.base import ModelResponse
from cafe_agent.parsing.parser import parse_response, strict_json
from cafe_agent.prompting.prompts import system_prompt
from cafe_agent.tools.schemas import tool_schemas

from .schema import Conversation
from .tokenization import mask_assistant_tokens


def validate_conversation(record: Conversation, tokenizer: Any, max_tokens: int) -> int:
    record = Conversation.model_validate(record.model_dump())
    if record.tools != tool_schemas():
        raise ValueError("Tool schemas differ from the shared registry")
    if record.messages[0].role != "system" or record.messages[0].content != system_prompt():
        raise ValueError("System prompt differs from the shared prompt")
    pending: dict[str, dict] = {}
    seen_ids: set[str] = set()
    advances: dict[int, int] = defaultdict(int)
    for step in record.clock_advances:
        if step.before_message > len(record.messages):
            raise ValueError("Clock advance is outside the conversation")
        advances[step.before_message] += step.minutes
    service = CafeService()
    try:
        for index, message in enumerate(record.messages):
            if index in advances:
                service.advance_time(advances[index])
            if message.role == "system" and index != 0:
                raise ValueError("Unexpected system message")
            if message.role in {"assistant", "user"} and pending:
                raise ValueError("Unanswered tool calls before the next message")
            if message.role == "assistant":
                native = message.to_message()
                parsed = parse_response(
                    ModelResponse(content=native.content or "", tool_calls=native.tool_calls or [])
                )
                for call in parsed.calls:
                    if call.id is None or call.id in seen_ids:
                        raise ValueError("Tool call IDs must be unique")
                    seen_ids.add(call.id)
                    pending[call.id] = service.call(call.name, call.arguments).model_dump(
                        mode="json"
                    )
            elif message.role == "tool":
                if message.tool_call_id not in pending:
                    raise ValueError("Tool result has no matching call")
                expected = pending.pop(message.tool_call_id)
                if strict_json(message.content or "") != expected:
                    raise ValueError("Stored tool result does not match backend replay")
        if pending:
            raise ValueError("Missing tool result")
        if len(record.messages) in advances:
            service.advance_time(advances[len(record.messages)])
        if service.snapshot() != record.expected_final_state:
            raise ValueError("Final state does not match backend replay")
    finally:
        service.close()
    masked = mask_assistant_tokens(
        tokenizer, [m.to_message() for m in record.messages], record.tools, max_tokens
    )
    return len(masked.input_ids)
