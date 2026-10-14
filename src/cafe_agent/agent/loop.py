"""Bounded model → parser → tool loop with trace and caller-owned history."""

import json
from typing import Any

from pydantic import Field

from cafe_agent.config import RuntimeConfig, StrictModel, load_runtime
from cafe_agent.models.base import Message, ModelClient, WireFunction, WireToolCall
from cafe_agent.parsing.parser import ParseError, ToolCall, parse_response
from cafe_agent.tools.executor import Executor
from cafe_agent.tools.schemas import tool_schemas


class TraceEntry(StrictModel):
    call: ToolCall
    result: dict[str, Any]


class AgentResult(StrictModel):
    reply: str
    messages: list[Message]
    trace: list[TraceEntry] = Field(default_factory=list)
    error: str | None = None


class Agent:
    def __init__(
        self, client: ModelClient, executor: Executor, config: RuntimeConfig | None = None
    ) -> None:
        self.client = client
        self.executor = executor
        self.config = config or load_runtime()

    def run(self, messages: list[Message]) -> AgentResult:
        history = [message.model_copy(deep=True) for message in messages]
        trace = []
        used_ids = {call.id for message in history for call in message.tool_calls or []}
        for round_number in range(self.config.agent.max_rounds):
            response = self.client.generate(history, tool_schemas())
            try:
                parsed = parse_response(response)
            except ParseError:
                return AgentResult(
                    reply="I couldn't process that response. Please try again.",
                    messages=history,
                    trace=trace,
                    error="PARSE_ERROR",
                )
            if not parsed.calls:
                history.append(Message(role="assistant", content=parsed.text))
                return AgentResult(reply=parsed.text, messages=history, trace=trace)
            wire_calls = []
            for index, call in enumerate(parsed.calls):
                call_id = call.id or f"call_{len(history)}_{round_number}_{index}"
                if call_id in used_ids:
                    return AgentResult(
                        reply="I couldn't process that response. Please try again.",
                        messages=history,
                        trace=trace,
                        error="DUPLICATE_CALL_ID",
                    )
                used_ids.add(call_id)
                call.id = call_id
                wire_calls.append(
                    WireToolCall(
                        id=call_id,
                        function=WireFunction(
                            name=call.name, arguments=json.dumps(call.arguments, sort_keys=True)
                        ),
                    )
                )
            history.append(
                Message(role="assistant", content=parsed.text or None, tool_calls=wire_calls)
            )
            for call in parsed.calls:
                result = self.executor.execute(call)
                trace.append(TraceEntry(call=call, result=result))
                history.append(
                    Message(
                        role="tool",
                        tool_call_id=call.id,
                        content=json.dumps(result, sort_keys=True),
                    )
                )
        return AgentResult(
            reply="I couldn't finish this request. Please try again.",
            messages=history,
            trace=trace,
            error="MAX_ROUNDS",
        )
