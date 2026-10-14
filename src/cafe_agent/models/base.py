"""Provider-neutral messages and synchronous generation protocol."""

from typing import Any, Literal, Protocol

from pydantic import Field

from cafe_agent.config import StrictModel


class WireFunction(StrictModel):
    name: str
    arguments: str


class WireToolCall(StrictModel):
    id: str
    type: Literal["function"] = "function"
    function: WireFunction


class Message(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[WireToolCall] | None = None
    tool_call_id: str | None = None

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class TokenUsage(StrictModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelResponse(StrictModel):
    content: str = ""
    tool_calls: list[WireToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cached: bool = False


class ModelClient(Protocol):
    def generate(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse: ...
