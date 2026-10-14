"""Validated conversation records, provenance and human gold annotations."""

import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from cafe_agent.config import StrictModel
from cafe_agent.domain.models import BackendState
from cafe_agent.models.base import Message, WireFunction, WireToolCall


class Intent(StrEnum):
    SINGLE_ORDER = "single_order"
    MULTI_ORDER = "multi_order"
    CLARIFICATION = "clarification"
    AVAILABILITY = "availability"
    ORDER_STATUS = "order_status"
    CANCEL_ORDER = "cancel_order"
    MENU = "menu"
    UNSUPPORTED = "unsupported"
    TOOL_ERROR = "tool_error"
    OUT_OF_SCOPE = "out_of_scope"


class ChatFunction(StrictModel):
    name: str
    arguments: dict[str, Any]


class ChatToolCall(StrictModel):
    id: str
    type: Literal["function"] = "function"
    function: ChatFunction


class ChatMessage(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ChatToolCall] | None = None
    tool_call_id: str | None = None

    def to_message(self) -> Message:
        return Message(
            role=self.role,
            content=self.content,
            tool_call_id=self.tool_call_id,
            tool_calls=None
            if self.tool_calls is None
            else [
                WireToolCall(
                    id=call.id,
                    function=WireFunction(
                        name=call.function.name, arguments=json.dumps(call.function.arguments)
                    ),
                )
                for call in self.tool_calls
            ],
        )


class Entity(StrictModel):
    kind: str
    value: str
    aliases: list[str] = Field(default_factory=list)


class TurnProvenance(StrictModel):
    original: str
    entities: list[Entity] = Field(default_factory=list)
    paraphrase_attempted: bool = False
    paraphrase_accepted: bool = False
    rejection_reason: str | None = None
    noise_applied: bool = False


class GoldAction(StrictModel):
    message_index: int = Field(ge=0)
    behaviour: Literal["tool", "clarify", "decline", "reply"]


class ClockAdvance(StrictModel):
    before_message: int = Field(ge=0)
    minutes: int = Field(ge=0)


class Conversation(StrictModel):
    id: str = Field(min_length=1)
    split: Literal["train", "validation", "test", "human_test"]
    intent: Intent
    template_id: str
    tools: list[dict[str, Any]]
    messages: list[ChatMessage] = Field(min_length=3)
    expected_final_state: BackendState
    scripted_user_turns: list[str] = Field(min_length=1)
    actions: list[GoldAction]
    provenance: list[TurnProvenance] = Field(default_factory=list)
    clock_advances: list[ClockAdvance] = Field(default_factory=list)
    author: Literal["synthetic", "codex_example", "human"] = "synthetic"

    @model_validator(mode="after")
    def check_alignment(self) -> "Conversation":
        users = [message.content for message in self.messages if message.role == "user"]
        if users != self.scripted_user_turns:
            raise ValueError("Scripted turns must exactly match user messages")
        if self.provenance and len(self.provenance) != len(users):
            raise ValueError("Every user turn needs provenance")
        indices = [i for i, message in enumerate(self.messages) if message.role == "assistant"]
        if [action.message_index for action in self.actions] != indices:
            raise ValueError("Annotate every assistant message exactly once, in order")
        for action in self.actions:
            has_calls = bool(self.messages[action.message_index].tool_calls)
            if (action.behaviour == "tool") != has_calls:
                raise ValueError("Tool behaviour must agree with tool_calls")
        return self

    @property
    def scoreable(self) -> bool:
        return self.author != "codex_example"
