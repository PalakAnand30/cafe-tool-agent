"""Programmatic gold conversations using the real in-process backend."""

import json
from typing import Any, Literal

from cafe_agent.backend.service import CafeService
from cafe_agent.prompting.prompts import system_prompt
from cafe_agent.tools.schemas import tool_schemas

from .schema import (
    ChatFunction,
    ChatMessage,
    ChatToolCall,
    ClockAdvance,
    Conversation,
    Entity,
    GoldAction,
    Intent,
    TurnProvenance,
)


class ConversationBuilder:
    def __init__(self) -> None:
        self.service = CafeService()
        self.messages = [ChatMessage(role="system", content=system_prompt())]
        self.provenance: list[TurnProvenance] = []
        self.actions: list[GoldAction] = []
        self.clock_advances: list[ClockAdvance] = []

    def user(self, text: str, entities: list[Entity] | None = None) -> None:
        self.messages.append(ChatMessage(role="user", content=text))
        self.provenance.append(TurnProvenance(original=text, entities=entities or []))

    def say(self, text: str, behaviour: Literal["clarify", "decline", "reply"] = "reply") -> None:
        self.actions.append(GoldAction(message_index=len(self.messages), behaviour=behaviour))
        self.messages.append(ChatMessage(role="assistant", content=text))

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        call_id = f"call_{len(self.messages)}"
        self.actions.append(GoldAction(message_index=len(self.messages), behaviour="tool"))
        self.messages.append(
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ChatToolCall(id=call_id, function=ChatFunction(name=name, arguments=arguments))
                ],
            )
        )
        result = self.service.call(name, arguments).model_dump(mode="json")
        self.messages.append(
            ChatMessage(
                role="tool", content=json.dumps(result, separators=(",", ":")), tool_call_id=call_id
            )
        )
        return result

    def advance(self, minutes: int) -> None:
        self.clock_advances.append(ClockAdvance(before_message=len(self.messages), minutes=minutes))
        self.service.advance_time(minutes)

    def finish(
        self,
        *,
        record_id: str,
        split: str,
        intent: Intent,
        template_id: str,
        author: str = "synthetic",
    ) -> Conversation:
        return Conversation(
            id=record_id,
            split=split,
            intent=intent,
            template_id=template_id,
            tools=tool_schemas(),
            messages=self.messages,
            scripted_user_turns=[m.content for m in self.messages if m.role == "user"],
            expected_final_state=self.service.snapshot(),
            actions=self.actions,
            provenance=self.provenance,
            clock_advances=self.clock_advances,
            author=author,
        )

    def close(self) -> None:
        self.service.close()
