"""Isolated bounded sessions and serialized access to one CPU model context."""

import secrets
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Protocol

from pydantic import Field

from cafe_agent.agent.loop import Agent, TraceEntry
from cafe_agent.backend.service import CafeService
from cafe_agent.config import StrictModel
from cafe_agent.domain.models import BackendState, Order
from cafe_agent.models.base import Message, ModelClient
from cafe_agent.parsing.parser import ToolCall
from cafe_agent.prompting.prompts import system_prompt
from cafe_agent.tools.executor import ToolExecutor

from .settings import ServeSettings


class ChatModel(ModelClient, Protocol):
    def reset(self) -> None: ...


class RecordingExecutor(ToolExecutor):
    def __init__(self, service: CafeService) -> None:
        super().__init__(service)
        self.trace: list[TraceEntry] = []

    def execute(self, call: ToolCall) -> dict[str, Any]:
        result = super().execute(call)
        self.trace.append(TraceEntry(call=call.model_copy(deep=True), result=result))
        return result


class SessionError(Exception):
    def __init__(self, status: int, code: str) -> None:
        self.status, self.code = status, code


class ChatRequest(StrictModel):
    session_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    message: str = Field(min_length=1)


class SessionRequest(StrictModel):
    session_id: str = Field(pattern=r"^[a-f0-9]{64}$")


class ChatResponse(StrictModel):
    session_id: str
    reply: str
    trace: list[TraceEntry]
    current_order: Order | None
    state: BackendState
    error: str | None = None


@dataclass
class Session:
    cafe: CafeService
    messages: list[Message] = field(default_factory=list)
    touched: float = field(default_factory=time.monotonic)
    turns: int = 0


class ChatService:
    def __init__(self, client: ChatModel, settings: ServeSettings) -> None:
        self.client, self.settings = client, settings
        self.sessions: dict[str, Session] = {}
        self.lock = RLock()

    def _session(self, session_id: str | None) -> tuple[str, Session]:
        now = time.monotonic()
        for key, value in list(self.sessions.items()):
            if now - value.touched > self.settings.session_ttl_seconds:
                self.sessions.pop(key).cafe.close()
        if session_id is not None:
            if session_id not in self.sessions:
                raise SessionError(404, "SESSION_EXPIRED")
            session = self.sessions[session_id]
            session.touched = now
            return session_id, session
        if len(self.sessions) >= self.settings.max_sessions:
            raise SessionError(503, "SESSION_CAPACITY")
        session_id = secrets.token_hex(32)
        session = Session(CafeService(), [Message(role="system", content=system_prompt())])
        self.sessions[session_id] = session
        return session_id, session

    def chat(self, request: ChatRequest) -> ChatResponse:
        if len(request.message) > self.settings.max_message_chars:
            raise SessionError(422, "MESSAGE_TOO_LONG")
        with self.lock:
            key, session = self._session(request.session_id)
            if session.turns >= self.settings.max_turns:
                raise SessionError(409, "TURN_LIMIT_RESET_REQUIRED")
            session.messages.append(Message(role="user", content=request.message))
            session.turns += 1
            # The shared model's KV context is never reused across customers.
            executor = RecordingExecutor(session.cafe)
            try:
                self.client.reset()
                result = Agent(self.client, executor).run(session.messages)
            except Exception:
                # Preserve any tool side effects and disable retry in a damaged session.
                session.turns = self.settings.max_turns
                state = session.cafe.snapshot()
                return self._response(
                    key,
                    state,
                    "Request interrupted. Inspect your orders and start a new session.",
                    executor.trace,
                    "MODEL_ERROR",
                )
            session.messages = result.messages
            session.touched = time.monotonic()
            if result.error:
                session.turns = self.settings.max_turns
                return self._response(
                    key,
                    session.cafe.snapshot(),
                    "Request incomplete. Inspect your orders and start a new session.",
                    result.trace,
                    result.error,
                )
            return self._response(
                key, session.cafe.snapshot(), result.reply, result.trace, result.error
            )

    @staticmethod
    def _response(
        key: str, state: BackendState, reply: str, trace: list[TraceEntry], error: str | None = None
    ) -> ChatResponse:
        return ChatResponse(
            session_id=key,
            reply=reply,
            trace=trace,
            state=state,
            current_order=state.orders[-1] if state.orders else None,
            error=error,
        )

    def advance(self, key: str) -> dict[str, Any]:
        with self.lock:
            _, session = self._session(key)
            session.cafe.advance_time(self.settings.advance_minutes)
            return session.cafe.snapshot().model_dump(mode="json")

    def reset(self, key: str) -> None:
        with self.lock:
            session = self.sessions.pop(key, None)
            if session is not None:
                session.cafe.close()

    def close(self) -> None:
        with self.lock:
            for session in self.sessions.values():
                session.cafe.close()
            self.sessions.clear()
