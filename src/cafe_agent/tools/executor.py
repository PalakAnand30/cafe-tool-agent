"""Interchangeable in-process and HTTP tool execution."""

from typing import Any, Protocol

import httpx

from cafe_agent.backend.service import CafeService
from cafe_agent.parsing.parser import ToolCall


class Executor(Protocol):
    def execute(self, call: ToolCall) -> dict[str, Any]: ...


class ToolExecutor:
    def __init__(self, service: CafeService) -> None:
        self.service = service

    def execute(self, call: ToolCall) -> dict[str, Any]:
        return self.service.call(call.name, call.arguments).model_dump(mode="json")


class HTTPToolExecutor:
    """The caller owns the HTTP client and its timeout/lifetime configuration."""

    def __init__(self, client: httpx.Client) -> None:
        self.client = client

    def execute(self, call: ToolCall) -> dict[str, Any]:
        response = self.client.post(f"/tools/{call.name}", json=call.arguments)
        value = response.json()
        if response.is_error and not (isinstance(value, dict) and "error" in value):
            response.raise_for_status()
        if not isinstance(value, dict):
            raise ValueError("The café API must return a JSON object")
        return value
