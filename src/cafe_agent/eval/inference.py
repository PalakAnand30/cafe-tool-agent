"""Canonical histories, response recording and account-aware cached Groq requests."""

import hashlib
import json
import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv
from groq import APIStatusError, Groq

from cafe_agent.config import RuntimeConfig
from cafe_agent.models.base import Message, ModelClient, ModelResponse
from cafe_agent.models.groq import GroqClient
from cafe_agent.parsing.parser import ToolCall, strict_json

from .actions import normalized_arguments
from .schema import Observation


class QuotaUnavailable(RuntimeError):
    def __init__(self, seconds: float, headers: dict[str, str]) -> None:
        super().__init__("Groq quota requires a longer wait; saved predictions can be resumed")
        self.seconds, self.headers = seconds, headers


class ProviderToolFailure(RuntimeError):
    """A provider-rejected model generation counts as an invalid output."""


def canonical_history(messages: list[Message]) -> list[Message]:
    history = [message.model_copy(deep=True) for message in messages]
    ids = {}
    for index, message in enumerate(history):
        for number, call in enumerate(message.tool_calls or []):
            original_id = call.id
            call.id = f"eval_call_{index}_{number}"
            ids[original_id] = call.id
            args = normalized_arguments(
                ToolCall(name=call.function.name, arguments=strict_json(call.function.arguments))
            )
            call.function.arguments = json.dumps(args, sort_keys=True, separators=(",", ":"))
        if message.role == "tool":
            message.tool_call_id = ids[message.tool_call_id]
            message.content = json.dumps(
                strict_json(message.content or ""), sort_keys=True, separators=(",", ":")
            )
    return history


def request_key(runtime: RuntimeConfig, messages: list[Message], tools: list[dict]) -> str:
    # Same payload/hash as GroqClient; covered by a real-SDK MockTransport test.
    request = {
        "model": runtime.groq.model,
        "messages": [message.to_wire() for message in messages],
        "tools": tools,
        "tool_choice": "auto",
        "seed": runtime.seed,
        **runtime.generation.model_dump(),
    }
    if not tools:
        del request["tools"], request["tool_choice"]
    if runtime.groq.reasoning_effort is not None:
        request["reasoning_effort"] = runtime.groq.reasoning_effort
    return hashlib.sha256(
        json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


class EvaluationGroqClient(GroqClient):
    def __init__(self, runtime: RuntimeConfig, max_quota_wait: float) -> None:
        super().__init__(runtime)
        self.max_quota_wait = max_quota_wait
        self.rate_headers: dict[str, str] = {}
        self.headers_at = 0.0
        self.reserve_tokens = runtime.generation.max_tokens
        self.last_wait_seconds = 0.0

    def _capture(self, response: httpx.Response) -> None:
        self.rate_headers = {
            key: value
            for key, value in response.headers.items()
            if key.startswith("x-ratelimit-") or key == "retry-after"
        }
        self.headers_at = time.monotonic()
        if response.status_code == 429:
            delay = float(response.headers.get("retry-after", "0"))
            if delay > self.max_quota_wait:
                response.read()
                raise QuotaUnavailable(delay, self.rate_headers)

    def _sdk(self) -> Any:
        if self.sdk is None:
            load_dotenv(self.env_path)
            key = os.environ.get("GROQ_API_KEY")
            if not key:
                raise RuntimeError("Set GROQ_API_KEY in .env before making a Groq request")
            self.sdk = Groq(
                api_key=key,
                timeout=self.config.groq.timeout_seconds,
                max_retries=0,
                http_client=httpx.Client(event_hooks={"response": [self._capture]}),
            )
        return self.sdk

    def generate(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        self.last_wait_seconds = 0.0
        key = request_key(self.config, messages, tools)
        error_path = self.config.groq.cache_dir / "eval_errors" / f"{key}.json"
        if error_path.is_file():
            # Deliberate provider rejection is reproducible evidence, not a retryable outage.
            value = json.loads(error_path.read_text())
            raise CachedToolFailure(value["failed_generation"], cached=True)
        path = self.config.groq.cache_dir / f"{key}.json"
        if not path.is_file():
            headers = self.rate_headers
            limit = float(headers.get("x-ratelimit-limit-tokens", "0"))
            if limit:
                elapsed = time.monotonic() - self.headers_at
                remaining = float(headers["x-ratelimit-remaining-tokens"]) + elapsed * limit / 60
                wait = max(0, (self.reserve_tokens - remaining) * 60 / limit)
                if wait > self.max_quota_wait:
                    raise QuotaUnavailable(wait, headers)
                if wait:
                    self.last_wait_seconds = wait
                    time.sleep(wait)
        try:
            result = super().generate(messages, tools)
        except APIStatusError as exc:
            body = exc.body if isinstance(exc.body, dict) else {}
            error = body.get("error", body)
            if exc.status_code == 400 and error.get("code") in {
                "tool_use_failed",
                "json_validate_failed",
            }:
                failed = str(error.get("failed_generation", ""))
                error_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = error_path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"failed_generation": failed}))
                temporary.replace(error_path)
                raise CachedToolFailure(failed, cached=False) from None
            raise
        self.reserve_tokens = result.usage.prompt_tokens + self.config.generation.max_tokens
        return result


class CachedToolFailure(ProviderToolFailure):
    def __init__(self, text: str, *, cached: bool) -> None:
        super().__init__("PROVIDER_TOOL_USE_FAILED")
        self.text, self.cached = text, cached


class RecordingClient:
    def __init__(self, client: ModelClient) -> None:
        self.client = client
        self.user_index = 0
        self.observations: list[Observation] = []
        self.last_history: list[Message] = []

    def generate(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        self.last_history = [message.model_copy(deep=True) for message in messages]
        history = canonical_history(messages)
        started = time.perf_counter()
        try:
            response = self.client.generate(history, tools)
        except CachedToolFailure as exc:
            self.observations.append(
                Observation(
                    user_index=self.user_index,
                    context_role=history[-1].role,
                    response=ModelResponse(content=exc.text, cached=exc.cached),
                    latency_seconds=max(
                        0,
                        time.perf_counter()
                        - started
                        - getattr(self.client, "last_wait_seconds", 0),
                    ),
                    error="PROVIDER_TOOL_USE_FAILED",
                )
            )
            raise
        self.observations.append(
            Observation(
                user_index=self.user_index,
                context_role=history[-1].role,
                response=response,
                latency_seconds=max(
                    0, time.perf_counter() - started - getattr(self.client, "last_wait_seconds", 0)
                ),
            )
        )
        return response
