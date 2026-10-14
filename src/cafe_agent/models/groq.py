"""Native Groq tool calling with content-addressed disk cache and bounded retries."""

import hashlib
import json
import os
import random
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import APIStatusError, Groq

from cafe_agent.config import RuntimeConfig, load_runtime

from .base import Message, ModelResponse, TokenUsage, WireToolCall


class GroqClient:
    def __init__(
        self,
        config: RuntimeConfig | None = None,
        *,
        sdk: Any = None,
        env_path: Path = Path(".env"),
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or load_runtime()
        self.sdk = sdk
        self.env_path = env_path
        self.sleep = sleep
        self.random = random.Random(self.config.seed)
        self.owns_sdk = sdk is None

    def _sdk(self) -> Any:
        if self.sdk is None:
            load_dotenv(self.env_path)
            key = os.environ.get("GROQ_API_KEY")
            if not key:
                raise RuntimeError("Set GROQ_API_KEY in .env before making a Groq request")
            self.sdk = Groq(api_key=key, timeout=self.config.groq.timeout_seconds, max_retries=0)
        return self.sdk

    def generate(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        request = {
            "model": self.config.groq.model,
            "messages": [message.to_wire() for message in messages],
            "tools": tools,
            "tool_choice": "auto",
            "seed": self.config.seed,
            **self.config.generation.model_dump(),
        }
        if not tools:
            del request["tools"]
            del request["tool_choice"]
        if self.config.groq.reasoning_effort is not None:
            request["reasoning_effort"] = self.config.groq.reasoning_effort
        key = hashlib.sha256(
            json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        path = self.config.groq.cache_dir / f"{key}.json"
        cached = path.is_file()
        if cached:
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            retry = self.config.groq
            for attempt in range(retry.max_attempts):
                try:
                    raw = self._sdk().chat.completions.create(**request).model_dump(mode="json")
                    break
                except APIStatusError as exc:
                    if (
                        exc.status_code != 429
                        and not 500 <= exc.status_code <= 599
                        or attempt + 1 == retry.max_attempts
                    ):
                        raise
                    delay = min(retry.backoff_cap_seconds, retry.backoff_base_seconds * 2**attempt)
                    self.sleep(delay + self.random.uniform(0, delay))
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                json.dump(raw, handle, ensure_ascii=False)
            temporary.replace(path)
        choice = raw["choices"][0]
        message = choice["message"]
        usage = raw.get("usage") or {}
        return ModelResponse(
            content=message.get("content") or "",
            tool_calls=[
                WireToolCall.model_validate(call) for call in message.get("tool_calls") or []
            ],
            finish_reason=choice.get("finish_reason"),
            cached=cached,
            usage=TokenUsage(**{k: usage.get(k, 0) for k in TokenUsage.model_fields}),
        )

    def close(self) -> None:
        if self.owns_sdk and self.sdk is not None:
            self.sdk.close()
