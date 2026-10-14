"""Lazy CPU-only GGUF client using the shared Qwen tokenizer template."""

from pathlib import Path
from typing import Any

from cafe_agent.config import RuntimeConfig, load_runtime
from cafe_agent.prompting.prompts import ChatTokenizer, render_chat

from .base import Message, ModelResponse, TokenUsage


class LlamaCppClient:
    def __init__(
        self,
        model_path: Path,
        tokenizer: ChatTokenizer,
        config: RuntimeConfig | None = None,
        *,
        engine: Any = None,
    ) -> None:
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.config = config or load_runtime()
        self.engine = engine

    def _engine(self) -> Any:
        if self.engine is None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError("Install the pinned cpu extra before GGUF inference") from exc
            self.engine = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=0,
                n_ctx=self.config.llama.n_ctx,
                n_threads=self.config.llama.n_threads,
                seed=self.config.seed,
                verbose=False,
            )
        return self.engine

    def generate(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        raw = self._engine().create_completion(
            prompt=render_chat(self.tokenizer, messages, tools),
            **self.config.generation.model_dump(),
            seed=self.config.seed,
            stop=self.config.llama.stop,
        )
        choice = raw["choices"][0]
        usage = raw.get("usage") or {}
        return ModelResponse(
            content=choice["text"],
            finish_reason=choice.get("finish_reason"),
            usage=TokenUsage(**{k: usage.get(k, 0) for k in TokenUsage.model_fields}),
        )

    def close(self) -> None:
        if self.engine is not None:
            self.engine.close()
