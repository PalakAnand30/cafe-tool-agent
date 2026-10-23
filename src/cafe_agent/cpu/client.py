"""Greedy CPU GGUF completion with the same renderer and token IDs as GPU systems."""

from pathlib import Path
from typing import Any

from cafe_agent.models.base import Message, ModelResponse, TokenUsage
from cafe_agent.prompting.prompts import render_chat

from .settings import LocalSettings


class CPUClient:
    def __init__(
        self,
        settings: LocalSettings,
        tokenizer: Any,
        *,
        max_tokens: int,
        model_path: Path,
        engine: Any = None,
    ) -> None:
        self.settings, self.tokenizer = settings, tokenizer
        self.max_tokens, self.model_path, self.engine = max_tokens, model_path, engine
        self.last_prompt_tokens = 0

    def load(self) -> Any:
        if self.engine is None:
            from llama_cpp import Llama

            cpu = self.settings.cpu
            self.engine = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=0,
                offload_kqv=False,
                n_ctx=cpu.context,
                n_threads=cpu.threads,
                n_threads_batch=cpu.threads,
                n_batch=cpu.batch,
                n_ubatch=cpu.micro_batch,
                seed=self.settings.seed,
                verbose=False,
            )
        return self.engine

    def generate(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        text = render_chat(self.tokenizer, messages, tools)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        self.last_prompt_tokens = len(ids)
        if len(ids) + self.max_tokens > self.settings.cpu.context:
            return ModelResponse(content="CONTEXT_LIMIT_EXCEEDED", finish_reason="length")
        engine = self.load()
        native_ids = engine.tokenize(text.encode(), add_bos=False, special=True)
        if native_ids != ids:
            raise ValueError("GGUF and frozen Qwen tokenizer disagree; refusing inference")
        result = engine.create_completion(
            prompt=ids,
            temperature=0,
            max_tokens=self.max_tokens,
            seed=self.settings.seed,
            stop=["<|im_end|>"],
            repeat_penalty=1.0,
        )
        choice, usage = result["choices"][0], result.get("usage", {})
        return ModelResponse(
            content=choice["text"],
            finish_reason=choice.get("finish_reason"),
            usage=TokenUsage(**{key: usage.get(key, 0) for key in TokenUsage.model_fields}),
        )

    def reset(self) -> None:
        self.load().reset()

    def close(self) -> None:
        if self.engine is not None:
            self.engine.close()
