"""Notebook-only interfaces; GPU implementations arrive with Phase 4."""

from typing import Any

from .base import Message, ModelResponse


class VLLMClient:
    def generate(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        raise NotImplementedError("vLLM inference is implemented only in Phase 4 Kaggle notebooks")


class HFClient:
    def generate(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        raise NotImplementedError(
            "HF GPU inference is implemented only in Phase 4 Kaggle notebooks"
        )
