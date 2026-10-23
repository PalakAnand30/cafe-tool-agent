"""Load the verified GGUF without changing evaluation manifests or downloading gold."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from cafe_agent.config import load_runtime
from cafe_agent.cpu.client import CPUClient
from cafe_agent.cpu.download import file_hash
from cafe_agent.cpu.settings import local_settings
from cafe_agent.data.tokenization import load_tokenizer
from cafe_agent.models.base import Message, ModelResponse

from .settings import ServeSettings


class ServingClient(CPUClient):
    def generate(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        response = super().generate(messages, tools)
        if response.finish_reason == "length":
            raise ValueError("Context or response limit reached")
        return response


def load_client(settings: ServeSettings) -> ServingClient:
    load_dotenv()
    local = local_settings()
    path = local.model.directory / local.model.filename
    if not path.is_file():
        path = Path(
            hf_hub_download(
                local.model.repo,
                local.model.filename,
                revision=local.model.revision,
                token=os.environ.get("HF_TOKEN"),
                local_dir=local.model.directory,
            )
        )
    if file_hash(path) != settings.model_sha256:
        raise ValueError("GGUF differs from the verified Phase 5 export")
    import llama_cpp

    if llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError("The demo requires the CPU-only build; run make setup-cpu")
    client = ServingClient(
        local,
        load_tokenizer(),
        max_tokens=load_runtime().generation.max_tokens,
        model_path=path,
    )
    client.load()
    return client
