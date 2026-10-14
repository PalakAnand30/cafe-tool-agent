"""Pinned tokenizer-only loading and assistant-only labels without tensors."""

from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from cafe_agent.config import StrictModel, read_config
from cafe_agent.models.base import Message
from cafe_agent.prompting.prompts import render_chat

from .config import load_data_config


class TokenizerConfig(StrictModel):
    seed: int
    model_id: str
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    max_sequence_length: int = Field(gt=0)


def load_tokenizer(*, local_files_only: bool = False) -> Any:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    values = read_config("train.yaml")
    train = TokenizerConfig.model_validate(
        {key: values[key] for key in TokenizerConfig.model_fields}
    )
    data = load_data_config()
    cached = (
        data.tokenizer_cache
        / ("models--" + train.model_id.replace("/", "--"))
        / "snapshots"
        / train.revision
    )
    required = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "config.json",
    ]
    location = (
        str(cached)
        if all((cached / name).is_file() for name in required)
        else snapshot_download(
            train.model_id,
            revision=train.revision,
            cache_dir=str(data.tokenizer_cache),
            local_files_only=local_files_only,
            token=False,
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
                "config.json",
            ],
        )
    )
    return AutoTokenizer.from_pretrained(
        Path(location), local_files_only=True, trust_remote_code=False
    )


class MaskedExample(StrictModel):
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]

    @model_validator(mode="after")
    def check_lengths(self) -> "MaskedExample":
        if not len(self.input_ids) == len(self.attention_mask) == len(self.labels):
            raise ValueError("Token and label lengths differ")
        if not any(label != -100 for label in self.labels):
            raise ValueError("No assistant tokens to supervise")
        return self


def mask_assistant_tokens(
    tokenizer: Any, messages: list[Message], tools: list[dict[str, Any]], max_tokens: int
) -> MaskedExample:
    text = render_chat(tokenizer, messages, tools, add_generation_prompt=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if len(encoded["input_ids"]) >= max_tokens:
        raise ValueError(f"Conversation is not under the {max_tokens}-token limit")
    spans = []
    for index, message in enumerate(messages):
        if message.role != "assistant":
            continue
        before = render_chat(tokenizer, messages[:index], tools, add_generation_prompt=True)
        after = render_chat(tokenizer, messages[: index + 1], tools, add_generation_prompt=False)
        if not text.startswith(before) or not text.startswith(after) or len(after) <= len(before):
            raise ValueError("Chat-template prefixes do not align; refusing to guess label spans")
        spans.append((len(before), len(after)))
    labels = [
        token
        if end > start and any(left <= start and end <= right for left, right in spans)
        else -100
        for token, (start, end) in zip(encoded["input_ids"], encoded["offset_mapping"], strict=True)
    ]
    return MaskedExample(
        input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], labels=labels
    )
