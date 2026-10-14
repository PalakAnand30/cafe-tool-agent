"""Validated configuration from the checkout or installed package resources."""

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def read_config(name: str, path: Path | None = None) -> dict[str, Any]:
    if path is not None:
        text = path.read_text(encoding="utf-8")
    else:
        checkout = Path(__file__).resolve().parents[2] / "configs" / name
        resource = files("cafe_agent").joinpath("configs", name)
        text = (checkout if checkout.is_file() else resource).read_text(encoding="utf-8")
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    return value


class GenerationConfig(StrictModel):
    temperature: float = Field(ge=0)
    max_tokens: int = Field(gt=0)


class GroqConfig(StrictModel):
    model: str
    cache_dir: Path
    timeout_seconds: float = Field(gt=0)
    max_attempts: int = Field(gt=0)
    backoff_base_seconds: float = Field(gt=0)
    backoff_cap_seconds: float = Field(gt=0)


class LlamaConfig(StrictModel):
    n_ctx: int = Field(gt=0)
    n_threads: int = Field(gt=0)
    stop: list[str]


class AgentConfig(StrictModel):
    max_rounds: int = Field(gt=0)


class APIConfig(StrictModel):
    database_path: Path


class RuntimeConfig(StrictModel):
    seed: int
    generation: GenerationConfig
    groq: GroqConfig
    llama: LlamaConfig
    agent: AgentConfig
    api: APIConfig


def load_runtime(path: Path | None = None) -> RuntimeConfig:
    return RuntimeConfig.model_validate(read_config("runtime.yaml", path))
