"""Shared local/notebook evaluation settings, with no GPU imports."""

from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field

from cafe_agent.config import StrictModel, read_config

SystemID = Literal["S1", "S2", "S3", "S4", "S5"]
Mode = Literal["turn", "end_to_end"]
Split = Literal["test", "independent_test"]


class SystemConfig(StrictModel):
    model: str
    backend: str
    enabled: bool


class DatasetConfig(StrictModel):
    path: Path
    author: Literal["synthetic", "claude"]
    conversations: int = Field(gt=0)


class PricingConfig(StrictModel):
    input_per_million_usd: Decimal = Field(ge=0)
    output_per_million_usd: Decimal = Field(ge=0)
    verified_date: str
    source: str
    assumption: str


class EvalConfig(StrictModel):
    seed: int
    temperature: float = Field(ge=0)
    max_tokens: int = Field(gt=0)
    systems: dict[SystemID, SystemConfig]
    datasets: dict[Split, DatasetConfig]
    output_dir: Path
    metrics_path: Path
    report_path: Path
    modes: list[Mode]
    bootstrap_samples: int = Field(gt=0)
    confidence: Literal[0.95]
    progress_every: int = Field(gt=0)
    max_quota_wait_seconds: float = Field(gt=0)
    pricing: PricingConfig


def load_eval_config(path: Path | None = None) -> EvalConfig:
    return EvalConfig.model_validate(read_config("eval.yaml", path))
