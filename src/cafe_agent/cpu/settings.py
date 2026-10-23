"""Phase 5 settings, separate from immutable data and GPU/S3 configuration."""

from pathlib import Path

from pydantic import Field

from cafe_agent.config import StrictModel, read_config


class ModelSettings(StrictModel):
    repo: str
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    filename: str
    directory: Path


class CPUSettings(StrictModel):
    context: int = Field(gt=0)
    threads: int = Field(gt=0)
    batch: int = Field(gt=0)
    micro_batch: int = Field(gt=0)


class LatencySettings(StrictModel):
    warmup_requests: int = Field(ge=1)
    measured_requests: int = Field(ge=100)
    split: str
    reset_context_each_request: bool


class CostSettings(StrictModel):
    usd_per_cpu_hour: float = Field(ge=0)
    assumption: str


class ReportSettings(StrictModel):
    systems: list[str]
    representative_failures: int = Field(ge=10)
    charts_directory: Path
    directory: Path


class LocalSettings(StrictModel):
    seed: int
    output_directory: Path
    model: ModelSettings
    cpu: CPUSettings
    latency: LatencySettings
    cost: CostSettings
    report: ReportSettings


def local_settings(path: Path | None = None) -> LocalSettings:
    return LocalSettings.model_validate(read_config("local.yaml", path))
