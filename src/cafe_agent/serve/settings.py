"""Demo limits, independent of the frozen evaluation configuration."""

from pathlib import Path

from pydantic import Field

from cafe_agent.config import StrictModel, read_config


class ServeSettings(StrictModel):
    host: str
    port: int = Field(gt=0, le=65535)
    max_sessions: int = Field(gt=0)
    session_ttl_seconds: int = Field(gt=0)
    max_turns: int = Field(gt=0)
    max_message_chars: int = Field(gt=0)
    queue_size: int = Field(gt=0)
    advance_minutes: int = Field(gt=0)
    model_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    examples: list[str] = Field(min_length=5, max_length=5)
    space_name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    featured_items: list[str] = Field(min_length=1)


def serve_settings(path: Path | None = None) -> ServeSettings:
    return ServeSettings.model_validate(read_config("serve.yaml", path))
