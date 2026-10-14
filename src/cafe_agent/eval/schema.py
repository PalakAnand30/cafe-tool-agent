"""Versioned provider-neutral prediction files shared with future Kaggle runs."""

from typing import Literal

from pydantic import Field, model_validator

from cafe_agent.agent.loop import TraceEntry
from cafe_agent.config import StrictModel
from cafe_agent.data.schema import Intent
from cafe_agent.domain.models import BackendState
from cafe_agent.models.base import Message, ModelResponse

from .config import Mode, Split, SystemID


class Observation(StrictModel):
    user_index: int = Field(ge=0)
    context_role: Literal["user", "tool"]
    response: ModelResponse
    latency_seconds: float = Field(ge=0, allow_inf_nan=False)
    error: str | None = None


class Prediction(StrictModel):
    schema_version: Literal[1] = 1
    system: SystemID
    split: Split
    conversation_id: str
    intent: Intent
    author: Literal["synthetic", "claude"]
    mode: Mode
    fingerprint: str
    assistant_index: int | None = Field(default=None, ge=0)
    observations: list[Observation] = Field(min_length=1)
    final_state: BackendState | None = None
    messages: list[Message] = Field(default_factory=list)
    trace: list[TraceEntry] = Field(default_factory=list)
    error: str | None = None

    @model_validator(mode="after")
    def check_mode(self) -> "Prediction":
        if self.mode == "turn":
            if self.assistant_index is None or len(self.observations) != 1:
                raise ValueError("Turn prediction needs an assistant index and one response")
            if self.final_state is not None:
                raise ValueError("Turn prediction cannot contain an executed final state")
        elif self.assistant_index is not None or self.final_state is None:
            raise ValueError("End-to-end prediction requires a final state and no gold index")
        if (self.split == "independent_test") != (self.author == "claude"):
            raise ValueError("Preserve independent Claude authorship")
        return self

    @property
    def key(self) -> tuple[str, str, int | None]:
        return self.conversation_id, self.mode, self.assistant_index
