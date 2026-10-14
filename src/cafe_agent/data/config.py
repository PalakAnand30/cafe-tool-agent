"""Data plan configuration; sizes, rates and holdouts are explicit."""

from pathlib import Path

from pydantic import Field, model_validator

from cafe_agent.config import StrictModel, read_config

from .schema import Intent


class HeldOutCombination(StrictModel):
    item_name: str
    modifier: str


class DataConfig(StrictModel):
    seed: int
    splits: dict[str, int]
    intent_mix: dict[Intent, float]
    paraphrase_rate: float = Field(ge=0, le=1)
    noise_rate: float = Field(ge=0, le=1)
    multi_turn_rate: float = Field(ge=0, le=1)
    template_variants: int = Field(ge=2)
    test_template_fraction: float = Field(gt=0, lt=1)
    max_attempts_per_id: int = Field(gt=0)
    progress_every: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    paraphrase_max_tokens: int = Field(gt=0)
    paraphrase_batch_size: int = Field(gt=0)
    output_dir: Path
    human_path: Path
    human_min_conversations: int = Field(gt=0)
    human_max_conversations: int = Field(gt=0)
    tokenizer_cache: Path
    quantities: list[int]
    first_names: list[str]
    last_names: list[str]
    held_out_combinations: list[HeldOutCombination]
    synonyms: dict[str, list[str]]

    @model_validator(mode="after")
    def check_plan(self) -> "DataConfig":
        if set(self.splits) != {"train", "validation", "test"} or min(self.splits.values()) < 1:
            raise ValueError("Provide positive train/validation/test sizes")
        if set(self.intent_mix) != set(Intent) or abs(sum(self.intent_mix.values()) - 1) > 1e-9:
            raise ValueError("Intent shares must cover all intents and sum to one")
        if any(value <= 0 for value in self.intent_mix.values()):
            raise ValueError("Intent shares must be positive")
        held = self.template_variants * self.test_template_fraction
        if held != int(held) or not 0 < held < self.template_variants:
            raise ValueError("Reserve an exact positive number of template variants")
        if not self.quantities or any(not 1 <= q <= 10 for q in self.quantities):
            raise ValueError("Quantities must be in 1–10")
        return self


def load_data_config(path: Path | None = None) -> DataConfig:
    return DataConfig.model_validate(read_config("data.yaml", path))
