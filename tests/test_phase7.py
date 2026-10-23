"""Regression guard integrity and publication claims derived from persisted evidence."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from cafe_agent.config import load_runtime
from cafe_agent.data.builder import ConversationBuilder
from cafe_agent.data.schema import Intent
from cafe_agent.eval.config import load_eval_config
from cafe_agent.eval.runner import end_prediction
from cafe_agent.eval.smoke import score_sample, select_cases, smoke_settings
from cafe_agent.models.base import ModelResponse
from cafe_agent.reporting.publication import render_publication, replace_block


def test_smoke_sampling_is_complete_order_independent_and_rejects_duplicates():
    records = [SimpleNamespace(id=f"test-{i:03}") for i in range(100)]
    sample = select_cases(records, 50, 42)
    assert sample == select_cases(list(reversed(records)), 50, 42)
    assert len({case.id for case in sample}) == 50
    with pytest.raises(ValueError, match="unique"):
        select_cases(records + [records[0]], 50, 42)
    with pytest.raises(ValueError, match="enough"):
        select_cases(records, 101, 42)


def test_smoke_pragma_does_not_change_frozen_config_and_fails_closed(tmp_path):
    path = tmp_path / "eval.yaml"
    original = Path("configs/eval.yaml").read_text()
    path.write_text(original)
    assert load_eval_config(path) == load_eval_config()
    assert smoke_settings(path).conversations == 50
    path.write_text(original + "\n# smoke_eval: {}\n")
    with pytest.raises(ValueError, match="exactly one"):
        smoke_settings(path)
    path.write_text("seed: 42\n")
    with pytest.raises(ValueError, match="exactly one"):
        smoke_settings(path)
    path.write_text(
        "# smoke_eval: {conversations: 0, split: test, min_action_and_state_success: 2}"
    )
    with pytest.raises(ValueError):
        smoke_settings(path)


def test_guard_rejects_wrong_action_despite_equal_empty_state_and_bad_provenance():
    builder = ConversationBuilder()
    try:
        builder.user("A latte, please.")
        builder.say("Which size?", behaviour="clarify")
        record = builder.finish(
            record_id="clarify", split="test", intent=Intent.CLARIFICATION, template_id="guard"
        )
    finally:
        builder.close()

    class Client:
        def __init__(self, text):
            self.text = text

        def generate(self, messages, tools):
            return ModelResponse(content=self.text)

    config = load_eval_config()
    wrong = end_prediction(record, Client("Hello!"), load_runtime(), system="S5", fingerprint="f")
    assert wrong.final_state == record.expected_final_state
    assert not score_sample([record], [wrong], config, 0.8, "f")["passed"]
    right = end_prediction(
        record, Client("Which size?"), load_runtime(), system="S5", fingerprint="f"
    )
    assert score_sample([record], [right], config, 1, "f")["passed"]
    with pytest.raises(ValueError, match="provenance"):
        score_sample([record], [right], config, 0.8, "other")
    for rows in [[], [right, right]]:
        with pytest.raises(ValueError, match="missing, duplicate"):
            score_sample([record], rows, config, 0.8, "f")


def test_publication_generation_is_deterministic_and_tracks_changed_metrics(tmp_path):
    (tmp_path / "results").mkdir()
    for name in [
        "metrics.json",
        "dataset_upload.json",
        "model_download.json",
        "kaggle_training_user_report.json",
    ]:
        shutil.copyfile(Path("results") / name, tmp_path / "results" / name)
    (tmp_path / "README.md").write_text(
        "<!-- HEADLINE:START --><!-- HEADLINE:END -->\n"
        "<!-- DATA_TRAINING:START --><!-- DATA_TRAINING:END -->\n"
    )
    first = render_publication(tmp_path)
    assert first == render_publication(tmp_path)
    assert "not uploaded" in first["docs/cards/dataset/README.md"]
    assert "license:" not in first["docs/cards/dataset/README.md"].split("---")[1]
    path = tmp_path / "results/metrics.json"
    data = json.loads(path.read_text())
    target = next(g for g in data["groups"] if g["system"] == "S5" and g["split"] == "test")
    target["modes"]["end_to_end"]["metrics"]["action_and_state_success"] = {
        "value": 0.123,
        "ci95": None,
    }
    path.write_text(json.dumps(data))
    assert "12.3%" in render_publication(tmp_path)["README.md"]
    data["all_complete"] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="complete"):
        render_publication(tmp_path)


def test_publication_marker_rejects_duplicate_sections():
    with pytest.raises(ValueError, match="exactly one"):
        replace_block("<!-- X:START --><!-- X:START --><!-- X:END -->", "X", "text")
