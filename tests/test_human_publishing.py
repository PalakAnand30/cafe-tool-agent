"""Human examples stay excluded; the actual uploader always starts private."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cafe_agent.data.human import annotate, load_human_records, seed_examples
from cafe_agent.data.publishing import upload_private


def test_seed_examples_are_valid_idempotent_and_excluded(tmp_path: Path, tokenizer) -> None:
    path = tmp_path / "human.jsonl"
    seed_examples(path, tokenizer)
    before = path.read_bytes()
    seed_examples(path, tokenizer)
    assert path.read_bytes() == before
    records = load_human_records(path, scoreable_only=False)
    assert len(records) == 5
    assert all(record.author == "codex_example" for record in records)
    assert load_human_records(path) == []


def test_human_annotation_cli_saves_a_replayable_gold_conversation(
    tmp_path: Path, tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    replies = iter(
        [
            "single_order",
            "I'd like one medium latte for Morgan, please.",
            "tool",
            "create_order",
            json.dumps(
                {
                    "customer_name": "Morgan",
                    "items": [{"item_name": "latte", "size": "medium", "quantity": 1}],
                }
            ),
            "reply",
            "Order ORD-00001 is received; total £3.50.",
            "/save",
            "",
            "/quit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    path = tmp_path / "human.jsonl"
    annotate(path, tokenizer)
    records = load_human_records(path)
    assert len(records) == 1 and records[0].author == "human"
    assert records[0].expected_final_state.orders[0].customer_name == "Morgan"


@pytest.mark.parametrize("private", [True, False])
def test_hub_upload_private_only(tmp_path: Path, private: bool) -> None:
    class API:
        created = None
        uploaded = None

        def create_repo(self, **kwargs):
            self.created = kwargs

        def repo_info(self, **kwargs):
            return SimpleNamespace(private=private)

        def upload_folder(self, **kwargs):
            self.uploaded = kwargs

    api = API()
    if private:
        upload_private(tmp_path, username="test-user", token="not-a-secret", api=api)
        assert api.uploaded["repo_type"] == "dataset"
    else:
        with pytest.raises(ValueError, match="public"):
            upload_private(tmp_path, username="test-user", token="not-a-secret", api=api)
        assert api.uploaded is None
    assert api.created["private"] is True
