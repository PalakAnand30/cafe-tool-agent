"""Entity survival, binding protection, caching journals and surface noise."""

import json
import random
from pathlib import Path

import pytest

from cafe_agent.data.config import load_data_config
from cafe_agent.data.noise import add_noise
from cafe_agent.data.paraphraser import Paraphraser, ParaphraseRequest, verify_entities
from cafe_agent.data.schema import Entity
from cafe_agent.models.base import ModelResponse


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        ("Please help Alex Adams order 2 small latte with oat milk.", None),
        ("Please help Sam Smith order 2 small latte with oat milk.", "missing_name"),
        ("Please help Alex Adams order 3 small latte with oat milk.", "missing_quantity"),
        ("Please help Alex Adams order 2 large latte with oat milk.", "missing_size"),
        ("Please help Alex Adams order 2 small latte.", "missing_modifier"),
        ("Please help Alex Adams order 2 small latte with oat milk and 8 muffins.", "new_number"),
        ("Please help Alex Adams order 2 small latte with oat milk and soy milk.", "new_modifier"),
        (
            "Please help Alex Adams order 2 small latte with oat milk and a cinnamon roll.",
            "new_item",
        ),
        (
            "Please do not help Alex Adams order 2 small latte with oat milk.",
            "new_negation_or_cancellation",
        ),
    ],
)
def test_entity_verification(candidate: str, reason: str | None) -> None:
    original = "Alex Adams here; order 2 small latte with oat milk."
    entities = [
        Entity(kind="name", value="Alex Adams"),
        Entity(kind="quantity", value="2"),
        Entity(kind="size", value="small"),
        Entity(kind="item", value="latte"),
        Entity(kind="modifier", value="oat_milk"),
    ]
    assert verify_entities(original, candidate, entities, load_data_config()) == reason


def test_line_binding_cannot_be_swapped() -> None:
    original = "Order 2 small latte with oat milk and 1 large americano with soy milk."
    entities = [
        Entity(kind="order_line", value="2 small latte with oat milk"),
        Entity(kind="order_line", value="1 large americano with soy milk"),
    ]
    swapped = "Please order 1 large latte with soy milk and 2 small americano with oat milk."
    assert verify_entities(original, swapped, entities, load_data_config()) == "missing_order_line"


def test_batched_paraphrases_resume_and_reject_invalid_entities(tmp_path: Path) -> None:
    class Client:
        calls = 0

        def generate(self, messages, tools):
            self.calls += 1
            assert not tools
            payload = json.loads(messages[-1].content)
            return ModelResponse(
                content=json.dumps(
                    [
                        {
                            "id": row["id"],
                            "text": "Please assist: " + row["text"]
                            if row["id"] != "b"
                            else "Different name",
                        }
                        for row in payload
                    ]
                )
            )

    config = load_data_config()
    config.paraphrase_batch_size = 2
    client = Client()
    requests = [
        ParaphraseRequest(
            id=key,
            text="Alex Adams wants 1 latte.",
            entities=[Entity(kind="name", value="Alex Adams")],
        )
        for key in ["a", "b", "c"]
    ]
    engine = Paraphraser(config, tmp_path / "paraphrases.jsonl", client=client)
    results = engine.run(requests)
    assert client.calls == 2
    assert results["a"].accepted and results["c"].accepted
    assert not results["b"].accepted and results["b"].text == requests[1].text
    assert engine.run(requests) == results and client.calls == 2


def test_noise_is_seeded_and_preserves_order_facts() -> None:
    original = "I'm Alex Adams. Please order 2 small latte with oat milk, thanks."
    for seed in range(20):
        candidate = add_noise(original, random.Random(seed))
        assert candidate == add_noise(original, random.Random(seed))
        assert "2 small latte with oat milk" in candidate.lower()
        assert "alex adams" in candidate.lower()
