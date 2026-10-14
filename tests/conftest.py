"""Isolated, deterministic café fixtures."""

from collections.abc import Iterator

import pytest

from cafe_agent.backend.service import CafeService
from cafe_agent.data.tokenization import load_tokenizer


@pytest.fixture(scope="session")
def tokenizer():
    return load_tokenizer()


@pytest.fixture
def cafe() -> Iterator[CafeService]:
    service = CafeService()
    try:
        yield service
    finally:
        service.close()


@pytest.fixture
def latte_order() -> dict:
    return {
        "items": [
            {
                "item_name": "latte",
                "size": "medium",
                "quantity": 2,
                "modifiers": ["oat_milk", "extra_shot"],
            }
        ],
        "customer_name": "Alex",
    }
