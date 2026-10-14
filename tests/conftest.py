"""Isolated, deterministic café fixtures."""

from collections.abc import Iterator

import pytest

from cafe_agent.backend.service import CafeService


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
