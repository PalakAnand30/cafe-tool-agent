"""Backend contract, exact prices, validation, state progression and persistence."""

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from cafe_agent.backend.service import CafeService
from cafe_agent.domain.models import (
    Category,
    DomainConfig,
    ErrorCode,
    ErrorResult,
    MenuItem,
    OrderStatus,
    load_domain,
)


def test_menu_configuration_and_browsing(cafe: CafeService) -> None:
    config = load_domain()
    assert len(config.items) == 25
    assert {item.category for item in config.items} == set(Category)
    assert sum(not item.available for item in config.items) == 3
    result = cafe.call("get_menu", {"category": "bakery"})
    assert {item.name for item in result.items} == {
        "butter croissant",
        "almond croissant",
        "blueberry muffin",
        "chocolate brownie",
        "cinnamon roll",
    }
    assert all(item.price is not None and not item.prices for item in result.items)


@pytest.mark.parametrize(
    "values",
    [
        {"name": "x", "category": "coffee", "price": "2.00"},
        {"name": "x", "category": "bakery", "prices": {"small": "2.00"}},
        {"name": "x", "category": "bakery", "price": "-1.00"},
        {"name": "x", "category": "bakery", "price": "1.001"},
    ],
)
def test_invalid_menu_prices(values: dict) -> None:
    with pytest.raises(ValidationError):
        MenuItem.model_validate(values)


def test_duplicate_menu_names_and_invalid_timing() -> None:
    values = load_domain().model_dump()
    values["items"].append(values["items"][0])
    with pytest.raises(ValidationError, match="unique"):
        DomainConfig.model_validate(values)
    values = load_domain().model_dump()
    values["collected_after_minutes"] = 5
    with pytest.raises(ValidationError, match="Collected"):
        DomainConfig.model_validate(values)


def test_exact_prices_and_canonical_names(cafe: CafeService, latte_order: dict) -> None:
    latte_order["items"][0]["item_name"] = "  LATTE  "
    latte_order["items"].append(
        {"item_name": "butter croissant", "quantity": 1, "modifiers": ["warmed"]}
    )
    result = cafe.call("create_order", latte_order)
    assert result.order_id == "ORD-00001"
    assert result.total == Decimal("11.50")
    stored = cafe.snapshot().orders[0]
    assert stored.items[0].unit_price == Decimal("4.50")
    assert stored.items[0].line_total == Decimal("9.00")
    assert stored.items[0].item_name == "latte"
    assert stored.customer_name == "Alex"
    assert stored.items[1].size is None


@pytest.mark.parametrize(
    ("name", "arguments", "code"),
    [
        (
            "create_order",
            {"items": [{"item_name": "pizza", "quantity": 1}]},
            ErrorCode.ITEM_NOT_FOUND,
        ),
        (
            "create_order",
            {"items": [{"item_name": "filter coffee", "quantity": 1, "size": "small"}]},
            ErrorCode.OUT_OF_STOCK,
        ),
        (
            "create_order",
            {
                "items": [
                    {"item_name": "latte", "quantity": 1, "size": "small", "modifiers": ["warmed"]}
                ]
            },
            ErrorCode.UNSUPPORTED_MODIFIER,
        ),
        (
            "create_order",
            {"items": [{"item_name": "latte", "quantity": 1}]},
            ErrorCode.SIZE_REQUIRED,
        ),
        (
            "create_order",
            {"items": [{"item_name": "lat", "quantity": 1, "size": "small"}]},
            ErrorCode.AMBIGUOUS_ITEM,
        ),
        (
            "create_order",
            {"items": [{"item_name": "butter croissant", "quantity": 1, "size": "small"}]},
            ErrorCode.INVALID_SIZE,
        ),
        ("get_order_status", {"order_id": "ORD-12345"}, ErrorCode.ORDER_NOT_FOUND),
        ("cancel_order", {"order_id": "ORD-12345"}, ErrorCode.ORDER_NOT_FOUND),
        ("create_order", {"items": []}, ErrorCode.INVALID_ARGUMENTS),
        ("refund", {}, ErrorCode.UNKNOWN_TOOL),
    ],
)
def test_errors_do_not_mutate_state(
    cafe: CafeService, name: str, arguments: dict, code: ErrorCode
) -> None:
    before = cafe.snapshot()
    result = cafe.call(name, arguments)
    assert isinstance(result, ErrorResult)
    assert result.error.code == code
    assert cafe.snapshot() == before


@pytest.mark.parametrize("quantity", [0, 11, -1, True, 1.5, "2", None])
def test_invalid_quantities(cafe: CafeService, latte_order: dict, quantity: object) -> None:
    latte_order["items"][0]["quantity"] = quantity
    result = cafe.call("create_order", latte_order)
    assert result.error.code == ErrorCode.INVALID_ARGUMENTS
    assert not cafe.snapshot().orders


@pytest.mark.parametrize(
    "patch",
    [
        {"size": "gigantic"},
        {"unknown": "field"},
        {"modifiers": ["oat_milk", "oat_milk"]},
        {"item_name": "   "},
        {"modifiers": "oat_milk"},
        {"quantity": None},
    ],
)
def test_item_validation(cafe: CafeService, latte_order: dict, patch: dict) -> None:
    latte_order["items"][0].update(patch)
    assert cafe.call("create_order", latte_order).error.code == ErrorCode.INVALID_ARGUMENTS


@pytest.mark.parametrize(
    "arguments",
    [
        {"category": "invalid"},
        {"category": "coffee", "injected": True},
    ],
)
def test_menu_argument_validation(cafe: CafeService, arguments: dict) -> None:
    assert cafe.call("get_menu", arguments).error.code == ErrorCode.INVALID_ARGUMENTS


@pytest.mark.parametrize(
    "order_id", ["ORD-1", "ORD-123456", "ORD-abcde", "ord-00001", "ORD-１２３４５"]
)
def test_order_id_validation(cafe: CafeService, order_id: str) -> None:
    assert (
        cafe.call("cancel_order", {"order_id": order_id}).error.code == ErrorCode.INVALID_ARGUMENTS
    )


def test_order_validation_is_atomic(cafe: CafeService, latte_order: dict) -> None:
    latte_order["items"].append({"item_name": "pizza", "quantity": 1})
    assert cafe.call("create_order", latte_order).error.code == ErrorCode.ITEM_NOT_FOUND
    assert not cafe.snapshot().orders
    latte_order["items"].pop()
    assert cafe.call("create_order", latte_order).order_id == "ORD-00001"


def test_availability_alternatives(cafe: CafeService) -> None:
    result = cafe.call("check_availability", {"item_name": "filter coffee"})
    assert not result.available
    assert result.alternatives and all(item.available for item in result.alternatives)
    assert all(item.category == Category.COFFEE for item in result.alternatives)
    assert cafe.call("check_availability", {"item_name": "latte"}).available
    assert not cafe.snapshot().orders


def test_progression_and_cancellation_boundaries(cafe: CafeService, latte_order: dict) -> None:
    created = cafe.call("create_order", latte_order)
    args = {"order_id": created.order_id}
    assert cafe.call("get_order_status", args).status == OrderStatus.RECEIVED
    before = cafe.snapshot()
    cafe.call("get_order_status", args)
    assert cafe.snapshot() == before
    cafe.advance_time(1)
    assert cafe.call("get_order_status", args).status == OrderStatus.PREPARING
    cafe.advance_time(created.eta_minutes - 1)
    assert cafe.call("get_order_status", args).status == OrderStatus.READY
    assert cafe.call("get_order_status", args).eta_minutes == 0
    assert cafe.call("cancel_order", args).error.code == ErrorCode.CANNOT_CANCEL
    cafe.advance_time(100)
    assert cafe.call("get_order_status", args).status == OrderStatus.COLLECTED
    assert cafe.call("cancel_order", args).error.code == ErrorCode.CANNOT_CANCEL


@pytest.mark.parametrize("elapsed", [0, 1])
def test_cancel_before_ready_is_idempotent(
    cafe: CafeService, latte_order: dict, elapsed: int
) -> None:
    args = {"order_id": cafe.call("create_order", latte_order).order_id}
    cafe.advance_time(elapsed)
    assert cafe.call("cancel_order", args).cancelled
    assert cafe.call("cancel_order", args).cancelled
    cafe.advance_time(100)
    assert cafe.call("get_order_status", args).status == OrderStatus.CANCELLED
    assert cafe.call("get_order_status", args).eta_minutes == 0


@pytest.mark.parametrize("minutes", [-1, True, 1.5, "1"])
def test_invalid_clock_advance(cafe: CafeService, minutes: object) -> None:
    with pytest.raises(ValueError):
        cafe.advance_time(minutes)


def test_persistence_and_fresh_replay(tmp_path: Path, latte_order: dict) -> None:
    path = tmp_path / "cafe.sqlite3"
    first = CafeService(database=path)
    second = CafeService()
    try:
        assert first.call("create_order", latte_order) == second.call("create_order", latte_order)
        first.advance_time(2)
        second.advance_time(2)
        assert first.snapshot() == second.snapshot()
        expected = first.snapshot()
    finally:
        first.close()
        second.close()
    reopened = CafeService(database=path)
    try:
        assert reopened.snapshot() == expected
        assert reopened.call("create_order", latte_order).order_id == "ORD-00002"
    finally:
        reopened.close()


def test_concurrent_orders_have_unique_ids(cafe: CafeService, latte_order: dict) -> None:
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: cafe.call("create_order", latte_order), range(12)))
    assert len({result.order_id for result in results}) == 12
    assert len(cafe.snapshot().orders) == 12


def test_order_id_capacity_error(cafe: CafeService, latte_order: dict) -> None:
    cafe.call("create_order", latte_order)
    order = cafe.snapshot().orders[0].model_copy(update={"order_id": "ORD-99999"})
    with cafe.store.transaction() as connection:
        connection.execute("INSERT INTO orders VALUES (?, ?)", (99999, order.model_dump_json()))
    result = cafe.call("create_order", latte_order)
    assert result.error.code == ErrorCode.ORDER_LIMIT_REACHED
