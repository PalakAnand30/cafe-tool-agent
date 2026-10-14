"""Validated café operations; failed orders never create partial side effects."""

import random
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cafe_agent.domain.models import (
    AvailabilityResult,
    BackendState,
    CancelResult,
    CreateOrderResult,
    DomainConfig,
    ErrorCode,
    ErrorResult,
    MenuItem,
    MenuResult,
    Order,
    OrderLine,
    OrderStatus,
    StatusResult,
    ToolError,
    ToolResult,
    load_domain,
    normalise,
)
from cafe_agent.tools.schemas import (
    CheckAvailabilityArgs,
    CreateOrderArgs,
    GetMenuArgs,
    OrderIDArgs,
    ToolName,
    validate_arguments,
)

from .store import SQLiteStore


class DomainFailure(Exception):
    def __init__(self, result: ErrorResult) -> None:
        self.result = result


def error(code: ErrorCode, message: str, **kwargs: Any) -> ErrorResult:
    return ErrorResult(error=ToolError(code=code, message=message, **kwargs))


class CafeService:
    def __init__(
        self, config: DomainConfig | None = None, database: str | Path = ":memory:"
    ) -> None:
        self.config = (config or load_domain()).model_copy(deep=True)
        self.store = SQLiteStore(database)

    def close(self) -> None:
        self.store.close()

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            tool = ToolName(name)
        except ValueError:
            return error(ErrorCode.UNKNOWN_TOOL, "Unknown café tool.")
        try:
            request = validate_arguments(name, arguments)
        except ValidationError as exc:
            return error(
                ErrorCode.INVALID_ARGUMENTS,
                "Tool arguments failed validation.",
                details={"fields": [".".join(map(str, e["loc"])) for e in exc.errors()]},
            )
        try:
            return getattr(self, tool.value)(request)
        except DomainFailure as exc:
            return exc.result

    def _alternatives(self, item: MenuItem | None = None) -> list[MenuItem]:
        return [
            candidate
            for candidate in self.config.items
            if candidate.available
            and (item is None or candidate.category == item.category)
            and (item is None or candidate.name != item.name)
        ][: self.config.alternatives_limit]

    def _find(self, name: str) -> MenuItem:
        key = normalise(name)
        for item in self.config.items:
            if normalise(item.name) == key:
                return item
        matches = [item for item in self.config.items if key in normalise(item.name)]
        if matches:
            raise DomainFailure(
                error(
                    ErrorCode.AMBIGUOUS_ITEM,
                    "Use the full menu item name.",
                    alternatives=matches[: self.config.alternatives_limit],
                )
            )
        raise DomainFailure(
            error(
                ErrorCode.ITEM_NOT_FOUND,
                "That item is not on the menu.",
                alternatives=self._alternatives(),
            )
        )

    @staticmethod
    def _check_size(item: MenuItem, size: str | None, required: bool = False) -> None:
        if item.prices and size is None and required:
            raise DomainFailure(error(ErrorCode.SIZE_REQUIRED, "Choose small, medium or large."))
        if size is not None and size not in item.prices:
            raise DomainFailure(
                error(ErrorCode.INVALID_SIZE, "That size is not available for this item.")
            )

    def get_menu(self, request: GetMenuArgs) -> MenuResult:
        return MenuResult(
            currency=self.config.currency,
            items=[
                item
                for item in self.config.items
                if request.category is None or item.category == request.category
            ],
        )

    def check_availability(self, request: CheckAvailabilityArgs) -> AvailabilityResult:
        item = self._find(request.item_name)
        self._check_size(item, request.size)
        return AvailabilityResult(
            available=item.available,
            alternatives=[] if item.available else self._alternatives(item),
        )

    def create_order(self, request: CreateOrderArgs) -> CreateOrderResult | ErrorResult:
        lines = []
        for requested in request.items:
            item = self._find(requested.item_name)
            self._check_size(item, requested.size, required=True)
            if not item.available:
                raise DomainFailure(
                    error(
                        ErrorCode.OUT_OF_STOCK,
                        f"{item.name} is out of stock.",
                        alternatives=self._alternatives(item),
                    )
                )
            unsupported = set(requested.modifiers) - item.modifiers.keys()
            if unsupported:
                raise DomainFailure(
                    error(
                        ErrorCode.UNSUPPORTED_MODIFIER,
                        "The item does not support these modifiers.",
                        details={"modifiers": sorted(unsupported)},
                        alternatives=self._alternatives(item),
                    )
                )
            price = item.prices[requested.size] if requested.size is not None else item.price
            assert price is not None
            unit_price = price + sum((item.modifiers[m] for m in requested.modifiers), Decimal("0"))
            values = requested.model_dump()
            values["item_name"] = item.name
            lines.append(
                OrderLine(
                    **values, unit_price=unit_price, line_total=unit_price * requested.quantity
                )
            )
        with self.store.transaction() as connection:
            number = connection.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM orders").fetchone()[0]
            if number > 99999:
                return error(ErrorCode.ORDER_LIMIT_REACHED, "Order ID capacity reached.")
            minute = connection.execute("SELECT minute FROM clock WHERE id=1").fetchone()[0]
            order_id = f"ORD-{number:05d}"
            jitter = random.Random(f"{self.config.seed}:{order_id}").randint(
                0, self.config.eta_jitter_minutes
            )
            order = Order(
                order_id=order_id,
                customer_name=request.customer_name,
                items=lines,
                total=sum((line.line_total for line in lines), Decimal("0")),
                created_minute=minute,
                preparing_minute=minute + self.config.preparing_after_minutes,
                ready_minute=minute + self.config.ready_after_minutes + jitter,
                collected_minute=minute + self.config.collected_after_minutes,
            )
            connection.execute(
                "INSERT INTO orders VALUES (?, ?)", (number, order.model_dump_json())
            )
        return CreateOrderResult(
            order_id=order.order_id, total=order.total, eta_minutes=order.ready_minute - minute
        )

    @staticmethod
    def _read_order(connection: sqlite3.Connection, order_id: str) -> Order:
        row = connection.execute(
            "SELECT body FROM orders WHERE id=?", (int(order_id[4:]),)
        ).fetchone()
        if row is None:
            raise DomainFailure(error(ErrorCode.ORDER_NOT_FOUND, "Order not found."))
        return Order.model_validate_json(row[0])

    def get_order_status(self, request: OrderIDArgs) -> StatusResult:
        with self.store.transaction() as connection:
            order = self._read_order(connection, request.order_id)
            minute = connection.execute("SELECT minute FROM clock WHERE id=1").fetchone()[0]
        eta = 0 if order.status == OrderStatus.CANCELLED else max(0, order.ready_minute - minute)
        return StatusResult(status=order.status, eta_minutes=eta)

    def cancel_order(self, request: OrderIDArgs) -> CancelResult:
        with self.store.transaction() as connection:
            order = self._read_order(connection, request.order_id)
            if order.status in {OrderStatus.READY, OrderStatus.COLLECTED}:
                raise DomainFailure(
                    error(ErrorCode.CANNOT_CANCEL, "The order is already ready or collected.")
                )
            order.status = OrderStatus.CANCELLED
            connection.execute(
                "UPDATE orders SET body=? WHERE id=?",
                (order.model_dump_json(), int(order.order_id[4:])),
            )
        return CancelResult(cancelled=True)

    def advance_time(self, minutes: int) -> None:
        """Advance the logical clock explicitly; observations never advance time."""
        if isinstance(minutes, bool) or not isinstance(minutes, int) or minutes < 0:
            raise ValueError("minutes must be a non-negative integer")
        with self.store.transaction() as connection:
            connection.execute("UPDATE clock SET minute=minute+? WHERE id=1", (minutes,))
            minute = connection.execute("SELECT minute FROM clock WHERE id=1").fetchone()[0]
            for number, body in connection.execute("SELECT id, body FROM orders").fetchall():
                order = Order.model_validate_json(body)
                if order.status == OrderStatus.CANCELLED:
                    continue
                for status, at in [
                    (OrderStatus.PREPARING, order.preparing_minute),
                    (OrderStatus.READY, order.ready_minute),
                    (OrderStatus.COLLECTED, order.collected_minute),
                ]:
                    if minute >= at:
                        order.status = status
                connection.execute(
                    "UPDATE orders SET body=? WHERE id=?", (order.model_dump_json(), number)
                )

    def snapshot(self) -> BackendState:
        with self.store.transaction() as connection:
            minute = connection.execute("SELECT minute FROM clock WHERE id=1").fetchone()[0]
            orders = [
                Order.model_validate_json(row[0])
                for row in connection.execute("SELECT body FROM orders ORDER BY id").fetchall()
            ]
        return BackendState(minute=minute, orders=orders)
