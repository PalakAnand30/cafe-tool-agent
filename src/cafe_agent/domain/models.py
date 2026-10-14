"""Domain values, order state, and shared structured tool results."""

from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, StrictInt, model_validator

from cafe_agent.config import StrictModel, read_config

Money = Annotated[Decimal, Field(ge=0, max_digits=8, decimal_places=2, allow_inf_nan=False)]
Name = Annotated[str, Field(min_length=1, max_length=100, strict=True)]
OrderID = Annotated[str, Field(pattern=r"^ORD-[0-9]{5}$", strict=True)]
Quantity = Annotated[StrictInt, Field(ge=1, le=10)]


class Category(StrEnum):
    COFFEE = "coffee"
    TEA = "tea"
    COLD_DRINKS = "cold_drinks"
    BAKERY = "bakery"
    SANDWICHES = "sandwiches"


class Size(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class OrderStatus(StrEnum):
    RECEIVED = "received"
    PREPARING = "preparing"
    READY = "ready"
    COLLECTED = "collected"
    CANCELLED = "cancelled"


class ErrorCode(StrEnum):
    ITEM_NOT_FOUND = "ITEM_NOT_FOUND"
    OUT_OF_STOCK = "OUT_OF_STOCK"
    UNSUPPORTED_MODIFIER = "UNSUPPORTED_MODIFIER"
    SIZE_REQUIRED = "SIZE_REQUIRED"
    ORDER_NOT_FOUND = "ORDER_NOT_FOUND"
    CANNOT_CANCEL = "CANNOT_CANCEL"
    INVALID_SIZE = "INVALID_SIZE"
    AMBIGUOUS_ITEM = "AMBIGUOUS_ITEM"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    ORDER_LIMIT_REACHED = "ORDER_LIMIT_REACHED"


class MenuItem(StrictModel):
    name: Name
    category: Category
    prices: dict[Size, Money] = Field(default_factory=dict)
    price: Money | None = None
    modifiers: dict[str, Money] = Field(default_factory=dict)
    available: bool = True

    @model_validator(mode="after")
    def check_prices(self) -> "MenuItem":
        drink = self.category in {Category.COFFEE, Category.TEA, Category.COLD_DRINKS}
        if drink and (set(self.prices) != set(Size) or self.price is not None):
            raise ValueError("Drinks require all three size prices and no unsized price")
        if not drink and (self.price is None or self.prices):
            raise ValueError("Food requires one unsized price")
        return self


class DomainConfig(StrictModel):
    seed: int
    currency: str
    preparing_after_minutes: int = Field(ge=0)
    ready_after_minutes: int = Field(gt=0)
    collected_after_minutes: int = Field(gt=0)
    eta_jitter_minutes: int = Field(ge=0)
    alternatives_limit: int = Field(gt=0)
    items: list[MenuItem] = Field(min_length=1)

    @model_validator(mode="after")
    def check_domain(self) -> "DomainConfig":
        names = [normalise(item.name) for item in self.items]
        if len(names) != len(set(names)):
            raise ValueError("Menu names must be unique")
        if not self.preparing_after_minutes < self.ready_after_minutes:
            raise ValueError("Preparing must precede ready")
        if self.ready_after_minutes + self.eta_jitter_minutes >= self.collected_after_minutes:
            raise ValueError("Collected must follow the latest ready time")
        return self


def normalise(value: str) -> str:
    return " ".join(value.casefold().split())


def load_domain(path: Path | None = None) -> DomainConfig:
    return DomainConfig.model_validate(read_config("domain.yaml", path))


class OrderItem(StrictModel):
    item_name: Name
    size: Size | None = None
    quantity: Quantity
    modifiers: list[Name] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_modifiers(self) -> "OrderItem":
        if len(self.modifiers) != len(set(self.modifiers)):
            raise ValueError("Duplicate modifiers are not allowed")
        return self


class OrderLine(OrderItem):
    unit_price: Money
    line_total: Money


class Order(StrictModel):
    order_id: OrderID
    customer_name: Name | None = None
    items: list[OrderLine]
    total: Money
    status: OrderStatus = OrderStatus.RECEIVED
    created_minute: int
    preparing_minute: int
    ready_minute: int
    collected_minute: int


class ToolError(StrictModel):
    code: ErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    alternatives: list[MenuItem] = Field(default_factory=list)


class ErrorResult(StrictModel):
    error: ToolError


class MenuResult(StrictModel):
    currency: str
    items: list[MenuItem]


class AvailabilityResult(StrictModel):
    available: bool
    alternatives: list[MenuItem] = Field(default_factory=list)


class CreateOrderResult(StrictModel):
    order_id: OrderID
    total: Money
    eta_minutes: int


class StatusResult(StrictModel):
    status: OrderStatus
    eta_minutes: int


class CancelResult(StrictModel):
    cancelled: bool
    reason: str | None = None


ToolResult = (
    ErrorResult | MenuResult | AvailabilityResult | CreateOrderResult | StatusResult | CancelResult
)


class BackendState(StrictModel):
    minute: int
    orders: list[Order]
