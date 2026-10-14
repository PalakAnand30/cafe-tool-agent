"""One registry drives Pydantic validation and OpenAI function schemas."""

from enum import StrEnum
from typing import Any

from pydantic import Field

from cafe_agent.config import StrictModel
from cafe_agent.domain.models import Category, Name, OrderID, OrderItem, Size


class ToolName(StrEnum):
    GET_MENU = "get_menu"
    CHECK_AVAILABILITY = "check_availability"
    CREATE_ORDER = "create_order"
    GET_ORDER_STATUS = "get_order_status"
    CANCEL_ORDER = "cancel_order"


class GetMenuArgs(StrictModel):
    category: Category | None = None


class CheckAvailabilityArgs(StrictModel):
    item_name: Name
    size: Size | None = None


class CreateOrderArgs(StrictModel):
    customer_name: Name | None = None
    items: list[OrderItem] = Field(min_length=1)


class OrderIDArgs(StrictModel):
    order_id: OrderID


ARGUMENT_MODELS: dict[ToolName, type[StrictModel]] = {
    ToolName.GET_MENU: GetMenuArgs,
    ToolName.CHECK_AVAILABILITY: CheckAvailabilityArgs,
    ToolName.CREATE_ORDER: CreateOrderArgs,
    ToolName.GET_ORDER_STATUS: OrderIDArgs,
    ToolName.CANCEL_ORDER: OrderIDArgs,
}

DESCRIPTIONS = {
    ToolName.GET_MENU: "Browse the menu, optionally filtering by category.",
    ToolName.CHECK_AVAILABILITY: "Check stock only when the customer asks about availability.",
    ToolName.CREATE_ORDER: "Place a complete order directly, without a preliminary stock check.",
    ToolName.GET_ORDER_STATUS: "Look up an existing order's status and remaining preparation time.",
    ToolName.CANCEL_ORDER: "Cancel an order before it is ready.",
}


def validate_arguments(name: str, arguments: dict[str, Any]) -> StrictModel:
    return ARGUMENT_MODELS[ToolName(name)].model_validate(arguments)


def tool_schemas() -> list[dict[str, Any]]:
    """Inline definitions so function schemas work independently on every provider."""

    def inline(value: Any, definitions: dict[str, Any]) -> Any:
        if isinstance(value, list):
            return [inline(item, definitions) for item in value]
        if isinstance(value, dict):
            if "$ref" in value:
                return inline(definitions[value["$ref"].split("/")[-1]], definitions)
            return {k: inline(v, definitions) for k, v in value.items() if k != "$defs"}
        return value

    tools = []
    for name, model in ARGUMENT_MODELS.items():
        schema = model.model_json_schema()
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name.value,
                    "description": DESCRIPTIONS[name],
                    "parameters": inline(schema, schema.get("$defs", {})),
                },
            }
        )
    return tools
