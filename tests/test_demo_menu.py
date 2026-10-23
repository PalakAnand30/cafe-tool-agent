"""Menu browsing stays read-only, isolated from model ordering and evaluation."""

import json

import pytest
from fastapi.testclient import TestClient

from cafe_agent.models.base import ModelResponse
from cafe_agent.serve.app import create_app
from cafe_agent.serve.menu import menu_request
from cafe_agent.serve.presentation import menu_cards, order_board


class OrderingModel:
    def __init__(self):
        self.calls = 0

    def reset(self):
        pass

    def generate(self, messages, tools):
        self.calls += 1
        if messages[-1].role == "tool":
            return ModelResponse(content="Your latte is ordered.")
        return ModelResponse(
            content="<tool_call>"
            + json.dumps(
                {
                    "name": "create_order",
                    "arguments": {
                        "items": [
                            {"item_name": "latte", "size": "medium", "quantity": 1, "modifiers": []}
                        ]
                    },
                }
            )
            + "</tool_call>"
        )


def test_full_menu_then_food_menu_then_order_works_in_one_session():
    model = OrderingModel()
    with TestClient(create_app(client=model, with_ui=False)) as client:
        first = client.post("/chat", json={"message": "Show me the menu"}).json()
        assert first["source"] == "menu" and first["error"] is None
        assert len(first["trace"][0]["result"]["items"]) == 25
        food = client.post(
            "/chat", json={"message": "show me the food menu", "session_id": first["session_id"]}
        ).json()
        assert food["source"] == "menu" and food["error"] is None
        assert [t["call"]["arguments"] for t in food["trace"]] == [
            {"category": "bakery"},
            {"category": "sandwiches"},
        ]
        assert "£2.50" in food["reply"] and "£4.50" in food["reply"]
        assert "out of stock" in food["reply"]
        assert food["state"]["orders"] == [] and food["state"]["minute"] == 0
        assert model.calls == 0
        ordered = client.post(
            "/chat",
            json={"message": "One medium latte, please.", "session_id": first["session_id"]},
        ).json()
        assert ordered["source"] == "model" and ordered["error"] is None
        assert ordered["current_order"]["total"] == "3.50"
        assert model.calls == 2


@pytest.mark.parametrize(
    "message",
    [
        "Show me the food menu and order a sandwich",
        "Cancel my order and show me the menu",
        "Show me the pizza menu",
        "Order from the food menu",
        "My name is Menu",
    ],
)
def test_menu_shortcuts_do_not_swallow_orders_or_unknown_requests(message):
    assert menu_request(message) is None


@pytest.mark.parametrize(
    ("message", "category"),
    [
        ("Show me the menu", "all"),
        ("show me the food menu", "food"),
        ("Could you show me your bakery menu please?", "bakery"),
        ("Can I see the tea menu?", "tea"),
        ("Cold drinks menu", "cold_drinks"),
    ],
)
def test_standalone_menu_requests(message, category):
    assert menu_request(message) == category


def test_food_cards_show_backend_stock_and_receipts_escape_customer_content():
    menu = menu_cards("food", [])
    assert menu.count("<article") == 8
    assert "Almond croissant" in menu and "Out of stock" in menu
    receipt = order_board(
        {
            "orders": [
                {
                    "order_id": "ORD-00001",
                    "status": "received",
                    "total": "3.90",
                    "items": [
                        {
                            "item_name": "<script>bad</script>",
                            "size": "medium",
                            "quantity": 1,
                            "modifiers": ["oat_milk"],
                            "line_total": "3.90",
                        }
                    ],
                }
            ]
        }
    )
    assert "<script>" not in receipt and "&lt;script&gt;" in receipt
    assert "Medium · oat milk" in receipt and "£3.90" in receipt
