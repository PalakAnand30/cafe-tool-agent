"""Shared tool schemas and HTTP/in-process execution must obey one contract."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from cafe_agent.backend.app import create_app
from cafe_agent.backend.service import CafeService
from cafe_agent.parsing.parser import ToolCall
from cafe_agent.tools.executor import HTTPToolExecutor, ToolExecutor
from cafe_agent.tools.schemas import ToolName, tool_schemas


def test_schemas_are_self_contained_and_constrained() -> None:
    tools = tool_schemas()
    assert {tool["function"]["name"] for tool in tools} == set(ToolName)
    assert "$ref" not in json.dumps(tools)
    order = next(
        tool["function"]["parameters"]
        for tool in tools
        if tool["function"]["name"] == "create_order"
    )
    item = order["properties"]["items"]["items"]
    assert item["properties"]["quantity"]["minimum"] == 1
    assert item["properties"]["quantity"]["maximum"] == 10
    assert item["additionalProperties"] is False
    assert "quantity" in item["required"]


def test_http_and_local_executor_parity(latte_order: dict) -> None:
    local_service, http_service = CafeService(), CafeService()
    try:
        with TestClient(create_app(http_service)) as client:
            remote = HTTPToolExecutor(client)
            local = ToolExecutor(local_service)
            calls = [
                ToolCall(name="get_menu", arguments={"category": "tea"}),
                ToolCall(name="check_availability", arguments={"item_name": "orange juice"}),
                ToolCall(name="create_order", arguments=latte_order),
                ToolCall(name="get_order_status", arguments={"order_id": "ORD-00001"}),
                ToolCall(name="cancel_order", arguments={"order_id": "ORD-00001"}),
                ToolCall(name="create_order", arguments={"items": []}),
                ToolCall(name="refund", arguments={}),
            ]
            for call in calls:
                assert remote.execute(call) == local.execute(call)
            assert local_service.snapshot() == http_service.snapshot()
    finally:
        local_service.close()
        http_service.close()


def test_rest_routes(cafe: CafeService, latte_order: dict) -> None:
    with TestClient(create_app(cafe)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/menu", params={"category": "bakery"}).status_code == 200
        assert client.post("/availability", json={"item_name": "latte"}).json()["available"]
        created = client.post("/orders", json=latte_order)
        assert created.status_code == 200
        assert created.json()["total"] == "9.00"
        order_id = created.json()["order_id"]
        assert client.get(f"/orders/{order_id}").json()["status"] == "received"
        assert client.post(f"/orders/{order_id}/cancel").json()["cancelled"]
        assert client.get("/orders/ORD-12345").status_code == 404
        assert client.get("/openapi.json").status_code == 200


@pytest.mark.parametrize(
    ("method", "path", "kwargs", "code"),
    [
        ("get", "/menu?category=pizza", {}, "INVALID_ARGUMENTS"),
        ("get", "/orders/oops", {}, "INVALID_ARGUMENTS"),
        ("post", "/orders", {"json": {"items": []}}, "INVALID_ARGUMENTS"),
        (
            "post",
            "/orders",
            {"content": "{bad", "headers": {"Content-Type": "application/json"}},
            "INVALID_ARGUMENTS",
        ),
        ("post", "/tools/get_menu", {"json": []}, "INVALID_ARGUMENTS"),
        ("post", "/tools/unknown", {"json": {}}, "UNKNOWN_TOOL"),
    ],
)
def test_structured_http_errors(
    cafe: CafeService, method: str, path: str, kwargs: dict, code: str
) -> None:
    with TestClient(create_app(cafe)) as client:
        result = client.request(method, path, **kwargs)
        assert result.status_code in {404, 422}
        assert result.json()["error"]["code"] == code
        assert not cafe.snapshot().orders


def test_http_conflicts(cafe: CafeService, latte_order: dict) -> None:
    with TestClient(create_app(cafe)) as client:
        order_id = client.post("/orders", json=latte_order).json()["order_id"]
        cafe.advance_time(100)
        assert client.post(f"/orders/{order_id}/cancel").status_code == 409
        result = client.post(
            "/orders",
            json={"items": [{"item_name": "filter coffee", "size": "small", "quantity": 1}]},
        )
        assert result.status_code == 409
        assert result.json()["error"]["alternatives"]


def test_http_transport_errors_are_not_domain_errors() -> None:
    with httpx.Client(
        base_url="https://cafe.test",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(503, json={"detail": "unavailable"})
        ),
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            HTTPToolExecutor(client).execute(ToolCall(name="get_menu", arguments={}))
