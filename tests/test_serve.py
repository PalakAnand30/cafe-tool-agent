"""Session boundaries and deployment integrity without loading a model in unit tests."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from cafe_agent.models.base import ModelResponse
from cafe_agent.serve.app import create_app
from cafe_agent.serve.deployment import prepare_bundle, publish_bundle
from cafe_agent.serve.service import ChatRequest, ChatService, SessionError
from cafe_agent.serve.settings import serve_settings


def call(name, arguments):
    return ModelResponse(
        content="<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"
    )


class Model:
    def __init__(self, responses=()):
        self.responses = iter(responses)
        self.histories = []

    def reset(self):
        pass

    def generate(self, messages, tools):
        self.histories.append([m.model_copy(deep=True) for m in messages])
        return next(self.responses, ModelResponse(content="What size would you like?"))


def order_model():
    return Model(
        [
            call(
                "create_order",
                {
                    "items": [
                        {"item_name": "latte", "size": "medium", "quantity": 1, "modifiers": []}
                    ]
                },
            ),
            ModelResponse(content="Your order is ORD-00001."),
        ]
    )


def test_chat_session_state_money_and_customer_isolation():
    model = order_model()
    with TestClient(create_app(client=model, with_ui=False)) as client:
        first = client.post("/chat", json={"message": "One medium latte."}).json()
        assert first["current_order"]["order_id"] == "ORD-00001"
        assert isinstance(first["current_order"]["total"], str)
        assert first["trace"][0]["call"]["name"] == "create_order"
        other = client.post("/chat", json={"message": "What is available?"}).json()
        assert other["session_id"] != first["session_id"]
        assert other["state"]["orders"] == []
        assert all("ORD-00001" not in (m.content or "") for m in model.histories[-1])
        key = first["session_id"]
        tick = client.post("/session/advance", json={"session_id": key}).json()
        assert tick["minute"] == serve_settings().advance_minutes
        assert tick["orders"][0]["status"] == "preparing"
        observed = client.post("/chat", json={"session_id": key, "message": "Hello?"}).json()
        assert observed["state"]["minute"] == tick["minute"]
        assert client.post("/session/reset", json={"session_id": key}).status_code == 204
        assert client.post("/chat", json={"session_id": key, "message": "Hi"}).status_code == 404
        assert (
            client.post(
                "/chat", json={"session_id": other["session_id"], "message": "Hi"}
            ).status_code
            == 200
        )


def test_session_expiry_capacity_turn_limits_and_invalid_messages(monkeypatch):
    settings = serve_settings().model_copy(update={"max_sessions": 1, "max_turns": 1})
    service = ChatService(Model(), settings)
    try:
        first = service.chat(ChatRequest(message="Hello"))
        with pytest.raises(SessionError) as error:
            service.chat(ChatRequest(message="Another customer"))
        assert error.value.code == "SESSION_CAPACITY"
        with pytest.raises(SessionError) as error:
            service.chat(ChatRequest(session_id=first.session_id, message="Again"))
        assert error.value.code == "TURN_LIMIT_RESET_REQUIRED"
        now = time.monotonic() + settings.session_ttl_seconds + 1
        monkeypatch.setattr("cafe_agent.serve.service.time.monotonic", lambda: now)
        with pytest.raises(SessionError) as error:
            service.chat(ChatRequest(session_id=first.session_id, message="Old"))
        assert error.value.code == "SESSION_EXPIRED"
        assert service.chat(ChatRequest(message="New")).session_id != first.session_id
    finally:
        service.close()
    with TestClient(create_app(client=Model(), with_ui=False)) as client:
        for body in [
            {"message": "   "},
            {"message": "x", "session_id": "../../etc/passwd"},
            {"message": "x", "extra": True},
            {"message": "x" * 2001},
        ]:
            assert client.post("/chat", json=body).status_code == 422


def test_failed_generation_preserves_tools_and_disables_unsafe_retry():
    class Failing(Model):
        def generate(self, messages, tools):
            if messages[-1].role == "tool":
                raise RuntimeError("provider details must not reach customers")
            return call(
                "create_order",
                {
                    "items": [
                        {"item_name": "latte", "size": "medium", "quantity": 1, "modifiers": []}
                    ]
                },
            )

    with TestClient(create_app(client=Failing(), with_ui=False)) as client:
        result = client.post("/chat", json={"message": "One medium latte"}).json()
        assert result["error"] == "MODEL_ERROR"
        assert len(result["state"]["orders"]) == 1
        assert result["trace"][0]["call"]["name"] == "create_order"
        assert "provider details" not in json.dumps(result)
        assert (
            client.post(
                "/chat", json={"session_id": result["session_id"], "message": "Retry"}
            ).status_code
            == 409
        )


def test_model_context_is_serialized_between_api_requests():
    class Serialized(Model):
        def __init__(self):
            self.active = threading.Lock()
            self.calls = 0

        def generate(self, messages, tools):
            assert self.active.acquire(blocking=False), "concurrent model context use"
            try:
                time.sleep(0.01)
                self.calls += 1
                return ModelResponse(content="Hello")
            finally:
                self.active.release()

    model = Serialized()
    service = ChatService(model, serve_settings())
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda i: service.chat(ChatRequest(message=str(i))), range(4)))
        assert model.calls == 4
        assert len({r.session_id for r in results}) == 4
    finally:
        service.close()


def test_parse_failure_after_order_keeps_trace_and_requires_reset():
    model = order_model()
    model.responses = iter(
        [
            call(
                "create_order",
                {
                    "items": [
                        {"item_name": "latte", "size": "medium", "quantity": 1, "modifiers": []}
                    ]
                },
            ),
            ModelResponse(content="<tool_call>not JSON</tool_call>"),
        ]
    )
    with TestClient(create_app(client=model, with_ui=False)) as client:
        result = client.post("/chat", json={"message": "One medium latte"}).json()
        assert result["error"] == "PARSE_ERROR"
        assert result["trace"][0]["result"]["order_id"] == "ORD-00001"
        assert result["state"]["orders"][0]["order_id"] == "ORD-00001"
        assert (
            client.post(
                "/chat", json={"session_id": result["session_id"], "message": "Retry"}
            ).status_code
            == 409
        )


@pytest.mark.filterwarnings("ignore:websockets.legacy is deprecated:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:'audioop' is deprecated:DeprecationWarning")
def test_gradio_page_and_api_share_the_initialized_service():
    with TestClient(create_app(client=order_model())) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").json()["n_gpu_layers"] == 0
        config = client.get("/demo/config").json()
        assert sum(c["type"] == "chatbot" for c in config["components"]) == 1
        result = client.post("/chat", json={"message": "One medium latte"}).json()
        assert result["current_order"]["order_id"] == "ORD-00001"


def test_prepared_space_is_allowlisted_reproducible_and_tamper_checked(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path, receipt = prepare_bundle(root, tmp_path)
    again, repeated = prepare_bundle(root, tmp_path)
    assert path == again and receipt == repeated and not receipt["published"]
    assert "Dockerfile" in receipt["files"]
    assert not any(
        name.startswith((".env", "data/", "results/", "models/", ".git/"))
        for name in receipt["files"]
    )
    (path / ".env").write_text("DO_NOT_UPLOAD=secret")
    with pytest.raises(ValueError, match="changed"):
        prepare_bundle(root, tmp_path)


def test_deploy_preserves_existing_space_visibility_and_rejects_wrong_sdk(tmp_path):
    actions = []
    api = SimpleNamespace(
        space_info=lambda repo: SimpleNamespace(sdk="docker", private=True),
        add_space_secret=lambda *a, **k: actions.append(("secret", k["key"])),
        upload_folder=lambda **k: actions.append(("upload", k["repo_id"]))
        or SimpleNamespace(oid="a" * 40),
    )
    assert publish_bundle(api, "test/cafe", tmp_path, "private-token") == "a" * 40
    assert actions == [("secret", "HF_TOKEN"), ("upload", "test/cafe")]
    api.space_info = lambda repo: SimpleNamespace(sdk="gradio")
    with pytest.raises(ValueError, match="Docker"):
        publish_bundle(api, "test/cafe", tmp_path, "private-token")
    assert len(actions) == 2
