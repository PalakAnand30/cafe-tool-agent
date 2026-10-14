"""Exercise the real Groq SDK over fake HTTP and a fake CPU engine, offline."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from groq import APIStatusError, Groq

from cafe_agent.config import load_runtime
from cafe_agent.models.base import Message, WireFunction, WireToolCall
from cafe_agent.models.groq import GroqClient
from cafe_agent.models.kaggle import HFClient, VLLMClient
from cafe_agent.models.llama_cpp import LlamaCppClient
from cafe_agent.parsing.parser import parse_response
from cafe_agent.prompting.prompts import render_chat, system_prompt
from cafe_agent.tools.schemas import tool_schemas


def completion() -> dict:
    return {
        "id": "test-completion",
        "object": "chat.completion",
        "created": 0,
        "model": "llama-3.3-70b-versatile",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-one",
                            "type": "function",
                            "function": {"name": "get_menu", "arguments": '{"category":"coffee"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def configured(tmp_path: Path):
    config = load_runtime()
    config.groq.cache_dir = tmp_path / "cache"
    return config


def test_groq_native_request_and_cache_key_inputs(tmp_path: Path) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=completion())

    config = configured(tmp_path)
    with Groq(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as sdk:
        client = GroqClient(config, sdk=sdk)
        messages = [Message(role="user", content="Show me coffees.")]
        first = client.generate(messages, tool_schemas())
        second = client.generate(messages, tool_schemas())
        assert not first.cached and second.cached
        assert len(requests) == 1
        assert first.usage.total_tokens == 15
        assert parse_response(first).calls[0].arguments == {"category": "coffee"}
        assert requests[0]["tools"] == tool_schemas()
        assert requests[0]["temperature"] == 0 and requests[0]["seed"] == 42
        assert requests[0]["max_tokens"] == 512 and requests[0]["tool_choice"] == "auto"
        client.generate([Message(role="user", content="Show teas.")], tool_schemas())
        client.generate(messages, tool_schemas()[:1])
        config.generation.max_tokens = 128
        client.generate(messages, tool_schemas())
        config.groq.model = "another-model"
        client.generate(messages, tool_schemas())
        assert len(requests) == 5
    assert len(list(config.groq.cache_dir.glob("*.json"))) == 5
    assert all("test-key" not in path.read_text() for path in config.groq.cache_dir.glob("*.json"))


@pytest.mark.parametrize("status", [429, 500, 503])
def test_groq_backoff_retries_transient_statuses(tmp_path: Path, status: int) -> None:
    attempts, delays = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return (
            httpx.Response(status, json={"error": {"message": "retry"}})
            if len(attempts) < 3
            else httpx.Response(200, json=completion())
        )

    with Groq(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as sdk:
        result = GroqClient(configured(tmp_path), sdk=sdk, sleep=delays.append).generate(
            [Message(role="user", content="Menu?")], tool_schemas()
        )
    assert result.tool_calls
    assert len(attempts) == 3
    assert 1 <= delays[0] <= 2 and 2 <= delays[1] <= 4


@pytest.mark.parametrize("status", [400, 401, 429, 503])
def test_groq_retry_limit_and_permanent_errors(tmp_path: Path, status: int) -> None:
    attempts, delays = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(status, json={"error": {"message": "failure"}})

    config = configured(tmp_path)
    config.groq.max_attempts = 3
    with Groq(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as sdk:
        with pytest.raises(APIStatusError):
            GroqClient(config, sdk=sdk, sleep=delays.append).generate(
                [Message(role="user", content="Menu?")], tool_schemas()
            )
    assert len(attempts) == (3 if status in {429, 503} else 1)
    assert len(delays) == len(attempts) - 1
    assert not list(config.groq.cache_dir.glob("*.json"))


def test_cache_hit_needs_no_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = configured(tmp_path)
    messages = [Message(role="user", content="Menu?")]
    with Groq(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=completion()))
        ),
    ) as sdk:
        GroqClient(config, sdk=sdk).generate(messages, tool_schemas())
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = GroqClient(config, env_path=tmp_path / "missing.env")
    assert client.generate(messages, tool_schemas()).cached
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        client.generate([Message(role="user", content="new")], tool_schemas())


class FakeTokenizer:
    def apply_chat_template(self, messages: list[dict], **kwargs) -> str:
        self.messages, self.kwargs = messages, kwargs
        return "rendered-qwen-prompt"


def test_render_chat_converts_arguments_without_mutation() -> None:
    tokenizer = FakeTokenizer()
    call = WireToolCall(id="a", function=WireFunction(name="get_menu", arguments="{}"))
    messages = [Message(role="assistant", tool_calls=[call])]
    assert render_chat(tokenizer, messages) == "rendered-qwen-prompt"
    assert tokenizer.messages[0]["tool_calls"][0]["function"]["arguments"] == {}
    assert call.function.arguments == "{}"
    assert tokenizer.kwargs == {
        "tools": tool_schemas(),
        "tokenize": False,
        "add_generation_prompt": True,
    }


def test_llama_lazy_load_is_cpu_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    initialisations, requests = [], []

    class FakeLlama:
        def __init__(self, **kwargs) -> None:
            initialisations.append(kwargs)

        def create_completion(self, **kwargs) -> dict:
            requests.append(kwargs)
            return {
                "choices": [
                    {
                        "text": '<tool_call>{"name":"get_menu","arguments":{}}</tool_call>',
                        "finish_reason": "stop",
                    }
                ]
            }

        def close(self) -> None:
            pass

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    config, tokenizer = load_runtime(), FakeTokenizer()
    client = LlamaCppClient(tmp_path / "model.gguf", tokenizer, config)
    assert not initialisations
    result = client.generate([Message(role="user", content="Menu?")], tool_schemas())
    assert initialisations[0]["n_gpu_layers"] == 0
    assert initialisations[0]["seed"] == 42
    assert requests[0]["prompt"] == "rendered-qwen-prompt"
    assert requests[0]["temperature"] == 0 and requests[0]["max_tokens"] == 512
    assert requests[0]["stop"] == ["<|im_end|>"]
    assert parse_response(result).calls[0].name == "get_menu"
    client.close()


@pytest.mark.parametrize("client", [HFClient(), VLLMClient()])
def test_gpu_clients_are_explicit_stubs(client) -> None:
    with pytest.raises(NotImplementedError, match="Phase 4"):
        client.generate([], [])


def test_prompt_encodes_required_behaviours() -> None:
    prompt = system_prompt()
    for text in [
        "create_order",
        "check_availability",
        "clarification",
        "decline",
        "alternatives",
        "unsupported modifiers",
        "quantity",
        "latte",
        "iced latte",
        "chai latte",
    ]:
        assert text in prompt
