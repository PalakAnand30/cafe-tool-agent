"""CPU-only contracts and evidence-based report validation, without real model inference."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cafe_agent.cpu.client import CPUClient
from cafe_agent.cpu.settings import local_settings
from cafe_agent.eval.actions import Action
from cafe_agent.eval.latency import benchmark_cases, summarize_latency
from cafe_agent.models.base import Message
from cafe_agent.parsing.parser import ToolCall
from cafe_agent.reporting.aggregate import checked_manifest, local_cost
from cafe_agent.reporting.cli import verified_latency
from cafe_agent.reporting.errors import action_failure


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is False
        return "prompt"

    def encode(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        return [1, 2, 3]


class Engine:
    def __init__(self, ids=None):
        self.ids = ids if ids is not None else [1, 2, 3]
        self.calls = []

    def tokenize(self, text, **kwargs):
        assert kwargs == {"add_bos": False, "special": True}
        return self.ids

    def create_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "choices": [{"text": "Which size?", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }


def test_cpu_exact_prompt_ids_and_unpenalized_greedy_decoding():
    engine = Engine()
    client = CPUClient(
        local_settings(),
        Tokenizer(),
        max_tokens=512,
        model_path=Path("ignored.gguf"),
        engine=engine,
    )
    response = client.generate([Message(role="user", content="latte")], [])
    assert response.content == "Which size?" and response.usage.total_tokens == 7
    assert engine.calls == [
        {
            "prompt": [1, 2, 3],
            "temperature": 0,
            "max_tokens": 512,
            "seed": 42,
            "stop": ["<|im_end|>"],
            "repeat_penalty": 1.0,
        }
    ]


def test_cpu_tokenizer_mismatch_never_calls_engine():
    engine = Engine([1, 9, 3])
    client = CPUClient(
        local_settings(),
        Tokenizer(),
        max_tokens=512,
        model_path=Path("ignored.gguf"),
        engine=engine,
    )
    with pytest.raises(ValueError, match="tokenizer disagree"):
        client.generate([Message(role="user", content="latte")], [])
    assert engine.calls == []


def test_cpu_context_overflow_never_loads_or_truncates():
    settings = local_settings()
    settings.cpu.context = 514
    client = CPUClient(settings, Tokenizer(), max_tokens=512, model_path=Path("absent.gguf"))
    response = client.generate([Message(role="user", content="x")], [])
    assert response.content == "CONTEXT_LIMIT_EXCEEDED"
    assert client.engine is None


def test_cpu_load_disables_layers_and_kqv(monkeypatch):
    import sys

    calls = []
    monkeypatch.setitem(
        sys.modules, "llama_cpp", SimpleNamespace(Llama=lambda **kw: calls.append(kw) or Engine())
    )
    client = CPUClient(local_settings(), Tokenizer(), max_tokens=512, model_path=Path("x.gguf"))
    client.load()
    assert calls[0]["n_gpu_layers"] == 0 and calls[0]["offload_kqv"] is False


def test_latency_prompt_selection_is_order_independent_and_complete():
    records = [
        SimpleNamespace(id=f"case-{i:03}", actions=[SimpleNamespace(message_index=2)])
        for i in range(110)
    ]
    selected = benchmark_cases(records, 100, 42)
    assert selected == benchmark_cases(list(reversed(records)), 100, 42)
    assert len({r.id for r, _ in selected}) == 100
    with pytest.raises(ValueError, match="Not enough"):
        benchmark_cases(records, 111, 42)
    summary = summarize_latency([{"seconds": 1}, {"seconds": 3}], 100)
    assert not summary["complete"] and summary["p50_seconds"] == 2
    assert summary["p95_seconds"] == pytest.approx(2.9)


def test_cpu_cost_uses_wall_time_not_groq_tokens():
    rows = [SimpleNamespace(mode="end_to_end", observations=[SimpleNamespace(latency_seconds=36)])]
    cost = local_cost(rows, local_settings(), bootstrap=True)
    assert cost["usd_per_1000_conversations"] == pytest.approx(1)
    assert cost["ci95"] == [cost["usd_per_1000_conversations"]] * 2


def test_report_rejects_mutated_manifest(tmp_path):
    value = {"system": "S1", "sources": {}}
    fingerprint = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    path = tmp_path / "S1__manifest.json"
    path.write_text(json.dumps({"fingerprint": fingerprint, **value}))
    assert checked_manifest(tmp_path, "S1")["system"] == "S1"
    path.write_text(path.read_text().replace('"S1"', '"S2"'))
    with pytest.raises(ValueError, match="fingerprint"):
        checked_manifest(tmp_path, "S1")


def test_failure_categories_distinguish_invalid_action_and_arguments():
    expected = Action(behaviour="tool", calls=[ToolCall(name="get_menu", arguments={})])
    assert (
        action_failure(expected, Action(behaviour="invalid", parse_valid=False)) == "invalid_output"
    )
    assert action_failure(expected, Action(behaviour="clarify")) == "wrong_action"
    actual = Action(
        behaviour="tool", calls=[ToolCall(name="get_menu", arguments={"category": "tea"})]
    )
    assert action_failure(expected, actual) == "wrong_arguments"
    assert action_failure(expected, expected) is None


def test_latency_report_rejects_duplicate_requests(tmp_path):
    manifest = tmp_path / "S5__manifest.json"
    manifest.write_text("{}")
    source = Path(__file__).resolve().parents[1] / "src/cafe_agent/eval/latency.py"
    (tmp_path / "cpu_latency_manifest.json").write_text(
        json.dumps(
            {
                "s5_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "cases": [[f"case-{i}", 2] for i in range(100)],
            }
        )
    )
    (tmp_path / "cpu_latency.jsonl").write_text('{"index": 0}\n' * 100)
    with pytest.raises(ValueError, match="100 complete, unique"):
        verified_latency(tmp_path, manifest)


@pytest.mark.parametrize("bad_hash", [False, True])
def test_download_verifies_hub_hash_and_keeps_credentials_out_of_receipt(
    tmp_path, monkeypatch, bad_hash
):
    from cafe_agent.config import read_config
    from cafe_agent.cpu import download

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(download, "load_dotenv", lambda: None)
    monkeypatch.setenv("HF_TOKEN", "FAKE_PRIVATE_CREDENTIAL")
    settings = local_settings()
    payload = b"GGUF"
    digest = hashlib.sha256(payload).hexdigest()
    entry = SimpleNamespace(
        rfilename=settings.model.filename,
        size=len(payload),
        lfs=SimpleNamespace(sha256="0" * 64 if bad_hash else digest),
    )

    class API:
        def __init__(self, token):
            assert token == "FAKE_PRIVATE_CREDENTIAL"

        def model_info(self, repo, **kwargs):
            assert kwargs["revision"] == settings.model.revision
            return SimpleNamespace(sha=settings.model.revision, siblings=[entry], private=True)

    def fetch(repo, filename, **kwargs):
        assert kwargs["token"] == "FAKE_PRIVATE_CREDENTIAL"
        path = kwargs["local_dir"] / filename
        if filename.endswith(".gguf"):
            path.write_bytes(payload)
        else:
            path.write_text(
                json.dumps({"train": {"revision": read_config("train.yaml")["revision"]}})
            )
        return str(path)

    monkeypatch.setattr(download, "HfApi", API)
    monkeypatch.setattr(download, "hf_hub_download", fetch)
    if bad_hash:
        with pytest.raises(ValueError, match="hash/size"):
            download.download_model(settings)
        assert not Path("results/model_download.json").exists()
    else:
        receipt = download.download_model(settings)
        assert receipt["sha256"] == digest and receipt["hub_lfs_sha256_verified"]
        assert "FAKE_PRIVATE_CREDENTIAL" not in json.dumps(receipt)


def test_full_report_uses_no_groq_prices_for_gpu(tmp_path, monkeypatch):
    from cafe_agent.eval.config import load_eval_config
    from cafe_agent.reporting import aggregate

    monkeypatch.chdir(tmp_path)
    settings = local_settings()
    settings.report.systems = ["S1"]
    directory = tmp_path / "predictions"
    directory.mkdir()
    (directory / "S1__manifest.json").write_text("{}")
    config = load_eval_config()
    manifest = {
        "eval": config.model_dump(mode="json"),
        "fingerprint": "fixture",
        "gpu": {"backend": "transformers", "device": "T4", "revision": "a" * 40},
    }
    monkeypatch.setattr(aggregate, "prediction_directory", lambda *args: directory)
    monkeypatch.setattr(aggregate, "checked_manifest", lambda *args: manifest)

    def compute(selected, output, system):
        assert selected.pricing.input_per_million_usd == 0
        assert selected.pricing.output_per_million_usd == 0
        mode = {
            "metrics": {"paid_equivalent_usd_per_1000_conversations": {"value": 0}},
            "per_intent": {},
        }
        return {
            "groups": [
                {"split": "test", "complete": True, "modes": {"turn": mode, "end_to_end": mode}}
            ],
            "definitions": {},
            "bootstrap": {},
            "scorer_sha256": "frozen",
        }

    monkeypatch.setattr(aggregate, "compute_results", compute)
    report = aggregate.compute_comparison(settings)
    group = report["groups"][0]
    assert group["cost"]["usd_per_1000_conversations"] is None
    assert (
        group["modes"]["turn"]["metrics"]["paid_equivalent_usd_per_1000_conversations"]["value"]
        is None
    )
