"""Notebook/data contracts, masking, fallback isolation and real shared-runner fixtures."""

import hashlib
import json
import random
import runpy
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from cafe_agent.config import load_runtime, read_config
from cafe_agent.data.config import load_data_config
from cafe_agent.data.schema import Intent
from cafe_agent.data.templates import generate_example
from cafe_agent.eval.config import load_eval_config
from cafe_agent.eval.runner import run_records
from cafe_agent.kaggle.bundle import bundle_predictions
from cafe_agent.kaggle.checkpoint import prepare_run, restore_predictions
from cafe_agent.kaggle.clients import ContextLimitError, PromptClient, VLLMClient
from cafe_agent.kaggle.data import encode_records, pad_batch, read_records, select_train
from cafe_agent.kaggle.evaluate import few_shot_messages, gpu_manifest
from cafe_agent.kaggle.notebook_validation import Notebook, validate_notebook
from cafe_agent.kaggle.settings import train_settings
from cafe_agent.kaggle.train import training_arguments
from cafe_agent.models.base import Message, ModelResponse
from cafe_agent.parsing.parser import ParseError, parse_response

ROOT = Path(__file__).resolve().parents[1]


def example(split="train", key="fixture"):
    return generate_example(
        load_data_config(),
        key,
        split,
        Intent.SINGLE_ORDER,
        4 if split == "test" else 0,
        False,
        random.Random(42),
    )


def notebook_source(name, needle):
    value = json.loads((ROOT / "notebooks" / name).read_text())
    return next(
        "".join(cell["source"]) for cell in value["cells"] if needle in "".join(cell["source"])
    )


def test_data_hash_and_authorship_guard(tmp_path):
    row = example()
    path = tmp_path / "train.jsonl"
    path.write_text(row.model_dump_json() + "\n")
    spec = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "conversations": 1}
    assert read_records(path, spec, "train") == [row]
    with pytest.raises(ValueError, match="split/authorship"):
        read_records(path, spec, "test")
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_records(path, spec, "train")


def test_selection_never_uses_held_out_data():
    rows = [example(key=f"train-{index}") for index in range(5)]
    assert select_train(rows, 0.4, 42) == select_train(list(reversed(rows)), 0.4, 42)
    assert len(select_train(rows, 0.4, 42)) == 2
    with pytest.raises(ValueError):
        select_train([example("test")], 1, 42)
    with pytest.raises(ValueError):
        select_train(rows, 0, 42)


def test_collator_preserves_assistant_masks_and_padding(tokenizer):
    encoded = encode_records([example(), example(key="other")], tokenizer, 4096)
    shorter = {name: value[:-1] for name, value in encoded[1].items()}
    batch = pad_batch([encoded[0], shorter], tokenizer.pad_token_id)
    assert batch["labels"][0] == encoded[0]["labels"]
    assert batch["labels"][1][:-1] == shorter["labels"]
    assert batch["labels"][1][-1] == -100 and batch["attention_mask"][1][-1] == 0
    assert any(label == -100 for label in batch["labels"][0])
    with pytest.raises(ValueError, match="Only train/validation"):
        encode_records([example("test")], tokenizer, 4096)


def test_training_arguments_preserve_labels_and_half_epoch_schedule():
    args = training_arguments(train_settings(), 3500, 4, 4)
    assert args["eval_steps"] == args["save_steps"] == 110
    assert args["dataset_kwargs"] == {"skip_prepare_dataset": True}
    assert args["remove_unused_columns"] is False and args["packing"] is False
    assert args["fp16"] is True and args["bf16"] is False
    assert args["load_best_model_at_end"] is True and args["max_length"] == 4096
    assert args["label_names"] == ["labels"]


def test_training_resume_rejects_different_inputs_and_reuses_complete_output(tmp_path):
    output = tmp_path / "run"
    manifest = {"rank": 16, "dataset_revision": "fixed"}
    assert prepare_run(output, manifest, None) is False
    checkpoint = output / "checkpoints/checkpoint-110"
    checkpoint.mkdir(parents=True)
    assert prepare_run(output, manifest, checkpoint) is False
    with pytest.raises(ValueError, match="changed"):
        prepare_run(output, {**manifest, "rank": 8}, checkpoint)
    with pytest.raises(ValueError, match="matching run"):
        prepare_run(tmp_path / "other", {**manifest, "rank": 8}, checkpoint)
    (output / "training_summary.json").write_text("{}")
    with pytest.raises(ValueError, match="no saved adapter"):
        prepare_run(output, manifest, None)
    (output / "adapter").mkdir()
    (output / "adapter/adapter_model.safetensors").write_bytes(b"fixture")
    assert prepare_run(output, manifest, None) is True


def test_restore_preserves_newer_predictions_and_rejects_divergence(tmp_path):
    source, destination = tmp_path / "old", tmp_path / "new"
    (source / "vllm").mkdir(parents=True)
    (destination / "vllm").mkdir(parents=True)
    old = source / "vllm/S1__test.jsonl"
    current = destination / "vllm/S1__test.jsonl"
    old.write_bytes(b'{"first":1}\n')
    current.write_bytes(old.read_bytes() + b'{"second":2}\n')
    restore_predictions(source, destination)
    assert b"second" in current.read_bytes()
    old.write_bytes(current.read_bytes() + b'{"third":3}\n')
    restore_predictions(source, destination)
    assert current.read_bytes() == old.read_bytes()
    old.write_bytes(b'{"divergent":true}\n')
    with pytest.raises(ValueError, match="differ"):
        restore_predictions(source, destination)


def test_frozen_examples_are_only_train_and_ids_do_not_collide(tokenizer):
    values = read_config("few_shot.yaml")
    assert len(values["examples"]) == 4
    assert values["train_sha256"] == read_config("kaggle.yaml")["datasets"]["train"]["sha256"]
    ids = [call.id for message in few_shot_messages() for call in message.tool_calls or []]
    assert len(ids) == len(set(ids)) and all(value.startswith("fewshot_train-") for value in ids)
    record = example("test")
    history = [message.to_message() for message in record.messages[:2]]
    prompt = PromptClient(tokenizer, 8192, 512, few_shot_messages())
    text, ids = prompt.prompt(history, record.tools)
    assert text.startswith("<|im_start|>system") and len(ids) + 512 < 8192
    assert record.scripted_user_turns[0] in text
    with pytest.raises(ContextLimitError):
        PromptClient(tokenizer, 1, 512).prompt(history, record.tools)


def test_injected_vllm_preserves_invalid_and_length_outputs(tokenizer):
    record = example("test")
    calls = []

    def generate(prompts, sampling, use_tqdm):
        calls.append(prompts)
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text='<tool_call>{"name":', finish_reason="length", token_ids=[1, 2]
                    )
                ]
            )
        ]

    client = VLLMClient(
        tokenizer, 8192, 512, engine=SimpleNamespace(generate=generate), sampling=object()
    )
    response = client.generate(
        [message.to_message() for message in record.messages[:2]], record.tools
    )
    assert response.finish_reason == "length" and len(calls) == 1
    with pytest.raises(ParseError):
        parse_response(response)
    client.context = 1
    assert (
        client.generate([Message(role="user", content="x")], []).content == "CONTEXT_LIMIT_EXCEEDED"
    )
    assert len(calls) == 1  # Context failure neither truncates nor calls the engine.


@pytest.mark.parametrize("name", ["train_kaggle.ipynb", "eval_gpu_kaggle.ipynb"])
def test_notebook_schema_syntax_and_no_outputs(name):
    report = validate_notebook(ROOT / "notebooks" / name)
    assert report["compiled_code_cells"] >= 5 and report["outputs"] == 0
    value = json.loads((ROOT / "notebooks" / name).read_text())
    code = next(cell for cell in value["cells"] if cell["cell_type"] == "code")
    code["outputs"] = [{"output_type": "stream", "text": "must not publish outputs"}]
    with pytest.raises(ValidationError, match="output-free"):
        Notebook.model_validate(value)


def test_generated_notebooks_are_reproducible(tmp_path, monkeypatch):
    (tmp_path / "notebooks").mkdir()
    builder = runpy.run_path(str(ROOT / "scripts/build_notebooks.py"))
    monkeypatch.chdir(tmp_path)
    builder["build"]()
    for name in ["train_kaggle.ipynb", "eval_gpu_kaggle.ipynb"]:
        generated = json.loads((tmp_path / "notebooks" / name).read_text())
        checked_in = json.loads((ROOT / "notebooks" / name).read_text())
        # Users fill in the first code cell before uploading to Kaggle.
        # Require every generated workflow cell to match, preserving those settings.
        assert generated["cells"][1]["cell_type"] == checked_in["cells"][1]["cell_type"] == "code"
        checked_in["cells"][1] = generated["cells"][1]
        assert generated == checked_in


@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_bootstrap_credentials_only_enter_environment(
    tmp_path, monkeypatch, capsys, private, training
):
    requested, commands = [], []

    class Secrets:
        def get_secret(self, name):
            requested.append(name)
            return "FAKE_" + name

    monkeypatch.setitem(sys.modules, "kaggle_secrets", SimpleNamespace(UserSecretsClient=Secrets))
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: commands.append(command))
    source = notebook_source("train_kaggle.ipynb", "from kaggle_secrets")
    source = source.replace("/tmp/cafe", str(tmp_path / "cafe"))
    values = {
        "GIT_REF": "a" * 40,
        "GITHUB_USERNAME": "tester",
        "REPO_NAME": "cafe-tool-agent",
        "HF_USERNAME": "tester",
        "PRIVATE_GITHUB": private,
        "IS_TRAINING": training,
    }
    exec(compile(source, "<bootstrap>", "exec"), values)
    assert requested == ["HF_TOKEN"] + (["WANDB_API_KEY"] if training else []) + (
        ["GITHUB_TOKEN"] if private else []
    )
    assert values["ENV"]["HF_TOKEN"] == "FAKE_HF_TOKEN"
    assert values["ENV"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert not any("FAKE_" in str(command) for command in commands)
    assert "FAKE_" not in capsys.readouterr().out


@pytest.mark.parametrize("codes", [[42, 0, 0, 0], [0, 42, 0, 0]])
def test_eval_fallback_is_a_new_backend_and_process(tmp_path, monkeypatch, codes):
    calls, environments = [], []
    sequence = iter(codes)

    def execute(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, next(sequence))

    monkeypatch.setattr(subprocess, "run", execute)
    source = notebook_source("eval_gpu_kaggle.ipynb", "selection, attempts =")
    source = source.replace("/kaggle/working/cafe-gpu", str(tmp_path / "output"))
    values = {
        "re": __import__("re"),
        "Path": Path,
        "json": json,
        "subprocess": subprocess,
        "MERGED_REVISION": "b" * 40,
        "RESUME_PREDICTIONS": "",
        "VLLM_READY": True,
        "FALLBACK_PYTHON": None,
        "PYTHON": "/vllm/python",
        "ENV": {},
        "DATASET_REPO": "test/data",
        "HF_USERNAME": "test",
        "make_environment": lambda path: environments.append(path) or "/hf/python",
    }
    exec(compile(source, "<evaluation cell>", "exec"), values)
    assert environments == ["/tmp/cafe-transformers-env"]
    assert calls[0][0] == "/vllm/python"
    fallback = next(command for command in calls if command[0] == "/hf/python")
    assert fallback[fallback.index("--backend") + 1] == "transformers"
    assert all(Path(path).name in {"vllm", "transformers"} for path in values["selection"].values())
    assert len(values["selection"]) == 3


def test_non_initialization_failure_does_not_switch_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 1)
    )
    source = notebook_source("eval_gpu_kaggle.ipynb", "selection, attempts =")
    source = source.replace("/kaggle/working/cafe-gpu", str(tmp_path / "output"))
    values = {
        "re": __import__("re"),
        "Path": Path,
        "json": json,
        "subprocess": subprocess,
        "MERGED_REVISION": "b" * 40,
        "RESUME_PREDICTIONS": "",
        "VLLM_READY": True,
        "FALLBACK_PYTHON": None,
        "PYTHON": "/vllm/python",
        "ENV": {},
        "DATASET_REPO": "test/data",
        "HF_USERNAME": "test",
    }
    with pytest.raises(subprocess.CalledProcessError):
        exec(compile(source, "<evaluation cell>", "exec"), values)
    assert values["VLLM_READY"] is True


def test_gpu_manifest_resume_and_bundle_complete_predictions(tmp_path):
    record = example("test")
    path = tmp_path / "test.jsonl"
    path.write_text(record.model_dump_json() + "\n")
    config = load_eval_config()
    config.datasets = {"test": config.datasets["test"]}
    config.datasets["test"].path = path
    config.datasets["test"].conversations = 1
    root = tmp_path / "predictions"
    config.output_dir = root
    selection = {}
    for system in ["S1", "S2", "S4"]:
        fingerprint = gpu_manifest(config, load_runtime(), root, system, {"backend": "fixture"})
        assert (
            gpu_manifest(config, load_runtime(), root, system, {"backend": "fixture"})
            == fingerprint
        )
        with pytest.raises(ValueError, match="changed"):
            gpu_manifest(config, load_runtime(), root, system, {"backend": "other"})
        responses = [
            ModelResponse(
                content=record.messages[a.message_index].content or "",
                tool_calls=record.messages[a.message_index].to_message().tool_calls or [],
            )
            for a in record.actions
        ] * 2
        responses_iter = iter(responses)
        client = SimpleNamespace(generate=lambda messages, tools, it=responses_iter: next(it))
        run_records(
            [record],
            client,
            load_runtime(),
            config,
            system=system,
            fingerprint=fingerprint,
            output=root,
        )
        selection[system] = str(root)
    archive = tmp_path / "gpu.zip"
    summary = bundle_predictions(selection, archive)
    assert summary["complete"] and set(summary["systems"]) == {"S1", "S2", "S4"}
    with zipfile.ZipFile(archive) as result:
        assert all("/" not in name for name in result.namelist())
        assert "S3__test.jsonl" not in result.namelist()
    path = root / "S2__test.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="partial"):
        bundle_predictions(selection, archive)
