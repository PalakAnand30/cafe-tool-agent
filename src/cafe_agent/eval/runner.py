"""Gold-history turn evaluation and fresh-backend end-to-end agent execution."""

import hashlib
import json
from pathlib import Path
from typing import Any

from cafe_agent.agent.loop import Agent, TraceEntry
from cafe_agent.backend.service import CafeService
from cafe_agent.config import RuntimeConfig, read_config
from cafe_agent.data.independent import load_independent_records
from cafe_agent.data.io import append_jsonl, read_jsonl
from cafe_agent.data.schema import Conversation
from cafe_agent.data.tokenization import load_tokenizer
from cafe_agent.data.validator import validate_conversation
from cafe_agent.models.base import Message, ModelClient
from cafe_agent.parsing.parser import ToolCall
from cafe_agent.prompting.prompts import system_prompt
from cafe_agent.tools.executor import ToolExecutor
from cafe_agent.tools.schemas import tool_schemas

from .config import EvalConfig, Split, SystemID
from .inference import ProviderToolFailure, RecordingClient
from .schema import Prediction


class RecordingExecutor(ToolExecutor):
    def __init__(self, service: CafeService) -> None:
        super().__init__(service)
        self.trace: list[TraceEntry] = []

    def execute(self, call: ToolCall) -> dict[str, Any]:
        result = super().execute(call)
        self.trace.append(TraceEntry(call=call.model_copy(deep=True), result=result))
        return result


def load_records(config: EvalConfig, split: Split, *, validate: bool = True) -> list[Conversation]:
    dataset = config.datasets[split]
    if split == "independent_test":
        records = load_independent_records(dataset.path)
    else:
        records = [Conversation.model_validate(row) for row in read_jsonl(dataset.path)]
    if len(records) != dataset.conversations or any(
        record.split != split or record.author != dataset.author for record in records
    ):
        raise ValueError(
            "Evaluation dataset is incomplete, mislabeled or contains excluded examples"
        )
    if len({record.id for record in records}) != len(records):
        raise ValueError("Duplicate evaluation conversation ID")
    if validate:
        tokenizer = load_tokenizer()
        for record in records:
            validate_conversation(record, tokenizer, read_config("data.yaml")["max_tokens"])
    return records


def user_segments(record: Conversation) -> list[tuple[int, int]]:
    positions = [index for index, message in enumerate(record.messages) if message.role == "user"]
    return list(zip(positions, [*positions[1:], len(record.messages)], strict=True))


def turn_prediction(
    record: Conversation,
    index: int,
    client: ModelClient,
    *,
    system: SystemID,
    fingerprint: str,
) -> Prediction:
    recorder = RecordingClient(client)
    recorder.user_index = sum(message.role == "user" for message in record.messages[:index]) - 1
    error = None
    try:
        recorder.generate(
            [message.to_message() for message in record.messages[:index]], record.tools
        )
    except ProviderToolFailure:
        error = "PROVIDER_TOOL_USE_FAILED"
    return Prediction(
        system=system,
        split=record.split,
        conversation_id=record.id,
        intent=record.intent,
        author=record.author,
        mode="turn",
        fingerprint=fingerprint,
        assistant_index=index,
        observations=recorder.observations,
        error=error,
    )


def end_prediction(
    record: Conversation,
    client: ModelClient,
    runtime: RuntimeConfig,
    *,
    system: SystemID,
    fingerprint: str,
) -> Prediction:
    service = CafeService()
    executor = RecordingExecutor(service)
    recorder = RecordingClient(client)
    messages = [Message(role="system", content=system_prompt())]
    error = None
    segments = user_segments(record)
    allowed = {start for start, _ in segments} | {len(record.messages)}
    if any(event.before_message not in allowed for event in record.clock_advances):
        service.close()
        raise ValueError("End-to-end clock advances must anchor to user turns or conversation end")
    advances: dict[int, int] = {}
    for event in record.clock_advances:
        advances[event.before_message] = advances.get(event.before_message, 0) + event.minutes
    try:
        for user_index, (start, _) in enumerate(segments):
            service.advance_time(advances.get(start, 0))
            messages.append(record.messages[start].to_message())
            recorder.user_index = user_index
            try:
                result = Agent(recorder, executor, runtime).run(messages)
            except ProviderToolFailure:
                messages = recorder.last_history
                error = "PROVIDER_TOOL_USE_FAILED"
                break
            messages, error = result.messages, result.error
            if error:
                break
        if error is None:
            service.advance_time(advances.get(len(record.messages), 0))
        return Prediction(
            system=system,
            split=record.split,
            conversation_id=record.id,
            intent=record.intent,
            author=record.author,
            mode="end_to_end",
            fingerprint=fingerprint,
            observations=recorder.observations,
            messages=messages,
            trace=executor.trace,
            error=error,
            final_state=service.snapshot(),
        )
    finally:
        service.close()


def run_manifest(
    config: EvalConfig,
    runtime: RuntimeConfig,
    output: Path,
    system: SystemID,
) -> str:
    root = Path(__file__).resolve().parents[1]
    sources = {}
    paths = [root / "config.py", root / "data/schema.py"]
    paths += [
        Path(__file__).parent / name
        for name in ["runner.py", "inference.py", "schema.py", "actions.py", "config.py"]
    ]
    for directory in ["agent", "domain", "tools", "backend", "parsing", "prompting", "models"]:
        paths.extend((root / directory).rglob("*.py"))
    for path in sorted(paths):
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    value: dict[str, Any] = {
        "system": system,
        "eval": config.model_dump(mode="json"),
        "runtime": runtime.model_dump(mode="json"),
        "domain": read_config("domain.yaml"),
        "prompt": system_prompt(),
        "tools": tool_schemas(),
        "sources": sources,
        "datasets": {
            split: hashlib.sha256(dataset.path.read_bytes()).hexdigest()
            for split, dataset in config.datasets.items()
        },
        "history_policy": (
            "Canonical tool IDs/JSON and schema defaults; identical for every system."
        ),
    }
    fingerprint = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{system}__manifest.json"
    if path.exists():
        if json.loads(path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Evaluation inputs/code changed; choose a new --output-dir")
    else:
        if list(output.glob(f"{system}__*.jsonl")):
            raise ValueError("Predictions have no manifest; choose a new output directory")
        path.write_text(json.dumps({"fingerprint": fingerprint, **value}, indent=2) + "\n")
    return fingerprint


def run_records(
    records: list[Conversation],
    client: ModelClient,
    runtime: RuntimeConfig,
    config: EvalConfig,
    *,
    system: SystemID,
    fingerprint: str,
    output: Path,
) -> None:
    if not records:
        return
    path = output / f"{system}__{records[0].split}.jsonl"
    existing = [Prediction.model_validate(row) for row in read_jsonl(path, repair_tail=True)]
    if any(row.fingerprint != fingerprint or row.system != system for row in existing):
        raise ValueError("Prediction provenance differs from the current run")
    done = {row.key for row in existing}
    if len(done) != len(existing):
        raise ValueError("Duplicate prediction key")
    expected = {
        (record.id, mode, action.message_index if mode == "turn" else None)
        for record in records
        for mode in config.modes
        for action in (record.actions if mode == "turn" else [None])
    }
    if done - expected:
        raise ValueError("Unexpected prediction IDs/turns")
    for number, record in enumerate(records, 1):
        # Complete both modes per case so interrupted runs retain paired results.
        for mode in config.modes:
            for action in record.actions if mode == "turn" else [None]:
                index = action.message_index if action else None
                key = (record.id, mode, index)
                if key in done:
                    continue
                prediction = (
                    turn_prediction(record, index, client, system=system, fingerprint=fingerprint)
                    if mode == "turn"
                    else end_prediction(
                        record, client, runtime, system=system, fingerprint=fingerprint
                    )
                )
                append_jsonl(path, prediction.model_dump(mode="json"))
                done.add(key)
        if number % config.progress_every == 0 or number == len(records):
            print(f"{system}/{record.split}: {number}/{len(records)} conversations", flush=True)
