"""Resumable drafts → cached paraphrases → noise → validated split pipeline."""

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from cafe_agent.config import read_config

from .config import DataConfig, load_data_config
from .io import append_jsonl, read_jsonl
from .noise import add_noise
from .paraphraser import Paraphraser, ParaphraseRequest
from .schema import Conversation
from .splitter import check_leakage, make_plan, normalise_user_text
from .stats import write_stats
from .templates import generate_example, template_ids
from .tokenization import load_tokenizer
from .validator import validate_conversation


def _manifest(config: DataConfig, output: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    sources = {"config.py": hashlib.sha256((root / "config.py").read_bytes()).hexdigest()}
    for directory in ["data", "domain", "backend", "tools", "prompting", "parsing", "models"]:
        for path in sorted((root / directory).rglob("*.py")):
            sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    value = {
        "config": config.model_dump(mode="json"),
        "domain": read_config("domain.yaml"),
        "train": read_config("train.yaml"),
        "runtime": read_config("runtime.yaml"),
        "sources": sources,
    }
    fingerprint = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    path = output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text())["fingerprint"] != fingerprint:
            raise ValueError(
                "Data inputs or code changed; choose a new --output-dir "
                "to preserve the existing run"
            )
    else:
        if any(output.glob("*.jsonl")) or (output / "drafts").exists():
            raise ValueError("Existing data has no manifest; choose a new output directory")
        path.write_text(
            json.dumps({"fingerprint": fingerprint, **value}, indent=2) + "\n", encoding="utf-8"
        )


def build_dataset(
    config: DataConfig,
    *,
    tokenizer: Any = None,
    paraphraser: Any = None,
    output_dir: Path | None = None,
) -> dict:
    output = output_dir or config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _manifest(config, output)
    tokenizer = tokenizer or load_tokenizer()
    plan = make_plan(config)
    slots = {slot.id: slot for slot in plan}
    drafts = []
    seen = set()
    completed = {}
    for split in config.splits:
        for value in read_jsonl(output / "drafts" / f"{split}.jsonl", repair_tail=True):
            record = Conversation.model_validate(value)
            if record.id in completed:
                raise ValueError("Duplicate draft ID")
            completed[record.id] = record
            for text in record.scripted_user_turns:
                key = normalise_user_text(text)
                if key in seen:
                    raise ValueError("Duplicate text in persisted drafts")
                seen.add(key)
    if set(completed) - slots.keys():
        raise ValueError("Unexpected persisted draft IDs")
    for slot in plan:
        if slot.id not in completed:
            for attempt in range(config.max_attempts_per_id):
                rng = random.Random(f"{config.seed}:{slot.id}:{attempt}")
                choices = [
                    name
                    for name in template_ids(config, test=slot.split == "test")
                    if name.startswith(slot.intent.value + ".")
                ]
                variant = int(rng.choice(choices).rsplit("v", 1)[1])
                record = generate_example(
                    config, slot.id, slot.split, slot.intent, variant, slot.multi_turn, rng
                )
                keys = [normalise_user_text(text) for text in record.scripted_user_turns]
                if len(set(keys)) != len(keys) or any(key in seen for key in keys):
                    continue
                # Reject oversized gold conversations before making paid paraphrase requests.
                validate_conversation(record, tokenizer, config.max_tokens)
                append_jsonl(
                    output / "drafts" / f"{slot.split}.jsonl", record.model_dump(mode="json")
                )
                seen.update(keys)
                completed[slot.id] = record
                break
            else:
                raise ValueError(f"Could not produce unique valid data for {slot.id}")
        drafts.append(completed[slot.id])
        if len(drafts) % config.progress_every == 0:
            print(f"Drafts ready: {len(drafts)}/{len(plan)}", flush=True)
    check_leakage(drafts, config)
    requests = []
    for record in drafts:
        for index, provenance in enumerate(record.provenance):
            if (
                random.Random(f"{config.seed}:{record.id}:paraphrase:{index}").random()
                < config.paraphrase_rate
            ):
                requests.append(
                    ParaphraseRequest(
                        id=f"{record.id}/u{index}",
                        text=provenance.original,
                        entities=provenance.entities,
                    )
                )
    journal_path = output / "paraphrases.jsonl"
    engine = paraphraser or Paraphraser(config, journal_path)
    paraphrases = engine.run(requests) if requests else {}
    reserved = {
        normalise_user_text(text): record.id
        for record in drafts
        for text in record.scripted_user_turns
    }
    records, lengths, final_seen = [], {}, set()
    persisted = {}
    for split in config.splits:
        for value in read_jsonl(output / f"{split}.jsonl", repair_tail=True):
            record = Conversation.model_validate(value)
            if record.id in persisted:
                raise ValueError("Duplicate completed ID")
            persisted[record.id] = record
    if set(persisted) - {slot.id for slot in plan}:
        raise ValueError("Unexpected persisted conversation IDs")
    for draft in drafts:
        record = persisted.get(draft.id)
        if record is None:
            record = draft.model_copy(deep=True)
            user_index = 0
            local_seen = set()
            for message in record.messages:
                if message.role != "user":
                    continue
                provenance = record.provenance[user_index]
                paraphrase = paraphrases.get(f"{record.id}/u{user_index}")
                candidate = provenance.original
                if paraphrase:
                    provenance.paraphrase_attempted = True
                    provenance.paraphrase_accepted = paraphrase.accepted
                    provenance.rejection_reason = paraphrase.rejection_reason
                    candidate = paraphrase.text
                rng = random.Random(f"{config.seed}:{record.id}:noise:{user_index}")
                before_noise = candidate
                if rng.random() < config.noise_rate:
                    candidate = add_noise(candidate, rng)
                key = normalise_user_text(candidate)
                if (
                    key in final_seen
                    or key in local_seen
                    or reserved.get(key, record.id) != record.id
                ):
                    candidate = provenance.original
                    provenance.paraphrase_accepted = False
                    provenance.rejection_reason = "deduplication"
                provenance.noise_applied = (
                    candidate != before_noise and candidate != provenance.original
                )
                message.content = candidate
                local_seen.add(normalise_user_text(candidate))
                user_index += 1
            record.scripted_user_turns = [m.content for m in record.messages if m.role == "user"]
            length = validate_conversation(record, tokenizer, config.max_tokens)
            append_jsonl(output / f"{record.split}.jsonl", record.model_dump(mode="json"))
        else:
            length = validate_conversation(record, tokenizer, config.max_tokens)
        slot = slots[record.id]
        if (
            record.split != slot.split
            or record.intent != slot.intent
            or record.author != "synthetic"
        ):
            raise ValueError("Persisted record does not match the planned split/intent")
        records.append(record)
        lengths[record.id] = length
        final_seen.update(normalise_user_text(text) for text in record.scripted_user_turns)
    leakage = check_leakage(records, config)
    summary = write_stats(output, records, lengths, leakage, read_jsonl(journal_path))
    print(
        f"Validated {len(records)} conversations; "
        f"maximum {summary['token_lengths']['max']} tokens.",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--no-paraphrase",
        action="store_true",
        help="Explicit offline variant; records zero API paraphrases",
    )
    args = parser.parse_args()
    config = load_data_config(args.config)
    if args.no_paraphrase:
        config.paraphrase_rate = 0
    build_dataset(config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
