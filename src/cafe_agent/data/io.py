"""Append-only JSONL with explicit, backed-up recovery of interrupted final writes."""

import json
import os
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path, *, repair_tail: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = path.read_bytes()
    result, offset = [], 0
    for line in data.splitlines(keepends=True):
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("JSONL records must be objects")
        except (json.JSONDecodeError, UnicodeDecodeError):
            if not repair_tail or line.endswith(b"\n") or offset + len(line) != len(data):
                raise ValueError(f"Invalid JSONL in {path.name} at byte {offset}") from None
            backup = path.with_suffix(path.suffix + ".incomplete-tail")
            if backup.exists():
                raise ValueError(
                    "An interrupted-tail backup already exists; inspect it first"
                ) from None
            backup.write_bytes(line)
            with path.open("r+b") as handle:
                handle.truncate(offset)
            break
        result.append(value)
        offset += len(line)
    if repair_tail and data and not data.endswith(b"\n") and offset == len(data):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return result
