"""Check complete prediction/provenance coverage on Kaggle; score locally later."""

import hashlib
import json
from pathlib import Path

from cafe_agent.data.io import read_jsonl
from cafe_agent.eval.config import EvalConfig
from cafe_agent.eval.runner import load_records
from cafe_agent.eval.schema import Prediction


def check_coverage(config: EvalConfig, root: Path, system: str) -> dict:
    manifest = json.loads((root / f"{system}__manifest.json").read_text())
    groups = []
    for split, dataset in config.datasets.items():
        records = load_records(config, split, validate=False)
        by_id = {record.id: record for record in records}
        digest = hashlib.sha256(dataset.path.read_bytes()).hexdigest()
        if digest != manifest["datasets"][split]:
            raise ValueError("Dataset differs from immutable prediction manifest")
        path = root / f"{system}__{split}.jsonl"
        rows = [Prediction.model_validate(row) for row in read_jsonl(path)]
        expected = {
            (record.id, mode, action.message_index if action else None)
            for record in records
            for mode in config.modes
            for action in (record.actions if mode == "turn" else [None])
        }
        if len({row.key for row in rows}) != len(rows):
            raise ValueError("Duplicate prediction key")
        for row in rows:
            if (
                row.key not in expected
                or row.system != system
                or row.split != split
                or row.fingerprint != manifest["fingerprint"]
                or row.author != by_id[row.conversation_id].author
                or row.intent != by_id[row.conversation_id].intent
            ):
                raise ValueError("Prediction key/provenance differs from the frozen source")
        groups.append(
            {
                "split": split,
                "complete": {row.key for row in rows} == expected,
                "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                if path.exists()
                else None,
            }
        )
    return {"groups": groups}
