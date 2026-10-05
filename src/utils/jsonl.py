from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error


def completed_question_ids(path: Path, retry_errors: bool) -> set[int]:
    completed: set[int] = set()
    for row in iter_jsonl(path):
        status = row.get("status", "ok")
        if status == "ok" or not retry_errors:
            completed.add(int(row["question_id"]))
    return completed
