from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PopeRow:
    question_id: int
    image: str
    question: str
    label: str


def normalize_binary_answer(raw_answer: str) -> str:
    """Normalize a generated answer to yes/no/invalid without guessing."""
    text = raw_answer.strip().lower()
    match = re.match(r"^(yes|no)\b", text)
    return match.group(1) if match else "invalid"


def extract_object_phrase(question: str) -> str:
    text = question.strip().lower()
    patterns = (
        r"^is there an? (.+?) in the image\?*$",
        r"^is there (.+?) in the image\?*$",
    )
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            return match.group(1).strip()
    raise ValueError(f"Unsupported POPE question template: {question}")


def load_pope_rows(path: Path) -> list[PopeRow]:
    rows: list[PopeRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                label = str(raw["label"]).strip().lower()
                if label not in {"yes", "no"}:
                    raise ValueError(f"invalid label: {label}")
                rows.append(
                    PopeRow(
                        question_id=int(raw["question_id"]),
                        image=str(raw["image"]),
                        question=str(raw.get("text", raw.get("question", ""))),
                        label=label,
                    )
                )
            except Exception as error:
                raise ValueError(f"Failed to parse {path}:{line_number}: {error}") from error
    if not rows:
        raise ValueError(f"No POPE rows found in {path}")
    return rows
