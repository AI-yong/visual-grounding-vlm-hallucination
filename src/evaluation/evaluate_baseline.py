from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.metrics import binary_metrics
from src.utils.jsonl import iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen POPE baseline JSONL file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    latest_by_question_id = {
        int(row["question_id"]): row for row in iter_jsonl(args.input)
    }
    metrics = binary_metrics(latest_by_question_id.values())
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
