from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.metrics import binary_metrics
from src.utils.jsonl import iter_jsonl


def latest_rows(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["question_id"]): row for row in iter_jsonl(path)}


def paired_changes(
    baseline: dict[int, dict[str, Any]], guided: dict[int, dict[str, Any]]
) -> dict[str, int]:
    result = {
        "same_answer": 0,
        "changed_answer": 0,
        "corrected": 0,
        "worsened": 0,
        "false_positive_fixed": 0,
        "true_positive_lost": 0,
        "false_negative_fixed": 0,
        "true_negative_lost": 0,
    }
    for question_id, new in guided.items():
        old = baseline[question_id]
        old_answer = old.get("pred_answer")
        new_answer = new.get("pred_answer")
        label = new["label"]
        if old_answer == new_answer:
            result["same_answer"] += 1
            continue
        result["changed_answer"] += 1
        old_correct = old_answer == label
        new_correct = new_answer == label
        result["corrected"] += int(not old_correct and new_correct)
        result["worsened"] += int(old_correct and not new_correct)
        result["false_positive_fixed"] += int(
            label == "no" and old_answer == "yes" and new_answer == "no"
        )
        result["true_positive_lost"] += int(
            label == "yes" and old_answer == "yes" and new_answer == "no"
        )
        result["false_negative_fixed"] += int(
            label == "yes" and old_answer == "no" and new_answer == "yes"
        )
        result["true_negative_lost"] += int(
            label == "no" and old_answer == "no" and new_answer == "yes"
        )
    return result


def comparison_report(
    baseline: dict[int, dict[str, Any]], guided: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "n": len(guided),
        "baseline": binary_metrics(baseline.values()),
        "bbox_guided": binary_metrics(guided.values()),
        "paired_changes": paired_changes(baseline, guided),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare bbox-guided Qwen predictions with the original baseline."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--guided", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--subset", choices=("all", "dev", "test"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = latest_rows(args.baseline)
    guided = latest_rows(args.guided)
    if not guided:
        raise ValueError("Guided prediction file is empty")
    missing = sorted(set(guided) - set(baseline))
    if missing:
        raise ValueError(f"Baseline is missing guided question IDs: {missing[:5]}")

    if args.subset != "all":
        if args.split is None:
            raise ValueError("--split is required for dev or test evaluation")
        split = json.loads(args.split.read_text(encoding="utf-8"))
        allowed_images = set(split[f"{args.subset}_images"])
        guided = {
            question_id: row
            for question_id, row in guided.items()
            if row["image"] in allowed_images
        }
    baseline = {question_id: baseline[question_id] for question_id in guided}
    for question_id in guided:
        if baseline[question_id]["label"] != guided[question_id]["label"]:
            raise ValueError(f"Label mismatch for question {question_id}")

    with_box = {question_id: row for question_id, row in guided.items() if row.get("has_box")}
    without_box = {question_id: row for question_id, row in guided.items() if not row.get("has_box")}
    report = {
        "subset": args.subset,
        "overall": comparison_report(baseline, guided),
        "with_box": comparison_report(
            {question_id: baseline[question_id] for question_id in with_box}, with_box
        ),
        "without_box": comparison_report(
            {question_id: baseline[question_id] for question_id in without_box}, without_box
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
