from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable, Iterable

from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.metrics import binary_metrics
from src.utils.jsonl import iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select OWLv2 thresholds on an image-level dev split and evaluate on test."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    parser.add_argument("--detector-output", type=Path, required=True)
    parser.add_argument("--gated-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-min", type=float, default=0.0)
    parser.add_argument("--threshold-max", type=float, default=0.5)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser.parse_args()


def latest_rows(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["question_id"]): row for row in iter_jsonl(path)}


def validate_join(
    baseline: dict[int, dict[str, Any]], grounding: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    if set(baseline) != set(grounding):
        missing_grounding = sorted(set(baseline) - set(grounding))
        missing_baseline = sorted(set(grounding) - set(baseline))
        raise ValueError(
            f"Question IDs do not match: missing_grounding={missing_grounding[:5]} "
            f"missing_baseline={missing_baseline[:5]}"
        )

    joined: list[dict[str, Any]] = []
    for question_id in sorted(baseline):
        base = baseline[question_id]
        ground = grounding[question_id]
        if base.get("status", "ok") != "ok" or ground.get("status", "ok") != "ok":
            raise ValueError(f"Question {question_id} has a non-ok input row")
        if base["image"] != ground["image"] or base["label"] != ground["label"]:
            raise ValueError(f"Question {question_id} metadata differs between inputs")
        score = float(ground["max_score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Question {question_id} has invalid grounding score {score}")
        joined.append(
            {
                "question_id": question_id,
                "image": base["image"],
                "question": base["question"],
                "label": base["label"],
                "baseline_answer": base["pred_answer"],
                "object_phrase": ground["object_phrase"],
                "max_score": score,
                "best_box": ground.get("best_box"),
            }
        )
    return joined


def make_or_load_image_split(
    rows: list[dict[str, Any]], path: Path, dev_ratio: float, seed: int
) -> dict[str, Any]:
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError("--dev-ratio must be between 0 and 1")
    all_images = sorted({row["image"] for row in rows})

    if path.exists():
        split = json.loads(path.read_text(encoding="utf-8"))
        split_images = set(split["dev_images"]) | set(split["test_images"])
        if split_images != set(all_images):
            raise ValueError("Existing split does not match the input image set")
        if set(split["dev_images"]) & set(split["test_images"]):
            raise ValueError("Existing split has overlapping dev/test images")
        return split

    shuffled = all_images.copy()
    random.Random(seed).shuffle(shuffled)
    dev_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * dev_ratio)))
    split = {
        "split_unit": "image",
        "seed": seed,
        "dev_ratio": dev_ratio,
        "n_images": len(shuffled),
        "dev_images": sorted(shuffled[:dev_count]),
        "test_images": sorted(shuffled[dev_count:]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return split


def threshold_grid(minimum: float, maximum: float, step: float) -> list[float]:
    if step <= 0 or maximum < minimum:
        raise ValueError("Invalid threshold range")
    count = int(round((maximum - minimum) / step))
    values = [round(minimum + index * step, 10) for index in range(count + 1)]
    if values[-1] < maximum:
        values.append(maximum)
    return values


def detector_prediction(row: dict[str, Any], threshold: float) -> str:
    return "yes" if row["max_score"] >= threshold else "no"


def gated_prediction(row: dict[str, Any], threshold: float) -> str:
    baseline = row["baseline_answer"]
    if baseline not in {"yes", "no"}:
        return "invalid"
    if baseline == "no":
        return "no"
    return "yes" if row["max_score"] >= threshold else "no"


def prediction_rows(
    rows: Iterable[dict[str, Any]],
    threshold: float,
    predictor: Callable[[dict[str, Any], float], str],
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "pred_answer": predictor(row, threshold),
            "threshold": threshold,
            "status": "ok",
        }
        for row in rows
    ]


def balanced_accuracy(metrics: dict[str, float | int]) -> float:
    tp = int(metrics["tp"])
    fn = int(metrics["fn"])
    tn = int(metrics["tn"])
    fp = int(metrics["fp"])
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return (sensitivity + specificity) / 2


def select_threshold(
    rows: list[dict[str, Any]],
    thresholds: list[float],
    predictor: Callable[[dict[str, Any], float], str],
) -> tuple[float, list[dict[str, Any]]]:
    curve: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = binary_metrics(prediction_rows(rows, threshold, predictor))
        curve.append(
            {
                "threshold": threshold,
                "balanced_accuracy": balanced_accuracy(metrics),
                "accuracy": metrics["accuracy_valid"],
                "f1": metrics["f1"],
                "false_positive_rate": metrics["false_positive_rate"],
                "recall": metrics["recall"],
            }
        )
    # Prefer the less interventionist (smaller) threshold when scores tie.
    best = max(curve, key=lambda item: (item["balanced_accuracy"], -item["threshold"]))
    return float(best["threshold"]), curve


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(rendered, encoding="utf-8")


def intervention_metrics(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "baseline_yes_changed_to_no": sum(
            row["baseline_answer"] == "yes" and row["pred_answer"] == "no" for row in rows
        ),
        "corrected_false_positives": sum(
            row["label"] == "no"
            and row["baseline_answer"] == "yes"
            and row["pred_answer"] == "no"
            for row in rows
        ),
        "introduced_false_negatives": sum(
            row["label"] == "yes"
            and row["baseline_answer"] == "yes"
            and row["pred_answer"] == "no"
            for row in rows
        ),
    }


def score_ranking_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    labels = [1 if row["label"] == "yes" else 0 for row in rows]
    scores = [row["max_score"] for row in rows]
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def main() -> None:
    args = parse_args()
    joined = validate_join(latest_rows(args.baseline), latest_rows(args.grounding))
    split = make_or_load_image_split(joined, args.split_output, args.dev_ratio, args.seed)
    dev_images = set(split["dev_images"])
    test_images = set(split["test_images"])
    dev_rows = [row for row in joined if row["image"] in dev_images]
    test_rows = [row for row in joined if row["image"] in test_images]
    thresholds = threshold_grid(args.threshold_min, args.threshold_max, args.threshold_step)

    detector_threshold, detector_curve = select_threshold(
        dev_rows, thresholds, detector_prediction
    )
    gate_threshold, gate_curve = select_threshold(dev_rows, thresholds, gated_prediction)

    detector_test = prediction_rows(test_rows, detector_threshold, detector_prediction)
    gated_test = prediction_rows(test_rows, gate_threshold, gated_prediction)
    baseline_test = [
        {**row, "pred_answer": row["baseline_answer"], "status": "ok"} for row in test_rows
    ]
    write_jsonl(args.detector_output, detector_test)
    write_jsonl(args.gated_output, gated_test)

    report = {
        "protocol": {
            "split_unit": "image",
            "seed": split["seed"],
            "dev_ratio": split["dev_ratio"],
            "dev_images": len(dev_images),
            "test_images": len(test_images),
            "dev_questions": len(dev_rows),
            "test_questions": len(test_rows),
            "selection_objective": "balanced_accuracy",
        },
        "selected_thresholds": {
            "detector_only": detector_threshold,
            "qwen_owlv2_gate": gate_threshold,
        },
        "test": {
            "qwen_only": binary_metrics(baseline_test),
            "owlv2_only": {
                **binary_metrics(detector_test),
                **score_ranking_metrics(test_rows),
            },
            "qwen_owlv2_gate": {
                **binary_metrics(gated_test),
                **intervention_metrics(gated_test),
            },
        },
        "dev_threshold_curves": {
            "detector_only": detector_curve,
            "qwen_owlv2_gate": gate_curve,
        },
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["protocol"], ensure_ascii=False, indent=2))
    print(json.dumps(report["selected_thresholds"], ensure_ascii=False, indent=2))
    print(json.dumps(report["test"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
