from __future__ import annotations

from typing import Any, Iterable


def binary_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    rows = list(rows)
    ok_rows = [row for row in rows if row.get("status", "ok") == "ok"]
    valid = [row for row in ok_rows if row.get("pred_answer") in {"yes", "no"}]

    tp = sum(row["label"] == "yes" and row["pred_answer"] == "yes" for row in valid)
    fp = sum(row["label"] == "no" and row["pred_answer"] == "yes" for row in valid)
    tn = sum(row["label"] == "no" and row["pred_answer"] == "no" for row in valid)
    fn = sum(row["label"] == "yes" and row["pred_answer"] == "no" for row in valid)
    invalid = len(ok_rows) - len(valid)
    errors = len(rows) - len(ok_rows)

    def safe_div(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "n_total": len(rows),
        "n_valid": len(valid),
        "invalid_count": invalid,
        "error_count": errors,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy_valid": safe_div(tp + tn, len(valid)),
        "accuracy_overall": safe_div(tp + tn, len(rows)),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "false_positive_rate": safe_div(fp, fp + tn),
        "yes_ratio": safe_div(tp + fp, len(valid)),
        "invalid_ratio": safe_div(invalid, len(rows)),
        "error_ratio": safe_div(errors, len(rows)),
    }
