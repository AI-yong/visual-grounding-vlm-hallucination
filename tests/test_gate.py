import unittest

from src.evaluation.evaluate_grounded_gate import (
    detector_prediction,
    gated_prediction,
    intervention_metrics,
    select_threshold,
    threshold_grid,
)


class GateTests(unittest.TestCase):
    def test_gate_only_changes_baseline_yes(self):
        yes_row = {"baseline_answer": "yes", "max_score": 0.1}
        no_row = {"baseline_answer": "no", "max_score": 0.9}
        self.assertEqual(gated_prediction(yes_row, 0.2), "no")
        self.assertEqual(gated_prediction(no_row, 0.2), "no")
        self.assertEqual(detector_prediction(no_row, 0.2), "yes")

    def test_threshold_grid_is_inclusive(self):
        self.assertEqual(threshold_grid(0.0, 0.03, 0.01), [0.0, 0.01, 0.02, 0.03])

    def test_threshold_is_selected_on_balanced_accuracy(self):
        rows = [
            {"label": "yes", "baseline_answer": "yes", "max_score": 0.8},
            {"label": "yes", "baseline_answer": "yes", "max_score": 0.6},
            {"label": "no", "baseline_answer": "yes", "max_score": 0.2},
            {"label": "no", "baseline_answer": "yes", "max_score": 0.1},
        ]
        threshold, _ = select_threshold(rows, [0.0, 0.3, 0.7], gated_prediction)
        self.assertEqual(threshold, 0.3)

    def test_intervention_metrics_separate_corrections_and_damage(self):
        rows = [
            {"label": "no", "baseline_answer": "yes", "pred_answer": "no"},
            {"label": "yes", "baseline_answer": "yes", "pred_answer": "no"},
            {"label": "yes", "baseline_answer": "yes", "pred_answer": "yes"},
        ]
        metrics = intervention_metrics(rows)
        self.assertEqual(metrics["baseline_yes_changed_to_no"], 2)
        self.assertEqual(metrics["corrected_false_positives"], 1)
        self.assertEqual(metrics["introduced_false_negatives"], 1)


if __name__ == "__main__":
    unittest.main()
