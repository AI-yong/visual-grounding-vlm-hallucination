from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.data.pope import extract_object_phrase, load_pope_rows, normalize_binary_answer
from src.evaluation.metrics import binary_metrics
from src.utils.jsonl import append_jsonl, iter_jsonl


class CoreTests(unittest.TestCase):
    def test_normalize_binary_answer(self):
        self.assertEqual(normalize_binary_answer("Yes"), "yes")
        self.assertEqual(normalize_binary_answer("No."), "no")
        self.assertEqual(normalize_binary_answer("Yes, there is one."), "yes")
        self.assertEqual(normalize_binary_answer("I cannot tell"), "invalid")

    def test_extract_object_phrase(self):
        self.assertEqual(extract_object_phrase("Is there a dog in the image?"), "dog")
        self.assertEqual(extract_object_phrase("Is there an orange in the image?"), "orange")

    def test_load_pope_rows(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample.json"
            path.write_text(
                '{"question_id": 1, "image": "a.jpg", "text": "Is there a dog in the image?", "label": "yes"}\n',
                encoding="utf-8",
            )
            rows = load_pope_rows(path)
            self.assertEqual(rows[0].question, "Is there a dog in the image?")

    def test_binary_metrics(self):
        rows = [
            {"label": "yes", "pred_answer": "yes", "status": "ok"},
            {"label": "no", "pred_answer": "yes", "status": "ok"},
            {"label": "no", "pred_answer": "no", "status": "ok"},
            {"label": "yes", "pred_answer": "no", "status": "ok"},
        ]
        metrics = binary_metrics(rows)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["accuracy_valid"], 0.5)

    def test_latest_retry_row_can_replace_error(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            append_jsonl(path, {"question_id": 1, "status": "error"})
            append_jsonl(
                path,
                {"question_id": 1, "status": "ok", "label": "yes", "pred_answer": "yes"},
            )
            latest = {int(row["question_id"]): row for row in iter_jsonl(path)}
            metrics = binary_metrics(latest.values())
            self.assertEqual(metrics["n_total"], 1)
            self.assertEqual(metrics["tp"], 1)


if __name__ == "__main__":
    unittest.main()
