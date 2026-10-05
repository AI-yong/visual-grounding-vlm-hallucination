import unittest

from src.grounding.run_owlv2 import group_rows_by_image, summarize_detections
from src.data.pope import PopeRow


class GroundingTests(unittest.TestCase):
    def test_summarize_detections_keeps_best_box_per_query(self):
        summaries = summarize_detections(
            scores=[0.2, 0.8, 0.6],
            labels=[0, 0, 1],
            boxes=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            query_count=3,
        )
        self.assertEqual(summaries[0]["max_score"], 0.8)
        self.assertEqual(summaries[0]["best_box"], [5.0, 6.0, 7.0, 8.0])
        self.assertEqual(summaries[0]["detection_count"], 2)
        self.assertEqual(summaries[1]["max_score"], 0.6)
        self.assertEqual(summaries[2]["max_score"], 0.0)
        self.assertIsNone(summaries[2]["best_box"])

    def test_group_rows_by_image_preserves_question_order(self):
        rows = [
            PopeRow(1, "a.jpg", "Is there a dog in the image?", "yes"),
            PopeRow(2, "b.jpg", "Is there a cat in the image?", "no"),
            PopeRow(3, "a.jpg", "Is there a chair in the image?", "no"),
        ]
        grouped = group_rows_by_image(rows)
        self.assertEqual([row.question_id for row in grouped["a.jpg"]], [1, 3])
        self.assertEqual([row.question_id for row in grouped["b.jpg"]], [2])


if __name__ == "__main__":
    unittest.main()
