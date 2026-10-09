import unittest

from src.grounding.run_grounding_dino import summarize_detections


class GroundingDinoTests(unittest.TestCase):
    def test_summarize_detections_keeps_best_per_query(self):
        result = summarize_detections(
            scores=[0.2, 0.7, 0.4],
            labels=[0, 0, 1],
            boxes=[[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]],
            query_count=3,
        )
        self.assertEqual(result[0]["max_score"], 0.7)
        self.assertEqual(result[0]["best_box"], [2.0, 2.0, 3.0, 3.0])
        self.assertEqual(result[0]["detection_count"], 2)
        self.assertEqual(result[1]["max_score"], 0.4)
        self.assertIsNone(result[2]["best_box"])


if __name__ == "__main__":
    unittest.main()
