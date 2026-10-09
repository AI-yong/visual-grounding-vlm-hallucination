import unittest

from PIL import Image

from src.evaluation.evaluate_bbox_guided import paired_changes
from src.inference.run_bbox_guided import clamp_box, draw_candidate_box


class BboxGuidedTests(unittest.TestCase):
    def test_clamp_box(self):
        self.assertEqual(clamp_box([-5.0, 2.2, 20.8, 30.1], 16, 24), (0, 2, 15, 23))

    def test_draw_candidate_box_does_not_mutate_input(self):
        image = Image.new("RGB", (32, 32), "white")
        rendered, box = draw_candidate_box(image, [5, 6, 20, 22])
        self.assertEqual(box, (5, 6, 20, 22))
        self.assertEqual(image.getpixel((5, 6)), (255, 255, 255))
        self.assertEqual(rendered.getpixel((5, 6)), (255, 0, 0))

    def test_paired_changes(self):
        baseline = {
            1: {"label": "no", "pred_answer": "yes"},
            2: {"label": "yes", "pred_answer": "yes"},
        }
        guided = {
            1: {"label": "no", "pred_answer": "no"},
            2: {"label": "yes", "pred_answer": "no"},
        }
        changes = paired_changes(baseline, guided)
        self.assertEqual(changes["corrected"], 1)
        self.assertEqual(changes["worsened"], 1)
        self.assertEqual(changes["false_positive_fixed"], 1)
        self.assertEqual(changes["true_positive_lost"], 1)


if __name__ == "__main__":
    unittest.main()
