import unittest

from PIL import Image

from src.inference.run_grounding_control import (
    box_iou,
    matched_random_box,
    padded_crop,
)


class GroundingControlTests(unittest.TestCase):
    def test_matched_random_box_preserves_size_and_is_deterministic(self):
        source = (10, 10, 30, 40)
        first = matched_random_box(source, 100, 80, seed=7)
        second = matched_random_box(source, 100, 80, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first[2] - first[0], 20)
        self.assertEqual(first[3] - first[1], 30)
        self.assertEqual(box_iou(source, first), 0.0)

    def test_padded_crop_stays_inside_image(self):
        image = Image.new("RGB", (40, 30), "white")
        crop = padded_crop(image, (0, 0, 10, 10))
        self.assertGreater(crop.width, 10)
        self.assertGreater(crop.height, 10)
        self.assertLessEqual(crop.width, image.width)
        self.assertLessEqual(crop.height, image.height)


if __name__ == "__main__":
    unittest.main()
