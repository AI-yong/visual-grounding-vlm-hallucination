import unittest

import numpy as np
from PIL import Image

from src.inference.run_sam2_guided import make_masked_crop, padded_bounds


class Sam2GuidedTests(unittest.TestCase):
    def test_padded_bounds_stay_inside_image(self):
        self.assertEqual(padded_bounds((0, 0, 10, 10), 20, 20), (0, 0, 13, 13))

    def test_make_masked_crop_replaces_background(self):
        image = Image.new("RGB", (20, 20), (255, 255, 255))
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:10, 5:10] = True
        crop, ratio = make_masked_crop(image, mask, (5, 5, 10, 10))
        self.assertAlmostEqual(ratio, 25 / 400)
        pixels = np.asarray(crop).reshape(-1, 3)
        self.assertTrue(np.any(np.all(pixels == (127, 127, 127), axis=1)))
        self.assertTrue(np.any(np.all(pixels == (255, 255, 255), axis=1)))


if __name__ == "__main__":
    unittest.main()
