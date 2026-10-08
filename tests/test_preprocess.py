import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocess


def fake_crop(h=80, w=160):
    """Dark roller with one bright blob (the cashew)."""
    img = np.full((h, w, 3), 20, np.uint8)
    img[30:55, 40:110] = 200
    return img


class PreprocessTest(unittest.TestCase):
    def test_find_cashew_rect(self):
        area, rect = preprocess.find_cashew_rect(fake_crop())
        self.assertGreater(area, 1000)
        self.assertEqual(tuple(rect), (40, 30, 70, 25))
        self.assertEqual(preprocess.find_cashew_rect(np.full((50, 50, 3), 20, np.uint8)), (0, None))

    def test_tighten_keeps_margin_and_falls_back(self):
        m = preprocess.TIGHT_MARGIN_PX
        self.assertEqual(preprocess.tighten(fake_crop()).shape[:2], (25 + 2 * m, 70 + 2 * m))
        empty = np.full((50, 60, 3), 20, np.uint8)
        self.assertEqual(preprocess.tighten(empty).shape, empty.shape)

    def test_square_for_any_shape(self):
        for h, w in ((25, 70), (70, 25), (33, 34), (50, 50)):
            sq = preprocess.to_square(np.zeros((h, w, 3), np.uint8))
            self.assertEqual(sq.shape[0], sq.shape[1])
            self.assertEqual(sq.shape[0], max(h, w))

    def test_labeler_and_live_paths_give_the_same_tensor(self):
        transform = preprocess.build_transform(train=False)
        crop = fake_crop()
        from_labeler = preprocess.crop_to_tensor(crop, transform, tight=True)          # auto-labeler / training
        from_live = preprocess.crop_to_tensor(preprocess.tighten(crop), transform, tight=False)   # grader cuts first
        self.assertEqual(tuple(from_labeler.shape), (3, preprocess.CNN_INPUT_SIZE, preprocess.CNN_INPUT_SIZE))
        self.assertTrue(bool((from_labeler == from_live).all()))


if __name__ == "__main__":
    unittest.main()
