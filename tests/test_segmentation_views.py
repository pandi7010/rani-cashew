import os
import sys
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import segmentation
import view_features as vf

SAMPLES = os.path.join(ROOT, "samples", "rolling_example")
BLUE = (150, 60, 20)        # BGR of a blue belt
CREAM = (190, 225, 235)


def synthetic(shape="ellipse"):
    img = np.full((260, 220, 3), BLUE, np.uint8)
    img[:, 100:104] = (10, 10, 10)                              # a dark roller gap
    if shape == "ellipse":
        cv2.ellipse(img, (110, 130), (50, 100), 0, 0, 360, CREAM, -1)
    else:                                                       # kidney: an ellipse with a bite out of one side
        cv2.ellipse(img, (110, 130), (70, 100), 0, 0, 360, CREAM, -1)
        cv2.circle(img, (190, 130), 55, BLUE, -1)
    return img


class SegmentationTest(unittest.TestCase):
    def test_blue_belt_and_dark_gap_are_removed(self):
        seg = segmentation.segment(synthetic())
        self.assertIsNotNone(seg)
        self.assertAlmostEqual(seg.area, np.pi * 50 * 100, delta=600)
        self.assertAlmostEqual(seg.cx, 110, delta=2)
        self.assertAlmostEqual(seg.cy, 130, delta=2)

    def test_empty_belt_and_black_piece_give_no_mask(self):
        self.assertIsNone(segmentation.segment(np.full((200, 200, 3), BLUE, np.uint8)))
        img = np.full((200, 200, 3), BLUE, np.uint8)
        cv2.circle(img, (100, 100), 60, (12, 12, 12), -1)
        self.assertIsNone(segmentation.segment(img))
        self.assertIsNone(segmentation.segment(None))

    def test_real_samples_are_segmented(self):
        for n in range(1, 5):
            img = cv2.imread(os.path.join(SAMPLES, f"frame{n}.jpg"))
            self.assertIsNotNone(img, "sample frame missing")
            seg = segmentation.segment(img)
            self.assertIsNotNone(seg)
            share = seg.area / (img.shape[0] * img.shape[1])
            self.assertTrue(0.3 < share < 0.6, share)

    def test_touches_edge(self):
        roi = (100, 100, 500, 400)
        self.assertFalse(segmentation.touches_edge([150, 150, 300, 300], roi, 2))
        self.assertTrue(segmentation.touches_edge([101, 150, 300, 300], roi, 2))
        self.assertTrue(segmentation.touches_edge([150, 150, 300, 399], roi, 2))

    def test_teach_belt(self):
        belt = np.full((100, 300, 3), BLUE, np.uint8)
        belt[:, ::20] = (8, 8, 8)
        p = segmentation.teach_belt(belt)
        hue = int(cv2.cvtColor(np.uint8([[BLUE]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
        self.assertTrue(p["blue_h_min"] <= hue <= p["blue_h_max"])
        self.assertLessEqual(p["blue_h_max"] - p["blue_h_min"], 12)
        self.assertIsNotNone(segmentation.segment(synthetic(), p))
        self.assertIsNone(segmentation.teach_belt(np.zeros((50, 50, 3), np.uint8)))      # dark picture
        mixed = np.concatenate([belt, np.full((100, 300, 3), (40, 60, 220), np.uint8)])  # half of it is red
        self.assertIsNone(segmentation.teach_belt(mixed))


class ViewFeatureTest(unittest.TestCase):
    def features(self, img):
        return vf.measure(img, segmentation.segment(img))

    def test_outline_features(self):
        oval, kidney = self.features(synthetic()), self.features(synthetic("kidney"))
        self.assertAlmostEqual(oval["aspect"], 2.0, delta=0.1)
        self.assertGreater(oval["solidity"], 0.97)
        self.assertLess(kidney["solidity"], 0.9)
        self.assertGreater(kidney["defect"], oval["defect"] + 0.1)
        self.assertGreater(oval["symmetry"], kidney["symmetry"])
        self.assertLess(oval["hollow"], 0.02)

    def test_dark_red_hollow_is_measured_inside_the_mask(self):
        img = synthetic()
        cv2.ellipse(img, (125, 130), (15, 40), 0, 0, 360, (70, 80, 150), -1)    # red-brown patch off the centre line
        f = self.features(img)
        self.assertGreater(f["hollow"], 0.08)
        self.assertGreater(f["hollow_off"], 0.08)
        self.assertGreater(f["red_pct"], 5)

    def test_no_view_without_limits(self):
        f = self.features(synthetic())
        self.assertEqual(vf.classify(f, {}), vf.OTHER)
        self.assertEqual(vf.classify(f, {"NECK": {}, "BACK": {}, "TOP": {"solidity": [None, None]}}), vf.OTHER)

    def test_view_from_limits(self):
        oval, kidney = self.features(synthetic()), self.features(synthetic("kidney"))
        rules = {"TOP": {"solidity": [None, 0.9]}, "BACK": {"solidity": [0.95, None], "hollow": [None, 0.05]},
                 "NECK": {"solidity": [0.95, None], "hollow": [0.05, None]}}
        self.assertEqual(vf.classify(kidney, rules), "TOP")
        self.assertEqual(vf.classify(oval, rules), "BACK")
        img = synthetic()
        cv2.ellipse(img, (125, 130), (15, 40), 0, 0, 360, (70, 80, 150), -1)
        self.assertEqual(vf.classify(self.features(img), rules), "NECK")

    def test_samples_give_all_features(self):
        img = cv2.imread(os.path.join(SAMPLES, "frame4.jpg"))
        f = self.features(img)
        for name in vf.FEATURES:
            self.assertIn(name, f)
        self.assertIn("A", vf.short_text(f))


if __name__ == "__main__":
    unittest.main()
