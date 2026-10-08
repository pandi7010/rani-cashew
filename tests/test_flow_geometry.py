import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flow_geometry as fg


class FlowGeometryTest(unittest.TestCase):
    def test_orientation_follows_flow(self):
        self.assertEqual(fg.line_orientation(fg.RIGHT_TO_LEFT), fg.VERTICAL)
        self.assertEqual(fg.line_orientation(fg.LEFT_TO_RIGHT), fg.VERTICAL)
        self.assertEqual(fg.line_orientation(fg.TOP_TO_BOTTOM), fg.HORIZONTAL)
        self.assertEqual(fg.line_orientation(fg.BOTTOM_TO_TOP), fg.HORIZONTAL)
        self.assertEqual(fg.valid_flow("nonsense"), fg.RIGHT_TO_LEFT)

    def test_progress_is_negative_before_and_positive_after_the_line(self):
        # (flow, point before the line, point after the line), line at 250
        cases = [(fg.RIGHT_TO_LEFT, (400, 80), (200, 80)), (fg.LEFT_TO_RIGHT, (100, 80), (300, 80)),
                 (fg.TOP_TO_BOTTOM, (80, 100), (80, 300)), (fg.BOTTOM_TO_TOP, (80, 400), (80, 200))]
        for flow, before, after in cases:
            self.assertLess(fg.progress(flow, 250, *before), 0, flow)
            self.assertGreater(fg.progress(flow, 250, *after), 0, flow)
            self.assertEqual(abs(fg.progress(flow, 250, *after)), 50, flow)

    def test_right_to_left_matches_the_old_rule(self):
        # old code: crossed when line_x - 400 <= cx <= line_x
        for cx in (251, 250, 100, -150, -151):
            old = (cx <= 250) and (cx >= 250 - 400)
            self.assertEqual(0 <= fg.progress(fg.RIGHT_TO_LEFT, 250, cx, 0) <= 400, old, cx)

    def test_step_toward_line_reduces_the_distance(self):
        for flow in fg.FLOWS:
            for point in ((100, 100), (400, 400)):
                vx, vy = fg.step_toward_line(flow, 250, *point, 20)
                before = abs(fg.progress(flow, 250, *point))
                after = abs(fg.progress(flow, 250, point[0] + vx, point[1] + vy))
                self.assertEqual(before - after, 20, (flow, point))

    def test_lane_overlap_uses_the_axis_across_the_flow(self):
        a, b_same_row, b_same_column = [0, 0, 50, 50], [200, 10, 250, 60], [10, 200, 60, 250]
        self.assertGreater(fg.lane_overlap(fg.RIGHT_TO_LEFT, a, b_same_row), 0)
        self.assertLess(fg.lane_overlap(fg.RIGHT_TO_LEFT, a, b_same_column), -20)
        self.assertGreater(fg.lane_overlap(fg.TOP_TO_BOTTOM, a, b_same_column), 0)
        self.assertLess(fg.lane_overlap(fg.TOP_TO_BOTTOM, a, b_same_row), -20)

    def test_stack_crops_wide_and_tall(self):
        wide = [np.full((50, 300, 3), 1, np.uint8), np.full((60, 200, 3), 2, np.uint8)]
        image, places = fg.stack_crops(wide)
        self.assertEqual(image.shape[:2], (110, 300))
        self.assertEqual(places, [(0, 0, 300, 50), (0, 50, 200, 60)])
        tall = [np.full((300, 50, 3), 1, np.uint8), np.full((200, 60, 3), 2, np.uint8)]
        image, places = fg.stack_crops(tall)
        self.assertEqual(image.shape[:2], (300, 110))
        self.assertEqual(places, [(0, 0, 50, 300), (50, 0, 60, 200)])
        self.assertEqual(fg.place_index(places, 70, 100), 1)
        self.assertEqual(fg.place_index(places, 70, 250), -1)      # padding below the shorter crop
        for (x, y, w, h), value in zip(places, (1, 2)):
            self.assertTrue((image[y:y + h, x:x + w] == value).all())


if __name__ == "__main__":
    unittest.main()
