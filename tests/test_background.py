"""A detection is a cashew only when it moves with the belt and is not tiny (owner, 2026-10-13: the roller and
stains on it were detected and sent to the reject ejector). Numbers are from logs/cashews_20261013.csv."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import engines
import flow_geometry as fg
import lane_tracker
from lane_tracker import Track, long_side_px, moves_with_belt

FLOW, BELT, FRACTION = fg.RIGHT_TO_LEFT, 2.48, lane_tracker.DEFAULTS["stuck_speed_fraction"]


def track(points):
    """points = [(t_ms, cx), ...] of real detections."""
    t0, x0 = points[0]
    trk = Track(id="R1-000001", roi=1, cx=x0, cy=300, box=[x0 - 30, 250, x0 + 30, 350], born_ms=t0, seen_ms=t0, now_ms=t0)
    for t, x in points:
        trk.path.append((t, x, 300))
    trk.hits = len(points)
    return trk


class MovesWithBelt(unittest.TestCase):
    def test_one_frame_at_the_belt_end_is_no_proof(self):               # R2-000019: one frame at x 25-79, after the line
        self.assertFalse(moves_with_belt(track([(0, 52)]), FLOW, BELT, FRACTION))

    def test_a_standing_mark_is_not_a_cashew(self):                     # R1-000003: 6 px in 4 frames, 0.16 px/ms
        self.assertFalse(moves_with_belt(track([(0, 284), (13, 283), (26, 280), (39, 278)]), FLOW, BELT, FRACTION))

    def test_a_creeping_mark_is_not_a_cashew(self):                     # R2-000002: 0.41 px/ms
        self.assertFalse(moves_with_belt(track([(0, 491), (20, 483), (39, 475)]), FLOW, BELT, FRACTION))

    def test_a_cashew_is_proven_after_three_frames(self):               # slowest real cashew of 2,385: 2.37 px/ms
        points = [(0, 240), (16, 202), (32, 164)]
        self.assertFalse(moves_with_belt(track(points[:2]), FLOW, BELT, FRACTION))
        self.assertTrue(moves_with_belt(track(points), FLOW, BELT, FRACTION))

    def test_a_cashew_entering_at_half_speed_still_counts(self):        # box centre of a partly visible cashew
        self.assertTrue(moves_with_belt(track([(0, 240), (16, 220), (32, 200)]), FLOW, BELT, FRACTION))

    def test_something_moving_against_the_belt_is_not_a_cashew(self):
        self.assertFalse(moves_with_belt(track([(0, 100), (16, 140), (32, 180)]), FLOW, BELT, FRACTION))

    def test_without_a_belt_speed_nothing_is_held_back(self):
        self.assertTrue(moves_with_belt(track([(0, 52)]), FLOW, 0.0, FRACTION))


class SmallObjects(unittest.TestCase):
    def test_long_side_is_the_middle_value_over_the_frames(self):
        self.assertEqual(long_side_px([[545, 1696, 581, 1721], [400, 1650, 453, 1700], [310, 1593, 375, 1698]]), 53.0)
        self.assertEqual(long_side_px([]), 0.0)
        self.assertEqual(long_side_px([None, [0, 0, 10, 228]]), 228.0)

    def test_the_limit_is_off_until_the_owner_sets_it(self):
        self.assertEqual(engines.RULE_DEFAULTS["min_object_mm"], 0.0)


if __name__ == "__main__":
    unittest.main()
