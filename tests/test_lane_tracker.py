import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flow_geometry as fg
from lane_tracker import Detection, LaneTracker, crossing_time, track_speed

FLOW = fg.RIGHT_TO_LEFT         # cashews move towards smaller x
ROI = (0, 0, 2400, 600)


def det(x, y, w=200, h=260):
    return Detection(x, y, [x - w / 2, y - h / 2, x + w / 2, y + h / 2])


def run(tracker, frames, dt=40.0, flow=FLOW):
    """frames: list of lists of (x, y). Returns the list of ID tuples per frame, in detection order."""
    ids = []
    for n, points in enumerate(frames):
        dets = [det(x, y) for x, y in points]
        tracker.update(dets, 1000.0 + n * dt, flow, ROI)
        ids.append(tuple(d.track.id for d in dets))
    return ids


class LaneTrackerTest(unittest.TestCase):
    def test_one_cashew_keeps_one_id_while_its_centroid_jitters(self):
        jitter = [0, 12, -9, 15, -14, 6, -11, 9, 0, -7]
        frames = [[(2000 - 30 * n + jitter[n], 300 + jitter[-n - 1])] for n in range(10)]
        ids = run(LaneTracker(1), frames)
        self.assertEqual(len({i for frame in ids for i in frame}), 1)
        self.assertEqual(ids[0][0], "R1-000001")

    def test_neighbours_never_swap_or_share_an_id(self):
        # Two cashews 150 px apart, moving 60 px per frame: the old greedy tracker could cross them.
        frames = [[(2000 - 60 * n, 300), (2150 - 60 * n, 310)] for n in range(15)]
        ids = run(LaneTracker(1), frames)
        self.assertTrue(all(frame == ids[0] for frame in ids))
        self.assertNotEqual(ids[0][0], ids[0][1])
        # the same with the detections listed in the other order every second frame
        frames = [list(reversed(f)) if n % 2 else f for n, f in enumerate(frames)]
        ids = run(LaneTracker(1), frames)
        for n, frame in enumerate(ids):
            self.assertEqual(tuple(reversed(frame)) if n % 2 else frame, ids[0])

    def test_missed_frames_keep_the_id(self):
        frames = [[(2000 - 40 * n, 300)] for n in range(6)] + [[], [], []] + [[(2000 - 40 * n, 300)] for n in range(9, 14)]
        ids = run(LaneTracker(1), frames)
        self.assertEqual({i for frame in ids for i in frame}, {"R1-000001"})

    def test_missed_longer_than_hold_time_gives_a_new_id(self):
        tracker = LaneTracker(1, {"hold_ms": 100})
        frames = [[(2000 - 40 * n, 300)] for n in range(5)] + [[]] * 6 + [[(1500, 300)]]
        ids = run(tracker, frames)
        self.assertEqual(ids[-1], ("R1-000002",))

    def test_a_cashew_behind_is_never_taken_for_the_one_in_front(self):
        tracker = LaneTracker(1)
        run(tracker, [[(1000 - 40 * n, 300)] for n in range(5)])       # first cashew is now at x = 840
        dets = [det(1040, 300)]                                         # a new one 200 px behind it
        tracker.update(dets, 1000.0 + 5 * 40, FLOW, ROI)
        self.assertEqual(dets[0].track.id, "R1-000002")

    def test_other_roller_channel_is_another_cashew(self):
        ids = run(LaneTracker(1), [[(2000, 150)], [(1960, 450)]])
        self.assertNotEqual(ids[0], ids[1])

    def test_new_cashew_starts_with_the_lane_speed(self):
        tracker = LaneTracker(1)
        run(tracker, [[(2000 - 150 * n, 300)] for n in range(8)])      # 150 px per 40 ms
        self.assertAlmostEqual(tracker.lane_speed, 150 / 40.0, places=2)
        # A new cashew that moves 150 px in its first step is still one cashew (gate from the lane speed).
        a, b = [det(2300, 300)], [det(2150, 300)]
        tracker.update(a, 2000.0, FLOW, ROI)
        tracker.update(b, 2040.0, FLOW, ROI)
        self.assertEqual(a[0].track.id, b[0].track.id)

    def test_ids_are_separate_per_roi_and_survive_clear(self):
        t1, t2 = LaneTracker(1), LaneTracker(2)
        self.assertEqual(run(t1, [[(2000, 300)]])[0][0], "R1-000001")
        self.assertEqual(run(t2, [[(2000, 300)]])[0][0], "R2-000001")
        t1.clear()
        self.assertEqual(run(t1, [[(2000, 300)]])[0][0], "R1-000002")

    def test_vertical_flow(self):
        frames = [[(300, 100 + 50 * n), (310, 260 + 50 * n)] for n in range(10)]
        ids = run(LaneTracker(1), frames, flow=fg.TOP_TO_BOTTOM)
        self.assertTrue(all(frame == ids[0] for frame in ids))

    def test_predicted_track_ends_when_it_leaves_the_roi(self):
        tracker = LaneTracker(1, {"hold_ms": 10000})
        run(tracker, [[(200 - 50 * n, 300)] for n in range(3)])        # at x = 100, 50 px per frame
        for n in range(3, 8):
            tracker.update([], 1000.0 + n * 40, FLOW, ROI)
        self.assertEqual(tracker.tracks, [])

    def test_crossing_time_is_interpolated(self):
        path = [(1000.0, 280, 300), (1040.0, 240, 300)]                # line at x = 250, right to left
        self.assertAlmostEqual(crossing_time(path, FLOW, 250), 1030.0)
        self.assertIsNone(crossing_time([(1000.0, 240, 300)], FLOW, 250))
        self.assertAlmostEqual(track_speed(path, FLOW), 1.0)


if __name__ == "__main__":
    unittest.main()
