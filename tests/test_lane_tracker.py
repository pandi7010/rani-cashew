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

    def test_n_cashews_in_give_n_ids_with_partial_entry_and_missed_frames(self):
        """20 cashews, 100 px per 40 ms, each one enters the ROI partly visible (its box centre then moves at half
        speed) and is missed for 1-3 frames at different places, also right after the entry: 20 IDs, not 21."""
        import random
        rng = random.Random(7)
        speed, length, edge = 2.5, 240.0, 2400.0             # px/ms, cashew length, ROI entry edge (x2)
        starts = [n * 460.0 + rng.uniform(0, 60) for n in range(20)]         # ms at which the leading edge enters
        gaps = {n: set(range(g, g + rng.randint(1, 3))) for n, g in enumerate(rng.randint(2, 14) for _ in range(20))}
        gaps[0], gaps[1] = {2, 3, 4}, {3, 4}                 # lost right after the partial entry (the real case)
        tracker, ids, seen = LaneTracker(1), {}, {}
        for frame in range(int((starts[-1] + 1200) / 40)):
            t = frame * 40.0
            dets, owners = [], []
            for n, t0 in enumerate(starts):
                lead = edge - speed * (t - t0)
                if t < t0 + 16 or lead < -length:
                    continue
                trail = min(edge, lead + length)
                lead = max(0.0, lead)
                if trail - lead < 20:
                    continue
                seen[n] = seen.get(n, 0) + 1
                if seen[n] - 1 in gaps[n]:
                    continue
                cx = (lead + trail) / 2 + rng.uniform(-6, 6)
                dets.append(Detection(cx, 300 + rng.uniform(-8, 8), [lead, 200, trail, 400]))
                owners.append(n)
            tracker.update(dets, 1000.0 + t, FLOW, ROI)
            for n, d in zip(owners, dets):
                ids.setdefault(n, set()).add(d.track.id)
        self.assertEqual(len(ids), 20)
        self.assertEqual([len(v) for v in ids.values()], [1] * 20)                  # one ID per cashew
        self.assertEqual(len({i for v in ids.values() for i in v}), 20)             # and no ID shared by two cashews

    def test_found_again_near_the_predicted_place_keeps_the_id_but_not_far_from_it(self):
        tracker = LaneTracker(1)
        run(tracker, [[(2000 - 100 * n, 300)] for n in range(6)])      # lane speed 2.5 px/ms, last seen at x = 1500
        for n in range(6, 10):                                          # four frames without a detection
            tracker.update([], 1000.0 + n * 40, FLOW, ROI)
        near, far = [det(1000 + 30, 300)], [det(300, 300)]            # predicted x = 1000 at frame 10
        tracker.update(near, 1000.0 + 10 * 40, FLOW, ROI)
        self.assertEqual(near[0].track.id, "R1-000001")
        other = LaneTracker(1)
        run(other, [[(2000 - 100 * n, 300)] for n in range(6)])
        for n in range(6, 10):
            other.update([], 1000.0 + n * 40, FLOW, ROI)
        other.update(far, 1000.0 + 10 * 40, FLOW, ROI)                 # 700 px further than it can be: another cashew
        self.assertEqual(far[0].track.id, "R1-000002")

    def test_weak_detection_continues_a_cashew_but_never_starts_one(self):
        tracker = LaneTracker(1)
        run(tracker, [[(2000 - 100 * n, 300)] for n in range(4)])
        faint = det(1600, 300)
        faint.weak = True
        ghost = det(900, 320)
        ghost.weak = True
        tracker.update([faint, ghost], 1000.0 + 4 * 40, FLOW, ROI)
        self.assertEqual(faint.track.id, "R1-000001")               # the known cashew goes on with a faint box
        self.assertIsNone(ghost.track)                               # a faint box alone is nothing
        self.assertEqual(ghost.ignored, "weak")
        self.assertEqual(len(tracker.tracks), 1)

    def test_second_box_on_the_same_cashew_gets_no_id(self):
        tracker = LaneTracker(1)
        run(tracker, [[(2000 - 100 * n, 300)] for n in range(3)])
        whole, part = det(1700, 300), Detection(1740, 250, [1690, 180, 1790, 320])     # a box on one half of it
        tracker.update([whole, part], 1000.0 + 3 * 40, FLOW, ROI)
        self.assertEqual(whole.track.id, "R1-000001")
        self.assertIsNone(part.track)
        self.assertEqual(part.ignored, "duplicate of R1-000001")
        neighbour = det(1600 + 260, 300)                             # a real neighbour 260 px behind: its own ID
        tracker.update([det(1600, 300), neighbour], 1000.0 + 4 * 40, FLOW, ROI)
        self.assertEqual(neighbour.track.id, "R1-000002")

    def test_companion_finds_the_track_at_the_same_place(self):
        tracker = LaneTracker(1)
        run(tracker, [[(2000 - 100 * n, 300), (1400 - 100 * n, 310)] for n in range(4)])
        first, second = tracker.tracks
        self.assertIsNone(tracker.companion(first, FLOW))            # 600 px apart: two cashews
        second.cx = first.cx + 40                                    # (as after a broken ID: both at one place)
        self.assertIs(tracker.companion(first, FLOW), second)

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
