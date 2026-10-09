import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alarms
import decision as dc
import engines
import fire_timing as ft
import flow_geometry as fg
import grading_config
from decision import FrameVote, decide
from lane_tracker import Detection, LaneTracker

GOOD = [FrameVote(0, "GOOD_CASHEW", 0.9), FrameVote(1, "GOOD_CASHEW", 0.9, view="NECK"), FrameVote(2, "GOOD_CASHEW", 0.9)]
NO_NECK = [FrameVote(n, "GOOD_CASHEW", 0.9) for n in range(3)]


class FireTimingTest(unittest.TestCase):
    def test_manual_example(self):
        # 5-track manual B1: 650 mm, 310 mm/s, pulse 150 ms -> about 2020 ms
        self.assertEqual(ft.delay_ms(650, 310, 150), 2022)

    def test_delays_add_up_per_roi_and_are_rising(self):
        spacing = {r: 200.0 for r in range(1, 41)}
        spacing[1], spacing[21] = 400.0, 300.0
        d = ft.delays_from_distances(spacing, 300.0, 50.0)
        self.assertEqual(d[1], round(400 / 300 * 1000 - 25))
        self.assertEqual(d[2], round(600 / 300 * 1000 - 25))
        self.assertEqual(d[20], round((400 + 19 * 200) / 300 * 1000 - 25))
        self.assertEqual(d[21], round(300 / 300 * 1000 - 25))       # ROI 2 starts again at its own line
        self.assertEqual(ft.inconsistent_delays(d), [])

    def test_inconsistent_delays_are_found(self):
        d = {r: 1000 * r for r in range(1, 41)}
        d[16], d[36] = 5500, 11026
        self.assertEqual(ft.inconsistent_delays(d), [16, 36])
        self.assertRaises(ValueError, ft.delay_ms, 100, 0, 50)


class AlwaysRejectTest(unittest.TestCase):
    def test_order_and_reasons(self):
        d = decide(GOOD, appeared_after_line=True, double=True, stuck=True, lost_ms=900)
        self.assertEqual((d.result, d.reason, d.reject), (dc.UNDEFINED, dc.APPEARED_AFTER_LINE, True))
        d = decide(GOOD, double=True, stuck=True)
        self.assertEqual((d.result, d.reason, d.reject), (dc.DOUBLE, dc.DOUBLE_OR_TOUCHING, True))
        d = decide(GOOD, stuck=True, lost_ms=900)
        self.assertEqual((d.result, d.reason), (dc.UNDEFINED, dc.STUCK))
        d = decide(GOOD, lost_ms=450)
        self.assertEqual((d.result, d.reason, d.reject), (dc.UNDEFINED, "LOST_450MS", True))
        self.assertEqual(decide(GOOD, lost_ms=300).result, dc.GOOD)     # exactly the limit is still fine

    def test_view_engine_modes(self):
        active = decide(NO_NECK, {"neck_min_frames": 1}, view_engine=dc.ACTIVE)
        self.assertEqual((active.result, active.reason), (dc.UNDEFINED, dc.NOT_ROLLED))
        shadow = decide(NO_NECK, {"neck_min_frames": 1}, view_engine=dc.SHADOW)
        self.assertEqual((shadow.result, shadow.reject, shadow.shadow), (dc.GOOD, False, [dc.VIEW_NOT_ROLLED]))
        off = decide(NO_NECK, {"neck_min_frames": 1}, view_engine=dc.OFF)
        self.assertEqual((off.result, off.shadow), (dc.GOOD, []))
        self.assertEqual(decide(GOOD, view_engine=dc.SHADOW).shadow, [])

    def test_shadow_never_hides_a_defect(self):
        votes = NO_NECK + [FrameVote(3, "PKW1", 0.9), FrameVote(4, "PKW1", 0.9)]
        self.assertEqual(decide(votes, view_engine=dc.SHADOW).result, "PKW1")


class EngineTest(unittest.TestCase):
    def feat(self, length_mm=26.0, width_mm=13.0, solidity=0.9, px_per_mm=10.0):
        return {"length_px": length_mm * px_per_mm, "width_px": width_mm * px_per_mm, "solidity": solidity,
                "area_px": length_mm * width_mm * 0.7 * px_per_mm ** 2}

    def test_shape(self):
        self.assertIsNone(engines.shape_check([self.feat()] * 3, 10.0))
        self.assertTrue(engines.shape_check([self.feat(solidity=0.6)] * 3, 10.0).startswith(engines.SHAPE_BROKEN))
        self.assertTrue(engines.shape_check([self.feat(10, 8)] * 3, 10.0).startswith(engines.SHAPE_TOO_SMALL))
        self.assertTrue(engines.shape_check([self.feat(15, 14)] * 3, 10.0).startswith(engines.SHAPE_SPLIT))
        self.assertIsNone(engines.shape_check([], 10.0))
        # the middle value of the views decides, one odd view does not
        self.assertIsNone(engines.shape_check([self.feat(), self.feat(solidity=0.5), self.feat()], 10.0))

    def test_double_is_longer_than_45_mm_in_enough_frames(self):
        self.assertFalse(engines.is_double([260, 270, 255], 10.0))
        self.assertTrue(engines.is_double([460, 470, 455], 10.0))
        self.assertFalse(engines.is_double([460, 270, 255, 260, 250], 10.0))    # one long frame is not enough
        self.assertTrue(engines.is_double([460, 470, 255, 260], 10.0))          # two frames and half of them
        self.assertFalse(engines.is_double([460, 470], 10.0, {"enabled": False}))
        self.assertTrue(engines.is_double([300, 310], 6.57))                    # 45.7 mm at the 5-track scale

    def test_new_start_values_reach_a_settings_file_written_before_them(self):
        old = {"voting": {"neck_min_frames": 1, "single_defect_blocks_good": True, "good_min_frames": 4}}
        cfg = grading_config.merged(old)
        self.assertEqual((cfg["voting"]["neck_min_frames"], cfg["voting"]["single_defect_blocks_good"]), (0, False))
        self.assertEqual(cfg["voting"]["good_min_frames"], 4)           # other stored values are kept
        # once saved with the new version, the owner's own choice is kept
        kept = grading_config.merged({"defaults_version": 2, "voting": {"neck_min_frames": 2, "single_defect_blocks_good": True}})
        self.assertEqual((kept["voting"]["neck_min_frames"], kept["voting"]["single_defect_blocks_good"]), (2, True))
        self.assertEqual(grading_config.defaults()["voting"]["neck_min_frames"], 0)

    def test_all_engines_start_in_shadow(self):
        cfg = grading_config.defaults()
        self.assertEqual(set(cfg["engines"].values()), {engines.SHADOW})
        self.assertEqual(cfg["double"]["max_single_length_mm"], 45.0)
        self.assertEqual(cfg["rules"]["max_lost_ms"], 300.0)
        self.assertEqual(cfg["avg_cashew_weight_g"], 1.8)
        self.assertEqual(grading_config.merged({"engines": {"view": "ACTIVE"}})["engines"]["shape"], engines.SHADOW)


class TrackerRuleTest(unittest.TestCase):
    ROI = (0, 0, 2400, 600)

    def det(self, x, y=300):
        return Detection(x, y, [x - 100, y - 130, x + 100, y + 130])

    def test_gap_is_measured_and_dropped_tracks_are_reported(self):
        tracker = LaneTracker(1)
        for n in range(4):
            tracker.update([self.det(2000 - 40 * n)], 1000 + 40 * n, fg.RIGHT_TO_LEFT, self.ROI)
        for n in range(4, 14):                                           # 400 ms without a detection
            tracker.update([], 1000 + 40 * n, fg.RIGHT_TO_LEFT, self.ROI)
        dets = [self.det(2000 - 40 * 14)]
        tracker.update(dets, 1000 + 40 * 14, fg.RIGHT_TO_LEFT, self.ROI)
        self.assertEqual(dets[0].track.id, "R1-000001")
        self.assertGreater(dets[0].track.max_gap_ms, 300)
        for n in range(15, 35):
            tracker.update([], 1000 + 40 * n, fg.RIGHT_TO_LEFT, self.ROI)
            if tracker.dropped:
                break
        self.assertEqual([t.id for t in tracker.dropped], ["R1-000001"])

    def test_stuck_cashew_and_stopped_conveyor(self):
        tracker = LaneTracker(1)
        for n in range(12):                                              # two cashews moving normally
            tracker.update([self.det(2200 - 40 * n), self.det(1700 - 40 * n)], 1000 + 40 * n, fg.RIGHT_TO_LEFT, self.ROI)
        self.assertFalse(tracker.stopped)
        lane = tracker.lane_speed
        stuck_x = 2200 - 40 * 11
        for n in range(12, 40):                                          # the first one stops, the second goes on
            dets = [self.det(stuck_x), self.det(1700 - 40 * n)]
            tracker.update(dets, 1000 + 40 * n, fg.RIGHT_TO_LEFT, self.ROI)
        self.assertTrue(dets[0].track.stuck)
        self.assertFalse(dets[1].track.stuck)
        self.assertFalse(tracker.stopped)
        self.assertAlmostEqual(tracker.lane_speed, lane, delta=0.2 * lane)      # the stuck one did not pull it down
        x2 = 1700 - 40 * 39
        for n in range(40, 60):                                          # now everything stands
            tracker.update([self.det(stuck_x), self.det(x2)], 1000 + 40 * n, fg.RIGHT_TO_LEFT, self.ROI)
        self.assertTrue(tracker.stopped)


class AlarmTest(unittest.TestCase):
    def collect(self, **kw):
        base = dict(camera_connected=True, camera_state="ONLINE", plc_connected=True, running=True, ai_fps=30.0,
                    min_fps=10.0, free_gb=200.0, missed_rejects=[])
        base.update(kw)
        return alarms.collect(**base)

    def test_no_alarm_when_all_is_well(self):
        self.assertEqual(self.collect(), [])

    def test_each_alarm(self):
        self.assertIn("CAMERA OFFLINE", self.collect(camera_connected=False, camera_state="OFFLINE")[0])
        self.assertIn("PLC NOT CONNECTED", self.collect(plc_connected=False)[0])
        self.assertIn("AI FPS 6.0 IS TOO LOW", self.collect(ai_fps=6.0)[0])
        self.assertEqual(self.collect(ai_fps=6.0, running=False), [])
        self.assertIn("DISK SPACE LOW (12 GB free)", self.collect(free_gb=12.0)[0])
        self.assertIn("CONVEYOR STOPPED WITH PENDING REJECTS: R1-000007 (R3)", self.collect(missed_rejects=["R1-000007 (R3)"])[0])
        self.assertEqual(len(self.collect(camera_connected=False, plc_connected=False, free_gb=1.0)), 3)

    def test_min_safe_fps(self):
        self.assertAlmostEqual(alarms.min_safe_fps(1600.0, 160.0), 10.0)


if __name__ == "__main__":
    unittest.main()
