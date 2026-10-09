"""Production capture: what is kept, the quotas, the limits, and that the frame loop never waits."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import production_capture as pc

CROP = np.zeros((40, 60, 3), np.uint8)


class FakeRandom:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def wait_for(condition, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.02)
    return False


class Classify(unittest.TestCase):
    cfg = pc.merged_config()

    def test_categories(self):
        good = [("GOOD", 0.97)] * 6
        self.assertEqual(pc.classify([("GOOD", 0.97), ("GOOD", 0.70)], "ACCEPT", "GOOD x2", None, self.cfg), "uncertain")
        self.assertEqual(pc.classify([("GOOD", 0.97), ("TESTA", 0.95)], "REJECT", "DEFECT:TESTA", None, self.cfg), "uncertain")
        self.assertEqual(pc.classify(good, "ACCEPT", "GOOD x6", {"would": ["color"]}, self.cfg), "uncertain")
        self.assertEqual(pc.classify(good[:1], "REJECT", "NOT_ENOUGH_VIEWS:1/3", None, self.cfg), "uncertain")
        self.assertEqual(pc.classify([("TESTA", 0.99)] * 5, "REJECT", "DEFECT:TESTA 0.99", None, self.cfg), "reject")
        self.assertEqual(pc.classify(good, "ACCEPT", "GOOD x6", None, self.cfg, FakeRandom(0.001)), "good")
        self.assertIsNone(pc.classify(good, "ACCEPT", "GOOD x6", None, self.cfg, FakeRandom(0.5)))
        self.assertIsNone(pc.classify(good, "LOST", "LEFT_ROI", None, self.cfg, FakeRandom(0.5)))

    def test_pick_views_keeps_the_doubtful_ones_and_spreads(self):
        views = [("GOOD", 0.95)] * 10
        views[3] = ("GOOD", 0.55)
        views[7] = ("TESTA", 0.90)
        chosen = pc.pick_views(views, 4)
        self.assertEqual(len(chosen), 4)
        self.assertEqual(len(set(chosen)), 4)
        self.assertIn(3, chosen)
        self.assertIn(7, chosen)
        self.assertEqual(pc.pick_views(views[:3], 4), [0, 1, 2])


class Capture(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.inbox = Path(self.dir.name) / "inbox"
        self.now = [time.mktime((2026, 10, 6, 8, 0, 0, 0, 0, -1))]
        self.capture = pc.ProductionCapture(clock=lambda: self.now[0])
        self.configure()
        self.capture.refresh_inbox()
        self.capture._take_quota('good', 0)          # production starts at 08:00: the working hours count from here

    def tearDown(self):
        self.capture.stop(wait=True)
        self.dir.cleanup()

    def configure(self, **changes):
        cfg = dict(enabled=True, daily_limit=100, working_hours=10, min_free_gb=0.0)
        cfg.update(changes)
        self.capture.configure(cfg, "Default", self.inbox, {"cnn": "model.pth"})

    def cashew(self, number, cls="TESTA", conf=0.99, decision="REJECT", views=6):
        track = f"L1-{number:06d}"
        for _ in range(views):
            self.capture.note_view(track, CROP, cls, conf)
        return self.capture.on_decision(track, 1, decision, f"DEFECT:{cls}")

    def images(self):
        return sorted(self.inbox.rglob("*.jpg"))

    def test_saves_views_and_metadata(self):
        self.now[0] += 9 * 3600
        self.assertEqual(self.cashew(1), "reject")
        self.assertTrue(wait_for(lambda: len(self.images()) == 4))
        meta = json.loads(next(self.inbox.rglob("*L1-000001.json")).read_text())
        self.assertEqual((meta["id"], meta["lane"], meta["decision"], meta["category"], meta["program"]), ("L1-000001", 1, "REJECT", "reject", "Default"))
        self.assertEqual(len(meta["saved_views"]), 4)
        self.assertEqual(len(meta["views_seen"]), 6)
        self.assertEqual(meta["models"], {"cnn": "model.pth"})

    def test_off_saves_nothing(self):
        self.configure(enabled=False)
        self.assertIsNone(self.cashew(1))
        time.sleep(0.2)
        self.assertEqual(self.images(), [])

    def test_hourly_cap_and_roll_over(self):
        # reject share 30 % of 100 = 30 images a day, 3 an hour: the first hour takes no 4-view cashew ...
        self.assertIsNone(self.cashew(1))
        # ... the unused hour rolls over: in the second hour 6 images are released
        self.now[0] += 3600
        self.assertEqual(self.cashew(2), "reject")
        self.assertIsNone(self.cashew(3))
        # late in the day everything that was not used is available, up to the share of the category
        self.now[0] += 9 * 3600
        taken = sum(1 for n in range(10, 30) if self.cashew(n) == "reject")
        self.assertEqual(taken, 6)                   # 30 images in total = 7 cashews of 4 views, one was taken before
        self.assertEqual(self.capture.status()["today_by_category"]["reject"], 28)

    def test_uncertain_has_its_own_share(self):
        self.now[0] += 9 * 3600
        taken = sum(1 for n in range(40) if self.cashew(n, "GOOD", 0.6, "REJECT") == "uncertain")
        self.assertEqual(taken, 15)                  # 60 images / 4 views

    def test_inbox_full_stops_and_resumes_without_deleting(self):
        self.now[0] += 9 * 3600
        self.configure(inbox_limit=4)
        self.assertEqual(self.cashew(1), "reject")
        self.assertTrue(wait_for(lambda: len(self.images()) == 4))
        self.assertEqual(self.capture.blocking_reason(), pc.INBOX_FULL)
        self.assertIsNone(self.cashew(2))
        self.assertEqual(len(self.images()), 4)      # nothing was deleted to make room
        for path in self.images():                   # the operator reviews / clears the inbox
            path.unlink()
        self.capture.refresh_inbox()
        self.assertEqual(self.capture.blocking_reason(), pc.OK)
        self.assertEqual(self.cashew(3), "reject")

    def test_low_disk_stops(self):
        self.configure(min_free_gb=10 ** 9)
        self.capture.refresh_inbox()
        self.assertEqual(self.capture.blocking_reason(), pc.DISK_LOW)
        self.assertIsNone(self.cashew(1))

    def test_slow_disk_never_blocks_the_frame_loop(self):
        self.capture.stop()
        slow = pc.ProductionCapture(write_image=lambda path, image: time.sleep(0.2), clock=lambda: self.now[0])
        try:
            slow.configure(dict(enabled=True, daily_limit=10 ** 6, working_hours=1, min_free_gb=0.0, share_reject=1.0), "Default", self.inbox)
            slow.refresh_inbox()
            started = time.perf_counter()
            for number in range(300):
                track = f"L1-{number:06d}"
                for _ in range(5):
                    slow.note_view(track, CROP, "TESTA", 0.99)
                slow.on_decision(track, 1, "REJECT", "DEFECT:TESTA")
            elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 0.5)            # 300 cashews handed over while every image write takes 0.2 s
            self.assertGreater(slow.skipped_busy, 200)
        finally:
            slow.stop(wait=True)

    def test_counts_continue_after_a_restart(self):
        self.now[0] += 9 * 3600
        self.cashew(1)
        self.assertTrue(wait_for(lambda: len(self.images()) == 4))
        self.capture._write_state()
        self.capture.stop()
        again = pc.ProductionCapture(clock=lambda: self.now[0])
        try:
            again.configure(dict(enabled=True, daily_limit=100, working_hours=10, min_free_gb=0.0), "Default", self.inbox)
            again.refresh_inbox()
            again._take_quota("reject", 0)
            self.assertEqual(again.status()["today_by_category"]["reject"], 4)
            self.assertEqual(again.status()["inbox"], 4)
        finally:
            again.stop(wait=True)


if __name__ == "__main__":
    unittest.main()
