import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import undefined_capture as uc


def make(root, **cfg):
    written = []
    cap = uc.UndefinedCapture(root, cfg, write_image=lambda path, image: (written.append(path), open(path, "wb").close(), True)[2],
                              clock=lambda: 1791578000.0, free_gb=lambda: cfg.pop("_free", 100.0) if "_free" in cfg else 100.0, start_thread=False)
    return cap, written


class UndefinedCaptureTest(unittest.TestCase):
    crop = np.zeros((20, 30, 3), np.uint8)
    frame = np.zeros((200, 300, 3), np.uint8)

    def test_undefined_cashew_gets_full_frame_all_views_and_a_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap, written = make(tmp)
            for _ in range(5):
                cap.note_view("R1-000007", self.crop)
            self.assertTrue(cap.on_result("R1-000007", uc.UNDEFINED, "too_few_frames", {"relay": 1, "votes": {"UPW": 1}}, self.frame))
            files = cap.drain()
            day = os.listdir(os.path.join(tmp, "undefined"))[0]
            names = sorted(os.listdir(os.path.join(tmp, "undefined", day)))
            self.assertEqual(len(names), 1 + 5 + 1)
            self.assertEqual(len(files), 7)
            self.assertTrue(names[0].endswith("_R1-000007_too_few_frames.json"))
            self.assertTrue(any(n.endswith("_full.jpg") for n in names))
            self.assertEqual(sum("_v0" in n for n in names), 5)
            info = json.load(open(os.path.join(tmp, "undefined", day, names[0])))
            self.assertEqual((info["reason"], info["views"], info["relay"]), ("too_few_frames", 5, 1))

    def test_a_decided_good_or_defect_cashew_is_not_saved_and_its_views_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap, written = make(tmp)
            cap.note_view("R1-000001", self.crop)
            self.assertFalse(cap.on_result("R1-000001", None, "upw_seen", {}, self.frame))
            self.assertEqual((cap.drain(), cap._views, os.listdir(tmp)), ([], {}, []))

    def test_double_goes_to_its_own_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap, _ = make(tmp)
            cap.note_view("R2-000003", self.crop)
            cap.on_result("R2-000003", uc.DOUBLE, "DOUBLE_OR_TOUCHING", {}, self.frame)
            cap.drain()
            self.assertEqual(os.listdir(tmp), ["double"])

    def test_day_limit_disk_limit_and_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap, _ = make(tmp, max_per_day=2)
            results = [cap.on_result(f"R1-{n}", uc.UNDEFINED, "no_clear_vote", {}, None) for n in range(4)]
            self.assertEqual(results, [True, True, False, False])
            self.assertIn("day limit", cap.blocking_reason())
            self.assertEqual(cap.skipped, 2)
            cap.drain()
            again, _ = make(tmp, max_per_day=2)             # after a restart the files of today still count
            self.assertFalse(again.on_result("R1-9", uc.UNDEFINED, "x", {}, None))
            full = uc.UndefinedCapture(tmp, {"min_free_gb": 20.0}, free_gb=lambda: 5.0, start_thread=False)
            self.assertIn("disk nearly full", full.blocking_reason())
            off, _ = make(tmp, enabled=False)
            off.note_view("a", self.crop)
            self.assertEqual((off._views, off.on_result("a", uc.UNDEFINED, "x")), ({}, False))

    def test_views_per_cashew_and_cashews_in_memory_are_limited(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap, _ = make(tmp)
            for _ in range(uc.MAX_VIEWS + 10):
                cap.note_view("one", self.crop)
            self.assertEqual(len(cap._views["one"]), uc.MAX_VIEWS)
            for n in range(uc.MAX_TRACKS + 20):
                cap.note_view(f"t{n}", self.crop)
            self.assertEqual(len(cap._views), uc.MAX_TRACKS)

    def test_names_are_safe_for_files(self):
        self.assertEqual(uc.safe("SIZE_NOT_READY:W320"), "SIZE_NOT_READY_W320")
        self.assertEqual(uc.safe("../x y"), "x_y")


if __name__ == "__main__":
    unittest.main()
