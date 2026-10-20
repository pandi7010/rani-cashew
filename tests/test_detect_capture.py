import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import detect_capture as dc


def make(root, free=100.0, **cfg):
    written = []
    cap = dc.DetectCapture(root, cfg, write_image=lambda path, image: (written.append(path), open(path, "wb").close(), True)[2],
                           clock=lambda: 1792480000.25, free_gb=lambda: free, start_thread=False)
    return cap, written


def frame(cnn, conf, h=20, w=30, edge=False):
    return {"cnn": cnn, "cnn_conf": conf, "edge_cut": edge, "crop_img": np.zeros((h, w, 3), np.uint8)}


class DetectCaptureTest(unittest.TestCase):
    crop = np.zeros((20, 30, 3), np.uint8)

    def test_fired_cashew_goes_into_the_folder_of_its_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap, _ = make(tmp)
            self.assertTrue(cap.on_fired("R1-000007", "UPW", 13, self.crop))
            self.assertTrue(cap.on_fired("R2-000008", "TESTA SMALL", 29, self.crop))
            self.assertTrue(cap.on_fired("R1-000009", "W450", 14, self.crop))
            files = cap.drain()
            self.assertEqual(len(files), 3)
            self.assertEqual(sorted(os.listdir(tmp)), ["TESTA_SMALL", "UPW", "W450"])
            day = os.listdir(os.path.join(tmp, "UPW"))[0]
            self.assertEqual(len(day), 8)
            name = os.listdir(os.path.join(tmp, "UPW", day))[0]
            self.assertTrue(name.endswith("_250_R1-000007_A13.jpg"), name)
            self.assertEqual(cap.status()["saved"], 3)

    def test_one_picture_per_cashew(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap, written = make(tmp)
            cap.on_fired("R1-000007", "OIL", 5, self.crop)
            cap.drain()
            self.assertEqual(len(written), 1)

    def test_nothing_saved_without_a_crop_or_when_switched_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap, _ = make(tmp)
            self.assertFalse(cap.on_fired("R1-000007", "UPW", 2, None))
            off, _ = make(tmp, enabled=False)
            self.assertFalse(off.on_fired("R1-000007", "UPW", 2, self.crop))
            self.assertEqual(cap.drain() + off.drain(), [])

    def test_disk_nearly_full_pauses_saving(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap, _ = make(tmp, free=19.0)
            self.assertFalse(cap.on_fired("R1-000007", "UPW", 2, self.crop))
            self.assertIn("disk nearly full", cap.status()["blocked"])
            self.assertEqual(cap.skipped, 1)

    def test_folder_names_are_safe(self):
        self.assertEqual(dc.folder_name("TESTA SMALL"), "TESTA_SMALL")
        self.assertEqual(dc.folder_name("double kernel"), "DOUBLE_KERNEL")
        self.assertEqual(dc.folder_name(""), dc.UNDEFINED)
        self.assertEqual(dc.folder_name("../x"), "x".upper())

    def test_best_frame_is_the_most_confident_uncut_vote_for_the_class(self):
        frames = [frame("UPW", 0.99, edge=True), frame("UPW", 0.90), frame("UPW", 0.95), frame("GOOD_CASHEW", 0.99)]
        self.assertIs(dc.best_frame(frames, ["UPW"]), frames[2])

    def test_best_frame_matches_good_names_and_canonical_names(self):
        frames = [frame("good cashew", 0.80), frame("UPW", 0.99)]
        self.assertIs(dc.best_frame(frames, ["GOOD_CASHEW"]), frames[0])
        alias = [frame("upw_model_name", 0.9), frame("TESTA", 0.99)]
        self.assertIs(dc.best_frame(alias, ["UPW"], canonical=lambda n: "UPW" if n == "upw_model_name" else n), alias[0])

    def test_best_frame_falls_back_to_the_biggest_crop(self):
        frames = [frame("TESTA", 0.9, 20, 30), frame("TESTA", 0.9, 40, 50), frame("TESTA", 0.9, 90, 90, edge=True)]
        self.assertIs(dc.best_frame(frames, ["UNDEFINED"]), frames[1])
        cut = [frame("TESTA", 0.9, 20, 30, edge=True), frame("TESTA", 0.9, 40, 50, edge=True)]
        self.assertIs(dc.best_frame(cut, []), cut[1])
        self.assertIsNone(dc.best_frame([{"cnn": "UPW", "crop_img": None}], ["UPW"]))


if __name__ == "__main__":
    unittest.main()
