import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dataset_import as di
import dataset_ops as ops
import preprocess
import supervisor
import training


def cashew(path, colour=(190, 225, 235), size=(120, 90), seed=0):
    """A small crop: a light blob on a dark blue background (what the grader's tight crop expects)."""
    rng = np.random.default_rng(seed)
    img = np.full((size[0], size[1], 3), (90, 30, 10), np.uint8)
    cv2.ellipse(img, (size[1] // 2 + int(rng.integers(-6, 6)), size[0] // 2), (size[1] // 4, size[0] // 3), int(rng.integers(0, 40)), 0, 360, colour, -1)
    img = np.clip(img.astype(np.int16) + rng.integers(-8, 8, img.shape), 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


class DatasetImportTest(unittest.TestCase):
    def test_final_class_names(self):
        self.assertEqual(len(di.FINAL_CLASSES), 14)
        for folder, expected in (("GOOD", "GOOD_CASHEW"), ("GOOD_CASHEW", "GOOD_CASHEW"), ("TESTA_SMALL", "TESTA SMALL"), ("TESTA SMALL", "TESTA SMALL"),
                                 ("BACK_CUT_OIL", "OIL"), ("OIL_BACKCUT", "OIL"), ("OIL", "OIL"), ("EDGECUT", "EDGECUT"), ("EDGE_CUT", "EDGECUT"),
                                 ("white_dw", "WHITE_DW"), ("DOUBLE_KERNELS", "DOUBLE"), ("SW", "GOOD_CASHEW"), ("Class_1", None), ("golden_images", None)):
            self.assertEqual(di.final_class(folder), expected, folder)

    def test_copies_merges_and_never_touches_the_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, data, test = os.path.join(tmp, "src"), os.path.join(tmp, "dataset"), os.path.join(tmp, "testset")
            for n in range(12):
                cashew(os.path.join(src, "GOOD", f"g{n}.jpg"), seed=n)
            for n in range(3):
                cashew(os.path.join(src, "BACK_CUT_OIL", f"b{n}.jpg"), colour=(60, 180, 200), seed=100 + n)
            cashew(os.path.join(src, "OIL", "o0.jpg"), colour=(60, 180, 200), seed=200)
            cashew(os.path.join(src, "DOUBLE_KERNELS", "d0.jpg"), seed=300)
            cashew(os.path.join(src, "SW", "s0.jpg"), seed=301)
            cashew(os.path.join(src, "golden_images", "x0.jpg"), seed=302)
            cashew(os.path.join(src, "TESTA", "frame.jpg"), size=(1200, 1400), seed=400)      # a full frame, not a crop
            with open(os.path.join(src, "TESTA", "broken.jpg"), "wb") as f:
                f.write(b"not an image")
            import shutil
            shutil.copy(os.path.join(src, "GOOD", "g0.jpg"), os.path.join(src, "GOOD", "copy_of_g0.jpg"))      # exact duplicate
            before = sorted(os.listdir(os.path.join(src, "GOOD")))

            report = di.build([("old", src)], data, test)
            self.assertEqual(sorted(os.listdir(data))[:3], ["BROWN", "DW1", "DW2"])           # every final class has a folder
            self.assertEqual(report["after"]["GOOD_CASHEW"], (12, 1))                        # 12 GOOD + 1 SW; the 10th went to the test set
            self.assertEqual(report["after"]["DOUBLE"], (1, 0))                              # DOUBLE_KERNELS is the class DOUBLE
            self.assertNotIn("SW", report["after"])
            self.assertEqual(report["after"]["OIL"], (4, 0))                                 # BACK_CUT_OIL merged into OIL
            self.assertEqual(report["skipped_folders"], {"old/golden_images": 1})
            self.assertEqual((report["duplicates"], len(report["unreadable"]), len(report["full_frames"])), (1, 1, 1))
            self.assertEqual(sorted(os.listdir(os.path.join(src, "GOOD"))), before)           # the source is untouched
            self.assertIn("| GOOD_CASHEW | 0 | 13 | 12 | 1 | 988 |", di.table(report))

            again = di.build([("old", src)], data, test)                                     # running it twice copies nothing twice
            self.assertEqual(sum(again["copied"].values()) + sum(again["to_testset"].values()), 0)
            self.assertEqual(again["after"]["GOOD_CASHEW"], (12, 1))
            only = di.build([("x", src, {"TESTA"})], data, test)                             # a source limited to some folders
            self.assertEqual(only["skipped_folders"], {})

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            cashew(os.path.join(src, "UPW", "u.jpg"))
            report = di.build([("old", src)], os.path.join(tmp, "dataset"), os.path.join(tmp, "testset"), dry_run=True)
            self.assertEqual(report["after"]["UPW"], (1, 0))
            self.assertFalse(os.path.exists(os.path.join(tmp, "dataset")))


class SupervisorPasswordTest(unittest.TestCase):
    def test_set_and_check_only_a_hash_is_stored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "machine_config.json")
            self.assertFalse(supervisor.is_set(path))
            self.assertFalse(supervisor.check(path, "anything"))
            self.assertRaises(ValueError, supervisor.set_password, path, "abc")
            supervisor.set_password(path, "open-sesame")
            self.assertTrue(supervisor.is_set(path))
            self.assertTrue(supervisor.check(path, "open-sesame"))
            self.assertTrue(supervisor.check(path, " open-sesame "))
            self.assertFalse(supervisor.check(path, "open-sesam"))
            self.assertFalse(supervisor.check(path, None))
            with open(path) as f:
                self.assertNotIn("open-sesame", f.read())


class TrainingTest(unittest.TestCase):
    def test_images_are_read_through_the_graders_preprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.jpg")
            cashew(path, size=(200, 120))
            expected = preprocess.prepare_crop(cv2.imread(path), tight=True)
            got = np.array(training._load_rgb(path))
            self.assertEqual(got.shape[0], got.shape[1])                                     # padded to a square
            self.assertLess(got.shape[0], 200)                                               # tight crop around the cashew
            self.assertTrue(np.array_equal(got, expected))

    def test_small_training_run_keeps_validation_apart_and_writes_the_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, out = os.path.join(tmp, "dataset"), os.path.join(tmp, "models")
            for n in range(10):
                cashew(os.path.join(data, "GOOD_CASHEW", f"g{n}.jpg"), colour=(200, 230, 240), seed=n)
                cashew(os.path.join(data, "BROWN", f"b{n}.jpg"), colour=(40, 70, 120), seed=50 + n)
            cashew(os.path.join(data, "OIL", "only_one.jpg"))
            self.assertRaises(ValueError, training.train_cnn, data, ["GOOD_CASHEW", "OIL"], 1, out, pretrained=False)   # too few images
            rows = []
            result = training.train_cnn(data, ["GOOD_CASHEW", "BROWN"], 2, out, name_prefix="cnn_test", progress=rows.append, pretrained=False)
            self.assertEqual([r["epoch"] for r in rows], [1, 2])
            self.assertTrue({"loss", "val_acc", "seconds_left"} <= set(rows[0]))
            self.assertEqual((result["train_images"], result["val_images"]), (16, 4))        # 20 % held back
            self.assertEqual(result["classes"], ["GOOD_CASHEW", "BROWN"])
            self.assertEqual(sum(row["total"] for row in result["per_class"].values()), 4)
            model_path = result["model_path"]
            self.assertTrue(os.path.exists(model_path) and model_path.endswith(".pth"))
            saved = training.read_metrics(model_path)
            self.assertEqual(saved["val_images"], 4)
            self.assertEqual(len(saved["history"]), 2)
            # the checkpoint is one the grader can load, with its class names
            model, names = preprocess.load_cnn_checkpoint(model_path, "cpu")
            self.assertEqual(names, ["GOOD_CASHEW", "BROWN"])
            predictions = training.predict_images(model_path, [os.path.join(data, "BROWN", "b0.jpg")])
            self.assertIn(list(predictions.values())[0][0], names)


if __name__ == "__main__":
    unittest.main()
