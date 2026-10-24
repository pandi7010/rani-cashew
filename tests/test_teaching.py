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
import training_report


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
            self.assertEqual(sorted(os.listdir(data))[:3], ["BROWN", "DOUBLE", "DW1"])        # every final class has a folder
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

    def test_frames_of_one_cashew_are_never_on_both_sides_of_the_split(self):
        # dataset capture: crop_<time ms>_<track id>; production capture: <HHMMSS>_<cashew id>_v<n>
        burst = [f"/d/UPW/crop_{1789975937470 + 50 * n}_1321.jpg" for n in range(9)]
        later = [f"/d/UPW/crop_{1789975937470 + 60000 + 50 * n}_1321.jpg" for n in range(4)]      # the id used again a minute later
        views = [f"/d/UPW/020332_R2-000002_v{n}.jpg" for n in range(1, 5)]
        other = ["/d/UPW/old_crop_0002.jpg", "/d/UPW/old_0088_cashew_1788563085198_1766_1650.jpg", "/d/UPW/old_crop_1789975937470_7.jpg"]
        groups = sorted(training.cashew_groups(burst + later + views + other), key=len)
        self.assertEqual([len(g) for g in groups], [1, 1, 1, 4, 4, 9])
        self.assertEqual(sorted(groups[-1]), sorted(burst))

        images = {"UPW": burst + later + views + other,
                  "GOOD_CASHEW": [f"/d/GOOD_CASHEW/crop_{1789975000000 + 50 * n}_{n // 10}.jpg" for n in range(60)],     # 6 cashews, 10 frames each
                  "ONE": [f"/d/ONE/crop_{1789975000000 + 50 * n}_5.jpg" for n in range(12)]}                              # a single cashew
        train, val = training.split_train_val(images)
        self.assertEqual(len(train) + len(val), sum(len(v) for v in images.values()))
        self.assertFalse({p for p, _ in train} & {p for p, _ in val})
        for index, name in enumerate(images):
            in_train, in_val = [p for p, l in train if l == index], [p for p, l in val if l == index]
            self.assertTrue(in_train and in_val, name)                                       # every class is in both parts
            if name != "ONE":                                                                # one cashew only: it has to be divided
                group_of = {p: n for n, g in enumerate(training.cashew_groups(images[name])) for p in g}
                self.assertFalse({group_of[p] for p in in_train} & {group_of[p] for p in in_val}, name)
        self.assertEqual(len([p for p, l in val if l == 1]), 10)                             # about 20 % of 60, whole cashews

    def test_an_unreadable_image_is_left_out_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            good, broken = os.path.join(tmp, "a.jpg"), os.path.join(tmp, "b.jpg")
            cashew(good)
            with open(broken, "wb") as f:
                f.write(b"not a picture")
            self.assertEqual(training.readable_images([good, broken]), ([good], [broken]))

    def test_rare_class_neither_takes_over_nor_keeps_a_right_model_unsure(self):
        # the image counts of the model of 2026-10-24 13:36 (DW1 8 ... UPW 1784)
        counts = [89, 50, 8, 35, 1010, 59, 85, 1000, 1063, 191, 102, 1631, 1417, 1784, 43]
        weights = training.class_weights(counts)
        self.assertEqual(weights[counts.index(1784)], 1.0)
        self.assertLessEqual(max(weights), training.MAX_CLASS_WEIGHT)
        self.assertLess(weights[counts.index(1631)], weights[counts.index(89)])     # rarer still counts more
        import torch
        criterion = training.smoothed_loss(torch.tensor(weights), training.LABEL_SMOOTHING)
        # the best answer the loss allows for a batch of UPW images: free logits, trained until they settle
        upw = counts.index(1784)
        logits = torch.zeros(1, len(counts), requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=0.1)
        for _ in range(500):
            optimizer.zero_grad()
            criterion(logits, torch.tensor([upw])).backward()
            optimizer.step()
        probs = torch.softmax(logits.detach(), 1)[0]
        self.assertEqual(int(probs.argmax()), upw)
        self.assertGreater(float(probs[upw]), 0.85)     # the old loss: DW1 0.48 > UPW 0.29
        # with equal weights it is PyTorch's own label smoothing
        x, y = torch.randn(6, 4), torch.tensor([0, 1, 2, 3, 0, 1])
        own = training.smoothed_loss(torch.ones(4), 0.1)(x, y)
        self.assertAlmostEqual(float(own), float(torch.nn.CrossEntropyLoss(label_smoothing=0.1)(x, y)), places=5)

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
            # GSN_V21: per-epoch training / validation loss and accuracy, confusion matrix, precision / recall, report pictures
            self.assertTrue({"train_acc", "val_loss"} <= set(rows[0]))
            self.assertEqual(sum(sum(r) for r in saved["confusion_matrix"]), 4)
            self.assertTrue({"precision", "recall", "accuracy"} <= set(saved["per_class"]["BROWN"]))
            self.assertEqual(saved["wrong_images_total"], len(saved["wrong_images"]))
            for kind in ("curves", "classes", "confusion"):
                self.assertTrue(os.path.getsize(result["reports"][kind]) > 5000, kind)
            self.assertEqual(sorted(t for t, _ in training_report.existing_reports(model_path)),
                             ["VALIDATION: CLASSES", "VALIDATION: CONFUSION", "VALIDATION: CURVES"])
            # label check with a confidence limit: every image is listed as a mismatch or as unsure
            check = training.verify_labels(model_path, data, low_confidence=1.01)
            self.assertEqual(check["checked"], 20)
            self.assertEqual(len(check["mismatches"]), 20)
            self.assertEqual({m["kind"] for m in check["mismatches"]} - {"MISMATCH", "LOW CONFIDENCE"}, set())
            self.assertEqual(len(training.verify_labels(model_path, data)["mismatches"]), sum(m["kind"] == "MISMATCH" for m in check["mismatches"]))
            # a check on labelled folders without touching the model
            before = os.path.getmtime(model_path)
            folder_result = training.confusion_on_folder(model_path, data)
            self.assertEqual((folder_result["images"], folder_result["classes"], folder_result["skipped"]), (20, names, {"OIL": 1}))
            self.assertEqual(os.path.getmtime(model_path), before)


class ReportAndChangesTest(unittest.TestCase):
    def test_class_metrics_and_fire_warning(self):
        matrix = [[98, 2, 0], [5, 45, 0], [0, 1, 9]]          # GOOD, TESTA, OIL
        m = training_report.class_metrics(matrix, ["GOOD_CASHEW", "TESTA", "OIL"])
        self.assertEqual((m["GOOD_CASHEW"]["accuracy"], m["TESTA"]["accuracy"], m["OIL"]["accuracy"]), (0.98, 0.9, 0.9))
        self.assertEqual(m["TESTA"]["precision"], round(45 / 48, 4))
        self.assertEqual(m["GOOD_CASHEW"]["recall"], 0.98)
        low = training_report.fire_warnings(m, pass_classes=["GOOD_CASHEW"])
        self.assertEqual(low, [("TESTA", 0.9), ("OIL", 0.9)])                 # the PASS class is not a FIRE class
        self.assertEqual(training_report.fire_warnings({"UPW": {"accuracy": 0.985}}), [])
        self.assertEqual(training_report.fire_warnings({"UPW": {"accuracy": None}}), [])

    def test_confusion_folder_undo_trash_and_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, logs = os.path.join(tmp, "prog", "dataset"), os.path.join(tmp, "logs")
            paths = []
            for n in range(4):
                path = os.path.join(data, "TESTA", f"t{n}.jpg")
                cashew(path, seed=n)
                paths.append(path)
            changes = ops.Changes(logs)
            self.assertIsNone(changes.undo())
            # 1. out of training into CONFUSION (beside the dataset, under the class it came from)
            changes.add("MOVE TO CONFUSION", ops.move_to_confusion(paths[:2], data))
            self.assertEqual(ops.class_counts(data)["TESTA"], 2)
            self.assertEqual([i["former"] for i in ops.list_confusion(data)], ["TESTA", "TESTA"])
            self.assertFalse(ops.confusion_dir(data).startswith(os.path.abspath(data)))
            self.assertNotIn("CONFUSION", training.list_class_images(data))  # training never sees it
            # 2. re-label one of them: it joins training again
            again = ops.list_confusion(data)[0]["path"]
            changes.add("RE-LABEL from CONFUSION to UPW", ops.confirm_images([again], "UPW", data, os.path.join(tmp, "prog", "testset"), use_testset=False))
            self.assertEqual((ops.class_counts(data)["UPW"], ops.confusion_count(data)), (1, 1))
            # 3. delete = trash, so it can be undone
            changes.add("DELETE", ops.trash_images([paths[3]], data))
            self.assertFalse(os.path.exists(paths[3]))
            self.assertEqual(ops.class_counts(data)["TESTA"], 1)
            # multi-step undo, newest first
            self.assertEqual(changes.last_action(), "DELETE (1 image(s))")
            self.assertEqual(changes.undo(), ("DELETE", 1))
            self.assertTrue(os.path.exists(paths[3]))
            self.assertEqual(changes.undo()[0], "RE-LABEL from CONFUSION to UPW")
            self.assertEqual((ops.class_counts(data).get("UPW", 0), ops.confusion_count(data)), (0, 2))
            self.assertEqual(changes.undo(), ("MOVE TO CONFUSION", 2))
            self.assertEqual((ops.class_counts(data)["TESTA"], ops.confusion_count(data)), (4, 0))
            self.assertFalse(changes.can_undo())
            with open(os.path.join(logs, ops.CHANGES_LOG), encoding="utf-8") as f:
                lines = f.read().splitlines()
            self.assertEqual(len(lines), 2 + 1 + 1 + 3)                       # one line per moved file, one per undo
            self.assertIn("MOVE TO CONFUSION: ", lines[0])
            self.assertIn(" -> ", lines[0])
            self.assertIn("UNDO DELETE: 1 of 1 image(s) put back", lines[4])


if __name__ == "__main__":
    unittest.main()
