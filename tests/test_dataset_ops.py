"""Dataset file operations: confirm into class folders, test set rule, undo, inbox listing, retrain counter."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataset_ops as ops


class DatasetOps(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self.tmp.name)
        self.dataset, self.testset, self.inbox, self.src = root / "dataset", root / "testset", root / "inbox", root / "src"
        for folder in (self.dataset, self.src):
            folder.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def images(self, n, folder=None, prefix="img"):
        folder = folder or self.src
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(n):
            path = folder / f"{prefix}_{i:03d}.jpg"
            path.write_bytes(b"x")
            paths.append(str(path))
        return paths

    def test_every_tenth_goes_to_the_test_set(self):
        moves = ops.confirm_images(self.images(25), "GOOD", str(self.dataset), str(self.testset))
        self.assertEqual(len(moves), 25)
        self.assertEqual(len(list((self.dataset / "GOOD").glob("*.jpg"))), 23)
        self.assertEqual(len(list((self.testset / "GOOD").glob("*.jpg"))), 2)
        self.assertEqual(list(self.src.glob("*.jpg")), [])
        # the count continues in the next call: the 30th is a test image again
        ops.confirm_images(self.images(5, prefix="more"), "GOOD", str(self.dataset), str(self.testset))
        self.assertEqual(len(list((self.testset / "GOOD").glob("*.jpg"))), 3)

    def test_undo_puts_everything_back(self):
        paths = self.images(12)
        moves = ops.confirm_images(paths, "TESTA", str(self.dataset), str(self.testset))
        self.assertEqual(ops.undo_moves(moves), 12)
        self.assertEqual(sorted(str(p) for p in self.src.glob("*.jpg")), sorted(paths))

    def test_same_file_name_is_not_overwritten(self):
        (self.dataset / "GOOD").mkdir()
        (self.dataset / "GOOD" / "img_000.jpg").write_bytes(b"old")
        ops.confirm_images(self.images(1), "GOOD", str(self.dataset), str(self.testset))
        self.assertEqual((self.dataset / "GOOD" / "img_000.jpg").read_bytes(), b"old")
        self.assertEqual(len(list((self.dataset / "GOOD").glob("*.jpg"))), 2)

    def test_retrain_recommended_after_300_new_images(self):
        ops.confirm_images(self.images(299), "TESTA", str(self.dataset), str(self.testset), use_testset=False)
        self.assertEqual(ops.retrain_classes(str(self.dataset)), {})
        ops.confirm_images(self.images(1, prefix="last"), "TESTA", str(self.dataset), str(self.testset), use_testset=False)
        self.assertEqual(ops.retrain_classes(str(self.dataset)), {"TESTA": 300})
        ops.training_finished(str(self.dataset))
        self.assertEqual(ops.retrain_classes(str(self.dataset)), {})

    def test_inbox_most_uncertain_first_and_clear(self):
        day = self.inbox / "20261006"
        self.images(3, day, "a")
        (day / "a.json").write_text(json.dumps({"id": "L1-000001", "lane": 1, "decision": "REJECT", "reason": "x", "category": "uncertain",
                                                "saved_views": [{"file": "a_000.jpg", "class": "GOOD", "confidence": 0.9},
                                                                {"file": "a_001.jpg", "class": "TESTA", "confidence": 0.5}]}))
        items = ops.list_inbox(str(self.inbox))
        self.assertEqual([os.path.basename(i["path"]) for i in items], ["a_002.jpg", "a_001.jpg", "a_000.jpg"])
        self.assertEqual(items[1]["suggested"], "TESTA")
        self.assertEqual(ops.inbox_days(str(self.inbox)), ["20261006"])
        self.assertEqual(ops.clear_inbox(str(self.inbox)), 3)
        self.assertEqual(ops.list_inbox(str(self.inbox)), [])

    def test_warnings(self):
        notes = ops.dataset_warnings({"GOOD": 600, "TESTA": 40, "PKW": 0})
        self.assertTrue(any("TESTA (40)" in n and "PKW (0)" in n for n in notes))
        self.assertTrue(any(n.startswith("Unbalanced") for n in notes))
        self.assertEqual(ops.dataset_warnings({"GOOD": 500, "TESTA": 400}), [])


if __name__ == "__main__":
    unittest.main()
