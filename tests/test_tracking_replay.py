import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import flow_geometry as fg
import lane_tracker
import replay_model


class TwentyInTwentyOut(unittest.TestCase):
    """One physical cashew = one ID. The run is modelled on the real one of 2026-10-10 03:15 (see replay_model.py)."""

    def test_20_cashews_give_20_ids(self):
        for seed in range(1, 41):
            ids, ghosts, everything = replay_model.run(lane_tracker, replay_model.detections(20, seed), fg.RIGHT_TO_LEFT)
            self.assertEqual(len(ids), 20, seed)
            self.assertEqual([len(v) for v in ids.values()], [1] * 20, f"seed {seed}: a cashew got more than one ID")
            self.assertEqual(len({i for v in ids.values() for i in v}), 20, f"seed {seed}: two cashews share an ID")

    def test_also_without_faint_detections(self):
        for seed in range(1, 21):
            ids, _, _ = replay_model.run(lane_tracker, replay_model.detections(20, seed, use_weak=False), fg.RIGHT_TO_LEFT)
            self.assertEqual([len(v) for v in ids.values()], [1] * 20, seed)

    def test_the_tracker_before_the_fix_split_these_cashews(self):
        """The same frames through the tracker as it was at commit 3618f26 give more than 20 IDs (the fault seen on the machine)."""
        root = os.path.dirname(HERE)
        try:
            old = subprocess.run(["git", "show", "3618f26:lane_tracker.py"], cwd=root, capture_output=True, text=True, timeout=20)
        except Exception:
            self.skipTest("git not available")
        if old.returncode != 0:
            self.skipTest("commit 3618f26 not available")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lane_tracker_before.py")
            with open(path, "w") as f:
                f.write(old.stdout)
            spec = importlib.util.spec_from_file_location("lane_tracker_before", path)
            before = importlib.util.module_from_spec(spec)
            sys.modules["lane_tracker_before"] = before
            spec.loader.exec_module(before)
        ids, _, _ = replay_model.run(before, replay_model.detections(20, 1), fg.RIGHT_TO_LEFT, weak_supported=False)
        self.assertGreater(sum(len(v) for v in ids.values()), 20)


if __name__ == "__main__":
    unittest.main()
