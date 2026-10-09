import csv
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cashew_log import CashewLog
import grading_config


class CashewLogTest(unittest.TestCase):
    def test_csv_row_and_frame_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = CashewLog(tmp).start()
            log.write({"id": "R1-000001", "roi": 1, "result": "GOOD", "votes": {"GOOD_CASHEW": 3}, "length_px": 251.26},
                      frames=[{"t": 1.0, "cnn": "GOOD_CASHEW", "conf": 0.9, "view": "NECK"}])
            log.write({"id": "R2-000001", "roi": 2, "result": "UNDEFINED", "reason": "not_rolled"})
            log.stop()
            day = datetime.now().strftime("%Y%m%d")
            with open(Path(tmp) / f"cashews_{day}.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["id"] for r in rows], ["R1-000001", "R2-000001"])
            self.assertEqual(rows[0]["votes"], "GOOD_CASHEW=3")
            self.assertEqual(rows[0]["length_px"], "251.260")
            self.assertEqual(rows[1]["reason"], "not_rolled")
            with open(Path(tmp) / f"frames_{day}.jsonl", encoding="utf-8") as f:
                lines = [json.loads(l) for l in f]
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0]["frames"][0]["view"], "NECK")

    def test_old_frame_logs_are_deleted_cashew_csv_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
            recent = (datetime.now() - timedelta(days=3)).strftime("%Y%m%d")
            for name in (f"frames_{old}.jsonl", f"frames_{recent}.jsonl", f"cashews_{old}.csv", "grader.log"):
                (Path(tmp) / name).write_text("x")
            CashewLog(tmp, keep_days=7).cleanup()
            self.assertEqual(sorted(os.listdir(tmp)), sorted([f"frames_{recent}.jsonl", f"cashews_{old}.csv", "grader.log"]))

    def test_frame_log_can_be_switched_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = CashewLog(tmp, frames_enabled=False).start()
            log.write({"id": "R1-000001"}, frames=[{"t": 1.0}])
            log.stop()
            self.assertEqual([n for n in os.listdir(tmp) if n.startswith("frames_")], [])


class GradingConfigTest(unittest.TestCase):
    def test_stored_values_win_and_missing_ones_get_defaults(self):
        cfg = grading_config.merged({"voting": {"good_min_frames": 5}, "priority": ["OIL"], "views": {"rules": {"NECK": {"hollow": [0.1, None]}}}})
        self.assertEqual(cfg["voting"]["good_min_frames"], 5)
        self.assertEqual(cfg["voting"]["defect_conf"], 0.5)
        self.assertEqual(cfg["voting"]["rejection_min_frames"], 2)
        self.assertEqual(cfg["priority"], ["OIL"])
        self.assertEqual(cfg["views"]["rules"]["NECK"], {"hollow": [0.1, None]})
        self.assertEqual(cfg["views"]["rules"]["TOP"], {})
        self.assertTrue(cfg["views"]["show_values"])
        self.assertEqual(grading_config.merged(None), grading_config.defaults())


if __name__ == "__main__":
    unittest.main()
