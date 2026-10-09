"""History of every graded cashew.

logs/cashews_YYYYMMDD.csv   one row per cashew (kept)
logs/frames_YYYYMMDD.jsonl  one line per cashew with its frame-by-frame votes (kept `keep_days` days)
Writing happens in its own thread so the grading loop never waits for the disk.
"""
import csv
import json
import logging
import queue
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("cashew_log")

CSV_FIELDS = ["time", "id", "roi", "frames", "frames_used", "result", "reason", "reject", "good_frames", "neck_frames",
              "seen", "votes", "views", "cnn_sw_votes", "length_px", "width_px", "area_px", "length_mm", "width_mm",
              "L", "a", "b", "dark_pct_max", "red_pct_max", "spots_max", "speed_px_ms", "t_cross_ms",
              "grade", "relay", "fire", "late", "shadow"]


class CashewLog:
    def __init__(self, folder, keep_days=7, frames_enabled=True):
        self.folder = Path(folder)
        self.keep_days = keep_days
        self.frames_enabled = frames_enabled
        self._queue = queue.Queue(maxsize=2000)
        self._stop = threading.Event()
        self._thread = None
        self.dropped = 0

    def start(self):
        if self._thread is None:
            self.folder.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(target=self._run, daemon=True, name="CashewLog")
            self._thread.start()
        return self

    def stop(self, timeout=2.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def write(self, row: dict, frames=None):
        """Queue one cashew. `row` uses CSV_FIELDS (missing ones stay empty); `frames` is a list of dicts."""
        try:
            self._queue.put_nowait((datetime.now(), row, frames))
        except queue.Full:
            self.dropped += 1

    def flush(self, timeout=2.0):
        end = time.time() + timeout
        while not self._queue.empty() and time.time() < end:
            time.sleep(0.01)
        time.sleep(0.05)

    # ------------------------------------------------------------------ writer thread
    def _run(self):
        last_cleanup = None
        while not (self._stop.is_set() and self._queue.empty()):
            try:
                when, row, frames = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                day = when.strftime("%Y%m%d")
                self._write_csv(self.folder / f"cashews_{day}.csv", when, row)
                if frames is not None and self.frames_enabled:
                    with open(self.folder / f"frames_{day}.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"time": when.isoformat(timespec="milliseconds"), "id": row.get("id"),
                                            "result": row.get("result"), "frames": frames}, separators=(",", ":")) + "\n")
                if last_cleanup != day:
                    last_cleanup = day
                    self.cleanup(when)
            except Exception:
                log.exception("Could not write the cashew log")

    @staticmethod
    def _write_csv(path, when, row):
        new = not path.exists()
        if not new:     # columns changed since this day's file was started: keep the old file, start a new one
            with open(path, "r", encoding="utf-8") as f:
                header = f.readline().strip()
            if header != ",".join(CSV_FIELDS):
                path.rename(path.with_name(f"{path.stem}_old_{when.strftime('%H%M%S')}.csv"))
                new = True
        out = {k: row.get(k, "") for k in CSV_FIELDS}
        out["time"] = when.strftime("%H:%M:%S.%f")[:-3]
        for key, value in out.items():
            if isinstance(value, dict):
                out[key] = " ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in value.items())
            elif isinstance(value, float):
                out[key] = f"{value:.3f}"
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(out)

    def cleanup(self, now=None):
        """Delete per-frame logs older than keep_days. The one-row-per-cashew CSV files are never deleted."""
        limit = ((now or datetime.now()) - timedelta(days=self.keep_days)).strftime("%Y%m%d")
        for path in self.folder.glob("frames_*.jsonl"):
            day = path.stem.split("_", 1)[1]
            if day.isdigit() and day < limit:
                try:
                    path.unlink()
                except OSError:
                    log.warning("Could not delete old frame log %s", path)
