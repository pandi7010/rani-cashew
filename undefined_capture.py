"""Automatic capture of every Undefined / lost / double cashew, for the owner to look at and to label later.

    captures/undefined/<YYYYMMDD>/<HHMMSS>_<id>_<reason>_full.jpg      the whole camera frame at the decision
    captures/undefined/<YYYYMMDD>/<HHMMSS>_<id>_<reason>_v01.jpg ...   every view (crop) of that cashew
    captures/undefined/<YYYYMMDD>/<HHMMSS>_<id>_<reason>.json          result, reason, votes, ejector
    captures/double/<YYYYMMDD>/...                                     the same for DOUBLE KERNEL

Images are written by an own thread; the frame loop only hands them over and never waits. Nothing is ever deleted
by the program. Capture pauses when the day's limit is reached or the disk is nearly full.
"""
import json
import os
import queue
import shutil
import threading
import time

MAX_VIEWS = 30              # crops kept per cashew while it is tracked
MAX_TRACKS = 200            # cashews remembered at once (older ones are forgotten)
DEFAULTS = {"enabled": True, "max_per_day": 5000, "min_free_gb": 20.0, "full_frame": True}
UNDEFINED, DOUBLE = "undefined", "double"


def safe(text, limit=40):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text))[:limit].strip("_") or "x"


class UndefinedCapture:
    def __init__(self, root, cfg=None, write_image=None, clock=time.time, free_gb=None, start_thread=True):
        self.root, self.cfg = str(root), {**DEFAULTS, **(cfg or {})}
        self._write_image, self._clock = write_image or self._imwrite, clock
        self._free_gb = free_gb or self._disk_free_gb
        self._views = {}                        # track id -> [crop, ...]
        self._lock = threading.Lock()
        self._queue = queue.Queue(maxsize=60)
        self._day, self._count = "", 0
        self.saved, self.skipped, self.last_folder, self.last_reason = 0, 0, "", ""
        self._stop = threading.Event()
        self._thread = None
        if start_thread:
            self._thread = threading.Thread(target=self._run, daemon=True, name="UndefinedCapture")
            self._thread.start()

    # ------------------------------------------------------------------ called from the frame loop (must be fast)
    def note_view(self, track_id, crop):
        """Remember one view of a tracked cashew (a copy of the crop)."""
        if not self.cfg["enabled"] or crop is None or getattr(crop, "size", 0) == 0:
            return
        with self._lock:
            views = self._views.setdefault(track_id, [])
            if len(views) < MAX_VIEWS:
                views.append(crop.copy())
            while len(self._views) > MAX_TRACKS:
                self._views.pop(next(iter(self._views)))

    def forget(self, track_id):
        with self._lock:
            self._views.pop(track_id, None)

    def on_result(self, track_id, category, reason, info=None, full_frame=None):
        """The cashew was decided. category None = not wanted (its views are dropped). Returns True when queued."""
        with self._lock:
            views = self._views.pop(track_id, [])
        if category is None or not self.cfg["enabled"]:
            return False
        blocked = self.blocking_reason()
        if blocked:
            self.skipped += 1
            return False
        now = self._clock()
        job = {"id": track_id, "category": category, "reason": reason, "info": dict(info or {}), "views": views,
               "full": full_frame if self.cfg["full_frame"] else None, "time": now}
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self.skipped += 1
            return False
        self._count += 1
        return True

    # ------------------------------------------------------------------ limits
    def _roll_day(self):
        day = time.strftime("%Y%m%d", time.localtime(self._clock()))
        if day != self._day:
            self._day = day
            folder = os.path.join(self.root, UNDEFINED, day)
            try:
                self._count = sum(1 for name in os.listdir(folder) if name.endswith(".json"))
            except OSError:
                self._count = 0

    def blocking_reason(self):
        """Why nothing is captured now, or "" (checked for every cashew)."""
        if not self.cfg["enabled"]:
            return "switched off"
        self._roll_day()
        if self._count >= int(self.cfg["max_per_day"]):
            return f"day limit reached ({self.cfg['max_per_day']})"
        free = self._free_gb()
        if free is not None and free < float(self.cfg["min_free_gb"]):
            return f"disk nearly full ({free:.0f} GB free)"
        return ""

    def _disk_free_gb(self):
        try:
            os.makedirs(self.root, exist_ok=True)
            return shutil.disk_usage(self.root).free / 1e9
        except OSError:
            return None

    def status(self):
        self._roll_day()
        return {"today": self._count, "saved": self.saved, "skipped": self.skipped, "blocked": self.blocking_reason(),
                "last_folder": self.last_folder, "last_reason": self.last_reason, "max_per_day": int(self.cfg["max_per_day"])}

    # ------------------------------------------------------------------ saver thread
    @staticmethod
    def _imwrite(path, image):
        import cv2
        return bool(cv2.imwrite(path, image))

    def save(self, job):
        """Write one cashew. Returns the list of files written."""
        stamp = time.localtime(job["time"])
        folder = os.path.join(self.root, job["category"], time.strftime("%Y%m%d", stamp))
        os.makedirs(folder, exist_ok=True)
        base = os.path.join(folder, f"{time.strftime('%H%M%S', stamp)}_{safe(job['id'])}_{safe(job['reason'])}")
        files = []
        if job["full"] is not None and self._write_image(base + "_full.jpg", job["full"]):
            files.append(base + "_full.jpg")
        for n, crop in enumerate(job["views"], start=1):
            if self._write_image(f"{base}_v{n:02d}.jpg", crop):
                files.append(f"{base}_v{n:02d}.jpg")
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump({"id": job["id"], "category": job["category"], "reason": job["reason"], "views": len(job["views"]),
                       "time": time.strftime("%Y-%m-%d %H:%M:%S", stamp), **job["info"]}, f, indent=1, default=str)
        files.append(base + ".json")
        self.saved += 1
        self.last_folder, self.last_reason = folder, f"{job['id']} {job['reason']}"
        return files

    def _run(self):
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                self.save(job)
            except Exception as exc:
                self.skipped += 1
                print(f"[UNDEFINED CAPTURE] could not save {job.get('id')}: {exc}")

    def drain(self):
        """Save everything that waits, in this thread (tests, closing the program)."""
        done = []
        while True:
            try:
                done += self.save(self._queue.get_nowait())
            except queue.Empty:
                return done

    def stop(self):
        self._stop.set()
