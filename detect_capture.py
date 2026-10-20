"""One picture of every fired cashew, in a folder named after what it was sorted as (owner, 2026-10-20).

    detects/<class or grade>/<YYYYMMDD>/<HHMMSS_mmm>_<cashew id>_A<ejector>.jpg

The folder is the decided class (UPW, TESTA, PKW1, OIL, REJECTION ...), the size grade for a GOOD (or PASS) cashew
(W450, SW210 ...) and UNDEFINED for a cashew sent to the reject ejector without a class. Only cashews for which a fire
command was sent are saved. The picture is the clearest crop of the cashew: a frame not cut by the image edge that
voted for the decided class with the highest CNN confidence.

Images are written by an own thread; the frame loop only hands the crop over and never waits. No daily limit (owner);
saving pauses while the disk has less than min_free_gb free. Nothing is ever deleted by the program.
"""
import os
import queue
import re
import shutil
import threading
import time

DEFAULTS = {"enabled": True, "min_free_gb": 20.0}
UNDEFINED = "UNDEFINED"
FREE_CHECK_S = 5.0          # free disk space is looked up at most this often


def folder_name(text):
    """A safe folder name: 'TESTA SMALL' -> 'TESTA_SMALL'."""
    out = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text).strip().upper())
    return out.strip("_")[:40] or UNDEFINED


def _norm(name):
    """'GOOD_CASHEW' and 'good cashew' are the same class (as class_registry.normalize_name)."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s-]", " ", str(name or "").lower())).strip()


def best_frame(frames, wanted_classes=(), canonical=None):
    """The frame whose crop shows the cashew best: not cut by the image edge, voting for one of `wanted_classes`,
    highest CNN confidence (`canonical` maps a model's class name to the grading name). Falls back to the biggest
    uncut crop, then to the biggest crop."""
    with_crop = [f for f in frames if f.get("crop_img") is not None and getattr(f["crop_img"], "size", 0) > 0]
    if not with_crop:
        return None
    whole = [f for f in with_crop if not f.get("edge_cut")]
    wanted = {_norm(c) for c in wanted_classes if c}
    voting = [f for f in whole if _norm(canonical(f.get("cnn")) if canonical else f.get("cnn")) in wanted]
    if voting:
        return max(voting, key=lambda f: float(f.get("cnn_conf") or 0.0))
    return max(whole or with_crop, key=lambda f: f["crop_img"].shape[0] * f["crop_img"].shape[1])


class DetectCapture:
    def __init__(self, root, cfg=None, write_image=None, clock=time.time, free_gb=None, start_thread=True):
        self.root, self.cfg = str(root), {**DEFAULTS, **(cfg or {})}
        self._write_image, self._clock = write_image or self._imwrite, clock
        self._free_gb = free_gb or self._disk_free_gb
        self._free_cache, self._free_at = None, -1e9
        self._queue = queue.Queue(maxsize=200)
        self.saved, self.skipped, self.last_file = 0, 0, ""
        self._stop = threading.Event()
        self._thread = None
        if start_thread:
            self._thread = threading.Thread(target=self._run, daemon=True, name="DetectCapture")
            self._thread.start()

    # ------------------------------------------------------------------ called from the frame loop (must be fast)
    def on_fired(self, track_id, name, relay, crop):
        """A fire command was sent for this cashew. `name` = class or grade (folder). Returns True when queued."""
        if not self.cfg["enabled"] or crop is None or getattr(crop, "size", 0) == 0:
            return False
        if self.blocking_reason():
            self.skipped += 1
            return False
        job = {"id": track_id, "folder": folder_name(name), "relay": relay, "crop": crop, "time": self._clock()}
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self.skipped += 1
            return False
        return True

    def blocking_reason(self):
        """Why nothing is saved now, or ""."""
        if not self.cfg["enabled"]:
            return "switched off"
        now = self._clock()
        if now - self._free_at >= FREE_CHECK_S:
            self._free_cache, self._free_at = self._free_gb(), now
        if self._free_cache is not None and self._free_cache < float(self.cfg["min_free_gb"]):
            return f"disk nearly full ({self._free_cache:.0f} GB free)"
        return ""

    def _disk_free_gb(self):
        try:
            os.makedirs(self.root, exist_ok=True)
            return shutil.disk_usage(self.root).free / 1e9
        except OSError:
            return None

    def status(self):
        return {"saved": self.saved, "skipped": self.skipped, "blocked": self.blocking_reason(), "last_file": self.last_file}

    # ------------------------------------------------------------------ saver thread
    @staticmethod
    def _imwrite(path, image):
        import cv2
        return bool(cv2.imwrite(path, image))

    def save(self, job):
        """Write one cashew. Returns the file written, or "" when writing failed."""
        stamp = time.localtime(job["time"])
        folder = os.path.join(self.root, job["folder"], time.strftime("%Y%m%d", stamp))
        os.makedirs(folder, exist_ok=True)
        ms = int((job["time"] % 1) * 1000)
        relay = f"_A{int(job['relay']):02d}" if job.get("relay") else ""
        safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(job["id"]))
        path = os.path.join(folder, f"{time.strftime('%H%M%S', stamp)}_{ms:03d}_{safe_id}{relay}.jpg")
        if not self._write_image(path, job["crop"]):
            self.skipped += 1
            return ""
        self.saved += 1
        self.last_file = path
        return path

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
                print(f"[DETECT CAPTURE] could not save {job.get('id')}: {exc}")

    def drain(self):
        """Save everything that waits, in this thread (tests, closing the program)."""
        done = []
        while True:
            try:
                done.append(self.save(self._queue.get_nowait()))
            except queue.Empty:
                return done

    def stop(self):
        self._stop.set()
        self.drain()
