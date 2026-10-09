"""Ported from the 5-track CASHEW X AI project (external drive, read only); adapted for GSN_V21 where noted.

Automatic capture of cashew views during production, into an unlabelled INBOX for later review.

The frame loop only hands over references; all encoding and disk work is done by one saver thread.
If the saver falls behind, cashews are skipped (and counted) - the frame loop never waits.
Nothing in the inbox is ever deleted by this module, and nothing is added to the training dataset.

    inbox/<YYYYMMDD>/<HHMMSS>_<cashew id>_v<n>.jpg      3-5 views per cashew
    inbox/<YYYYMMDD>/<HHMMSS>_<cashew id>.json          what the machine saw and decided
"""
import json
import os
import queue
import random
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

DEFAULTS = {
    "enabled": True,              # GSN_V21: on (owner, 2026-10-09); switch on the Teaching CNN page
    "daily_limit": 5000,          # images per day
    "inbox_limit": 100000,        # images in the inbox in total; capture stops (never deletes) when reached
    "min_free_gb": 20.0,          # capture stops below this free disk space
    "working_hours": 10,          # the daily limit is spread over this many hours; unused hours roll over
    "share_uncertain": 0.60,      # shares of the daily limit
    "share_reject": 0.30,
    "share_good": 0.10,
    "good_sample_pct": 1.0,       # share of the confident GOOD cashews that are candidates at all
    "uncertain_conf_low": 0.40,
    "uncertain_conf_high": 0.85,
    "views_per_cashew": 4,        # 3-5: the most different views of one cashew
}
CATEGORIES = ("uncertain", "reject", "good")
OK, OFF, DAY_LIMIT, INBOX_FULL, DISK_LOW, NOT_READY = "OK", "OFF", "DAY LIMIT REACHED", "INBOX FULL", "DISK SPACE LOW", "NOT READY"
MAX_VIEWS_KEPT = 12               # per cashew in memory, thinned evenly when more arrive
IMAGE_TYPES = (".jpg", ".jpeg", ".png")


def merged_config(cfg=None):
    out = dict(DEFAULTS)
    for key, value in (cfg or {}).items():
        if key in out:
            out[key] = value
    out["views_per_cashew"] = max(3, min(5, int(out["views_per_cashew"])))
    return out


def count_inbox(root):
    """Images in the whole inbox (all days)."""
    total = 0
    try:
        for day in os.scandir(root):
            if day.is_dir():
                total += sum(1 for entry in os.scandir(day.path) if entry.name.lower().endswith(IMAGE_TYPES))
    except OSError:
        pass
    return total


def classify(views, decision, reason, engine_info, cfg, rng=random):
    """uncertain / reject / good / None for one decided cashew. views: [(class, confidence)]."""
    lo, hi = float(cfg["uncertain_conf_low"]), float(cfg["uncertain_conf_high"])
    reason = str(reason or "")
    uncertain = (any(lo <= conf <= hi for _, conf in views) or len({cls for cls, _ in views if cls}) > 1
                 or bool((engine_info or {}).get("would")) or reason.startswith(("NOT_ENOUGH_VIEWS", "LOW_CONFIDENCE")))
    if uncertain:
        return "uncertain"
    if decision == "REJECT":
        return "reject"
    if decision == "ACCEPT" and rng.random() * 100.0 < float(cfg["good_sample_pct"]):
        return "good"
    return None


def pick_views(views, n):
    """Indices of the n most different views: the least certain one, one that disagrees with the
    majority class, and the rest evenly spread over the travel (different sides of the rolling cashew)."""
    count = len(views)
    if count <= n:
        return list(range(count))
    chosen = [min(range(count), key=lambda i: views[i][1])]
    classes = [cls for cls, _ in views]
    majority = max(set(classes), key=classes.count)
    odd = [i for i in range(count) if classes[i] != majority and i not in chosen]
    if odd:
        chosen.append(min(odd, key=lambda i: views[i][1]))
    step = (count - 1) / (n - 1)
    for k in range(n):
        index = int(round(k * step))
        if len(chosen) >= n:
            break
        if index not in chosen:
            chosen.append(index)
    for index in range(count):                       # rounding collisions: fill up
        if len(chosen) >= n:
            break
        if index not in chosen:
            chosen.append(index)
    return sorted(chosen)


class ProductionCapture:
    def __init__(self, write_image=None, clock=time.time):
        self.cfg = merged_config()
        self.program = ""
        self.inbox_root = None
        self.model_info = {}
        self._clock = clock
        self._write_image = write_image          # tests replace the disk writer
        self._lock = threading.Lock()
        self._views = {}                         # track id -> [(crop, class, confidence)]
        self._jobs = queue.Queue(maxsize=50)
        self._day = ""
        self._day_start = 0.0                    # first capture decision of the day (start of the working hours)
        self._used = {name: 0 for name in CATEGORIES}
        self._inbox_total = None                 # None until the first count
        self._free_gb = None
        self._last_scan = 0.0
        self.skipped_busy = 0                    # cashews dropped because the saver was behind
        self.write_errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ProductionCapture")
        self._thread.start()

    # --------------------------------------------------------------- settings
    def configure(self, cfg, program, inbox_root, model_info=None):
        """Called from the UI thread (cheap). A program change starts new counters from that inbox."""
        root = Path(inbox_root) if inbox_root else None
        changed = root != self.inbox_root
        self.cfg = merged_config(cfg)
        self.program, self.model_info = program, dict(model_info or {})
        if changed:
            with self._lock:
                self.inbox_root, self._day, self._inbox_total, self._last_scan = root, "", None, 0.0

    def stop(self, wait=False):
        self._stop.set()
        if wait:
            self._thread.join(timeout=3.0)

    # ----------------------------------------------------- called while grading
    def wanted(self):
        return bool(self.cfg["enabled"]) and self.inbox_root is not None and self.blocking_reason() == OK

    def note_view(self, track_id, crop, cls, confidence):
        """CNN worker thread: remember one classified view of a cashew that is still travelling."""
        if not cls or not self.wanted():
            return
        with self._lock:
            views = self._views.setdefault(track_id, [])
            views.append((crop, cls, float(confidence)))
            if len(views) > MAX_VIEWS_KEPT:
                del views[1::2]                  # keep the spread over the travel, halve the memory
            if len(self._views) > 400:           # cashews that left without a decision
                for stale in list(self._views)[:100]:
                    del self._views[stale]

    def on_decision(self, track_id, lane, decision, reason, engine_info=None):
        """Frame loop, at the decision line: decide whether this cashew is kept and hand it to the saver.
        Only dictionary work here; never waits."""
        with self._lock:
            views = self._views.pop(track_id, None)
        if not views or not self.wanted():
            return None
        category = classify([(cls, conf) for _, cls, conf in views], decision, reason, engine_info, self.cfg)
        if category is None:
            return None
        n = min(len(views), int(self.cfg["views_per_cashew"]))
        if not self._take_quota(category, n):
            return None
        job = {"id": track_id, "lane": lane, "decision": decision, "reason": str(reason or ""), "category": category,
               "engine": dict(engine_info or {}), "views": views, "n": n, "time": self._clock(), "program": self.program,
               "models": dict(self.model_info)}
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            self.skipped_busy += 1
            with self._lock:
                self._used[category] -= n        # the quota was not used after all
            return None
        return category

    # ------------------------------------------------------------------ quota
    def _roll_day(self, now):
        day = datetime.fromtimestamp(now).strftime("%Y%m%d")
        if day != self._day:
            self._day, self._day_start = day, now
            self._used = {name: 0 for name in CATEGORIES}
            state = self._read_state(day)
            if state:                            # same day after a restart: continue the counts
                self._day_start = float(state.get("start", now))
                for name in CATEGORIES:
                    self._used[name] = int(state.get("used", {}).get(name, 0))

    def _allowance(self, category, now):
        """Images this category may have used by now: its share of the daily limit, released hour by hour.
        Hours that were not used stay available later in the day."""
        daily = float(self.cfg["daily_limit"]) * float(self.cfg[f"share_{category}"])
        hours = max(1.0, float(self.cfg["working_hours"]))
        released_hours = min(hours, int((now - self._day_start) // 3600) + 1)
        return daily * released_hours / hours

    def _take_quota(self, category, n):
        now = self._clock()
        with self._lock:
            self._roll_day(now)
            if sum(self._used.values()) + n > int(self.cfg["daily_limit"]):
                return False
            if self._used[category] + n > self._allowance(category, now):
                return False
            self._used[category] += n
            return True

    def blocking_reason(self):
        if not self.cfg["enabled"]:
            return OFF
        if self.inbox_root is None or self._inbox_total is None:
            return NOT_READY
        if self._inbox_total >= int(self.cfg["inbox_limit"]):
            return INBOX_FULL
        if self._free_gb is not None and self._free_gb < float(self.cfg["min_free_gb"]):
            return DISK_LOW
        if sum(self._used.values()) >= int(self.cfg["daily_limit"]) and self._day == datetime.fromtimestamp(self._clock()).strftime("%Y%m%d"):
            return DAY_LIMIT
        return OK

    def status(self):
        """Numbers for the UI."""
        with self._lock:
            used = dict(self._used) if self._day == datetime.fromtimestamp(self._clock()).strftime("%Y%m%d") else {n: 0 for n in CATEGORIES}
        return {"state": self.blocking_reason(), "today": sum(used.values()), "today_by_category": used,
                "daily_limit": int(self.cfg["daily_limit"]), "inbox": self._inbox_total, "inbox_limit": int(self.cfg["inbox_limit"]),
                "free_gb": self._free_gb, "min_free_gb": float(self.cfg["min_free_gb"]), "skipped_busy": self.skipped_busy,
                "write_errors": self.write_errors, "inbox_root": str(self.inbox_root) if self.inbox_root else ""}

    # ------------------------------------------------------------ saver thread
    def _state_file(self, day):
        return self.inbox_root / f"_capture_state_{day}.json"

    def _read_state(self, day):
        try:
            with open(self._state_file(day), "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError, TypeError):
            return None

    def _write_state(self):
        if self.inbox_root is None or not self._day:
            return
        try:
            self.inbox_root.mkdir(parents=True, exist_ok=True)
            with self._lock:
                state = {"start": self._day_start, "used": dict(self._used)}
            with open(self._state_file(self._day), "w", encoding="utf-8") as f:
                json.dump(state, f)
        except OSError as exc:
            print(f"[CAPTURE] Could not write the capture state: {exc}")

    def refresh_inbox(self):
        """Count the inbox and the free disk space (saver thread; also after a review or clear)."""
        root = self.inbox_root
        if root is None:
            return
        total = count_inbox(root)
        try:
            probe = root if root.exists() else root.parent
            free_gb = shutil.disk_usage(probe if probe.exists() else Path.cwd()).free / 1e9
        except OSError:
            free_gb = None
        with self._lock:
            if root == self.inbox_root:
                self._inbox_total, self._free_gb, self._last_scan = total, free_gb, time.monotonic()

    def _save(self, job):
        import cv2
        root = self.inbox_root
        stamp = datetime.fromtimestamp(job["time"])
        folder = root / stamp.strftime("%Y%m%d")
        folder.mkdir(parents=True, exist_ok=True)
        base = f"{stamp:%H%M%S}_{job['id']}"
        indices = pick_views([(cls, conf) for _, cls, conf in job["views"]], job["n"])
        saved = []
        for number, index in enumerate(indices, start=1):
            crop, cls, conf = job["views"][index]
            name = f"{base}_v{number}.jpg"
            if self._write_image is not None:
                self._write_image(str(folder / name), crop)
            elif not cv2.imwrite(str(folder / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise OSError(f"could not write {name}")
            saved.append({"file": name, "class": cls, "confidence": round(conf, 3), "view_index": index})
        meta = {"id": job["id"], "lane": job["lane"], "time": stamp.strftime("%Y-%m-%d %H:%M:%S"), "program": job["program"],
                "models": job["models"], "category": job["category"], "decision": job["decision"], "reason": job["reason"],
                "views_seen": [{"class": cls, "confidence": round(conf, 3)} for _, cls, conf in job["views"]],
                "saved_views": saved, "engines": job["engine"]}
        with open(folder / f"{base}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        with self._lock:
            if self._inbox_total is not None:
                self._inbox_total += len(saved)
        return len(saved)

    def _run(self):
        last_state = 0.0
        while not self._stop.is_set():
            if self.inbox_root is not None and time.monotonic() - self._last_scan > 60.0:
                self.refresh_inbox()
            try:
                job = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._save(job)
            except Exception as exc:
                self.write_errors += 1
                with self._lock:
                    self._used[job["category"]] = max(0, self._used[job["category"]] - job["n"])
                print(f"[CAPTURE] Could not save {job['id']}: {exc}")
            if time.monotonic() - last_state > 5.0:
                last_state = time.monotonic()
                self._write_state()
