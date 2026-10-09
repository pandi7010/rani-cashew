"""AUTO LEARNING switch (the learning itself is a later phase; this is the protected switch and its rules).

  * Default OFF. Turning it ON or OFF needs the admin password (checked by the caller, supervisor.py).
  * It cannot be turned ON before the sample target is reached (default 200,000; the admin can change it).
  * Until the self-learning phase is built, turning it ON changes nothing and says so.
  * Setting and target are stored in machine_config.json; every change is written to logs/admin_changes.log
    (time, old -> new, sample count). No password ever goes into a file or a log.
"""
import json
import os
import time

import dataset_ops as ops

KEY = "auto_learning"
DEFAULT_TARGET = 200_000
IMPLEMENTED = False             # set True when the self-learning phase exists
PLANNED_TEXT = "Auto learning is planned, not active in this version"
LOG_NAME = "admin_changes.log"


def _read(path) -> dict:
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _write(path, data):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, str(path))


def settings(path) -> dict:
    """{"enabled": bool, "target": int} (defaults when nothing is stored)."""
    stored = _read(path).get(KEY) or {}
    try:
        target = int(stored.get("target", DEFAULT_TARGET))
    except (TypeError, ValueError):
        target = DEFAULT_TARGET
    return {"enabled": bool(stored.get("enabled", False)), "target": max(1, target)}


def sample_count(dataset_dir, testset_dir) -> int:
    """Human-confirmed images: everything in the class folders of the dataset and of the test set.
    (Images confirmed from the INBOX are moved into these folders, so they are counted here once.)"""
    return sum(ops.class_counts(str(dataset_dir)).values()) + sum(ops.class_counts(str(testset_dir)).values())


def _log(log_dir, text):
    if not log_dir:
        return
    os.makedirs(str(log_dir), exist_ok=True)
    with open(os.path.join(str(log_dir), LOG_NAME), "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {text}\n")


def set_enabled(path, want: bool, samples: int, log_dir=None):
    """Called only after the admin password was accepted. Returns (enabled now, message for the screen)."""
    current = settings(path)
    old, target = current["enabled"], current["target"]
    state = lambda on: "ON" if on else "OFF"
    if want and samples < target:
        message = f"Cannot be turned ON yet: {samples:,} of {target:,} samples ({target - samples:,} more needed)."
        new = old
    elif want and not IMPLEMENTED:
        message, new = PLANNED_TEXT, False
    else:
        message, new = f"AUTO LEARNING is {state(want)}.", bool(want)
    if new != old:
        data = _read(path)
        data[KEY] = {"enabled": new, "target": target}
        _write(path, data)
    _log(log_dir, f"AUTO LEARNING {state(old)} -> {state(new)} (asked for {state(want)}); samples {samples} / {target}; {message}")
    return new, message


def set_target(path, target, samples: int, log_dir=None):
    """Called only after the admin password was accepted. Returns the stored target; raises ValueError for a bad value."""
    target = int(target)
    if not 1 <= target <= 100_000_000:
        raise ValueError("The target must be between 1 and 100,000,000.")
    current = settings(path)
    data = _read(path)
    data[KEY] = {"enabled": current["enabled"], "target": target}
    _write(path, data)
    _log(log_dir, f"AUTO LEARNING target {current['target']} -> {target}; samples {samples}; AUTO LEARNING is {'ON' if current['enabled'] else 'OFF'}")
    return target
