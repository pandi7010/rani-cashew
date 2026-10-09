"""Ported from the 5-track CASHEW X AI project (external drive, read only); adapted for GSN_V21 where noted.

File operations behind the Teaching CNN page: unlabelled images, the production inbox, confirming
images into the dataset (every 10th of a class goes to the fixed test set), undo, and the retrain counter.
No UI. Images are only moved when the operator confirms them; nothing here labels by itself.
"""
import json
import os
import shutil
from datetime import datetime

IMAGE_TYPES = (".jpg", ".jpeg", ".png")
TEST_EVERY = 10            # every 10th confirmed image of a class goes to the test set
RETRAIN_AFTER = 300        # new confirmed images of one class since the last training
STATE_FILE = "_confirmed.json"


def list_images(folder):
    """Images of a folder, newest first."""
    try:
        paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMAGE_TYPES)]
    except OSError:
        return []
    return sorted(paths, key=os.path.getmtime, reverse=True)


def class_counts(dataset_dir):
    """{class folder: image count}, sorted by name."""
    out = {}
    if os.path.isdir(dataset_dir):
        for name in sorted(os.listdir(dataset_dir)):
            folder = os.path.join(dataset_dir, name)
            if os.path.isdir(folder):
                out[name] = sum(1 for f in os.listdir(folder) if f.lower().endswith(IMAGE_TYPES))
    return out


def dataset_warnings(counts, minimum=100):
    """Plain sentences about classes that are too small or very unbalanced."""
    notes = []
    used = {name: count for name, count in counts.items()}
    small = [name for name, count in used.items() if count < minimum]
    if small:
        notes.append(f"Fewer than {minimum} images: {', '.join(f'{n} ({used[n]})' for n in small)}.")
    filled = [count for count in used.values() if count > 0]
    if len(filled) >= 2 and max(filled) > 5 * min(filled):
        biggest, smallest = max(used, key=used.get), min((n for n in used if used[n] > 0), key=used.get)
        notes.append(f"Unbalanced: {biggest} {used[biggest]} against {smallest} {used[smallest]}.")
    if notes:
        notes.append("A model trained on this will not be reliable for the small classes.")
    return notes


# ------------------------------------------------------------------ production inbox
def list_inbox(inbox_root, day=None):
    """Unreviewed production images, most uncertain first.
    Each item: path, day, suggested (class), confidence, lane, cashew id, decision, reason, category."""
    items = []
    try:
        days = sorted(d for d in os.listdir(inbox_root) if os.path.isdir(os.path.join(inbox_root, d)))
    except OSError:
        return items
    for folder_name in days:
        if day and folder_name != day:
            continue
        folder = os.path.join(inbox_root, folder_name)
        listed = set()
        for name in os.listdir(folder):
            if not name.endswith(".json") or name.startswith("_"):
                continue
            try:
                with open(os.path.join(folder, name), "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                continue
            for view in meta.get("saved_views", []):
                path = os.path.join(folder, view.get("file", ""))
                if os.path.exists(path):
                    listed.add(view["file"])
                    items.append({"path": path, "day": folder_name, "suggested": view.get("class", ""),
                                  "confidence": float(view.get("confidence", 0.0)), "lane": meta.get("lane"), "id": meta.get("id", ""),
                                  "decision": meta.get("decision", ""), "reason": meta.get("reason", ""), "category": meta.get("category", "")})
        for name in os.listdir(folder):               # images whose metadata is missing are still shown
            if name.lower().endswith(IMAGE_TYPES) and name not in listed:
                items.append({"path": os.path.join(folder, name), "day": folder_name, "suggested": "", "confidence": 0.0, "lane": None,
                              "id": "", "decision": "", "reason": "", "category": ""})
    items.sort(key=lambda item: item["confidence"])
    return items


def inbox_days(inbox_root):
    try:
        return sorted((d for d in os.listdir(inbox_root) if os.path.isdir(os.path.join(inbox_root, d))), reverse=True)
    except OSError:
        return []


def clear_inbox(inbox_root):
    """Delete every image and metadata file of the inbox (only after password and confirmation in the UI). Returns the image count."""
    removed = 0
    for day in inbox_days(inbox_root):
        folder = os.path.join(inbox_root, day)
        for name in os.listdir(folder):
            if name.lower().endswith(IMAGE_TYPES + (".json",)):
                try:
                    os.remove(os.path.join(folder, name))
                    removed += name.lower().endswith(IMAGE_TYPES)
                except OSError:
                    pass
    return removed


# ------------------------------------------------------------- confirming into the dataset
def _state_path(dataset_dir):
    return os.path.join(dataset_dir, STATE_FILE)


def read_state(dataset_dir):
    """{"confirmed": {class: total confirmed}, "since_training": {class: confirmed since the last training}}"""
    try:
        with open(_state_path(dataset_dir), "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError):
        state = {}
    state.setdefault("confirmed", {})
    state.setdefault("since_training", {})
    return state


def _write_state(dataset_dir, state):
    os.makedirs(dataset_dir, exist_ok=True)
    with open(_state_path(dataset_dir), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)


def _free_name(folder, name):
    target = os.path.join(folder, name)
    stem, ext = os.path.splitext(name)
    number = 1
    while os.path.exists(target):
        target = os.path.join(folder, f"{stem}_{number}{ext}")
        number += 1
    return target


def confirm_images(paths, class_name, dataset_dir, testset_dir, use_testset=True):
    """Move images the operator confirmed into the class folder. With use_testset every 10th confirmed
    image of the class goes to the test set instead (never used for training).
    Returns the moves [(old path, new path)] for undo."""
    state = read_state(dataset_dir)
    moves = []
    for path in paths:
        if not os.path.exists(path):
            continue
        count = int(state["confirmed"].get(class_name, 0)) + 1
        to_test = use_testset and count % TEST_EVERY == 0
        folder = os.path.join(testset_dir if to_test else dataset_dir, class_name)
        os.makedirs(folder, exist_ok=True)
        target = _free_name(folder, os.path.basename(path))
        shutil.move(path, target)
        moves.append((path, target))
        state["confirmed"][class_name] = count
        if not to_test:
            state["since_training"][class_name] = int(state["since_training"].get(class_name, 0)) + 1
    if moves:
        _write_state(dataset_dir, state)
    return moves


def undo_moves(moves):
    """Put the images of one confirm action back where they were."""
    restored = 0
    for old_path, new_path in reversed(moves):
        try:
            if os.path.exists(new_path):
                os.makedirs(os.path.dirname(old_path), exist_ok=True)
                shutil.move(new_path, old_path)
                restored += 1
        except OSError:
            pass
    return restored


def training_finished(dataset_dir):
    """A training has just used the dataset: the 'new since training' counters start again."""
    state = read_state(dataset_dir)
    state["since_training"] = {}
    state["last_training"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    _write_state(dataset_dir, state)


def retrain_classes(dataset_dir):
    """Classes that gained RETRAIN_AFTER or more confirmed images since the last training."""
    since = read_state(dataset_dir)["since_training"]
    return {name: count for name, count in since.items() if int(count) >= RETRAIN_AFTER}
