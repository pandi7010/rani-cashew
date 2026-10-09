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


# ====================================================================== GSN_V21 additions (not in the 5-track file)
# CONFUSION folder (images the model keeps getting wrong: left out of training until re-labelled), deletes that can
# be undone (a trash folder), a multi-step undo journal and logs/dataset_changes.log.
import time

CONFUSION = "CONFUSION"
CHANGES_LOG = "dataset_changes.log"


def confusion_dir(dataset_dir):
    """programs/<name>/confusion: beside the dataset, so no training ever reads it."""
    return os.path.join(os.path.dirname(os.path.abspath(str(dataset_dir))), "confusion")


def trash_dir(dataset_dir):
    return os.path.join(os.path.dirname(os.path.abspath(str(dataset_dir))), "trash")


def list_confusion(dataset_dir):
    """Images in the CONFUSION folder, newest first; 'suggested' is the class they were labelled as before."""
    root, items = confusion_dir(dataset_dir), []
    if os.path.isdir(root):
        for former in sorted(os.listdir(root)):
            folder = os.path.join(root, former)
            if os.path.isdir(folder):
                items += [{"path": path, "suggested": former, "confidence": 0.0, "former": former} for path in list_images(folder)]
    items.sort(key=lambda item: os.path.getmtime(item["path"]), reverse=True)
    return items


def confusion_count(dataset_dir):
    return len(list_confusion(dataset_dir))


def move_images(paths, target_folder):
    """Move images into a folder (name clashes get a number). Returns [(old path, new path)] for undo."""
    moves = []
    os.makedirs(target_folder, exist_ok=True)
    for path in paths:
        if os.path.exists(path) and os.path.abspath(os.path.dirname(path)) != os.path.abspath(target_folder):
            target = _free_name(target_folder, os.path.basename(path))
            shutil.move(path, target)
            moves.append((path, target))
    return moves


def move_to_confusion(paths, dataset_dir):
    """Take images out of training: confusion/<the class folder they were in>/. Returns the moves."""
    moves = []
    for path in paths:
        former = os.path.basename(os.path.dirname(os.path.abspath(path))) or "UNLABELLED"
        moves += move_images([path], os.path.join(confusion_dir(dataset_dir), former))
    return moves


def trash_images(paths, dataset_dir):
    """"Delete": the images go into trash/<time>/<their folder name>/ so the delete can be undone.
    The trash is never emptied by the program."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    moves = []
    for path in paths:
        former = os.path.basename(os.path.dirname(os.path.abspath(path)))
        moves += move_images([path], os.path.join(trash_dir(dataset_dir), stamp, former))
    return moves


class Changes:
    """Multi-step undo for every move and delete of the teaching pages, with a line per file in the log."""

    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        self.stack = []             # [(action text, [(old path, new path)])]

    def _log(self, text):
        if not self.log_dir:
            return
        try:
            os.makedirs(str(self.log_dir), exist_ok=True)
            with open(os.path.join(str(self.log_dir), CHANGES_LOG), "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {text}\n")
        except OSError:
            pass

    def add(self, action, moves):
        """Remember one action (a list of moves) so it can be undone, and log every file of it."""
        if not moves:
            return 0
        self.stack.append((action, list(moves)))
        for old, new in moves:
            self._log(f"{action}: {old} -> {new}")
        return len(moves)

    def can_undo(self):
        return bool(self.stack)

    def last_action(self):
        return f"{self.stack[-1][0]} ({len(self.stack[-1][1])} image(s))" if self.stack else ""

    def undo(self):
        """Take back the last action. Returns (action, images restored) or None when there is nothing to undo."""
        if not self.stack:
            return None
        action, moves = self.stack.pop()
        restored = undo_moves(moves)
        self._log(f"UNDO {action}: {restored} of {len(moves)} image(s) put back")
        return action, restored
