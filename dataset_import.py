"""Build the clean training dataset: one folder per FINAL class name (the names the grading rules use).

Images are COPIED from older dataset folders; the originals are never moved, changed or deleted.
Left out (and listed): folders that are not a final class, unreadable files, full camera frames, exact duplicates.
Every 10th image of a class goes to the test set, by the same counter the Teaching CNN page uses.

    python3 dataset_import.py            # import from the two known sources and print the table
    python3 dataset_import.py --dry-run  # only show what would happen
"""
import argparse
import hashlib
import os
import shutil
from typing import Dict, List, Optional

import dataset_ops as ops

# Owner, 2026-10-09: no SW class (SW / WW is decided by colour measurement; old SW images are GOOD_CASHEW);
# DOUBLE is a CNN class (two cashews stuck together -> reject ejector), in addition to the size rule.
FINAL_CLASSES = ["GOOD_CASHEW", "UPW", "PKW1", "PKW2", "OIL", "BROWN", "DW1", "DW2", "TESTA", "TESTA SMALL",
                 "WHITE_DW", "REJECTION", "EDGECUT", "DOUBLE"]
TARGET = 1000
MAX_CROP_SIDE = 900         # larger than this is a full camera frame, not a cashew crop
# Older folder names -> final class (owner, 2026-10-09: BACK_CUT_OIL / OIL_BACKCUT are OIL).
ALIASES = {"good": "GOOD_CASHEW", "good cashew": "GOOD_CASHEW", "sw": "GOOD_CASHEW", "testa small": "TESTA SMALL", "back cut oil": "OIL",
           "oil backcut": "OIL", "backcut oil": "OIL", "oil back cut": "OIL", "edge cut": "EDGECUT", "white dw": "WHITE_DW",
           "double kernels": "DOUBLE", "double kernel": "DOUBLE", "double cashews": "DOUBLE"}


def _key(name: str) -> str:
    return " ".join(str(name).lower().replace("_", " ").split())


def final_class(folder_name: str) -> Optional[str]:
    """The final class a source folder belongs to, or None when it is not one of them."""
    key = _key(folder_name)
    if key in ALIASES:
        return ALIASES[key]
    for name in FINAL_CLASSES:
        if _key(name) == key:
            return name
    return None


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def counts(dataset_dir, testset_dir) -> Dict[str, tuple]:
    """{class: (images in the dataset, images in the test set)} for every final class."""
    data, test = ops.class_counts(str(dataset_dir)), ops.class_counts(str(testset_dir))
    return {name: (data.get(name, 0), test.get(name, 0)) for name in FINAL_CLASSES}


def build(sources: List[tuple], dataset_dir, testset_dir, dry_run=False, read_image=None) -> dict:
    """sources: [(tag, folder with class sub-folders)] or [(tag, folder, {sub-folders to take})].
    Returns a report dict; see the keys below."""
    import cv2
    read_image = read_image or cv2.imread
    dataset_dir, testset_dir = str(dataset_dir), str(testset_dir)
    report = {"before": counts(dataset_dir, testset_dir), "copied": {n: 0 for n in FINAL_CLASSES}, "to_testset": {n: 0 for n in FINAL_CLASSES},
              "skipped_folders": {}, "duplicates": 0, "unreadable": [], "full_frames": [], "per_source": {}}
    if not dry_run:
        for name in FINAL_CLASSES:                          # every final class has its folder, also when it is still empty
            os.makedirs(os.path.join(dataset_dir, name), exist_ok=True)
    seen = set()
    for root in (dataset_dir, testset_dir):                 # what is already there is never copied twice
        for folder, _, files in os.walk(root):
            seen.update(_md5(os.path.join(folder, f)) for f in files if f.lower().endswith(ops.IMAGE_TYPES))
    state = ops.read_state(dataset_dir)
    for entry in sources:
        tag, source, only = entry[0], entry[1], (entry[2] if len(entry) > 2 else None)
        per = report["per_source"].setdefault(tag, {})
        if not os.path.isdir(source):
            continue
        for folder in sorted(os.listdir(source)):
            path = os.path.join(source, folder)
            if not os.path.isdir(path) or (only is not None and folder not in only):
                continue
            images = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith(ops.IMAGE_TYPES)]
            target_class = final_class(folder)
            if target_class is None:
                if images:
                    report["skipped_folders"][f"{tag}/{folder}"] = len(images)
                continue
            for image_path in images:
                digest = _md5(image_path)
                if digest in seen:
                    report["duplicates"] += 1
                    continue
                image = read_image(image_path)
                if image is None:
                    report["unreadable"].append(image_path)
                    continue
                if max(image.shape[:2]) > MAX_CROP_SIDE:
                    report["full_frames"].append(image_path)
                    continue
                seen.add(digest)
                count = int(state["confirmed"].get(target_class, 0)) + 1
                to_test = count % ops.TEST_EVERY == 0
                state["confirmed"][target_class] = count
                if not to_test:
                    state["since_training"][target_class] = int(state["since_training"].get(target_class, 0)) + 1
                report["to_testset" if to_test else "copied"][target_class] += 1
                per[f"{folder} -> {target_class}"] = per.get(f"{folder} -> {target_class}", 0) + 1
                if not dry_run:
                    dest_folder = os.path.join(testset_dir if to_test else dataset_dir, target_class)
                    os.makedirs(dest_folder, exist_ok=True)
                    shutil.copy2(image_path, ops._free_name(dest_folder, f"{tag}_{os.path.basename(image_path)}"))
    if not dry_run:
        ops._write_state(dataset_dir, state)
    report["after"] = counts(dataset_dir, testset_dir) if not dry_run else {
        n: (report["before"][n][0] + report["copied"][n], report["before"][n][1] + report["to_testset"][n]) for n in FINAL_CLASSES}
    return report


def table(report: dict) -> str:
    lines = ["| Class | Before | Imported | After (train) | Test set | Still needed to 1000 |", "|---|---|---|---|---|---|"]
    for name in FINAL_CLASSES:
        before, (after, test) = report["before"][name][0], report["after"][name]
        lines.append(f"| {name} | {before} | {report['copied'][name] + report['to_testset'][name]} | {after} | {test} | {max(0, TARGET - after)} |")
    total_after = sum(v[0] for v in report["after"].values())
    lines.append(f"| **Total** | {sum(v[0] for v in report['before'].values())} | {sum(report['copied'].values()) + sum(report['to_testset'].values())} "
                 f"| {total_after} | {sum(v[1] for v in report['after'].values())} | {sum(max(0, TARGET - v[0]) for v in report['after'].values())} |")
    return "\n".join(lines)


if __name__ == "__main__":
    import app_paths
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    SOURCES = [("old", "/home/phoenix/Documents/CNN_final/CNN/dataset"), ("proj", str(app_paths.DATASET_DIR)),
               ("oldsw", "/home/phoenix/Documents/CNN_final/CNN/dataset_disabled", {"SW"})]       # SW images -> GOOD_CASHEW
    result = build(SOURCES, app_paths.CLEAN_DATASET_DIR, app_paths.TESTSET_DIR, dry_run=args.dry_run)
    print(table(result))
    for tag, rows in result["per_source"].items():
        print(f"\nfrom {tag}: " + ", ".join(f"{k}: {v}" for k, v in rows.items()))
    print(f"\nnot a final class (left out): {result['skipped_folders']}")
    print(f"exact duplicates left out: {result['duplicates']}; unreadable: {len(result['unreadable'])}; full frames left out: {len(result['full_frames'])}")
