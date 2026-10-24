"""Ported from the 5-track CASHEW X AI project (external drive, read only); adapted for GSN_V21 where noted.
GSN_V21: every image goes through preprocess.py, exactly as in live grading (tight crop around the
cashew, padded to a square, 224 x 224, same normalisation).

Model training and label checking, without any UI.

    train_cnn(...)        MobileNetV2 classifier on the class folders, with a held-out validation part
    verify_labels(...)    run a CNN on the labelled images and list those where label and prediction disagree
    train_yolo_detect(...) YOLO DETECTION model (boxes) from images with YOLO .txt labels

Every function reports progress through a callback so the UI can show it live. Results (accuracy per
class, when, on how many images) are written next to the model as <model>.json.
"""
import json
import os
import random
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp")
MIN_VAL_PER_CLASS = 2


def list_class_images(dataset_dir):
    """{class folder name: [image paths]} for every class folder."""
    out = {}
    if os.path.isdir(dataset_dir):
        for name in sorted(os.listdir(dataset_dir)):
            folder = os.path.join(dataset_dir, name)
            if os.path.isdir(folder):
                out[name] = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(IMAGE_TYPES)]
    return out


def metrics_path(model_path):
    return str(Path(model_path).with_suffix(".json"))


def read_metrics(model_path):
    """The saved training result of a model, or None when it was trained without one."""
    try:
        with open(metrics_path(model_path), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# GSN_V21: frames of ONE cashew. The capture names carry the cashew: crop_<time ms>_<track id>.jpg (dataset capture) and
# <HHMMSS>_<cashew id>_v<n>.jpg (production capture). A cashew is photographed many times while it rolls past.
_VIEW_NAME = re.compile(r"^(.+)_v\d+$")
_TRACK_NAME = re.compile(r"^(.*?)(\d{10,})_(\d+)$")
SAME_CASHEW_GAP_MS = 3000       # a track id is used again later for another cashew; a pause this long starts a new one


def cashew_groups(paths):
    """[[paths of one cashew], ...]. An image whose name does not say which cashew it is forms a group of its own."""
    views, tracks, groups = {}, {}, []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        view, track = _VIEW_NAME.match(stem), _TRACK_NAME.match(stem)
        if view:
            views.setdefault(view.group(1), []).append(path)
        elif track:
            tracks.setdefault((track.group(1), track.group(3)), []).append((int(track.group(2)), path))
        else:
            groups.append([path])
    groups += list(views.values())
    for frames in tracks.values():
        frames.sort()
        current = [frames[0][1]]
        for (before, _), (now, path) in zip(frames, frames[1:]):
            if now - before > SAME_CASHEW_GAP_MS:
                groups.append(current)
                current = []
            current.append(path)
        groups.append(current)
    return groups


def split_train_val(images_by_class, val_fraction=0.2, seed=1):
    """Shuffled split per class, so every class is in both parts.
    GSN_V21: all frames of one cashew go to the same part (cashew_groups). With frames of the same cashew on both sides the
    validation only measures whether the model remembers that cashew, and the accuracy looks better than it is."""
    rng = random.Random(seed)
    train, val = [], []
    for index, (name, paths) in enumerate(images_by_class.items()):
        paths = list(paths)
        n_val = max(MIN_VAL_PER_CLASS, int(round(len(paths) * val_fraction)))
        n_val = min(n_val, max(0, len(paths) - 1))
        groups = cashew_groups(paths)
        rng.shuffle(groups)
        if len(groups) < 2:                         # one cashew only: its frames have to be divided
            rng.shuffle(paths)
            chosen, rest = paths[:n_val], paths[n_val:]
        else:
            chosen, rest = [], []
            for position, group in enumerate(groups):
                last_for_training = not rest and position == len(groups) - 1
                too_many = bool(chosen) and len(chosen) + len(group) > 1.5 * n_val
                if len(chosen) >= n_val or last_for_training or too_many or (not chosen and len(group) > 1.5 * n_val):
                    rest += group
                else:
                    chosen += group
            if not chosen:                          # every cashew has more frames than the validation part: the smallest one
                smallest = min(groups, key=len)
                chosen = list(smallest)
                rest = [path for group in groups if group is not smallest for path in group]
        val += [(path, index) for path in chosen]
        train += [(path, index) for path in rest]
    return train, val


def readable_images(paths):
    """(paths OpenCV can read, paths it cannot). GSN_V21: one broken file must not end a training run."""
    import cv2
    good, bad = [], []
    for path in paths:
        (good if cv2.imread(path) is not None else bad).append(path)
    return good, bad


def _load_rgb(path):
    """A dataset image as the live grader gives it to the CNN: preprocess.prepare_crop (tight crop + square)."""
    import cv2
    from PIL import Image
    import preprocess
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"cannot read image {path}")
    return Image.fromarray(preprocess.prepare_crop(image, tight=True))


# GSN_V21: class weights and label smoothing. The 5-track code weighted each class by 1 / its image count and gave the
# weights to CrossEntropyLoss(label_smoothing=0.1), which weights the smoothing part too. Model of 2026-10-24 13:36:
# DW1 with 8 images got weight 71, UPW with 1784 got 0.32, and the model called almost everything DW1. Without DW1
# (15:09) it was right but never sure: every UPW stayed at 50 % confidence or less, so on the machine it gave no vote.
MAX_CLASS_WEIGHT = 5.0      # the rarest class counts at most this many times as much as the most frequent one
LABEL_SMOOTHING = 0.1


def class_weights(counts):
    """Loss weight per class: sqrt(largest count / count), at most MAX_CLASS_WEIGHT; the most frequent class gets 1."""
    largest = max(counts)
    return [min(MAX_CLASS_WEIGHT, (largest / max(1, n)) ** 0.5) for n in counts]


def smoothed_loss(weights, smoothing):
    """Cross entropy with the class weights, plus label smoothing WITHOUT the weights, so a rare class cannot pull
    every image towards itself and a model that is right can still be sure."""
    import torch.nn as nn
    import torch.nn.functional as F
    weighted = nn.CrossEntropyLoss(weight=weights)

    def loss(outputs, labels):
        uniform = -F.log_softmax(outputs, dim=1).mean(dim=1).mean()
        return (1.0 - smoothing) * weighted(outputs, labels) + smoothing * uniform
    return loss


def train_cnn(dataset_dir, classes, epochs, out_dir, name_prefix="cnn", progress=None, stop=None, batch_size=32, pretrained=True,
              pass_classes=()):
    """Train MobileNetV2 on the chosen class folders.

    progress(dict) is called after every epoch with: epoch, epochs, loss, val_acc, seconds_left.
    stop() returning True ends the training after the current epoch (the best model so far is saved).
    Returns the result dict that is also written to <model>.json. Raises ValueError for a bad dataset.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    all_images = list_class_images(dataset_dir)
    images, unreadable = {}, []
    for name in classes:
        images[name], broken = readable_images(all_images.get(name, []))
        unreadable += broken
    if unreadable:
        print(f"[TRAIN] {len(unreadable)} image(s) cannot be read and are left out: {unreadable[:5]}")
    empty = [name for name, paths in images.items() if len(paths) < MIN_VAL_PER_CLASS + 1]
    if empty:
        raise ValueError(f"These classes have fewer than {MIN_VAL_PER_CLASS + 1} images: {', '.join(empty)}")
    if len(images) < 2:
        raise ValueError("At least 2 classes are needed")
    class_names = list(images.keys())
    train_items, val_items = split_train_val(images)
    val_cashews = {name: len(cashew_groups([path for path, label in val_items if label == i])) for i, name in enumerate(class_names)}

    import preprocess
    normalize = transforms.Normalize(preprocess.IMAGENET_MEAN, preprocess.IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(), normalize])
    eval_tf = preprocess.build_transform(train=False)     # the grader's own transform

    class Items(Dataset):
        def __init__(self, items, transform):
            self.items, self.transform = items, transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path, label = self.items[i]
            return self.transform(_load_rgb(path)), label

    train_loader = DataLoader(Items(train_items, train_tf), batch_size=min(batch_size, len(train_items)), shuffle=True, num_workers=0)
    val_loader = DataLoader(Items(val_items, eval_tf), batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)        # GSN_V21: the same images give the same order and augmentation in every run
    try:
        if not pretrained:
            raise RuntimeError("pre-trained weights not wanted")
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        pretrained = True
    except Exception as exc:                      # no internet and no cached weights
        print(f"[TRAIN] Pre-trained MobileNetV2 weights not available ({exc}); training from scratch")
        model = models.mobilenet_v2(weights=None)
        pretrained = False
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model = model.to(device)
    # Rare classes count more than frequent ones (GSN_V21: limited, see class_weights).
    weights = torch.tensor(class_weights([len(images[name]) for name in class_names]), dtype=torch.float32)
    # GSN_V21: label smoothing, weight decay and a learning rate that rises and then falls to zero. With a fixed rate the
    # validation accuracy jumped by 20 points from epoch to epoch and the "best epoch" was luck; now the last epochs settle.
    criterion = smoothed_loss(weights.to(device), LABEL_SMOOTHING)
    peak_lr = 1e-3 if not pretrained else 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=1e-2)
    total_steps = max(2, int(epochs) * len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=peak_lr, total_steps=total_steps, pct_start=0.15)
    steps_done = 0

    def evaluate():
        """Validation: correct / total per class, the loss, and the confusion matrix (rows = label, columns = prediction)."""
        model.eval()
        correct = [0] * len(class_names)
        total = [0] * len(class_names)
        matrix = [[0] * len(class_names) for _ in class_names]
        loss_sum, count = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs.to(device))
                loss_sum += criterion(outputs, labels.to(device)).item() * inputs.size(0)
                count += inputs.size(0)
                predicted = outputs.argmax(dim=1).cpu()
                for label, pred in zip(labels.tolist(), predicted.tolist()):
                    total[label] += 1
                    correct[label] += int(label == pred)
                    matrix[label][pred] += 1
        evaluate.loss, evaluate.matrix = loss_sum / max(1, count), matrix
        return correct, total

    history, best, started = [], None, time.time()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        running, seen, train_right = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            steps_done += 1
            if steps_done < total_steps:
                scheduler.step()
            running += loss.item() * inputs.size(0)
            seen += inputs.size(0)
            train_right += int((outputs.argmax(dim=1) == labels).sum().item())
        correct, total = evaluate()
        val_acc = sum(correct) / max(1, sum(total))
        # Mean of the per-class accuracies: a model that ignores a small class does not look good.
        balanced = sum(c / t for c, t in zip(correct, total) if t) / max(1, sum(1 for t in total if t))
        row = {"epoch": epoch, "loss": round(running / max(1, seen), 4), "val_acc": round(val_acc, 4), "val_balanced_acc": round(balanced, 4),
               "train_acc": round(train_right / max(1, seen), 4), "val_loss": round(evaluate.loss, 4)}
        history.append(row)
        if best is None or balanced >= best["balanced"]:
            best = {"balanced": balanced, "epoch": epoch, "correct": correct, "total": total, "val_acc": val_acc, "matrix": evaluate.matrix,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if progress:
            elapsed = time.time() - started
            progress(dict(row, epochs=int(epochs), seconds_left=elapsed / epoch * (int(epochs) - epoch)))
        if stop and stop():
            break

    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now()
    model_path = os.path.join(out_dir, f"{name_prefix}_{stamp:%Y%m%d_%H%M%S}.pth")
    torch.save({"model_state_dict": best["state"], "class_names": class_names}, model_path)

    # GSN_V21: the images the finished (best) model still gets wrong, over the whole training set, for the CONFUSION review.
    import training_report
    model.load_state_dict(best["state"])
    model.eval()
    wrong_images = []
    every = train_items + val_items
    with torch.no_grad():
        for start in range(0, len(every), batch_size):
            batch = every[start:start + batch_size]
            probabilities = torch.softmax(model(torch.stack([eval_tf(_load_rgb(path)) for path, _ in batch]).to(device)), dim=1).cpu()
            for (path, label), probs in zip(batch, probabilities):
                index = int(probs.argmax())
                if index != label:
                    wrong_images.append({"path": path, "label": class_names[label], "predicted": class_names[index],
                                         "confidence": round(float(probs[index]), 3), "kind": "WRONG AFTER TRAINING"})
    wrong_images.sort(key=lambda m: -m["confidence"])
    metrics = training_report.class_metrics(best["matrix"], class_names)
    result = {
        "type": "cnn", "architecture": "MobileNetV2", "model": os.path.basename(model_path), "trained": stamp.strftime("%Y-%m-%d %H:%M"),
        "dataset": str(dataset_dir), "classes": class_names, "images": {name: len(images[name]) for name in class_names},
        "train_images": len(train_items), "val_images": len(val_items), "val_cashews": val_cashews, "unreadable_images": unreadable,
        "epochs_run": len(history), "best_epoch": best["epoch"],
        "val_accuracy": round(best["val_acc"], 4), "val_balanced_accuracy": round(best["balanced"], 4),
        "per_class": {name: {"correct": best["correct"][i], "total": best["total"][i],
                             "accuracy": round(best["correct"][i] / best["total"][i], 4) if best["total"][i] else None,
                             "precision": metrics[name]["precision"], "recall": metrics[name]["recall"]}
                      for i, name in enumerate(class_names)},
        "confusion_matrix": best["matrix"], "wrong_images": wrong_images[:2000], "wrong_images_total": len(wrong_images),
        "pass_classes": [str(c) for c in pass_classes],
        "pretrained": pretrained, "device": str(device), "history": history,
        "note": "Accuracy is measured on validation images that were not used for training. All frames of one cashew are in the "
                "same part, so a validation cashew was never seen in training.",
    }
    result["reports"] = training_report.save_all(model_path, history, best["matrix"], class_names, pass_classes)
    with open(metrics_path(model_path), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["model_path"] = model_path
    return result


def load_cnn(model_path):
    """(model in eval mode, class names, device) of a saved CNN checkpoint."""
    import torch
    import preprocess
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, class_names = preprocess.load_cnn_checkpoint(model_path, device)     # the loader the grader uses
    except KeyError:
        raise ValueError("This model file has no class names stored; it cannot be used for the label check")
    return model, class_names, device


def verify_labels(model_path, dataset_dir, progress=None, batch_size=32, low_confidence=0.0):
    """Run the CNN on every labelled image. With low_confidence above 0 the list also has images whose prediction
    agrees with the label but with less confidence than that ("kind": LOW CONFIDENCE). Returns a dict:
        mismatches     [{path, label, predicted, confidence}] sorted by confidence (most certain first)
        checked        images checked
        skipped        {class folder: image count} whose class the model does not know (cannot be checked)
        model_classes  class names of the model
    """
    import torch
    from torchvision import transforms
    model, class_names, device = load_cnn(model_path)
    known = {name.upper().strip(): name for name in class_names}
    import preprocess
    eval_tf = preprocess.build_transform(train=False)     # the grader's own transform
    items, skipped = [], {}
    for folder, paths in list_class_images(dataset_dir).items():
        if folder.upper().strip() in known:
            items += [(path, known[folder.upper().strip()]) for path in paths]
        elif paths:
            skipped[folder] = len(paths)
    mismatches = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        tensors, kept = [], []
        for path, label in batch:
            try:
                tensors.append(eval_tf(_load_rgb(path)))
                kept.append((path, label))
            except ValueError:
                continue
        if not tensors:
            continue
        with torch.no_grad():
            probabilities = torch.softmax(model(torch.stack(tensors).to(device)), dim=1).cpu()
        for (path, label), probs in zip(kept, probabilities):
            index = int(probs.argmax())
            if class_names[index] != label:
                mismatches.append({"path": path, "label": label, "predicted": class_names[index], "confidence": round(float(probs[index]), 3),
                                   "kind": "MISMATCH"})
            elif float(probs[index]) < low_confidence:
                mismatches.append({"path": path, "label": label, "predicted": class_names[index], "confidence": round(float(probs[index]), 3),
                                   "kind": "LOW CONFIDENCE"})
        if progress:
            progress(min(len(items), start + batch_size), len(items))
    mismatches.sort(key=lambda m: (m["kind"] != "MISMATCH", -m["confidence"]))
    return {"mismatches": mismatches, "checked": len(items), "skipped": skipped, "model_classes": class_names}


def predict_images(model_path, paths, batch_size=32):
    """Class and confidence of a CNN for a list of images: {path: (class name, confidence)}. Nothing is moved."""
    import torch
    from torchvision import transforms
    model, class_names, device = load_cnn(model_path)
    import preprocess
    eval_tf = preprocess.build_transform(train=False)     # the grader's own transform
    out = {}
    for start in range(0, len(paths), batch_size):
        tensors, kept = [], []
        for path in paths[start:start + batch_size]:
            try:
                tensors.append(eval_tf(_load_rgb(path)))
                kept.append(path)
            except ValueError:
                continue
        if not tensors:
            continue
        with torch.no_grad():
            probabilities = torch.softmax(model(torch.stack(tensors).to(device)), dim=1).cpu()
        for path, probs in zip(kept, probabilities):
            index = int(probs.argmax())
            out[path] = (class_names[index], float(probs[index]))
    return out


def evaluate_cnn(model_path, dataset_dir, batch_size=32):
    """Accuracy of a CNN on labelled class folders (the fixed test set): {CLASS NAME: (correct, total)}.
    Only folders whose class the model knows are in the result."""
    import torch
    from torchvision import transforms
    model, class_names, device = load_cnn(model_path)
    known = {name.upper().strip(): index for index, name in enumerate(class_names)}
    import preprocess
    eval_tf = preprocess.build_transform(train=False)     # the grader's own transform
    result = {}
    for folder, paths in list_class_images(dataset_dir).items():
        key = folder.upper().strip()
        if key not in known or not paths:
            continue
        correct = total = 0
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start:start + batch_size]:
                try:
                    tensors.append(eval_tf(_load_rgb(path)))
                except ValueError:
                    continue
            if not tensors:
                continue
            with torch.no_grad():
                predicted = model(torch.stack(tensors).to(device)).argmax(dim=1).cpu().tolist()
            correct += sum(1 for p in predicted if p == known[key])
            total += len(predicted)
        result[key] = (correct, total)
    return result


def confusion_on_folder(model_path, dataset_dir, batch_size=32):
    """GSN_V21: confusion matrix of a CNN on labelled class folders (for example the test set), without changing the model.
    Returns {"classes": [...], "matrix": rows = label, columns = prediction, "images": n, "skipped": {folder: count}}.
    Classes are the model's classes; folders the model does not know are skipped."""
    import torch
    import preprocess
    model, class_names, device = load_cnn(model_path)
    eval_tf = preprocess.build_transform(train=False)
    known = {name.upper().strip(): index for index, name in enumerate(class_names)}
    matrix = [[0] * len(class_names) for _ in class_names]
    images, skipped = 0, {}
    for folder, paths in list_class_images(dataset_dir).items():
        key = folder.upper().strip()
        if key not in known:
            if paths:
                skipped[folder] = len(paths)
            continue
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start:start + batch_size]:
                try:
                    tensors.append(eval_tf(_load_rgb(path)))
                except ValueError:
                    continue
            if not tensors:
                continue
            with torch.no_grad():
                predicted = model(torch.stack(tensors).to(device)).argmax(dim=1).cpu().tolist()
            for pred in predicted:
                matrix[known[key]][pred] += 1
                images += 1
    return {"classes": class_names, "matrix": matrix, "images": images, "skipped": skipped}


# ------------------------------------------------------------------ YOLO detection
def detection_dataset_stats(images_dir, class_names):
    """Images, labelled images and boxes per class of a folder with YOLO .txt labels next to the images."""
    images = [f for f in sorted(os.listdir(images_dir)) if f.lower().endswith(IMAGE_TYPES)] if os.path.isdir(images_dir) else []
    stats = {"images": len(images), "labelled": 0, "boxes": 0, "per_class": {name: 0 for name in class_names}, "bad_lines": 0, "labelled_files": []}
    for name in images:
        label_file = os.path.join(images_dir, os.path.splitext(name)[0] + ".txt")
        if not os.path.exists(label_file):
            continue
        boxes = 0
        with open(label_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                try:
                    index = int(parts[0])
                    values = [float(v) for v in parts[1:]]
                    # a box is "x y w h"; a polygon (magic wand) is "x1 y1 x2 y2 ..." - YOLO turns polygons into boxes
                    is_box, is_polygon = len(values) == 4, len(values) >= 6 and len(values) % 2 == 0
                    if not (is_box or is_polygon) or not 0 <= index < len(class_names) or not all(0.0 <= v <= 1.0 for v in values):
                        raise ValueError
                except ValueError:
                    stats["bad_lines"] += 1
                    continue
                stats["per_class"][class_names[index]] += 1
                boxes += 1
        if boxes:
            stats["labelled"] += 1
            stats["boxes"] += boxes
            stats["labelled_files"].append(name)
    return stats


def train_yolo_detect(images_dir, class_names, epochs, base_model, out_dir, work_dir, name_prefix="yolo",
                      progress=None, stop=None, imgsz=640, single_class=False):
    """Train a YOLO DETECTION model (boxes + classes), the kind the grader needs.

    images_dir holds images with a YOLO .txt label file next to each labelled image.
    progress(dict): epoch, epochs, box_loss, map50, seconds_left. Raises ValueError when there is nothing to train on.
    """
    from ultralytics import YOLO
    if str(base_model).lower().endswith("-cls.pt"):
        raise ValueError("A classification base model (-cls) cannot be used: the grader needs a detection model")
    stats = detection_dataset_stats(images_dir, class_names)
    if stats["labelled"] < 10:
        raise ValueError(f"Only {stats['labelled']} image(s) with box labels in {images_dir}. Draw boxes on at least 10 images "
                         "(Teaching YOLO page) before training a detection model.")
    files = list(stats["labelled_files"])
    random.Random(1).shuffle(files)
    n_val = max(2, int(round(len(files) * 0.2)))
    work = Path(work_dir)
    if work.exists():
        shutil.rmtree(work)
    for part, names in (("val", files[:n_val]), ("train", files[n_val:])):
        (work / "images" / part).mkdir(parents=True)
        (work / "labels" / part).mkdir(parents=True)
        for name in names:
            shutil.copy(os.path.join(images_dir, name), work / "images" / part / name)
            source = os.path.join(images_dir, os.path.splitext(name)[0] + ".txt")
            target = work / "labels" / part / (os.path.splitext(name)[0] + ".txt")
            if single_class:
                # Detection only: every box becomes class 0 "CASHEW". The label files in the dataset are not changed.
                with open(source, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [" ".join(["0"] + line.split()[1:]) for line in f if len(line.split()) >= 5]
                target.write_text("\n".join(lines) + "\n", encoding="utf-8")
            else:
                shutil.copy(source, target)
    if single_class:
        class_names = ["CASHEW"]
    data_yaml = work / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {work.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n")
        for index, name in enumerate(class_names):
            f.write(f"  {index}: {json.dumps(name)}\n")

    model = YOLO(str(base_model))
    started, history = time.time(), []

    def on_epoch(trainer):
        if trainer.epoch + 1 <= len(history):      # called once more for the final validation of the best weights
            return
        metrics = dict(trainer.metrics or {})
        loss = 0.0
        try:                                       # mean training losses of the epoch, by name
            named = trainer.label_loss_items(trainer.tloss, prefix="train")
            loss = float(next((v for k, v in named.items() if "box" in k), 0.0))
        except Exception:
            pass
        row = {"epoch": trainer.epoch + 1, "epochs": int(epochs), "box_loss": round(loss, 4),
               "map50": round(float(metrics.get("metrics/mAP50(B)", 0.0)), 4)}
        history.append(row)
        if progress:
            elapsed = time.time() - started
            progress(dict(row, seconds_left=elapsed / row["epoch"] * (int(epochs) - row["epoch"])))
        if stop and stop():
            trainer.stop = True
    model.add_callback("on_fit_epoch_end", on_epoch)
    results = model.train(data=str(data_yaml), epochs=int(epochs), imgsz=imgsz, workers=0, verbose=False,
                          project=str(work / "runs"), name="detect", exist_ok=True, plots=False)
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError("Training ended without a best.pt file")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now()
    model_path = os.path.join(out_dir, f"{name_prefix}_{stamp:%Y%m%d_%H%M%S}.pt")
    shutil.copy(best, model_path)
    box = getattr(results, "box", None)
    per_class = {}
    if box is not None and getattr(box, "ap_class_index", None) is not None:
        for position, class_index in enumerate(box.ap_class_index):
            per_class[class_names[int(class_index)]] = {"map50": round(float(box.ap50[position]), 4)}
    result = {
        "type": "yolo-detect", "architecture": os.path.basename(str(base_model)), "model": os.path.basename(model_path),
        "trained": stamp.strftime("%Y-%m-%d %H:%M"), "dataset": str(images_dir), "classes": list(class_names),
        "images": stats["labelled"], "boxes": stats["boxes"], "boxes_per_class": stats["per_class"],
        "train_images": len(files) - n_val, "val_images": n_val, "epochs_run": len(history),
        "map50": round(float(box.map50), 4) if box is not None else None, "map50_95": round(float(box.map), 4) if box is not None else None,
        "per_class": per_class, "history": history, "note": "mAP is measured on validation images that were not used for training.",
    }
    with open(metrics_path(model_path), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["model_path"] = model_path
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the CNN from the command line (same code as the Teaching CNN page).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--classes", default="", help="comma separated; default: every class folder with at least 3 images")
    args = parser.parse_args()
    chosen = [c for c in args.classes.split(",") if c] or [n for n, p in list_class_images(args.dataset).items() if len(p) >= 3]
    outcome = train_cnn(args.dataset, chosen, args.epochs, args.out,
                        progress=lambda p: print(f"epoch {p['epoch']}/{p['epochs']}  loss {p['loss']:.3f}  val acc {p['val_acc']:.3f}  "
                                                 f"left {p['seconds_left']:.0f}s", flush=True))
    print(json.dumps({k: v for k, v in outcome.items() if k != "history"}, indent=2))
