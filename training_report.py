"""Pictures and numbers of a training result (GSN_V21): Colab-style curves, per-class bars, confusion matrix.

PNG files are written next to the model:  <model>_curves.png  <model>_classes.png  <model>_confusion.png
(and <model>_testset_classes.png / _testset_confusion.png for a check on the test set).
"""
import os
from pathlib import Path

FIRE_LIMIT = 0.98       # owner: a FIRE (defect) class below this accuracy is shown in red
BG, PANEL, TEXT, MUTED, GRID = "#0E1420", "#161E2E", "#E6EAF2", "#8A94A8", "#263044"
GOOD, WARN, BAD, ACCENT = "#2FBF71", "#F2A93B", "#E5484D", "#22B8CF"


def report_paths(model_path, prefix=""):
    """{kind: path} of the report pictures of a model (prefix "testset_" for the test-set check)."""
    stem = str(Path(model_path).with_suffix(""))
    kinds = ("classes", "confusion") if prefix else ("curves", "classes", "confusion")
    return {kind: f"{stem}_{prefix}{kind}.png" for kind in kinds}


def existing_reports(model_path):
    """[(title, path)] of the report pictures that exist for a model."""
    out = []
    for prefix, label in (("", "VALIDATION"), ("testset_", "TEST SET")):
        for kind, path in report_paths(model_path, prefix).items():
            if os.path.exists(path):
                out.append((f"{label}: {kind.upper()}", path))
    return out


def class_metrics(matrix, classes):
    """{class: {total, correct, accuracy (= recall), precision, recall}} from a confusion matrix
    (rows = true class, columns = predicted class)."""
    out = {}
    for i, name in enumerate(classes):
        total = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        correct = matrix[i][i]
        out[name] = {"total": total, "correct": correct,
                     "accuracy": round(correct / total, 4) if total else None,
                     "recall": round(correct / total, 4) if total else None,
                     "precision": round(correct / predicted, 4) if predicted else None}
    return out


def fire_warnings(per_class, pass_classes=(), limit=FIRE_LIMIT):
    """[(class, accuracy)] of the FIRE classes (everything that is not a PASS class) below the limit, worst first."""
    passed = {str(c).upper().strip() for c in pass_classes}
    low = [(name, row["accuracy"]) for name, row in per_class.items()
           if str(name).upper().strip() not in passed and row.get("accuracy") is not None and row["accuracy"] < limit]
    return sorted(low, key=lambda item: item[1])


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "text.color": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID, "font.size": 11, "savefig.facecolor": BG})
    return plt


def save_curves(history, path, title=""):
    """Loss and accuracy per epoch, training and validation, side by side."""
    plt = _pyplot()
    epochs = [row["epoch"] for row in history]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.6))
    for axis, keys, label in ((left, (("loss", "training loss", WARN), ("val_loss", "validation loss", ACCENT)), "loss"),
                              (right, (("train_acc", "training accuracy", WARN), ("val_acc", "validation accuracy", GOOD)), "accuracy")):
        for key, name, colour in keys:
            values = [row.get(key) for row in history]
            if any(v is not None for v in values):
                axis.plot(epochs, values, color=colour, linewidth=2, label=name)
        axis.set_xlabel("epoch")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.5)
        axis.legend(facecolor=PANEL, edgecolor=GRID)
    right.set_ylim(0, 1.02)
    fig.suptitle(title or "Training", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def save_class_bars(per_class, path, pass_classes=(), title=""):
    """Precision, recall and accuracy per class; FIRE classes below 98 % accuracy are marked red."""
    plt = _pyplot()
    names = list(per_class)
    passed = {str(c).upper().strip() for c in pass_classes}
    fig, axis = plt.subplots(figsize=(13, 5.2))
    width = 0.27
    for offset, (key, label, colour) in enumerate((("precision", "precision", ACCENT), ("recall", "recall", WARN), ("accuracy", "accuracy", GOOD))):
        values = [per_class[n].get(key) or 0.0 for n in names]
        colours = [colour] * len(names)
        if key == "accuracy":
            colours = [BAD if (n.upper().strip() not in passed and (per_class[n].get("accuracy") or 0.0) < FIRE_LIMIT) else GOOD for n in names]
        bars = axis.bar([i + (offset - 1) * width for i in range(len(names))], values, width, color=colours, label=label)
        if key == "accuracy":
            for bar, value in zip(bars, values):
                axis.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value * 100:.0f}", ha="center", va="bottom", fontsize=9, color=TEXT)
    axis.axhline(FIRE_LIMIT, color=BAD, linestyle="--", linewidth=1)
    axis.text(len(names) - 0.5, FIRE_LIMIT + 0.005, "98 % (FIRE classes)", color=BAD, ha="right", fontsize=9)
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels([f"{n}\n({per_class[n]['total']})" + ("" if n.upper().strip() not in passed else "\nPASS") for n in names], fontsize=9)
    axis.set_ylim(0, 1.2)
    axis.grid(True, axis="y", alpha=0.5)
    axis.legend(facecolor=PANEL, edgecolor=GRID, loc="upper left", ncol=3)
    axis.set_title(title or "Per class (red accuracy bar = FIRE class below 98 %)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def save_confusion(matrix, classes, path, title=""):
    """Confusion matrix heatmap: rows = true class, columns = what the model said; numbers = images."""
    plt = _pyplot()
    import numpy as np
    data = np.array(matrix, dtype=float)
    share = data / np.maximum(data.sum(axis=1, keepdims=True), 1.0)
    size = max(6.5, 0.62 * len(classes) + 2.5)
    fig, axis = plt.subplots(figsize=(size + 1.5, size))
    image = axis.imshow(share, cmap="viridis", vmin=0.0, vmax=1.0)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if data[i, j]:
                axis.text(j, i, f"{int(data[i, j])}", ha="center", va="center", fontsize=9, color="#06131A" if share[i, j] > 0.6 else TEXT)
    axis.set_xticks(range(len(classes)))
    axis.set_yticks(range(len(classes)))
    axis.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    axis.set_yticklabels(classes, fontsize=9)
    axis.set_xlabel("model says")
    axis.set_ylabel("label (true class)")
    axis.set_title(title or "Confusion matrix", fontweight="bold")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="share of the true class")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def save_all(model_path, history=None, matrix=None, classes=None, pass_classes=(), prefix="", name=""):
    """Write the report pictures of a model. Returns {kind: path} of what was written (never raises)."""
    written = {}
    paths = report_paths(model_path, prefix)
    label = name or os.path.basename(str(model_path))
    where = "test set" if prefix else "validation images"
    try:
        if history and "curves" in paths:
            written["curves"] = save_curves(history, paths["curves"], f"{label}: loss and accuracy per epoch")
        if matrix and classes:
            written["classes"] = save_class_bars(class_metrics(matrix, classes), paths["classes"], pass_classes,
                                                 f"{label}: precision / recall / accuracy on the {where} (red = FIRE class below 98 %)")
            written["confusion"] = save_confusion(matrix, classes, paths["confusion"], f"{label}: confusion matrix on the {where}")
    except Exception as exc:
        print(f"[REPORT] Could not write the report pictures: {exc}")
    return written
