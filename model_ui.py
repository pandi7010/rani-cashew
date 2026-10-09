"""Ported from the 5-track CASHEW X AI project (external drive, read only); adapted for GSN_V21 where noted.

AI MODEL SELECT page: model cards and the class-change confirmation shown before a model is activated."""
import os
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

import ui_theme as hu
import training

SMALL, SMALL_BOLD, BODY, BODY_BOLD, HEAD_BOLD = ((hu.FONT, hu.F_SMALL), (hu.FONT, hu.F_SMALL, "bold"), (hu.FONT, hu.F_BODY),
                                                 (hu.FONT, hu.F_BODY, "bold"), (hu.FONT, hu.F_HEAD, "bold"))
FIXED_FIRE = "DOUBLE"        # GSN_V21: the CNN class DOUBLE always goes to the reject ejector


def norm(name):
    return str(name).upper().strip()


def is_good_name(name):
    return norm(name).startswith("GOOD")


def read_model_classes(path):
    """Class names stored in a model file ([] when they cannot be read). Slow for YOLO files: call from a thread."""
    try:
        if str(path).lower().endswith(".pth"):
            import torch
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
            return list(checkpoint.get("class_names", [])) if isinstance(checkpoint, dict) else []
        from ultralytics import YOLO
        model = YOLO(str(path))
        if model.task != "detect":
            return [f"NOT A DETECTION MODEL ({model.task})"]
        return [model.names[i] for i in sorted(model.names)]
    except Exception as exc:
        print(f"[MODELS] Could not read the classes of {path}: {exc}")
        return []


def plan_class_changes(new_classes, old_classes, fire_classes, good_classes):
    """One row per class of the new model: (name, status, was, will be).
    Known classes keep their setting; new ones start as FIRE, except names starting with GOOD (PASS)."""
    fire, good = {norm(c) for c in fire_classes}, {norm(c) for c in good_classes}
    old = {norm(c) for c in old_classes}
    rows = []
    for name in new_classes:
        key = norm(name)
        if key == FIXED_FIRE:
            rows.append((key, "fixed rule", "FIRE", "FIRE"))
        elif key in fire:
            rows.append((key, "kept" if key in old else "known", "FIRE", "FIRE"))
        elif key in good:
            rows.append((key, "kept" if key in old else "known", "PASS", "PASS"))
        else:
            # "CASHEW" is the single class of a detection-only YOLO model: it says nothing about the grade.
            rows.append((key, "NEW", "-", "PASS" if (is_good_name(key) or key == "CASHEW") else "FIRE"))
    removed = sorted(old - {norm(c) for c in new_classes})
    return rows, removed


def confirm_activation(app, kind, model_name, new_classes, old_name, old_classes, fire_classes, good_classes, comparison, on_confirm):
    """Popup before a model becomes active. on_confirm({class: 'PASS' | 'FIRE'}) is called only on CONFIRM."""
    rows, removed = plan_class_changes(new_classes, old_classes, fire_classes, good_classes)
    height = min(980, 330 + 38 * len(rows) + (60 if removed else 0) + (60 + 30 * len(comparison["rows"]) if comparison else 40))
    top = tk.Toplevel(app)
    top.title("Activate model")
    top.configure(bg=hu.BG)
    top.transient(app)
    width = 900
    app.update_idletasks()
    top.geometry(f"{width}x{height}+{app.winfo_rootx() + (app.winfo_width() - width) // 2}+{app.winfo_rooty() + max(0, (app.winfo_height() - height) // 2)}")
    head = tk.Frame(top, bg=hu.PANEL)
    head.pack(fill=tk.X)
    tk.Label(head, text=f"ACTIVATE {kind} MODEL - CLASS CHANGES", font=HEAD_BOLD, fg=hu.TEXT, bg=hu.PANEL).pack(side=tk.LEFT, padx=16, pady=12)
    tk.Frame(top, bg=hu.BORDER, height=1).pack(fill=tk.X)
    tk.Label(top, text=f"New model: {model_name}        Now active: {old_name or 'none'}\nCheck every class. Nothing changes until you press CONFIRM.",
             font=SMALL, fg=hu.MUTED, bg=hu.BG, anchor="w", justify="left").pack(fill=tk.X, padx=16, pady=(12, 8))

    grid = tk.Frame(top, bg=hu.BG)
    grid.pack(fill=tk.X, padx=16)
    for column, (title, minsize) in enumerate((("CLASS", 250), ("STATUS", 150), ("WAS", 110), ("WILL BE (click to change)", 260))):
        grid.columnconfigure(column, minsize=minsize)
        tk.Label(grid, text=title, font=(hu.FONT, -11, "bold"), fg=hu.MUTED, bg=hu.BG, anchor="w").grid(row=0, column=column, sticky="w", pady=(0, 4))
    choice = {}
    colours = {"PASS": hu.GOOD, "FIRE": hu.BAD}
    for index, (name, status, was, will) in enumerate(rows, start=1):
        tk.Label(grid, text=name, font=BODY_BOLD, fg=hu.TEXT, bg=hu.BG, anchor="w").grid(row=index, column=0, sticky="w", pady=4)
        tk.Label(grid, text=status, font=SMALL_BOLD, fg=hu.ACCENT if status == "NEW" else hu.MUTED, bg=hu.BG, anchor="w").grid(row=index, column=1, sticky="w")
        tk.Label(grid, text=was, font=SMALL, fg=hu.MUTED, bg=hu.BG, anchor="w").grid(row=index, column=2, sticky="w")
        choice[name] = will
        if status == "fixed rule":
            tk.Label(grid, text="FIRE  (double / touching is always rejected)", font=SMALL_BOLD, fg=hu.BAD, bg=hu.BG).grid(row=index, column=3, sticky="w")
            continue
        selector = hu.Segmented(grid, ("PASS", "FIRE"), will, None, colours)
        selector.command = lambda value, n=name, s=selector: (choice.__setitem__(n, value), s.set(value))
        selector.grid(row=index, column=3, sticky="w")

    if removed:
        box = tk.Frame(top, bg=hu.mix(hu.PANEL, hu.WARN, 0.12), highlightthickness=1, highlightbackground=hu.WARN)
        box.pack(fill=tk.X, padx=16, pady=(12, 0))
        tk.Label(box, text="REMOVED - the new model does not know these classes any more (their PASS / FIRE settings stay stored):\n"
                           + ", ".join(removed), font=SMALL, fg=hu.TEXT, bg=box.cget("bg"), anchor="w", justify="left",
                 wraplength=width - 60).pack(fill=tk.X, padx=12, pady=8)

    # Old against new on the fixed test set (images never used for training).
    if comparison is None:
        text = "Model comparison: not available for this kind of model."
    elif not comparison["rows"]:
        text = ("Model comparison: there is no test set yet. It fills by itself: every 10th image you confirm in the inbox goes to the "
                "test set and is never used for training.")
    else:
        text = f"Model comparison on the test set ({comparison['images']} images never used for training):"
    tk.Label(top, text=text, font=SMALL, fg=hu.MUTED, bg=hu.BG, anchor="w", justify="left", wraplength=width - 40).pack(fill=tk.X, padx=16, pady=(12, 4))
    if comparison and comparison["rows"]:
        tree = ttk.Treeview(top, columns=("class", "n", "old", "new"), show="headings", style="Home.Treeview", height=len(comparison["rows"]))
        for col, title, w in (("class", "CLASS", 240), ("n", "TEST IMAGES", 150), ("old", "ACTIVE MODEL", 220), ("new", "NEW MODEL", 220)):
            tree.heading(col, text=title, anchor="w")
            tree.column(col, width=w, anchor="w")
        tree.tag_configure("better", foreground=hu.GOOD)
        tree.tag_configure("worse", foreground=hu.BAD)
        for name, total, old_acc, new_acc in comparison["rows"]:
            def show(value):
                return "cannot compare (class unknown)" if value is None else f"{value * 100:.0f} %"
            tag = "" if old_acc is None or new_acc is None else ("better" if new_acc > old_acc else ("worse" if new_acc < old_acc else ""))
            tree.insert("", "end", values=(name, f"{total}" + ("  (too few)" if total < 30 else ""), show(old_acc), show(new_acc)), tags=(tag,))
        tree.pack(fill=tk.X, padx=16)

    buttons = tk.Frame(top, bg=hu.BG)
    buttons.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=16)

    def confirm():
        top.destroy()
        on_confirm(dict(choice))
    hu.FlatButton(buttons, "CONFIRM AND ACTIVATE", confirm, kind="primary", height=44, width=240, font=BODY_BOLD).pack(side=tk.RIGHT)
    hu.FlatButton(buttons, "CANCEL", top.destroy, kind="muted", height=44, width=120).pack(side=tk.RIGHT, padx=(0, 8))
    tk.Label(buttons, text="New classes start as FIRE. Classes named GOOD... start as PASS.", font=SMALL, fg=hu.MUTED, bg=hu.BG).pack(side=tk.LEFT)
    top.grab_set()
    return top


class ModelPage:
    """Model cards for YOLO (.pt) and CNN (.pth) files in the model folder."""

    def __init__(self, app, parent, model_root, get_active, activate, is_running, get_class_settings, apply_class_settings, testset_dir):
        self.app, self.model_root = app, str(model_root)
        self.get_active, self.activate, self.is_running = get_active, activate, is_running
        self.get_class_settings, self.apply_class_settings, self.testset_dir = get_class_settings, apply_class_settings, testset_dir
        self._classes = {}              # (path, mtime) -> class list, read in a helper thread
        self._reading = set()
        self._signature = None
        self._buttons = []
        self.columns = {}
        parent.columnconfigure(0, weight=1, uniform="models")
        parent.columnconfigure(1, weight=1, uniform="models")
        parent.rowconfigure(0, weight=1)
        for column, (kind, title, note) in enumerate((("YOLO", "YOLO MODELS", "detection: finds the cashews"),
                                                      ("CNN", "CNN MODELS", "classification: grades every view"))):
            frame = tk.Frame(parent, bg=hu.PANEL, highlightthickness=1, highlightbackground=hu.BORDER)
            frame.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0 if column == 1 else 8), pady=(16, 0))
            head = tk.Frame(frame, bg=hu.PANEL)
            head.pack(fill=tk.X, padx=16, pady=(12, 8))
            tk.Label(head, text=title, font=SMALL_BOLD, fg=hu.MUTED, bg=hu.PANEL).pack(side=tk.LEFT)
            tk.Label(head, text=note, font=SMALL, fg=hu.MUTED, bg=hu.PANEL).pack(side=tk.LEFT, padx=(16, 0))
            hu.FlatButton(head, "RESCAN FOLDER", self.rescan, kind="muted", height=28, pad=12).pack(side=tk.RIGHT)
            body = tk.Frame(frame, bg=hu.PANEL)
            body.pack(fill=tk.BOTH, expand=True, padx=16)
            note_label = tk.Label(frame, text="", font=SMALL, fg=hu.MUTED, bg=hu.PANEL, anchor="w")
            note_label.pack(fill=tk.X, padx=16, pady=(0, 12))
            self.columns[kind] = (body, note_label)

    def rescan(self):
        self._signature = None
        self.refresh()

    def _files(self, kind):
        ext = ".pt" if kind == "YOLO" else ".pth"
        try:
            names = [f for f in os.listdir(self.model_root) if f.lower().endswith(ext)]
        except OSError:
            names = []
        return sorted(names, key=lambda f: -os.path.getmtime(os.path.join(self.model_root, f)))

    def _class_list(self, path):
        key = (path, os.path.getmtime(path))
        if key in self._classes:
            return self._classes[key]
        if key not in self._reading:
            self._reading.add(key)

            def read():
                self._classes[key] = read_model_classes(path)
            threading.Thread(target=read, daemon=True, name="ModelClasses").start()
        return None

    def refresh(self):
        """Rebuild the cards when the files, the active model, the class lists or the machine state changed."""
        running = bool(self.is_running())
        signature = [running]
        data = {}
        for kind in ("YOLO", "CNN"):
            active = self.get_active(kind)
            rows = []
            for name in self._files(kind):
                path = os.path.join(self.model_root, name)
                classes = self._class_list(path)
                rows.append((name, path, os.path.getmtime(path), os.path.getsize(path), classes, name == active))
                signature.append((name, os.path.getmtime(path), None if classes is None else len(classes), name == active))
            data[kind] = rows
        if signature == self._signature:
            return
        self._signature = signature
        self._buttons = []
        for kind, rows in data.items():
            body, note_label = self.columns[kind]
            for child in body.winfo_children():
                child.destroy()
            note_label.config(text="With the machine RUNNING, SET ACTIVE is disabled." if running else
                              "SET ACTIVE shows the class changes first; the model changes only after CONFIRM.", fg=hu.WARN if running else hu.MUTED)
            if not rows:
                tk.Label(body, text=f"No {kind} model files in {self.model_root}", font=SMALL, fg=hu.WARN, bg=hu.PANEL, anchor="w").pack(fill=tk.X, pady=12)
            for name, path, mtime, size, classes, active in rows[:6]:
                self._card(body, kind, name, path, mtime, size, classes, active, running)
            if len(rows) > 6:
                tk.Label(body, text=f"{len(rows) - 6} older model file(s) not shown.", font=SMALL, fg=hu.MUTED, bg=hu.PANEL, anchor="w").pack(fill=tk.X)

    def _card(self, body, kind, name, path, mtime, size, classes, active, running):
        bg = hu.mix(hu.PANEL, hu.GOOD, 0.08) if active else hu.PANEL
        card = tk.Frame(body, bg=bg, highlightthickness=1, highlightbackground=hu.GOOD if active else hu.BORDER)
        card.pack(fill=tk.X, pady=(0, 10))
        top = tk.Frame(card, bg=bg)
        top.pack(fill=tk.X, padx=16, pady=(10, 0))
        tk.Label(top, text=name, font=HEAD_BOLD, fg=hu.TEXT, bg=bg, anchor="w").pack(side=tk.LEFT)
        if active:
            tk.Label(top, text="  ACTIVE  ", font=(hu.FONT, -11, "bold"), fg="#06222A", bg=hu.GOOD).pack(side=tk.RIGHT)
        tk.Label(card, text=f"{datetime.fromtimestamp(mtime):%d.%m.%Y %H:%M}     {size / 1e6:.1f} MB", font=SMALL, fg=hu.MUTED, bg=bg,
                 anchor="w").pack(fill=tk.X, padx=16, pady=(4, 0))
        if classes is None:
            class_text = "Classes: reading ..."
        elif not classes:
            class_text = "Classes: not stored in this file"
        else:
            class_text = f"Classes ({len(classes)}): " + ", ".join(classes)
        tk.Label(card, text=class_text, font=SMALL, fg=hu.TEXT, bg=bg, anchor="w", justify="left", wraplength=780).pack(fill=tk.X, padx=16, pady=(4, 0))
        metrics = training.read_metrics(path)
        if metrics and metrics.get("type") == "cnn":
            accuracy, colour = (f"validation accuracy {metrics['val_accuracy'] * 100:.1f} %  ({metrics['val_images']} images, trained {metrics['trained']})", hu.GOOD)
        elif metrics and metrics.get("map50") is not None:
            accuracy, colour = f"validation mAP50 {metrics['map50']:.3f}  ({metrics['val_images']} images, trained {metrics['trained']})", hu.GOOD
        else:
            accuracy, colour = "accuracy not recorded", "#566178"
        bottom = tk.Frame(card, bg=bg)
        bottom.pack(fill=tk.X, padx=16, pady=(6, 12))
        tk.Label(bottom, text=accuracy, font=SMALL_BOLD, fg=colour, bg=bg, anchor="w").pack(side=tk.LEFT)
        if active:
            tk.Label(bottom, text="in use now", font=SMALL_BOLD, fg=hu.GOOD, bg=bg).pack(side=tk.RIGHT)
        else:
            button = hu.FlatButton(bottom, "SET ACTIVE", lambda: self._set_active(kind, name, path), height=32, width=130)
            button.pack(side=tk.RIGHT)
            button.set_enabled(not running and classes is not None)
        if kind == "CNN":           # GSN_V21: training graphs, and a check on the test set that does not change the model
            import training_report
            graphs = hu.FlatButton(bottom, "GRAPHS", lambda: self._show_reports(name, path), kind="muted", height=32, width=96)
            graphs.pack(side=tk.RIGHT, padx=(0, 8))
            graphs.set_enabled(bool(training_report.existing_reports(path)))
            test = hu.FlatButton(bottom, "TEST SET REPORT", lambda: self._testset_report(name, path), kind="muted", height=32, width=160)
            test.pack(side=tk.RIGHT, padx=(0, 8))
            test.set_enabled(not running)
            if metrics and metrics.get("type") == "cnn":
                low = training_report.fire_warnings(metrics.get("per_class", {}), self.get_class_settings()[1])
                if low:
                    tk.Label(card, text="FIRE classes below 98 % (validation):  " + ",  ".join(f"{n} {a * 100:.0f} %" for n, a in low), font=SMALL_BOLD,
                             fg=hu.BAD, bg=bg, anchor="w", justify="left", wraplength=780).pack(fill=tk.X, padx=16, pady=(0, 10))

    def _show_reports(self, name, path):
        import training_report
        import training_ui
        return training_ui.open_image_viewer(self.app, f"Report: {name}", training_report.existing_reports(path))

    def _testset_report(self, name, path):
        """Confusion matrix and per-class precision / recall / accuracy of this model on the test set (pictures never
        used for training). Writes only <model>_testset_*.png; the model and its result file are not changed."""
        import training_report
        testset = self.testset_dir()
        if self.is_running():
            return None
        if not testset or not any(training.list_class_images(testset).values()):
            messagebox.showinfo("Test set report", "There is no test set yet.")
            return None
        self.app.config(cursor="watch")
        self.app.update_idletasks()
        try:
            result = training.confusion_on_folder(path, testset)
        except Exception as exc:
            messagebox.showerror("Test set report", f"The model could not be run: {exc}")
            return None
        finally:
            self.app.config(cursor="")
        training_report.save_all(path, None, result["matrix"], result["classes"], self.get_class_settings()[1], prefix="testset_", name=name)
        self._signature = None
        return self._show_reports(name, path)

    def _set_active(self, kind, name, path):
        if self.is_running():
            return
        classes = self._class_list(path)
        if classes is None:
            return
        if not classes:
            messagebox.showerror("Activate model", f"{name} has no class names stored, so its classes cannot be checked.\n\nNot activated.")
            return
        if classes[0].startswith("NOT A DETECTION MODEL"):
            messagebox.showerror("Activate model", f"{name} is not a detection model. The grader needs YOLO detection (boxes).\n\nNot activated.")
            return
        old_name = self.get_active(kind)
        old_path = os.path.join(self.model_root, old_name) if old_name else ""
        old_classes = (self._class_list(old_path) or []) if old_path and os.path.exists(old_path) else []
        fire, good = self.get_class_settings()
        comparison = None
        if kind == "CNN":
            comparison = {"rows": [], "images": 0}
            testset = self.testset_dir()
            if testset and any(training.list_class_images(testset).values()):
                self.app.config(cursor="watch")
                self.app.update_idletasks()
                try:
                    new_result = training.evaluate_cnn(path, testset)
                    old_result = training.evaluate_cnn(old_path, testset) if old_path and os.path.exists(old_path) else {}
                finally:
                    self.app.config(cursor="")
                for folder, paths in training.list_class_images(testset).items():
                    if not paths:
                        continue
                    key = norm(folder)

                    def accuracy(result):
                        return result[key][0] / result[key][1] if key in result and result[key][1] else None
                    comparison["rows"].append((key, len(paths), accuracy(old_result), accuracy(new_result)))
                    comparison["images"] += len(paths)

        def confirmed(choice):
            self.apply_class_settings(choice)
            self.activate(kind, name, path)
            self._signature = None
            self.refresh()
        confirm_activation(self.app, kind, name, classes, old_name, old_classes, fire, good, comparison, confirmed)
