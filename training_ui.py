"""Windows for training: class selection, live progress with result, and the label-mismatch review."""
import os
import shutil
import time
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

import ui_theme as hu

SMALL, SMALL_BOLD, BODY, BODY_BOLD = (hu.FONT, hu.F_SMALL), (hu.FONT, hu.F_SMALL, "bold"), (hu.FONT, hu.F_BODY), (hu.FONT, hu.F_BODY, "bold")
MIN_GOOD_IMAGES = 100          # below this a class is marked as too small to train well


def _window(app, title, width, height, modal=False):
    top = tk.Toplevel(app)
    top.title(title)
    top.configure(bg=hu.BG)
    top.transient(app)
    app.update_idletasks()
    x = app.winfo_rootx() + (app.winfo_width() - width) // 2
    y = app.winfo_rooty() + max(0, (app.winfo_height() - height) // 2)
    top.geometry(f"{width}x{height}+{max(0, x)}+{max(0, y)}")
    head = tk.Frame(top, bg=hu.PANEL)
    head.pack(fill=tk.X)
    tk.Label(head, text=title.upper(), font=(hu.FONT, hu.F_HEAD, "bold"), fg=hu.TEXT, bg=hu.PANEL).pack(side=tk.LEFT, padx=16, pady=12)
    tk.Frame(top, bg=hu.BORDER, height=1).pack(fill=tk.X)
    if modal:
        top.grab_set()
    return top


def choose_classes(app, images_by_class, epochs, on_start):
    """Pick the class folders for a CNN training run. on_start(classes) is called on CONFIRM."""
    top = _window(app, "Train CNN - choose classes", 620, min(900, 250 + 40 * len(images_by_class)), modal=True)
    tk.Label(top, text=f"Epochs: {epochs}.  20 % of every class is held back to measure the accuracy.", font=SMALL, fg=hu.MUTED,
             bg=hu.BG, anchor="w").pack(fill=tk.X, padx=16, pady=(12, 4))
    variables = {}
    for name, paths in images_by_class.items():
        count = len(paths)
        row = tk.Frame(top, bg=hu.BG)
        row.pack(fill=tk.X, padx=16, pady=2)
        var = tk.BooleanVar(value=count >= 3)
        variables[name] = var
        tk.Checkbutton(row, text=f"  {name}", variable=var, font=BODY_BOLD, fg=hu.TEXT, bg=hu.BG, selectcolor=hu.FIELD,
                       activebackground=hu.BG, activeforeground=hu.TEXT, state=tk.NORMAL if count >= 3 else tk.DISABLED,
                       disabledforeground="#566178", cursor="hand2", anchor="w", width=18).pack(side=tk.LEFT)
        colour = hu.GOOD if count >= 1000 else (hu.WARN if count >= MIN_GOOD_IMAGES else hu.BAD)
        note = "" if count >= MIN_GOOD_IMAGES else ("  cannot be trained (needs 3)" if count < 3 else f"  fewer than {MIN_GOOD_IMAGES}: result will be weak")
        tk.Label(row, text=f"{count} images{note}", font=SMALL, fg=colour, bg=hu.BG).pack(side=tk.LEFT)

    def start():
        chosen = [name for name, var in variables.items() if var.get()]
        if len(chosen) < 2:
            messagebox.showerror("Train CNN", "Select at least 2 classes.", parent=top)
            return
        top.destroy()
        on_start(chosen)

    buttons = tk.Frame(top, bg=hu.BG)
    buttons.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=16)
    hu.FlatButton(buttons, "CONFIRM & TRAIN", start, kind="primary", height=40, width=200, font=BODY_BOLD).pack(side=tk.RIGHT)
    hu.FlatButton(buttons, "CANCEL", top.destroy, kind="muted", height=40, width=120).pack(side=tk.RIGHT, padx=(0, 8))


def _duration(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 60} min {seconds % 60:02d} s" if seconds >= 60 else f"{seconds} s"


def open_training_window(app):
    """Live progress of the running training (app.train_status), then its result. Real numbers only."""
    status = app.train_status
    top = _window(app, f"{status['kind']} training", 760, 760)
    lbl_epoch = tk.Label(top, text="starting ...", font=BODY_BOLD, fg=hu.TEXT, bg=hu.BG, anchor="w")
    lbl_epoch.pack(fill=tk.X, padx=16, pady=(12, 0))
    lbl_left = tk.Label(top, text="", font=SMALL, fg=hu.MUTED, bg=hu.BG, anchor="w")
    lbl_left.pack(fill=tk.X, padx=16)
    bar = hu.ProgressBar(top, hu.BG)
    bar.pack(fill=tk.X, padx=16, pady=(8, 12))
    charts = tk.Frame(top, bg=hu.BG)
    charts.pack(fill=tk.X, padx=16)
    charts.columnconfigure(0, weight=1, uniform="chart")
    charts.columnconfigure(1, weight=1, uniform="chart")
    loss_key, score_key = ("loss", "val_acc") if status["kind"] == "CNN" else ("box_loss", "map50")
    chart_loss = hu.LineChart(charts, "LOSS" if status["kind"] == "CNN" else "BOX LOSS", hu.WARN)
    chart_loss.grid(row=0, column=0, sticky="ew", padx=(0, 8))
    chart_score = hu.LineChart(charts, "VALIDATION ACCURACY" if status["kind"] == "CNN" else "mAP50 (validation)", hu.GOOD)
    chart_score.grid(row=0, column=1, sticky="ew", padx=(8, 0))
    lbl_now = tk.Label(top, text="", font=(hu.FONT, hu.F_HEAD, "bold"), fg=hu.TEXT, bg=hu.BG, anchor="w")
    lbl_now.pack(fill=tk.X, padx=16, pady=(8, 0))

    result_box = tk.Frame(top, bg=hu.PANEL, highlightthickness=1, highlightbackground=hu.BORDER)
    result_box.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)
    lbl_result = tk.Label(result_box, text="The per-class result appears here when the training has finished.", font=SMALL, fg=hu.MUTED,
                          bg=hu.PANEL, anchor="w", justify="left", wraplength=700)
    lbl_result.pack(fill=tk.X, padx=16, pady=(12, 6))
    tree = ttk.Treeview(result_box, columns=("class", "images", "score"), show="headings", style="Home.Treeview", height=8)
    for col, title, width in (("class", "CLASS", 220), ("images", "VALIDATION IMAGES", 200), ("score", "ACCURACY", 200)):
        tree.heading(col, text=title, anchor="w")
        tree.column(col, width=width, anchor="w")
    tree.tag_configure("good", foreground=hu.GOOD)
    tree.tag_configure("warn", foreground=hu.WARN)
    tree.tag_configure("bad", foreground=hu.BAD)

    buttons = tk.Frame(top, bg=hu.BG)
    buttons.pack(fill=tk.X, padx=16, pady=(0, 16))

    def stop():
        status["stop"] = True
        btn_stop.set_text("STOPPING AFTER THIS EPOCH")
        btn_stop.set_enabled(False)
    btn_stop = hu.FlatButton(buttons, "STOP TRAINING", stop, kind="danger", height=36, width=260)
    btn_stop.pack(side=tk.LEFT)
    hu.FlatButton(buttons, "CLOSE", top.destroy, kind="muted", height=36, width=120).pack(side=tk.RIGHT)
    shown = {"n": -1, "done": False}

    def refresh():
        if not top.winfo_exists():
            return
        history = status["history"]
        if len(history) != shown["n"]:
            shown["n"] = len(history)
            if history:
                last = history[-1]
                lbl_epoch.config(text=f"Epoch {last['epoch']} of {status['epochs']}")
                bar.set(last["epoch"] / max(1, status["epochs"]))
                chart_loss.show([row[loss_key] for row in history])
                chart_score.show([row[score_key] for row in history])
                lbl_now.config(text=f"loss {last[loss_key]:.3f}      " + (f"accuracy {last[score_key] * 100:.1f} %" if status["kind"] == "CNN"
                                                                             else f"mAP50 {last[score_key]:.3f}"))
        if status["running"]:
            lbl_left.config(text=f"about {_duration(status.get('seconds_left', 0))} left" if history else
                            f"preparing ... {_duration(time.time() - status['started'])}")
        elif not shown["done"]:
            shown["done"] = True
            btn_stop.set_enabled(False)
            if status.get("error"):
                lbl_epoch.config(text="TRAINING FAILED", fg=hu.BAD)
                lbl_left.config(text="")
                lbl_result.config(text=status["error"], fg=hu.BAD)
            else:
                result = status["result"]
                bar.set(1.0)
                lbl_epoch.config(text="TRAINING FINISHED" + ("  (stopped early)" if status.get("stop") else ""), fg=hu.GOOD)
                lbl_left.config(text=f"took {_duration(time.time() - status['started'])}")
                if result["type"] == "cnn":
                    summary = (f"Model saved: {result['model']}\nOverall validation accuracy {result['val_accuracy'] * 100:.1f} %  "
                               f"(best epoch {result['best_epoch']}, {result['val_images']} validation images, {result['train_images']} training images).")
                    for name, row in result["per_class"].items():
                        accuracy = row["accuracy"]
                        tag = "bad" if accuracy is None or accuracy < 0.8 else ("warn" if accuracy < 0.9 else "good")
                        tree.insert("", "end", values=(name, f"{row['correct']} of {row['total']} correct",
                                                       "-" if accuracy is None else f"{accuracy * 100:.0f} %"), tags=(tag,))
                else:
                    tree.heading("images", text="BOXES IN DATASET")
                    tree.heading("score", text="mAP50")
                    summary = (f"Model saved: {result['model']}\nValidation mAP50 {result['map50']:.3f}  "
                               f"({result['val_images']} validation images, {result['train_images']} training images).")
                    for name in result["classes"]:
                        row = result["per_class"].get(name)
                        tree.insert("", "end", values=(name, result["boxes_per_class"].get(name, 0), "-" if not row else f"{row['map50']:.3f}"),
                                    tags=("warn" if not row else ("good" if row["map50"] >= 0.8 else "bad"),))
                summary += "\nThe model is NOT active yet: activate it on the AI Model Select page."
                small = [n for n, c in result.get("images", {}).items() if c < MIN_GOOD_IMAGES] if result["type"] == "cnn" else []
                if small:
                    summary += f"\nWarning: fewer than {MIN_GOOD_IMAGES} images in {', '.join(small)}; these numbers are not reliable."
                lbl_result.config(text=summary, fg=hu.TEXT)
                tree.pack(fill=tk.BOTH, expand=True, padx=1, pady=(0, 1))
        top.after(300, refresh)

    refresh()
    return top


def open_mismatch_window(app, result, model_name, class_folders, on_changed=None):
    """Images where the folder label and the CNN prediction disagree, for review one by one."""
    mismatches = list(result["mismatches"])
    top = _window(app, "Verify label mismatches", 1100, 760)
    skipped = ", ".join(f"{name} ({count})" for name, count in result["skipped"].items())
    intro = (f"Model: {model_name}.   Checked {result['checked']} images, {len(mismatches)} disagree with their folder label."
             + (f"\nNot checked, because the model does not know these classes: {skipped}." if skipped else "")
             + "\nA disagreement is a hint, not proof: the label can be wrong, or the model can be wrong. Look at each image.")
    tk.Label(top, text=intro, font=SMALL, fg=hu.MUTED, bg=hu.BG, anchor="w", justify="left", wraplength=1060).pack(fill=tk.X, padx=16, pady=(12, 8))
    body = tk.Frame(top, bg=hu.BG)
    body.pack(fill=tk.BOTH, expand=True, padx=16)
    tree = ttk.Treeview(body, columns=("file", "label", "predicted", "conf"), show="headings", style="Home.Treeview")
    for col, title, width in (("file", "IMAGE", 260), ("label", "LABEL (FOLDER)", 150), ("predicted", "MODEL SAYS", 150), ("conf", "CONFIDENCE", 110)):
        tree.heading(col, text=title, anchor="w")
        tree.column(col, width=width, anchor="w")
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    side = tk.Frame(body, bg=hu.PANEL, highlightthickness=1, highlightbackground=hu.BORDER, width=380)
    side.pack(side=tk.LEFT, fill=tk.Y, padx=(16, 0))
    side.pack_propagate(False)
    picture = tk.Label(side, bg="#0A1626", bd=0, text="select an image", fg=hu.MUTED, font=SMALL)
    picture.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
    lbl_info = tk.Label(side, text="", font=BODY_BOLD, fg=hu.TEXT, bg=hu.PANEL, wraplength=350, justify="left", anchor="w")
    lbl_info.pack(fill=tk.X, padx=12)

    for index, item in enumerate(mismatches):
        tree.insert("", "end", iid=str(index), values=(os.path.basename(item["path"]), item["label"], item["predicted"], f"{item['confidence'] * 100:.0f} %"))

    def current():
        selection = tree.selection()
        return (int(selection[0]), mismatches[int(selection[0])]) if selection else (None, None)

    def show(event=None):
        _, item = current()
        if not item:
            return
        try:
            image = Image.open(item["path"]).convert("RGB")
            image = Image.eval(image, lambda v: min(255, int(v * 2.0)))        # the crops are dark; brighter for the eye only
            image.thumbnail((350, 330), Image.Resampling.LANCZOS)
            picture.image = ImageTk.PhotoImage(image)
            picture.config(image=picture.image, text="")
        except Exception as exc:
            picture.config(image="", text=f"cannot show image: {exc}")
        lbl_info.config(text=f"Label: {item['label']}     Model: {item['predicted']} ({item['confidence'] * 100:.0f} %)")
    tree.bind("<<TreeviewSelect>>", show)

    def remove_row(index):
        tree.delete(str(index))
        picture.config(image="", text="select an image")
        lbl_info.config(text="")
        if on_changed:
            on_changed()

    def move():
        index, item = current()
        if not item:
            return
        target = class_folders.get(item["predicted"].upper().strip())
        if not target:
            messagebox.showerror("Move", f"There is no class folder for {item['predicted']}.", parent=top)
            return
        try:
            shutil.move(item["path"], os.path.join(target, os.path.basename(item["path"])))
        except Exception as exc:
            messagebox.showerror("Move", f"Could not move the image: {exc}", parent=top)
            return
        remove_row(index)

    def delete():
        index, item = current()
        if not item or not messagebox.askyesno("Delete image", f"Delete {os.path.basename(item['path'])} from the dataset?\nThis cannot be undone.", parent=top):
            return
        try:
            os.remove(item["path"])
        except Exception as exc:
            messagebox.showerror("Delete", f"Could not delete the image: {exc}", parent=top)
            return
        remove_row(index)

    def keep():
        index, item = current()
        if item:
            tree.delete(str(index))                 # the label is right: only taken off this list

    actions = tk.Frame(side, bg=hu.PANEL)
    actions.pack(fill=tk.X, padx=12, pady=12)
    hu.FlatButton(actions, "LABEL IS RIGHT - KEEP", keep, kind="muted", height=36).pack(fill=tk.X)
    hu.FlatButton(actions, "MOVE TO THE MODEL'S CLASS", move, height=36).pack(fill=tk.X, pady=(8, 0))
    hu.FlatButton(actions, "DELETE IMAGE", delete, kind="danger", height=36).pack(fill=tk.X, pady=(8, 0))
    hu.FlatButton(top, "CLOSE", top.destroy, kind="muted", height=36, width=120).pack(side=tk.RIGHT, padx=16, pady=16)
    if mismatches:
        tree.selection_set("0")
    return top
