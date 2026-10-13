"""Ported from the 5-track CASHEW X AI project (external drive, read only); adapted for GSN_V21 where noted.
GSN_V21: smaller thumbnails and class rows (1920 px screen, 14 classes), a switch for the production capture.

TEACHING CNN page: 1 CAPTURE, 2 LABEL, 3 TRAIN, with the production INBOX as a second image source.

Images are only moved when the operator labels them. File work is in dataset_ops.py, training in
training.py; this module is the screen.
"""
import json
import os
import queue
import time
from collections import OrderedDict
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, messagebox, simpledialog

from PIL import Image, ImageTk

import dataset_ops as ops
import ui_theme as hu
import production_capture
import training

SMALL, SMALL_BOLD, BODY, BODY_BOLD = (hu.FONT, hu.F_SMALL), (hu.FONT, hu.F_SMALL, "bold"), (hu.FONT, hu.F_BODY), (hu.FONT, hu.F_BODY, "bold")
TINY = (hu.FONT, -11)
TARGET = 1000
THUMB_W, THUMB_H, COLS, ROWS = 152, 112, 5, 5
PAGE = COLS * ROWS
SCAN_EVERY_S = 3.0          # GSN_V21: folders are listed in the background, at most this often while the page is shown
THUMB_CACHE = 500           # thumbnails kept in memory


class _Background:
    """GSN_V21: one helper thread for the slow work of the page (listing folders, moving files), in order.
    A job returns a result that the UI thread collects with results(); the UI itself never waits for the disk."""

    def __init__(self):
        self._jobs, self._done = queue.Queue(), queue.Queue()
        self._busy = 0
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True, name="TeachPageWork").start()

    def submit(self, job):
        with self._lock:
            self._busy += 1
        self._jobs.put(job)

    def _run(self):
        while True:
            job = self._jobs.get()
            try:
                self._done.put(job())
            except Exception as exc:
                self._done.put(("error", f"{type(exc).__name__}: {exc}"))
            finally:
                with self._lock:
                    self._busy -= 1

    def busy(self):
        with self._lock:
            return self._busy > 0

    def results(self):
        out = []
        while True:
            try:
                out.append(self._done.get_nowait())
            except queue.Empty:
                return out

    def wait(self, timeout=30.0):
        end = time.time() + timeout
        while self.busy() and time.time() < end:
            time.sleep(0.005)
PALETTE = ["#E5484D", "#F2A93B", "#2FBF71", "#4C8DF6", "#9B6CF6", "#E667AF", "#22B8CF", "#C8D04A", "#FF8A5B"]
CAPTURES, INBOX, CONFUSION = "CAPTURES", "INBOX", "CONFUSION"      # CONFUSION: GSN_V21


class TeachPage:
    def __init__(self, app, parent, golden_dir, model_root):
        self.app, self.golden_dir, self.model_root = app, str(golden_dir), str(model_root)
        self.source = CAPTURES
        self.page = 0
        self.items = []                 # [{path, suggested, confidence, ...}] of the current source
        self.selected = set()
        self.changes = ops.Changes(getattr(app, "log_dir", None))      # multi-step undo + logs/dataset_changes.log (GSN_V21)
        self.suggestions = {}           # path -> (class, confidence) from AI SUGGEST
        self.active_class = ""
        self.classes = []               # class names in card order (shortcut 1-9)
        self._thumbs = []
        self._cells = {}
        self._signature = None
        self._class_signature = None
        self._suggesting = False
        # GSN_V21: nothing on this page reads a folder in the UI thread. Lists and counts come from a helper thread.
        self._work = _Background()
        self._data = {"source": None, "items": [], "counts": {}, "confusion": 0, "captures": 0, "loaded": False}
        self._generation = 0            # raised by every change made here; an older folder listing is thrown away
        self._scan_wanted, self._scan_running, self._scan_time = True, False, 0.0
        self._thumb_cache = OrderedDict()
        self.last_click_ms = 0.0        # how long the last label / delete click kept the UI busy
        self._build(parent)
        app.bind_all("<Key>", self._on_key, add="+")
        self.refresh(force=True)

    # ------------------------------------------------------------------ paths
    def dataset_dir(self):
        return self.app._active_dataset_dir()

    def testset_dir(self):
        return str(Path(self.dataset_dir()).parent / "testset")

    def inbox_dir(self):
        return str(Path(self.dataset_dir()).parent / "inbox")

    # ------------------------------------------------------------------ layout
    def _card(self, parent, title, note=""):
        frame = tk.Frame(parent, bg=hu.PANEL, highlightthickness=1, highlightbackground=hu.BORDER)
        head = tk.Frame(frame, bg=hu.PANEL)
        head.pack(fill=tk.X, padx=16, pady=(12, 8))
        tk.Label(head, text=title, font=SMALL_BOLD, fg=hu.MUTED, bg=hu.PANEL).pack(side=tk.LEFT)
        note_label = tk.Label(head, text=note, font=SMALL, fg=hu.MUTED, bg=hu.PANEL)
        note_label.pack(side=tk.LEFT, padx=(16, 0))
        return frame, head, note_label

    def _entry(self, parent, var, width):
        return tk.Entry(parent, textvariable=var, width=width, font=BODY, bg=hu.FIELD, fg=hu.TEXT, insertbackground=hu.TEXT, relief=tk.FLAT,
                        highlightthickness=1, highlightbackground=hu.BORDER, highlightcolor=hu.ACCENT, justify="center")

    def _build(self, parent):
        app = self.app
        # ---- three steps
        steps = tk.Frame(parent, bg=hu.BG)
        steps.pack(fill=tk.X, pady=(16, 0))
        boxes = []
        for column, (number, title, note) in enumerate((("1", "CAPTURE", "collect cashew images"), ("2", "LABEL", "put every image in its class"),
                                                        ("3", "TRAIN", "teach the model, see how good it is"))):
            steps.columnconfigure(column, weight=1, uniform="step")
            box = tk.Frame(steps, bg=hu.PANEL, highlightthickness=1, highlightbackground=hu.BORDER)
            box.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0 if column == 2 else 8))
            head = tk.Frame(box, bg=hu.PANEL)
            head.pack(fill=tk.X, padx=16, pady=(10, 6))
            tk.Label(head, text=f" {number} ", font=SMALL_BOLD, fg="#06222A", bg=hu.ACCENT).pack(side=tk.LEFT)
            tk.Label(head, text=title, font=BODY_BOLD, fg=hu.TEXT, bg=hu.PANEL).pack(side=tk.LEFT, padx=(10, 0))
            tk.Label(head, text=note, font=SMALL, fg=hu.MUTED, bg=hu.PANEL).pack(side=tk.LEFT, padx=(12, 0))
            row = tk.Frame(box, bg=hu.PANEL)
            row.pack(fill=tk.X, padx=16, pady=(0, 12))
            boxes.append(row)
        capture, label, train = boxes
        app.tcm_auto_cap_btn = hu.FlatButton(capture, "START AUTO CAPTURE", app.toggle_golden, height=36, width=166)
        app.tcm_auto_cap_btn.pack(side=tk.LEFT)
        app.tcm_manual_cap_btn = hu.FlatButton(capture, "SNAPSHOT", app.shared_state.trigger_capture, kind="muted", height=36, width=88)
        app.tcm_manual_cap_btn.pack(side=tk.LEFT, padx=(8, 0))
        tk.Label(capture, text="Limit", font=SMALL, fg=hu.MUTED, bg=hu.PANEL).pack(side=tk.LEFT, padx=(10, 6))
        self._entry(capture, app.golden_count_var, 5).pack(side=tk.LEFT, ipady=4)
        names = {"cropped": "CROP", "full": "FULL", "both": "BOTH"}
        self.mode_names = names
        self.mode_selector = hu.Segmented(capture, tuple(names.values()), names.get(app.capture_mode_var.get(), "CROP"),
                                          lambda value: app.capture_mode_var.set(next(k for k, v in names.items() if v == value)))
        self.mode_selector.pack(side=tk.LEFT, padx=(8, 0))

        hu.FlatButton(label, "AI SUGGEST", self.suggest, height=36, width=110).pack(side=tk.LEFT)
        hu.FlatButton(label, "AUTO-LABEL TOOL", self.launch_auto_labeler, kind="muted", height=36, width=150).pack(side=tk.LEFT, padx=(8, 0))
        app.tcm_verify_btn = hu.FlatButton(label, "CHECK CONFUSION", app._verify_mismatches, kind="muted", height=36, width=138)
        app.tcm_verify_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.lbl_unlabelled = tk.Label(label, text="", font=SMALL_BOLD, fg=hu.WARN, bg=hu.PANEL)
        self.lbl_unlabelled.pack(side=tk.LEFT, padx=(12, 0))

        app.tcm_arch_var = tk.StringVar(value="CNN (MobileNet)")
        app.tcm_epochs = tk.IntVar(value=30)
        app.is_training = False
        tk.Label(train, text="CNN MobileNetV2", font=BODY, fg=hu.TEXT, bg=hu.PANEL).pack(side=tk.LEFT)
        tk.Label(train, text="Epochs", font=SMALL, fg=hu.MUTED, bg=hu.PANEL).pack(side=tk.LEFT, padx=(16, 6))
        self._entry(train, app.tcm_epochs, 5).pack(side=tk.LEFT, ipady=4)
        app.tcm_train_btn = hu.FlatButton(train, "START TRAINING", app._start_training_router, kind="primary", height=36, font=BODY_BOLD)
        app.tcm_train_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(14, 0))

        body = tk.Frame(parent, bg=hu.BG)
        body.pack(fill=tk.BOTH, expand=True, pady=(16, 0))

        # ---- training column (right)
        train_card, _, _ = self._card(body, "TRAINING")
        train_card.config(width=400)
        train_card.pack(side=tk.RIGHT, fill=tk.Y, padx=(16, 0))
        train_card.pack_propagate(False)
        self.lbl_train_state = tk.Label(train_card, text="", font=BODY_BOLD, fg=hu.TEXT, bg=hu.PANEL, anchor="w", justify="left", wraplength=368)
        self.lbl_train_state.pack(fill=tk.X, padx=16)
        self.train_bar = hu.ProgressBar(train_card, hu.PANEL)
        self.train_bar.pack(fill=tk.X, padx=16, pady=(8, 8))
        self.chart_loss = hu.LineChart(train_card, "LOSS", hu.WARN, height=110)
        self.chart_loss.pack(fill=tk.X, padx=16)
        self.chart_acc = hu.LineChart(train_card, "VALIDATION ACCURACY", hu.GOOD, height=110)
        self.chart_acc.pack(fill=tk.X, padx=16, pady=(8, 0))
        tk.Frame(train_card, bg=hu.BORDER, height=1).pack(fill=tk.X, padx=16, pady=(12, 8))
        tk.Label(train_card, text="LAST RESULT   on images not used for training", font=TINY, fg=hu.MUTED, bg=hu.PANEL, anchor="w").pack(fill=tk.X, padx=16)
        self.result_box = tk.Frame(train_card, bg=hu.PANEL)
        self.result_box.pack(fill=tk.X, padx=16, pady=(6, 0))
        self.lbl_result = tk.Label(train_card, text="", font=SMALL, fg=hu.MUTED, bg=hu.PANEL, anchor="w", justify="left", wraplength=368)
        self.lbl_result.pack(fill=tk.X, padx=16, pady=(6, 0))
        hu.FlatButton(train_card, "ACTIVATE A MODEL ...", lambda: app._switch_page("AI Model Select"), height=34).pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=(0, 16))
        extra = tk.Frame(train_card, bg=hu.PANEL)
        extra.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=(0, 8))
        hu.FlatButton(extra, "GRAPHS", self.show_report, kind="muted", height=30, width=96).pack(side=tk.LEFT)
        self.btn_wrong = hu.FlatButton(extra, "STILL WRONG: -", app._review_wrong_images, kind="muted", height=30)
        self.btn_wrong.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        self.lbl_fire = tk.Label(train_card, text="", font=SMALL_BOLD, fg=hu.BAD, bg=hu.PANEL, anchor="w", justify="left", wraplength=340)
        self.lbl_fire.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=(0, 6))
        # GSN_V21: AUTO LEARNING switch (admin password; the learning itself is a later phase)
        auto = tk.Frame(train_card, bg=hu.PANEL, highlightthickness=1, highlightbackground=hu.BORDER)
        auto.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=(0, 10))
        auto_top = tk.Frame(auto, bg=hu.PANEL)
        auto_top.pack(fill=tk.X, padx=10, pady=(8, 0))
        self.lbl_auto = tk.Label(auto_top, text="AUTO LEARNING: OFF", font=BODY_BOLD, fg=hu.MUTED, bg=hu.PANEL)
        self.lbl_auto.pack(side=tk.LEFT)
        self.auto_switch = hu.ToggleSwitch(auto_top, hu.PANEL, on=False, command=app._auto_learning_toggle)
        self.auto_switch.pack(side=tk.RIGHT)
        auto_low = tk.Frame(auto, bg=hu.PANEL)
        auto_low.pack(fill=tk.X, padx=10, pady=(2, 8))
        self.lbl_auto_samples = tk.Label(auto_low, text="Samples: - / 200,000", font=SMALL, fg=hu.MUTED, bg=hu.PANEL)
        self.lbl_auto_samples.pack(side=tk.LEFT)
        hu.FlatButton(auto_low, "TARGET ...", app._auto_learning_target, kind="muted", height=24, pad=8).pack(side=tk.RIGHT)
        self.lbl_auto_note = tk.Label(auto, text="admin password", font=TINY, fg=hu.MUTED, bg=hu.PANEL, anchor="w", justify="left", wraplength=340)
        self.lbl_auto_note.pack(fill=tk.X, padx=10, pady=(0, 6))
        self.lbl_retrain = tk.Label(train_card, text="", font=SMALL_BOLD, fg=hu.WARN, bg=hu.PANEL, anchor="w", justify="left", wraplength=368)
        self.lbl_retrain.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=(0, 10))

        # ---- classes column
        class_card, class_head, _ = self._card(body, "CLASSES", f"target {TARGET}")
        class_card.config(width=400)
        class_card.pack(side=tk.RIGHT, fill=tk.Y, padx=(16, 0))
        class_card.pack_propagate(False)
        tools = tk.Frame(class_card, bg=hu.PANEL)
        tools.pack(fill=tk.X, padx=16, pady=(0, 8))
        for text, command, kind in (("UNDO", self.undo, "muted"), ("ADD", self.add_class, "outline"), ("DELETE", self.delete_class, "danger"),
                                    ("RENAME", self.rename_class, "muted"), ("COLOR", self.change_color, "muted")):
            hu.FlatButton(tools, text, command, kind=kind, height=28, pad=9).pack(side=tk.LEFT, padx=(0, 6))
        self.class_box = tk.Frame(class_card, bg=hu.PANEL)
        self.class_box.pack(fill=tk.X, padx=16)
        capture_row = tk.Frame(class_card, bg=hu.PANEL)
        capture_row.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=(0, 10))
        self.capture_switch = hu.ToggleSwitch(capture_row, hu.PANEL, on=bool(app.shared_state.capture_cfg.get("enabled", True)),
                                              command=app._set_production_capture)
        self.capture_switch.pack(side=tk.RIGHT)
        self.lbl_capture = tk.Label(capture_row, text="", font=TINY, fg=hu.MUTED, bg=hu.PANEL, anchor="w", justify="left", wraplength=290)
        self.lbl_capture.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.warn_box = tk.Frame(class_card, bg=hu.mix(hu.PANEL, hu.WARN, 0.12), highlightthickness=1, highlightbackground=hu.WARN)
        self.lbl_warn = tk.Label(self.warn_box, text="", font=SMALL, fg=hu.TEXT, bg=self.warn_box.cget("bg"), anchor="w", justify="left", wraplength=340)
        self.lbl_warn.pack(fill=tk.X, padx=12, pady=8)

        # ---- gallery (left, takes the rest)
        gallery_card, gallery_head, self.lbl_gallery_note = self._card(body, "IMAGES TO LABEL", "click to select, then press 1-9 or click a class")
        gallery_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.source_selector = hu.Segmented(gallery_head, (CAPTURES, INBOX, CONFUSION), CAPTURES, self.set_source)
        self.source_selector.pack(side=tk.RIGHT)
        self.lbl_selected = tk.Label(gallery_head, text="", font=SMALL_BOLD, fg=hu.ACCENT, bg=hu.PANEL)
        self.lbl_selected.pack(side=tk.RIGHT, padx=(0, 16))
        bottom = tk.Frame(gallery_card, bg=hu.PANEL)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=(4, 14))
        hu.FlatButton(bottom, "< PREV", lambda: self.turn(-1), kind="muted", height=32, width=84).pack(side=tk.LEFT)
        hu.FlatButton(bottom, "NEXT >", lambda: self.turn(1), kind="muted", height=32, width=84).pack(side=tk.LEFT, padx=(8, 0))
        self.lbl_page = tk.Label(bottom, text="", font=SMALL, fg=hu.MUTED, bg=hu.PANEL)
        self.lbl_page.pack(side=tk.LEFT, padx=(14, 0))
        hu.FlatButton(bottom, "DELETE SELECTED", self.delete_selected, kind="danger", height=32, width=160).pack(side=tk.RIGHT)
        hu.FlatButton(bottom, "DESELECT ALL", self.deselect_all, kind="muted", height=32, width=124).pack(side=tk.RIGHT, padx=8)
        hu.FlatButton(bottom, "SELECT PAGE", self.select_page, kind="muted", height=32, width=116).pack(side=tk.RIGHT)
        hu.FlatButton(bottom, "OPEN FOLDER", self.open_source_folder, kind="muted", height=32, width=120).pack(side=tk.RIGHT, padx=8)   # GSN_V21
        self.inbox_bar = tk.Frame(gallery_card, bg=hu.PANEL)
        self.btn_confirm_page = hu.FlatButton(self.inbox_bar, "CONFIRM ALL ON THIS PAGE AS SUGGESTED", self.confirm_page, height=30, width=340)
        self.btn_confirm_page.pack(side=tk.LEFT)
        hu.FlatButton(self.inbox_bar, "CLEAR INBOX ...", self.clear_inbox, kind="danger", height=30, width=150).pack(side=tk.RIGHT)
        self.lbl_inbox = tk.Label(self.inbox_bar, text="", font=SMALL, fg=hu.MUTED, bg=hu.PANEL)
        self.lbl_inbox.pack(side=tk.LEFT, padx=(14, 0))
        self.grid = tk.Frame(gallery_card, bg=hu.PANEL)
        self.grid.pack(fill=tk.BOTH, expand=True, padx=12)
        for column in range(COLS):
            self.grid.columnconfigure(column, weight=1)

    # --------------------------------------------------------------- image source
    def set_source(self, source):
        self.source, self.page = source, 0
        self.selected.clear()
        self.source_selector.set(source)
        self.items = []                 # the list of the other source comes from the helper thread in a moment
        self._data["loaded"] = False
        self._generation += 1
        self.refresh(force=True)

    def _load_items(self, source=None):
        """The image list of a source, read from the disk (helper thread, or the self-test)."""
        source = source or self.source
        if source == INBOX:
            return ops.list_inbox(self.inbox_dir())
        if source == CONFUSION:
            return ops.list_confusion(self.dataset_dir())
        return [{"path": path, "suggested": "", "confidence": 0.0} for path in ops.list_images(self.golden_dir)]

    # ------------------------------------------------------------------ GSN_V21: background listing
    def _request_scan(self):
        """List the shown source and count the classes in the helper thread (never in the UI thread)."""
        if self._scan_running:
            self._scan_wanted = True
            return
        self._scan_running, self._scan_wanted, self._scan_time = True, False, time.time()
        source, generation, dataset_dir, golden_dir = self.source, self._generation, self.dataset_dir(), self.golden_dir

        def job():
            items = self._load_items(source)
            try:
                captures = len(items) if source == CAPTURES else sum(1 for f in os.listdir(golden_dir) if f.lower().endswith(ops.IMAGE_TYPES))
            except OSError:
                captures = 0
            os.makedirs(dataset_dir, exist_ok=True)
            return ("scan", generation, {"source": source, "items": items, "counts": ops.class_counts(dataset_dir),
                                          "confusion": len(items) if source == CONFUSION else ops.confusion_count(dataset_dir),
                                          "captures": captures, "loaded": True})
        self._work.submit(job)
        self._poll_soon()

    def _poll_soon(self):
        if not getattr(self, "_polling", False):
            self._polling = True
            self.app.after(40, self._poll)

    def _poll(self):
        self._polling = False
        try:
            self.refresh()
        except tk.TclError:
            return
        if self._work.busy() or self._scan_running:
            self._poll_soon()

    def _take_results(self):
        """Apply what the helper thread finished (UI thread). True when the shown data changed."""
        changed = False
        for result in self._work.results():
            kind = result[0]
            if kind == "scan":
                self._scan_running = False
                _, generation, data = result
                if generation == self._generation and data["source"] == self.source:
                    self._data, self.items, changed = data, data["items"], True
                else:
                    self._scan_wanted = True        # the folders were changed meanwhile: list again
            elif kind == "moved":
                _, action, moves = result
                self.changes.add(action, moves)
                self._scan_wanted = True
            elif kind == "error":
                print(f"[TEACH] {result[1]}")
                self._scan_running = False
                self._scan_wanted = True
        return changed

    def wait_idle(self, timeout=30.0):
        """Wait until every file move and folder listing is finished and shown (UNDO, tests)."""
        end = time.time() + timeout
        while time.time() < end:
            self._work.wait(timeout)
            self._take_results()
            if self._scan_wanted and not self._scan_running:
                self._request_scan()
                continue
            if not self._work.busy() and not self._scan_running:
                break
        self.refresh(force=True, scan=False)

    def invalidate(self):
        """Something outside this page moved images (review windows, import): list the folders again."""
        self._generation += 1
        self._scan_wanted = True
        self._class_signature = None

    def _thumbnail(self, path):
        """Small picture of an image, from memory when it was shown before."""
        photo = self._thumb_cache.get(path)
        if photo is not None:
            self._thumb_cache.move_to_end(path)
            return photo
        try:
            image = Image.open(path)
            image.draft("RGB", (THUMB_W * 2, THUMB_H * 2))           # JPEG: decoded small, much faster for full frames
            image = image.convert("RGB")
            image.thumbnail((THUMB_W, THUMB_H), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (THUMB_W, THUMB_H), hu.rgb("#0A1626"))
            canvas.paste(image, ((THUMB_W - image.width) // 2, (THUMB_H - image.height) // 2))
            photo = ImageTk.PhotoImage(canvas)
        except Exception:
            return ImageTk.PhotoImage(Image.new("RGB", (THUMB_W, THUMB_H), hu.rgb(hu.FIELD)))
        self._thumb_cache[path] = photo
        while len(self._thumb_cache) > THUMB_CACHE:
            self._thumb_cache.popitem(last=False)
        return photo

    def _change_view(self, removed_paths, class_name=None):
        """Show a label / delete at once: the images leave the grid and the class count rises; the files follow."""
        gone = set(removed_paths)
        self._generation += 1
        self.items = [item for item in self.items if item["path"] not in gone]
        self._data["items"] = self.items
        if self.source == CAPTURES:
            self._data["captures"] = len(self.items)
        if self.source == CONFUSION:
            self._data["confusion"] = len(self.items)
        if class_name:
            counts = dict(self._data["counts"])
            counts[class_name] = counts.get(class_name, 0) + len(gone)
            self._data["counts"] = counts
        for path in gone:
            self._thumb_cache.pop(path, None)
        self.selected.clear()
        self._class_signature = None
        self.refresh(force=True, scan=False)

    def _page_items(self):
        return self.items[self.page * PAGE:(self.page + 1) * PAGE]

    def _suggestion(self, item):
        if item["path"] in self.suggestions:
            return self.suggestions[item["path"]]
        return (item["suggested"], item["confidence"]) if item.get("suggested") else None

    # ------------------------------------------------------------------ drawing
    def refresh(self, force=False, scan=True):
        """Redraw from what is known (called from the UI loop while the page is shown). The folders themselves are
        listed by the helper thread: at most every few seconds, and right after a change."""
        changed = self._take_results()
        stale = time.time() - self._scan_time > SCAN_EVERY_S
        if scan and (self._scan_wanted or force or stale or self._class_signature is None):
            self._request_scan()
        items = self.items
        signature = (self.source, len(items), items[0]["path"] if items else "", self.page, len(self.suggestions), self._data["loaded"])
        if force or changed or signature != self._signature:
            pages = max(1, (len(items) + PAGE - 1) // PAGE)
            self.page = max(0, min(self.page, pages - 1))
            self._signature = (self.source, len(items), items[0]["path"] if items else "", self.page, len(self.suggestions), self._data["loaded"])
            self.selected &= {item["path"] for item in items}
            self._draw_gallery()
        self._draw_classes()
        self._draw_status()

    def _draw_gallery(self):
        for child in self.grid.winfo_children():
            child.destroy()
        self._thumbs, self._cells = [], {}
        page_items = self._page_items()
        if not page_items:
            text = ("loading ..." if not self._data["loaded"] else
                    "The CONFUSION folder is empty." if self.source == CONFUSION else
                    "The inbox is empty. Production capture fills it while the machine runs." if self.source == INBOX
                    else "No unlabelled captures. Use START AUTO CAPTURE or SNAPSHOT.")
            tk.Label(self.grid, text=text, font=SMALL, fg=hu.MUTED, bg=hu.PANEL).grid(row=0, column=0, columnspan=COLS, pady=60)
        for index, item in enumerate(page_items):
            path = item["path"]
            cell = tk.Frame(self.grid, bg=hu.PANEL, highlightthickness=2, highlightbackground=hu.BORDER, cursor="hand2")
            cell.grid(row=index // COLS, column=index % COLS, padx=5, pady=4)
            photo = self._thumbnail(path)
            self._thumbs.append(photo)
            picture = tk.Label(cell, image=photo, bg="#0A1626", bd=0, cursor="hand2")
            picture.pack()
            suggestion = self._suggestion(item)
            if suggestion:
                name, confidence = suggestion
                caption = f" was: {name} " if item.get("former") and item["path"] not in self.suggestions else f" AI: {name} {confidence * 100:.0f}% "
                tk.Label(picture, text=caption, font=(hu.FONT, -11, "bold"), bg="#0A0F18",
                         fg=hu.GOOD if confidence >= 0.85 else hu.WARN).place(x=4, y=THUMB_H - 22)
            name_label = tk.Label(cell, text=os.path.basename(path)[:28], font=(hu.FONT, -10), fg="#566178", bg=hu.PANEL, anchor="w")
            name_label.pack(fill=tk.X)
            for widget in (cell, picture, name_label):
                widget.bind("<Button-1>", lambda e, p=path: self.toggle(p))
                widget.bind("<Double-Button-1>", lambda e, p=path: self.view_large(p))
            self._cells[path] = cell
        self._paint_selection()

    def _paint_selection(self):
        for path, cell in self._cells.items():
            cell.config(highlightbackground=hu.ACCENT if path in self.selected else hu.BORDER)
        self.lbl_selected.config(text=f"{len(self.selected)} selected" if self.selected else "")

    def _draw_classes(self):
        counts, confusion = self._data["counts"], self._data["confusion"]
        if getattr(self, "_colour_cache", None) is None or self._class_signature is None:
            self._colour_cache = self._colours()
        colours = self._colour_cache
        signature = (tuple(counts.items()), self.active_class, tuple(sorted(colours.items())), confusion)
        if signature == self._class_signature:
            return
        self._class_signature = signature
        self.classes = list(counts)
        for child in self.class_box.winfo_children():
            child.destroy()
        for index, (name, count) in enumerate(counts.items()):
            active = name == self.active_class
            bg = hu.mix(hu.PANEL, hu.ACCENT, 0.14) if active else hu.PANEL
            card = tk.Frame(self.class_box, bg=bg, highlightthickness=1, highlightbackground=hu.ACCENT if active else hu.BORDER, cursor="hand2")
            card.pack(fill=tk.X, pady=(0, 3))
            top = tk.Frame(card, bg=bg)
            top.pack(fill=tk.X, padx=8, pady=(3, 0))
            key = tk.Label(top, text=f" {index + 1} " if index < 9 else "   ", font=SMALL_BOLD, fg=hu.TEXT, bg=hu.FIELD)
            key.pack(side=tk.LEFT)
            dot = tk.Label(top, text="●", font=SMALL, fg=colours.get(name, PALETTE[index % len(PALETTE)]), bg=bg)
            dot.pack(side=tk.LEFT, padx=(8, 4))
            label = tk.Label(top, text=name, font=SMALL_BOLD, fg=hu.TEXT, bg=bg, anchor="w")
            label.pack(side=tk.LEFT)
            colour = hu.GOOD if count >= TARGET else (hu.WARN if count >= 100 else hu.BAD)
            folder = tk.Label(top, text="▤", font=SMALL_BOLD, fg=hu.MUTED, bg=bg, cursor="hand2")
            folder.pack(side=tk.RIGHT, padx=(6, 0))
            folder.bind("<Button-1>", lambda e, n=name: (self.open_folder(os.path.join(self.dataset_dir(), n)), "break")[1])
            number = tk.Label(top, text=f"{count} / {TARGET}", font=SMALL_BOLD, fg=colour, bg=bg)
            number.pack(side=tk.RIGHT)
            bar = hu.ProgressBar(card, bg, colour, height=4)
            bar.pack(fill=tk.X, padx=8, pady=(1, 4))
            bar.set(count / TARGET, colour)
            for widget in (card, top, key, dot, label, number, bar):
                widget.bind("<Button-1>", lambda e, n=name: self.class_clicked(n))
        row = tk.Frame(self.class_box, bg=hu.PANEL, highlightthickness=1, highlightbackground=hu.WARN if confusion else hu.BORDER, cursor="hand2")
        row.pack(fill=tk.X, pady=(0, 3))
        text = tk.Label(row, text="   CONFUSION   left out of training", font=SMALL_BOLD, fg=hu.WARN if confusion else hu.MUTED, bg=hu.PANEL, anchor="w")
        text.pack(side=tk.LEFT, pady=3)
        opener = tk.Label(row, text="▤", font=SMALL_BOLD, fg=hu.MUTED, bg=hu.PANEL, cursor="hand2")
        opener.pack(side=tk.RIGHT, padx=(6, 8))
        opener.bind("<Button-1>", lambda e: (self.open_folder(ops.confusion_dir(self.dataset_dir())), "break")[1])
        amount = tk.Label(row, text=str(confusion), font=SMALL_BOLD, fg=hu.WARN if confusion else hu.MUTED, bg=hu.PANEL)
        amount.pack(side=tk.RIGHT)
        for widget in (row, text, amount):
            widget.bind("<Button-1>", lambda e: self.set_source(CONFUSION))
        notes = ops.dataset_warnings(counts)
        if notes:
            self.lbl_warn.config(text="DATASET WARNINGS\n" + "\n".join(notes))
            self.warn_box.pack(fill=tk.X, padx=16, pady=(6, 0))
        else:
            self.warn_box.pack_forget()

    def _draw_status(self):
        app = self.app
        total = len(self.items)
        pages = max(1, (total + PAGE - 1) // PAGE)
        self.lbl_page.config(text=f"page {self.page + 1 if total else 0} of {pages if total else 0}")
        unlabelled = self._data["captures"] if self.source != CAPTURES else total
        self.lbl_unlabelled.config(text=f"{unlabelled} unlabelled" if unlabelled else "all labelled",
                                   fg=hu.WARN if unlabelled else hu.GOOD)
        self.mode_selector.set(self.mode_names.get(app.capture_mode_var.get(), "CROP"))
        if self.source == CONFUSION:
            self.lbl_gallery_note.config(text="left out of training: re-label (1-9 or class) or delete. Double click = large view.")
            self.inbox_bar.pack_forget()
        elif self.source == INBOX:
            self.lbl_gallery_note.config(text="production inbox, most uncertain first: confirm (1-9 or class) or delete")
            self.inbox_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=(4, 0), before=self.grid)
            self.lbl_inbox.config(text=f"{total} images to review")
        else:
            self.lbl_gallery_note.config(text="click to select, then press 1-9 or click a class. Double click = large view.")
            self.inbox_bar.pack_forget()
        if getattr(app, "processing_thread", None) is None:      # the page is built before the threads exist
            return
        status = app.processing_thread.capture.status()
        free = "?" if status["free_gb"] is None else f"{status['free_gb']:.0f} GB"
        inbox = "counting" if status["inbox"] is None else f"{hu.thousands(status['inbox'])} / {hu.thousands(status['inbox_limit'])}"
        state = status["state"]
        self.lbl_capture.config(
            text=(f"PRODUCTION CAPTURE: {state}" + ("  -  review or clear to continue" if state == production_capture.INBOX_FULL else "")
                  + f"\ntoday {hu.thousands(status['today'])} / {hu.thousands(status['daily_limit'])}     inbox {inbox}     free disk {free}"
                  + (f"\nskipped because the disk was busy: {status['skipped_busy']}" if status["skipped_busy"] else "")),
            fg=hu.BAD if state in (production_capture.INBOX_FULL, production_capture.DISK_LOW) else hu.MUTED)
        retrain = ops.retrain_classes(self.dataset_dir())
        self.lbl_retrain.config(text=("RETRAIN RECOMMENDED: " + ", ".join(f"{n} +{c}" for n, c in retrain.items()) + " new images since the last training")
                                if retrain else "")
        enabled, samples, target = app._auto_learning_status()
        self.lbl_auto.config(text=f"AUTO LEARNING: {'ON' if enabled else 'OFF'}", fg=hu.GOOD if enabled else hu.MUTED)
        self.lbl_auto_samples.config(text=f"Samples: {hu.thousands(samples)} / {hu.thousands(target)}", fg=hu.GOOD if samples >= target else hu.MUTED)
        if self.auto_switch.on != enabled:
            self.auto_switch.set(enabled)
        self._draw_training()

    def _draw_training(self):
        status = getattr(self.app, "train_status", None)
        if status and status["kind"] == "CNN" and (status["running"] or status.get("error")):
            history = status["history"]
            if status.get("error"):
                self.lbl_train_state.config(text="TRAINING FAILED: " + status["error"][:160], fg=hu.BAD)
            elif history:
                left = int(status.get("seconds_left", 0))
                self.lbl_train_state.config(text=f"Epoch {history[-1]['epoch']} of {status['epochs']}     about {left // 60} min {left % 60:02d} s left", fg=hu.TEXT)
                self.train_bar.set(history[-1]["epoch"] / max(1, status["epochs"]))
            else:
                self.lbl_train_state.config(text="training is starting ...", fg=hu.TEXT)
            self.chart_loss.show([row["loss"] for row in history])
            self.chart_acc.show([row["val_acc"] for row in history])
        else:
            self.lbl_train_state.config(text="no training running", fg=hu.MUTED)
            self.train_bar.set(0.0)
        result = self._last_result()
        key = (result or {}).get("model")
        if key != getattr(self, "_result_key", "-"):
            self._result_key = key
            for child in self.result_box.winfo_children():
                child.destroy()
            if not result:
                self.lbl_result.config(text="No model has a recorded result yet. Models trained from now on store their accuracy.")
            else:
                self.chart_loss.show([row["loss"] for row in result.get("history", [])])
                self.chart_acc.show([row["val_acc"] for row in result.get("history", [])])
                import training_report
                passed = {str(c).upper().strip() for c in self.app._class_settings()[1]}
                for name, row in result["per_class"].items():
                    accuracy = row["accuracy"]
                    fire = name.upper().strip() not in passed
                    # GSN_V21: a FIRE (defect) class must reach 98 %; a PASS class is judged by the softer steps
                    if fire:
                        colour = hu.GOOD if accuracy is not None and accuracy >= training_report.FIRE_LIMIT else hu.BAD
                    else:
                        colour = hu.BAD if accuracy is None or accuracy < 0.8 else (hu.WARN if accuracy < 0.9 else hu.GOOD)
                    line = tk.Frame(self.result_box, bg=hu.PANEL)
                    line.pack(fill=tk.X, pady=0)
                    tk.Label(line, text="FIRE" if fire else "PASS", font=(hu.FONT, -9, "bold"), fg=hu.BAD if fire else hu.GOOD, bg=hu.PANEL, width=5, anchor="w").pack(side=tk.LEFT)
                    tk.Label(line, text=name, font=SMALL_BOLD, fg=hu.TEXT, bg=hu.PANEL, width=13, anchor="w").pack(side=tk.LEFT)
                    tk.Label(line, text=f"{row['correct']} of {row['total']}", font=SMALL, fg=hu.MUTED, bg=hu.PANEL, width=12, anchor="w").pack(side=tk.LEFT)
                    tk.Label(line, text="-" if accuracy is None else f"{accuracy * 100:.0f} %", font=SMALL_BOLD, fg=colour, bg=hu.PANEL).pack(side=tk.RIGHT)
                self.lbl_result.config(text=f"{result['model']}\noverall {result['val_accuracy'] * 100:.1f} %, trained {result['trained']}. "
                                            "A new model is not active until you activate it.")
                low = training_report.fire_warnings(result["per_class"], passed)
                self.lbl_fire.config(text=("WARNING: FIRE classes below 98 %:  " + ",  ".join(f"{n} {a * 100:.0f} %" for n, a in low)) if low
                                     else "All FIRE classes are at 98 % or above.", fg=hu.BAD if low else hu.GOOD)
                wrong = result.get("wrong_images_total")
                self.btn_wrong.set_text("STILL WRONG: not recorded" if wrong is None else f"STILL WRONG: {wrong} - REVIEW")
                self.btn_wrong.set_enabled(bool(wrong))

    def _last_result(self):
        best = None
        try:
            for name in os.listdir(self.model_root):
                if name.lower().endswith(".pth"):
                    metrics = training.read_metrics(os.path.join(self.model_root, name))
                    if metrics and metrics.get("type") == "cnn" and (best is None or metrics["trained"] > best["trained"]):
                        best = metrics
        except OSError:
            pass
        return best

    # ---------------------------------------------------------------- selection
    def toggle(self, path):
        self.selected.symmetric_difference_update({path})
        self._paint_selection()

    def select_page(self):
        self.selected |= {item["path"] for item in self._page_items()}
        self._paint_selection()

    def deselect_all(self):
        self.selected.clear()
        self._paint_selection()

    def turn(self, step):
        pages = max(1, (len(self.items) + PAGE - 1) // PAGE)
        new_page = max(0, min(pages - 1, self.page + step))
        if new_page != self.page:
            self.page = new_page
            self.refresh(force=True, scan=False)

    def _on_key(self, event):
        """1-9 labels the selected images, arrows / PgUp / PgDn turn the page. Only on this page, not while typing."""
        if not self.grid.winfo_ismapped() or isinstance(event.widget, (tk.Entry, tk.Text)) or event.widget.winfo_toplevel() is not self.app:
            return
        if event.keysym in ("Left", "Prior"):
            self.turn(-1)
        elif event.keysym in ("Right", "Next"):
            self.turn(1)
        elif event.char and event.char in "123456789":
            index = int(event.char) - 1
            if index < len(self.classes) and self.selected:
                self.label_selected(self.classes[index])

    # ------------------------------------------------------------------ labelling
    def class_clicked(self, name):
        self.active_class = name
        if self.selected:
            self.label_selected(name)
        else:
            self._class_signature = None
            self._draw_classes()

    def label_selected(self, class_name):
        """Move the selected images into the class folder. Inbox images count for the test set rule (every 10th).
        GSN_V21: the page changes at once; the files are moved by the helper thread (UNDO waits for it)."""
        started = time.perf_counter()
        paths = [item["path"] for item in self.items if item["path"] in self.selected]
        if not paths:
            return
        source, dataset_dir, testset_dir = self.source, self.dataset_dir(), self.testset_dir()
        action = f"{'RE-LABEL from CONFUSION' if source == CONFUSION else 'LABEL'} as {class_name}"
        self.active_class = class_name
        self._change_view(paths, class_name)

        def job():
            moves = ops.confirm_images(paths, class_name, dataset_dir, testset_dir, use_testset=source == INBOX)
            if source == INBOX:
                self.app.processing_thread.capture.refresh_inbox()
            return ("moved", action, moves)
        self._work.submit(job)
        self._poll_soon()
        self.last_click_ms = 1000.0 * (time.perf_counter() - started)

    def confirm_page(self):
        """Inbox: accept the suggested class for every image on this page whose suggestion is an existing class."""
        known = {name.upper(): name for name in self.classes}
        plan = {}
        for item in self._page_items():
            suggestion = self._suggestion(item)
            if suggestion and suggestion[0].upper() in known:
                plan.setdefault(known[suggestion[0].upper()], []).append(item["path"])
        total = sum(len(paths) for paths in plan.values())
        if not total:
            messagebox.showinfo("Confirm page", "No image on this page has a suggestion for one of the dataset classes.")
            return
        summary = ", ".join(f"{name}: {len(paths)}" for name, paths in plan.items())
        if not messagebox.askyesno("Confirm page", f"Move {total} images into the dataset as suggested?\n\n{summary}\n\n"
                                                   "Look at the page first: the suggestion is the machine's opinion, not a checked label."):
            return
        dataset_dir, testset_dir = self.dataset_dir(), self.testset_dir()
        for name, paths in plan.items():
            self._change_view(paths, name)

        def job():
            moves = []
            for name, paths in plan.items():
                moves += ops.confirm_images(paths, name, dataset_dir, testset_dir, use_testset=True)
            self.app.processing_thread.capture.refresh_inbox()
            return ("moved", "CONFIRM PAGE AS SUGGESTED", moves)
        self._work.submit(job)
        self._poll_soon()

    def undo(self):
        """Multi-step undo: every press takes back one more move or delete (also those made in the review windows)."""
        self.wait_idle()                # moves still on their way are finished first
        result = self.changes.undo()
        if result is None:
            self.lbl_selected.config(text="nothing to undo")
            return
        self.invalidate()
        self.wait_idle()
        self.lbl_selected.config(text=f"UNDO {result[0]}: {result[1]} image(s) put back")

    def delete_selected(self):
        if not self.selected:
            return
        if not messagebox.askyesno("Delete images", f"Delete {len(self.selected)} selected image(s)?\n\nUNDO brings them back (they are kept in the trash folder of the program).",
                                   icon="warning"):
            return
        paths, source, dataset_dir = sorted(self.selected), self.source, self.dataset_dir()
        self._change_view(paths)

        def job():
            moves = ops.trash_images(paths, dataset_dir)
            if source == INBOX:
                self.app.processing_thread.capture.refresh_inbox()
            return ("moved", "DELETE", moves)
        self._work.submit(job)
        self._poll_soon()

    def clear_inbox(self):
        total = len(ops.list_inbox(self.inbox_dir()))
        if not total:
            messagebox.showinfo("Clear inbox", "The inbox is already empty.")
            return
        if not self.app._ask_supervisor_password("Clear inbox"):
            return
        if not messagebox.askyesno("Clear inbox", f"Delete ALL {total} unreviewed images in the inbox?\n\nThis cannot be undone.", icon="warning"):
            return
        removed = ops.clear_inbox(self.inbox_dir())
        self.app.processing_thread.capture.refresh_inbox()
        self.selected.clear()
        self.invalidate()
        self.refresh(force=True)
        messagebox.showinfo("Clear inbox", f"{removed} images deleted.")

    # ------------------------------------------------------------------- classes
    def _colours(self):
        try:
            with open(os.path.join(self.dataset_dir(), "colors.json"), "r") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def add_class(self):
        name = simpledialog.askstring("Add class", "New class name:", parent=self.app)
        if name and name.strip():
            os.makedirs(os.path.join(self.dataset_dir(), name.strip().replace(" ", "_")), exist_ok=True)
            self._after_class_change()

    def delete_class(self):
        name = self.active_class
        if not name:
            messagebox.showinfo("Delete class", "Click a class first.")
            return
        count = ops.class_counts(self.dataset_dir()).get(name, 0)
        if messagebox.askyesno("Delete class", f"Delete class '{name}' and ALL its {count} images?\n\nThis cannot be undone.", icon="warning"):
            import shutil
            try:
                shutil.rmtree(os.path.join(self.dataset_dir(), name))
                self.active_class = ""
                self._after_class_change()
            except OSError as exc:
                messagebox.showerror("Delete class", str(exc))

    def rename_class(self):
        old = self.active_class
        if not old:
            messagebox.showinfo("Rename class", "Click a class first.")
            return
        new = simpledialog.askstring("Rename class", f"Rename '{old}' to:", parent=self.app)
        if new and new.strip():
            new = new.strip().replace(" ", "_")
            try:
                os.rename(os.path.join(self.dataset_dir(), old), os.path.join(self.dataset_dir(), new))
                self.active_class = new
                self._after_class_change()
            except OSError as exc:
                messagebox.showerror("Rename class", str(exc))

    def change_color(self):
        name = self.active_class
        if not name:
            messagebox.showinfo("Class colour", "Click a class first.")
            return
        colour = colorchooser.askcolor(title=f"Colour for {name}")[1]
        if colour:
            colours = self._colours()
            colours[name] = colour
            with open(os.path.join(self.dataset_dir(), "colors.json"), "w") as f:
                json.dump(colours, f)
            self._class_signature = None
            self._draw_classes()

    def _after_class_change(self):
        self.invalidate()
        try:
            import recipe_manager
            recipe_manager.generate_data_yaml(self.app.active_program_name)
        except Exception as exc:
            print(f"[TEACH] data.yaml not updated: {exc}")
        self._class_signature = None
        self._draw_classes()

    # ------------------------------------------------------------------- AI help
    def suggest(self):
        """Run the active CNN on the listed images and show its class and confidence on each thumbnail (nothing is moved)."""
        if self._suggesting:
            return
        if self.app.shared_state.phase == "RUNNING":
            messagebox.showwarning("AI suggest", "Stop the machine first: this uses the same graphics card as the grader.")
            return
        paths = [item["path"] for item in self.items][:2000]
        if not paths:
            return
        model_path = str(self.app._active_cnn_path())
        self._suggesting = True

        def run():
            try:
                result = training.predict_images(model_path, paths)
                self.app.after(0, lambda: (self.suggestions.update(result), self.refresh(force=True, scan=False)))
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self.app.after(0, lambda: messagebox.showerror("AI suggest", f"The active CNN could not be run:\n{message}"))
            finally:
                self._suggesting = False
        threading.Thread(target=run, daemon=True, name="AISuggest").start()

    # ------------------------------------------------------------------- GSN_V21: folders, large view, graphs
    def open_folder(self, folder):
        """Open a folder in the Ubuntu file manager."""
        try:
            os.makedirs(folder, exist_ok=True)
            subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:
            messagebox.showerror("Open folder", f"Could not open {folder}: {exc}")

    def open_source_folder(self):
        self.open_folder({CAPTURES: self.golden_dir, INBOX: self.inbox_dir(), CONFUSION: ops.confusion_dir(self.dataset_dir())}[self.source])

    def view_large(self, path):
        import training_ui
        item = next((i for i in self.items if i["path"] == path), {})
        suggestion = self._suggestion(item) if item else None
        info = os.path.basename(path) + (f"     AI: {suggestion[0]} {suggestion[1] * 100:.0f} %" if suggestion and suggestion[1] else "") \
            + (f"     was labelled {item['former']}" if item.get("former") else "")
        return training_ui.open_image_viewer(self.app, "Image", [("IMAGE", path)], info)

    def show_report(self, model_path=None):
        """Training graphs of a model (curves, per-class bars, confusion matrix) in a large window."""
        import training_report
        import training_ui
        result = self._last_result()
        path = model_path or (os.path.join(self.model_root, result["model"]) if result else "")
        reports = training_report.existing_reports(path) if path else []
        if not reports:
            messagebox.showinfo("Training graphs", "This model has no graphs. Models trained from now on get them; "
                                                   "for an older model use TEST SET REPORT on the AI Model Select page.")
            return None
        return training_ui.open_image_viewer(self.app, f"Report: {os.path.basename(path)}", reports)

    def launch_auto_labeler(self, target_image=None):
        script = Path(__file__).resolve().parent / "ai_auto_labeler.py"
        if not script.exists():
            messagebox.showerror("AI auto-label tool", "ai_auto_labeler.py not found.")
            return
        try:
            subprocess.Popen([sys.executable, str(script)] + ([target_image] if target_image else []))
        except Exception as exc:
            messagebox.showerror("AI auto-label tool", f"Could not start the tool: {exc}")
