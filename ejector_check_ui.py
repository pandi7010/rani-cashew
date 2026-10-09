"""EJECTOR CHECK window (relays 1-40, a few pulses each, read-back from the PLC, operator marks the air).

Only shows and starts what ejector_check.EjectorCheck does; all firing rules are there and in relay_output.py.
"""
import tkinter as tk
from tkinter import messagebox, ttk

import ejector_check as ec
import relay_output as ro
import ui_theme as ui

STATUS_COLOUR = {ec.PLC_OK: ui.GOOD, ec.PLC_FAIL: ui.BAD, ec.DRY: ui.WARN, ec.UNFINISHED: ui.WARN, ec.RUNNING: ui.ACCENT, ec.NOT_RUN: ui.MUTED}
MARK_COLOUR = {ec.AIR_OK: ui.GOOD, ec.NO_AIR: ui.BAD, ec.WRONG_VALVE: ui.BAD, ec.NOT_CHECKED: ui.MUTED}
RANGES = (("ALL 1-40", "ALL"), ("ROI 1  (1-20)", "ROI1"), ("ROI 2  (21-40)", "ROI2"), ("SINGLE", "SINGLE"))


def open_ejector_check(parent, check: ec.EjectorCheck, relay_output, is_stopped, conveyor_off):
    """Open the window (one at a time). Returns the Toplevel."""
    existing = getattr(parent, "_ejector_check_window", None)
    if existing is not None and existing.winfo_exists():
        existing.lift()
        return existing
    win = EjectorCheckWindow(parent, check, relay_output, is_stopped, conveyor_off)
    parent._ejector_check_window = win
    return win


class EjectorCheckWindow(tk.Toplevel):
    def __init__(self, parent, check, relay_output, is_stopped, conveyor_off):
        super().__init__(parent, bg=ui.BG)
        self.check, self.out, self.is_stopped, self.conveyor_off = check, relay_output, is_stopped, conveyor_off
        self.title("Ejector check 1-40")
        self.geometry("1180x980")
        self.transient(parent)
        self.selected = 1
        self._auto_selected = None
        F = ui.FONT

        # ── head: title, DRY-RUN state ──
        head = tk.Frame(self, bg=ui.PANEL)
        head.pack(fill=tk.X)
        tk.Label(head, text="EJECTOR CHECK", font=(F, ui.F_HEAD, "bold"), fg=ui.TEXT, bg=ui.PANEL).pack(side=tk.LEFT, padx=20, pady=14)
        self.lbl_mode = tk.Label(head, text="", font=(F, ui.F_BODY, "bold"), bg=ui.PANEL, padx=14, pady=6)
        self.lbl_mode.pack(side=tk.RIGHT, padx=20)
        self.pill_machine = ui.Pill(head, "MACHINE", "—", ui.MUTED)
        self.pill_machine.pack(side=tk.RIGHT, padx=4)
        self.pill_conveyor = ui.Pill(head, "CONVEYOR", "—", ui.MUTED)
        self.pill_conveyor.pack(side=tk.RIGHT, padx=4)
        self.pill_plc = ui.Pill(head, "PLC", "—", ui.MUTED)
        self.pill_plc.pack(side=tk.RIGHT, padx=4)
        tk.Frame(self, bg=ui.BORDER, height=1).pack(fill=tk.X)

        # ── settings ──
        box = tk.Frame(self, bg=ui.PANEL, highlightthickness=1, highlightbackground=ui.BORDER)
        box.pack(fill=tk.X, padx=16, pady=(14, 8))
        row = tk.Frame(box, bg=ui.PANEL)
        row.pack(fill=tk.X, padx=16, pady=(12, 4))
        self.vars = {}
        for key, label, unit in (("pulses", "Pulses per relay", ""), ("pulse_ms", "Pulse", "ms"), ("gap_ms", "Gap between pulses", "ms"),
                                 ("relay_gap_ms", "Gap between relays", "ms"), ("countdown_s", "Countdown", "s")):
            tk.Label(row, text=label, font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT, padx=(0, 6))
            var = tk.StringVar(value=str(ec.DEFAULTS[key]))
            tk.Entry(row, textvariable=var, width=5, bg=ui.FIELD, fg=ui.TEXT, insertbackground=ui.TEXT, bd=0, justify="center",
                     font=(F, ui.F_BODY, "bold"), highlightthickness=1, highlightbackground=ui.BORDER).pack(side=tk.LEFT, ipady=3)
            tk.Label(row, text=unit, font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL, width=3, anchor="w").pack(side=tk.LEFT, padx=(4, 14))
            self.vars[key] = var
        row = tk.Frame(box, bg=ui.PANEL)
        row.pack(fill=tk.X, padx=16, pady=(6, 12))
        tk.Label(row, text="Range", font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT, padx=(0, 8))
        self.range_choice = "ALL 1-40"
        self.range_switch = ui.Segmented(row, [label for label, _ in RANGES], self.range_choice, command=self._set_range, font=(F, ui.F_SMALL, "bold"), pad=14)
        self.range_switch.pack(side=tk.LEFT)
        tk.Label(row, text="Single relay", font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT, padx=(20, 6))
        self.single_var = tk.StringVar(value="1")
        tk.Entry(row, textvariable=self.single_var, width=4, bg=ui.FIELD, fg=ui.TEXT, insertbackground=ui.TEXT, bd=0, justify="center",
                 font=(F, ui.F_BODY, "bold"), highlightthickness=1, highlightbackground=ui.BORDER).pack(side=tk.LEFT, ipady=3)
        tk.Label(row, text="Repeat", font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT, padx=(20, 6))
        self.repeat_var = tk.StringVar(value="1")
        tk.Entry(row, textvariable=self.repeat_var, width=4, bg=ui.FIELD, fg=ui.TEXT, insertbackground=ui.TEXT, bd=0, justify="center",
                 font=(F, ui.F_BODY, "bold"), highlightthickness=1, highlightbackground=ui.BORDER).pack(side=tk.LEFT, ipady=3)
        tk.Label(row, text="x", font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT, padx=4)
        self.btn_start = ui.FlatButton(row, "START CHECK", command=self._start, kind="primary", height=40, width=200, font=(F, ui.F_BODY, "bold"))
        self.btn_start.pack(side=tk.RIGHT)

        # ── progress ──
        self.lbl_progress = tk.Label(self, text="Not started.", font=(F, ui.F_HEAD, "bold"), fg=ui.TEXT, bg=ui.BG, anchor="w")
        self.lbl_progress.pack(fill=tk.X, padx=18, pady=(6, 0))
        self.lbl_message = tk.Label(self, text="", font=(F, ui.F_BODY), fg=ui.MUTED, bg=ui.BG, anchor="w", justify="left", wraplength=1140)
        self.lbl_message.pack(fill=tk.X, padx=18, pady=(0, 6))

        # ── bottom first (so it never leaves the window): STOP, marks, hint ──
        self.btn_stop = tk.Label(self, text="■  STOP  /  ALL RELAYS OFF NOW", font=(F, ui.F_HEAD, "bold"), fg="#FFFFFF", bg=ui.BAD, cursor="hand2", pady=14)
        self.btn_stop.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=(8, 14))
        self.btn_stop.bind("<Button-1>", lambda e: self._stop())
        tk.Label(self, text="PLC OK = the PLC took the ON and OFF write and reported the coil in that state. The software cannot see the air: "
                            "mark each relay by eye. Click a row to choose another relay. Every result is written to logs/ejector_check_<date>.csv.",
                 font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.BG, justify="left", anchor="w", wraplength=1140).pack(side=tk.BOTTOM, fill=tk.X, padx=18)
        marks = tk.Frame(self, bg=ui.BG)
        marks.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=8)
        self.lbl_selected = tk.Label(marks, text="", font=(F, ui.F_BODY, "bold"), fg=ui.TEXT, bg=ui.BG)
        self.lbl_selected.pack(side=tk.LEFT, padx=(2, 16))
        for key, mark, kind in (("1", ec.AIR_OK, "outline"), ("2", ec.NO_AIR, "danger"), ("3", ec.WRONG_VALVE, "danger")):
            ui.FlatButton(marks, f"{key}   {mark}", command=lambda m=mark: self._mark(m), kind=kind, height=40, width=190,
                          font=(F, ui.F_BODY, "bold")).pack(side=tk.LEFT, padx=(0, 10))
            self.bind(f"<Key-{key}>", lambda e, m=mark: self._mark(m))
            self.bind(f"<KP_{key}>", lambda e, m=mark: self._mark(m))
        ui.FlatButton(marks, "CLOSE", command=self._close, kind="muted", height=40, width=120).pack(side=tk.RIGHT)

        # ── table of the 40 relays: two blocks of 20 ──
        self.lbl_error = tk.Label(self, text="", font=(F, ui.F_SMALL), fg=ui.BAD, bg=ui.BG, anchor="w", justify="left", wraplength=1140)
        self.lbl_error.pack(side=tk.BOTTOM, fill=tk.X, padx=18)
        style = ttk.Style(self)
        style.configure("Check.Treeview", rowheight=21, font=(F, -13), background=ui.PANEL, fieldbackground=ui.PANEL, foreground=ui.TEXT, borderwidth=0)
        style.configure("Check.Treeview.Heading", font=(F, -12, "bold"), background=ui.BORDER, foreground=ui.ACCENT, borderwidth=0)
        style.layout("Check.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        tables = tk.Frame(self, bg=ui.BG)
        tables.pack(fill=tk.BOTH, expand=True, padx=16)
        self.trees = {}
        for i, (title, relays) in enumerate((("ROI 1   relays 1-20", range(1, 21)), ("ROI 2   relays 21-40", range(21, 41)))):
            col = tk.Frame(tables, bg=ui.BG)
            col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8) if i == 0 else (8, 0))
            tk.Label(col, text=title, font=(F, ui.F_SMALL, "bold"), fg=ui.MUTED, bg=ui.BG, anchor="w").pack(fill=tk.X, pady=(0, 4))
            tree = ttk.Treeview(col, columns=("relay", "addr", "pulses", "plc", "air"), show="headings", height=20, selectmode="browse", style="Check.Treeview")
            for name, text, width, anchor in (("relay", "RELAY", 70, "center"), ("addr", "PLC", 70, "center"), ("pulses", "PULSES", 80, "center"),
                                              ("plc", "PLC RESULT", 110, "w"), ("air", "OPERATOR", 130, "w")):
                tree.heading(name, text=text)
                tree.column(name, width=width, anchor=anchor)
            for tag, colour in (("ok", ui.GOOD), ("bad", ui.BAD), ("warn", ui.WARN), ("now", ui.ACCENT), ("idle", ui.MUTED)):
                tree.tag_configure(tag, foreground=colour)
            for relay in relays:
                tree.insert("", "end", iid=str(relay), values=(relay, "", "", "", ""))
            tree.bind("<<TreeviewSelect>>", lambda e, t=tree: self._on_select(t))
            tree.pack(fill=tk.BOTH, expand=True)
            self.trees[i] = tree

        self.protocol("WM_DELETE_WINDOW", self._close)
        self._tick()

    # ------------------------------------------------------------------ actions
    def _set_range(self, label):
        self.range_choice = label
        self.range_switch.set(label)

    def _start(self):
        if self.check.busy():
            return
        if not self.is_stopped():
            messagebox.showwarning("Ejector check", "The machine is RUNNING. Press STOP first.", parent=self)
            return
        if not self.conveyor_off():
            messagebox.showwarning("Ejector check", "The conveyor is ON. Switch the conveyor OFF first.", parent=self)
            return
        try:
            params = ec.clean_params({key: var.get() for key, var in self.vars.items()})
            relays = ec.relays_for(dict(RANGES)[self.range_choice], int(self.single_var.get() or 0))
            repeat = int(self.repeat_var.get())
        except ValueError as e:
            messagebox.showerror("Ejector check", f"Wrong value: {e}", parent=self)
            return
        seconds = params["countdown_s"] + repeat * len(relays) * (params["pulses"] * (params["pulse_ms"] + params["gap_ms"]) + params["relay_gap_ms"]) / 1000.0
        if self.out.dry_run:
            mode = f"{ec.DRY_RUN_TEXT}. The check only simulates."
        else:
            mode = "DRY-RUN is OFF: the REAL valves will fire."
        what = f"relay {relays[0]}" if len(relays) == 1 else f"relays {relays[0]}-{relays[-1]}"
        if not messagebox.askyesno("Ejector check", f"Make sure nobody is at the nozzles and air is ON.\n\n{mode}\n\n"
                                   f"{what}, {params['pulses']} pulses each" + (f", {repeat} times" if repeat > 1 else "") +
                                   f". About {seconds:.0f} seconds.\n\nStart?", parent=self):
            return
        reason = self.check.start(relays, params, repeat)
        if reason:
            messagebox.showwarning("Ejector check", reason, parent=self)

    def _stop(self):
        done, error = self.check.stop()
        if error:
            messagebox.showerror("ALL RELAYS OFF", error, parent=self)

    def _mark(self, mark):
        result = self.check.results.get(self.selected)
        if result is None or result.status in (ec.NOT_RUN, ec.RUNNING):
            return
        self.check.mark(self.selected, mark)

    def _on_select(self, tree):
        sel = tree.selection()
        if sel:
            self.selected = int(sel[0])
            for other in self.trees.values():
                if other is not tree and other.selection():
                    other.selection_remove(other.selection())

    def _select(self, relay):
        tree = self.trees[0 if relay <= 20 else 1]
        if tree.selection() != (str(relay),):
            tree.selection_set(str(relay))
            tree.see(str(relay))

    def _close(self):
        if self.check.busy():
            if not messagebox.askyesno("Ejector check", "The check is running. Stop it, switch all relays OFF and close?", parent=self):
                return
            self.check.stop()
        self.destroy()

    # ------------------------------------------------------------------ display
    def _tick(self):
        if not self.winfo_exists():
            return
        check, dry = self.check, self.out.dry_run
        self.lbl_mode.config(text=f"DRY-RUN ON:  {ec.DRY_RUN_TEXT.split(': ')[1]}" if dry else "DRY-RUN OFF:  REAL VALVES",
                             fg="#1A1300" if dry else "#FFFFFF", bg=ui.WARN if dry else ui.BAD)
        stopped, belt_off, plc = self.is_stopped(), self.conveyor_off(), self.out.is_connected()
        self.pill_machine.set("STOPPED" if stopped else "RUNNING", ui.GOOD if stopped else ui.BAD)
        self.pill_conveyor.set("OFF" if belt_off else "ON", ui.GOOD if belt_off else ui.BAD)
        self.pill_plc.set("ONLINE" if plc else "OFFLINE", ui.GOOD if plc else (ui.WARN if dry else ui.BAD))
        self.btn_start.set_enabled(not check.busy() and stopped and belt_off)

        relay = check.current_relay
        if check.phase == ec.COUNTDOWN:
            self.lbl_progress.config(text=f"Starting in {check.countdown} s ...   stand clear of the nozzles", fg=ui.WARN)
        elif check.phase == ec.RUNNING and relay:
            address = check.results[relay].address
            self.lbl_progress.config(text=f"Relay {relay}   →   M{address}   •   pulse {check.current_pulse} / {check.pulses}" +
                                          ("     (DRY-RUN, simulated)" if check.dry_run_at_start else ""), fg=ui.ACCENT)
        else:
            colour = {ec.DONE: ui.GOOD, ec.STOPPED: ui.WARN, ec.ERROR: ui.BAD}.get(check.phase, ui.TEXT)
            self.lbl_progress.config(text={ec.IDLE: "Not started.", ec.DONE: "FINISHED", ec.STOPPED: "STOPPED", ec.ERROR: "ERROR"}.get(check.phase, check.phase), fg=colour)
        self.lbl_message.config(text=check.message, fg=ui.BAD if check.phase == ec.ERROR else ui.MUTED)

        for r, res in check.results.items():
            tree = self.trees[0 if r <= 20 else 1]
            plc_text = res.status
            tag = {ec.PLC_OK: "ok", ec.PLC_FAIL: "bad", ec.DRY: "warn", ec.UNFINISHED: "warn", ec.RUNNING: "now"}.get(res.status, "idle")
            if res.mark in (ec.NO_AIR, ec.WRONG_VALVE):
                tag = "bad"
            pulses = "" if res.status == ec.NOT_RUN else (f"{res.pulses_ok} / {check.pulses}" if not check.dry_run_at_start else "simulated")
            values = (r, f"M{res.address}" if res.address is not None else "?", pulses, plc_text, res.mark if res.status != ec.NOT_RUN else "")
            if tuple(str(v) for v in tree.item(str(r), "values")) != tuple(str(v) for v in values) or tree.item(str(r), "tags") != (tag,):
                tree.item(str(r), values=values, tags=(tag,))
        # The relay being checked is selected, so keys 1 / 2 / 3 mark it; a click on another row is respected until the next relay.
        if check.busy() and relay and relay != self._auto_selected:
            self._auto_selected = relay
            self.selected = relay
            self._select(relay)
        sel = check.results.get(self.selected)
        self.lbl_selected.config(text=f"Mark relay {self.selected} (M{sel.address}):" if sel else "")
        failed = [f"Relay {r.relay} (M{r.address}): {r.error}" for r in check.results.values() if r.error]
        self.lbl_error.config(text=("PLC FAIL   " + "     ".join(failed[:4]) + ("   ..." if len(failed) > 4 else "")) if failed else "")
        self.after(100, self._tick)
