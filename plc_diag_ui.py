"""PLC DIAGNOSTICS window: the 5-track "Ejector Hardware Test" for the Inovance PLC.

Connection, counters since program start, the ejector -> M coil table with a pulse button per ejector and SAVE
MAPPING (supervisor password, logs/mapping_changes.log), SEQUENCE ALL, FIRE ALL, ejector check, ALL RELAYS OFF NOW.
Every button writes to the real PLC through relay_output: only while STOPPED, relays 1-40, never M140 / M141.
"""
import tkinter as tk
from tkinter import messagebox

import plc_mapping
import relay_output as ro
import ui_theme as ui


def open_plc_diagnostics(app):
    existing = getattr(app, "_plc_diag_window", None)
    if existing is not None and existing.winfo_exists():
        existing.lift()
        return existing
    app._plc_diag_window = PlcDiagnostics(app)
    return app._plc_diag_window


class PlcDiagnostics(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app, bg=ui.BG)
        self.app, self.out, self.plc = app, app.relay_output, app.controller
        import os
        if os.environ.get("GSN_UI_OFFSCREEN"):
            self.overrideredirect(True)
            self.geometry(f"1240x1000+{app.winfo_screenwidth() + 4200}+0")
        else:
            self.geometry("1240x1000")
        self.title("PLC diagnostics")
        self.transient(app)
        F = ui.FONT
        head = tk.Frame(self, bg=ui.PANEL)
        head.pack(fill=tk.X)
        tk.Label(head, text="PLC DIAGNOSTICS", font=(F, ui.F_HEAD, "bold"), fg=ui.TEXT, bg=ui.PANEL).pack(side=tk.LEFT, padx=20, pady=14)
        self.lbl_mode = tk.Label(head, text="", font=(F, ui.F_BODY, "bold"), bg=ui.PANEL, padx=14, pady=6)
        self.lbl_mode.pack(side=tk.RIGHT, padx=20)
        self.pill_machine = ui.Pill(head, "MACHINE", "—", ui.MUTED)
        self.pill_machine.pack(side=tk.RIGHT, padx=4)
        tk.Frame(self, bg=ui.BORDER, height=1).pack(fill=tk.X)

        def card(title, note=""):
            box = tk.Frame(self, bg=ui.PANEL, highlightthickness=1, highlightbackground=ui.BORDER)
            box.pack(fill=tk.X, padx=16, pady=(12, 0))
            top = tk.Frame(box, bg=ui.PANEL)
            top.pack(fill=tk.X, padx=16, pady=(10, 4))
            tk.Label(top, text=title, font=(F, ui.F_SMALL, "bold"), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT)
            tk.Label(top, text=note, font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT, padx=14)
            body = tk.Frame(box, bg=ui.PANEL)
            body.pack(fill=tk.X, padx=16, pady=(0, 10))
            return body, top

        # ── connection ──
        body, _ = card("CONNECTION")
        self.pill_plc = ui.Pill(body, "PLC", "—", ui.MUTED)
        self.pill_plc.pack(side=tk.LEFT)
        self.lbl_conn = tk.Label(body, text="", font=(F, ui.F_BODY), fg=ui.TEXT, bg=ui.PANEL)
        self.lbl_conn.pack(side=tk.LEFT, padx=16)

        # ── counters ──
        body, top = card("DIAGNOSTIC COUNTERS", "since program start")
        self.tiles = {}
        for i, (key, title) in enumerate((("sent", "PULSES OK"), ("reconnects", "RECONNECTS"), ("write_fail", "WRITE / READ-BACK FAIL"),
                                          ("lost", "LOST (PLC OFFLINE)"), ("blocked", "BLOCKED"), ("cancelled", "CANCELLED BY STOP"))):
            tile = tk.Frame(body, bg=ui.FIELD, highlightthickness=1, highlightbackground=ui.BORDER, width=188, height=62)
            tile.pack(side=tk.LEFT, padx=(0, 8))
            tile.pack_propagate(False)
            tk.Label(tile, text=title, font=(F, -10, "bold"), fg=ui.MUTED, bg=ui.FIELD, anchor="w").pack(fill=tk.X, padx=10, pady=(8, 0))
            value = tk.Label(tile, text="0", font=(F, ui.F_HEAD, "bold"), fg=ui.GOOD, bg=ui.FIELD, anchor="w")
            value.pack(fill=tk.X, padx=10)
            self.tiles[key] = value
        tk.Label(top, text="BLOCKED = refused (not an ejector, or a manual button while RUNNING).  Write timeouts are not counted by the PLC controller: NOT AVAILABLE.",
                 font=(F, -11), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.RIGHT)

        # ── mapping ──
        body, top = card("EJECTORS AND PLC COILS", "ejector -> M address;  M140 conveyor and M141 feeder can never be ejectors")
        tk.Label(top, text="⚡ fires that ejector at once with its pulse from the Pneumatic page", font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.RIGHT)
        grid = tk.Frame(body, bg=ui.PANEL)
        grid.pack(fill=tk.X)
        self.addr_vars, self.addr_entries, self.row_status, self._fired = {}, {}, {}, {}
        self.saved = {r: self.out.address_of(r) for r in plc_mapping.EJECTORS}
        for relay in plc_mapping.EJECTORS:
            block, row = (relay - 1) // 10, (relay - 1) % 10
            cell = tk.Frame(grid, bg=ui.PANEL)
            cell.grid(row=row, column=block, sticky="w", padx=(0, 18), pady=2)
            reject = relay in (1, 21)
            tk.Label(cell, text=f"A{relay:02d}", font=(F, ui.F_BODY, "bold"), fg=ui.WARN if reject else ui.TEXT, bg=ui.PANEL, width=4, anchor="w").pack(side=tk.LEFT)
            tk.Label(cell, text="M", font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT)
            var = tk.StringVar(value="" if self.saved[relay] is None else str(self.saved[relay]))
            entry = tk.Entry(cell, textvariable=var, width=5, bg=ui.FIELD, fg=ui.TEXT, insertbackground=ui.TEXT, bd=0, justify="center",
                             font=(F, ui.F_BODY), highlightthickness=1, highlightbackground=ui.BORDER)
            entry.pack(side=tk.LEFT, padx=(2, 6), ipady=1)
            var.trace_add("write", lambda *_: self._mapping_edited())
            ui.FlatButton(cell, "⚡", command=lambda r=relay: self._test(r), kind="danger", height=24, width=40, font=(F, -12, "bold")).pack(side=tk.LEFT)
            status = tk.Label(cell, text="", font=(F, -11), fg=ui.MUTED, bg=ui.PANEL, width=13, anchor="w")
            status.pack(side=tk.LEFT, padx=(6, 0))
            self.addr_vars[relay], self.addr_entries[relay], self.row_status[relay] = var, entry, status
        low = tk.Frame(body, bg=ui.PANEL)
        low.pack(fill=tk.X, pady=(8, 0))
        self.lbl_mapping = tk.Label(low, text="", font=(F, ui.F_SMALL, "bold"), fg=ui.GOOD, bg=ui.PANEL, anchor="w", justify="left", wraplength=780)
        self.lbl_mapping.pack(side=tk.LEFT)
        self.btn_save = ui.FlatButton(low, "SAVE MAPPING", command=self._save_mapping, kind="primary", height=34, width=160)
        self.btn_save.pack(side=tk.RIGHT)
        ui.FlatButton(low, "RESET TO DEFAULT", command=self._reset_default, kind="muted", height=34, width=170).pack(side=tk.RIGHT, padx=8)

        tk.Label(self, text="⚡ fires the REAL valve of that row through the saved table, only when STOPPED. SAVE MAPPING asks for the supervisor password. "
                            "The software cannot sense air or the valve: confirm each ejector by eye, or use the ejector check.",
                 font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.BG, justify="left", anchor="w", wraplength=1200).pack(fill=tk.X, padx=18, pady=(10, 0))
        self.lbl_result = tk.Label(self, text="", font=(F, ui.F_BODY, "bold"), fg=ui.TEXT, bg=ui.BG, anchor="w")
        self.lbl_result.pack(fill=tk.X, padx=18, pady=(6, 0))

        self.btn_off = tk.Label(self, text="■  ALL RELAYS OFF NOW", font=(F, ui.F_HEAD, "bold"), fg="#FFFFFF", bg=ui.BAD, cursor="hand2", pady=12)
        self.btn_off.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=(8, 14))
        self.btn_off.bind("<Button-1>", lambda e: self._all_off())
        buttons = tk.Frame(self, bg=ui.BG)
        buttons.pack(side=tk.BOTTOM, fill=tk.X, padx=16)
        ui.FlatButton(buttons, "SEQUENCE ALL", command=app._sequence_all, kind="danger", height=40, width=160).pack(side=tk.LEFT)
        ui.FlatButton(buttons, "FIRE ALL", command=app._fire_all, kind="danger", height=40, width=120).pack(side=tk.LEFT, padx=8)
        ui.FlatButton(buttons, "EJECTOR CHECK 1-40", command=app._open_ejector_check, height=40, width=210).pack(side=tk.LEFT)
        ui.FlatButton(buttons, "COPY DIAGNOSTICS", command=self._copy, kind="muted", height=40, width=190).pack(side=tk.LEFT, padx=8)
        ui.FlatButton(buttons, "CLOSE", command=self.destroy, kind="muted", height=40, width=110).pack(side=tk.RIGHT)
        ui.FlatButton(buttons, "VENDOR TOOL ...", command=app._open_plc_vendor_tool, kind="muted", height=40, width=170).pack(side=tk.RIGHT, padx=8)
        self._mapping_edited()
        self._tick()

    # ------------------------------------------------------------------ mapping
    def table(self):
        """The table as typed: {ejector: address or the raw text when it is not a number}."""
        out = {}
        for relay, var in self.addr_vars.items():
            text = var.get().strip()
            out[relay] = int(text) if text.isdigit() else text
        return out

    def _protected(self):
        return [a for a in (self.out.address_of(ro.CONVEYOR_RELAY), self.out.address_of(ro.FEEDER_RELAY)) if a is not None]

    def _mapping_edited(self):
        table = self.table()
        faults = plc_mapping.problems(table, self._protected())
        changed = [r for r in plc_mapping.EJECTORS if table.get(r) != self.saved.get(r)]
        bad = {int(text.split(":")[0].split()[-1]) for text in faults if text.startswith("Ejector")}
        for text in faults:
            if text.startswith("M"):
                bad |= {int(r) for r in text.split("ejectors ")[1].split(" (")[0].split(", ")}
        for relay, entry in self.addr_entries.items():
            entry.config(highlightbackground=ui.BAD if relay in bad else (ui.WARN if relay in changed else ui.BORDER),
                         fg=ui.BAD if relay in bad else (ui.WARN if relay in changed else ui.TEXT))
        if faults:
            self.lbl_mapping.config(text="SAVE IS BLOCKED:  " + ";  ".join(faults[:3]) + (" ..." if len(faults) > 3 else ""), fg=ui.BAD)
        elif changed:
            self.lbl_mapping.config(text=f"Changed, not saved: {', '.join(f'A{r:02d}' for r in changed)}", fg=ui.WARN)
        else:
            self.lbl_mapping.config(text="Table as saved in the PLC settings (inovance_config.json).", fg=ui.GOOD)
        self.btn_save.set_enabled(bool(changed) and not faults and self._stopped())

    def _reset_default(self):
        for relay, address in plc_mapping.default_table().items():
            self.addr_vars[relay].set(str(address))

    def _save_mapping(self):
        table = self.table()
        if plc_mapping.problems(table, self._protected()) or not self._stopped():
            return
        lines = plc_mapping.changes(self.saved, table)
        if not lines or not self.app._ask_supervisor_password("Save mapping"):
            return
        self.plc.update_config(relay_addresses={relay: table[relay] for relay in plc_mapping.EJECTORS})     # the controller's own save
        plc_mapping.log_changes(self.app.log_dir, self.saved, table)
        self.saved = {r: self.out.address_of(r) for r in plc_mapping.EJECTORS}
        self.lbl_result.config(text=f"Mapping saved: {len(lines)} ejector(s) changed. Check each changed ejector with TEST.", fg=ui.GOOD)
        self._mapping_edited()

    # ------------------------------------------------------------------ actions
    def _stopped(self):
        return self.app.shared_state.phase != "RUNNING"

    def _test(self, relay):
        """Lightning button of a row: fires that ejector at once (real PLC), with its pulse of the Pneumatic page."""
        if not self._stopped():
            self.lbl_result.config(text="Refused: the machine is RUNNING. Manual fire only when STOPPED.", fg=ui.WARN)
            return
        if self.table().get(relay) != self.saved.get(relay):
            self.lbl_result.config(text=f"A{relay:02d}: the changed address is not saved yet. The button uses the saved table: SAVE MAPPING first.", fg=ui.WARN)
            return
        pulse = int(self.app.shared_state.get_timing(relay)[1])
        self._fired[relay] = self.app.relay_output._clock()
        result = self.out.manual_pulse(relay, pulse, action="MAPPING WINDOW PULSE")
        self.row_status[relay].config(text="firing ..." if result == ro.SCHEDULED else ro.RESULT_TEXT.get(result, result)[:16],
                                      fg=ui.ACCENT if result == ro.SCHEDULED else ui.BAD)
        self.lbl_result.config(text=f"A{relay:02d} -> M{self.saved.get(relay)}, {pulse} ms:  {ro.RESULT_TEXT.get(result, result)}",
                               fg=ui.ACCENT if result == ro.SCHEDULED else ui.BAD)

    def _all_off(self):
        done, error = self.out.all_ejectors_off()
        self.app._sequence_running = False
        self.lbl_result.config(text=f"ALL RELAYS OFF: {done} of 40 written OFF." + (f"  PLC error: {error}" if error else ""), fg=ui.BAD if error else ui.GOOD)

    def diagnostics_text(self):
        wanted, sent, not_sent = self.out.counts()
        lines = ["PLC DIAGNOSTICS", f"PLC {getattr(self.plc, 'plc_ip', '?')}:{getattr(self.plc, 'plc_port', '?')} unit {getattr(self.plc, 'plc_unit_id', '?')}  "
                 f"{'ONLINE' if self.out.is_connected() else 'OFFLINE'}  type {getattr(self.plc, 'register_type', '?')}",
                 f"machine {'STOPPED' if self._stopped() else 'RUNNING'}",
                 f"pulses sent {sum(sent.values())}  not sent {not_sent}  reconnects {self.app.plc_reconnects}",
                 "table " + " ".join(f"A{r:02d}=M{self.saved.get(r)}" for r in plc_mapping.EJECTORS)]
        return "\n".join(lines)

    def _copy(self):
        self.clipboard_clear()
        self.clipboard_append(self.diagnostics_text())
        self.lbl_result.config(text="Diagnostics copied to the clipboard.", fg=ui.MUTED)

    def _tick(self):
        if not self.winfo_exists():
            return
        online, stopped = self.out.is_connected(), self._stopped()
        self.lbl_mode.config(text="REAL VALVES  -  manual fire only when STOPPED", fg="#FFFFFF", bg=ui.BAD)
        for relay, since in list(self._fired.items()):          # OK / PLC error next to the button when the pulse is over
            last = self.out.last_result.get(relay)
            if last and last[0] >= since:
                ok = last[1] == ro.SENT
                self.row_status[relay].config(text="OK" if ok else "PLC error", fg=ui.GOOD if ok else ui.BAD)
                self.lbl_result.config(text=f"A{relay:02d} -> M{self.saved.get(relay)}:  {ro.RESULT_TEXT.get(last[1], last[1])}" + (f"  ({last[2]})" if last[2] else ""),
                                       fg=ui.GOOD if ok else ui.BAD)
                del self._fired[relay]
        self.pill_machine.set("STOPPED" if stopped else "RUNNING", ui.GOOD if stopped else ui.BAD)
        self.pill_plc.set("ONLINE" if online else "OFFLINE", ui.GOOD if online else ui.BAD)
        self.lbl_conn.config(text=f"Inovance PLC, Modbus TCP  {getattr(self.plc, 'plc_ip', '?')} : {getattr(self.plc, 'plc_port', '?')}    unit {getattr(self.plc, 'plc_unit_id', '?')}"
                                  f"    {getattr(self.plc, 'register_type', '')}")
        wanted, sent, not_sent = self.out.counts()
        values = {"sent": sum(sent.values()), "reconnects": self.app.plc_reconnects,
                  "write_fail": not_sent.get(ro.WRITE_FAIL, 0) + not_sent.get(ro.READBACK_FAIL, 0), "lost": not_sent.get(ro.PLC_OFFLINE, 0),
                  "blocked": not_sent.get(ro.REFUSED, 0) + not_sent.get(ro.NOT_STOPPED, 0), "cancelled": not_sent.get(ro.CANCELLED, 0)}
        for key, value in values.items():
            bad = value > 0 and key in ("reconnects", "write_fail", "lost", "blocked")
            if key == "cancelled":
                bad = False
            self.tiles[key].config(text=str(value), fg=ui.BAD if bad else ui.GOOD)
        self.after(300, self._tick)
