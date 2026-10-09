"""Automatic ejector check: every chosen relay gets a few pulses, each confirmed by reading the coil back.

No GUI here (the window is ejector_check_ui.py). All firing goes through RelayOutput.check_pulse(): real PLC,
only while the machine is STOPPED, only relays 1-40; conveyor (M140) and feeder (M141) are never touched.

The software cannot see the air. PLC OK only means: the PLC took the ON and the OFF write and reported the
coil in that state. Whether the right valve blew is marked by the operator: AIR OK / NO AIR / WRONG VALVE.
"""
import csv
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import relay_output as ro

log = logging.getLogger("ejector_check")

DEFAULTS = {"pulses": 5, "pulse_ms": 100, "gap_ms": 400, "relay_gap_ms": 1000, "countdown_s": 3}
LIMITS = {"pulses": (1, 20), "pulse_ms": (10, 1000), "gap_ms": (50, 5000), "relay_gap_ms": (0, 10000), "countdown_s": (0, 30)}

IDLE, COUNTDOWN, RUNNING, DONE, STOPPED, ERROR = "IDLE", "COUNTDOWN", "RUNNING", "DONE", "STOPPED", "ERROR"
PLC_OK, PLC_FAIL, NOT_RUN, UNFINISHED = "PLC OK", "PLC FAIL", "not run", "STOPPED"
AIR_OK, NO_AIR, WRONG_VALVE, NOT_CHECKED = "AIR OK", "NO AIR", "WRONG VALVE", "not checked"
MARKS = (AIR_OK, NO_AIR, WRONG_VALVE)
LOG_FIELDS = ["time", "relay", "m_address", "pulses_sent", "plc", "operator_mark", "error"]


def relays_for(choice: str, single: int = 1) -> List[int]:
    """Relays of a range choice: 'ALL', 'ROI1', 'ROI2' or 'SINGLE'."""
    if choice == "ROI1":
        return list(range(1, 21))
    if choice == "ROI2":
        return list(range(21, 41))
    if choice == "SINGLE":
        if not ro.is_ejector(single):
            raise ValueError("relay must be 1 to 40")
        return [single]
    return list(range(ro.FIRST_EJECTOR, ro.LAST_EJECTOR + 1))


def clean_params(params: Optional[dict]) -> dict:
    """Parameters inside their limits; raises ValueError with a readable text otherwise."""
    out = dict(DEFAULTS)
    for key, value in (params or {}).items():
        if key not in LIMITS:
            continue
        lo, hi = LIMITS[key]
        value = int(float(value))
        if not lo <= value <= hi:
            raise ValueError(f"{key.replace('_', ' ')} must be {lo} to {hi}")
        out[key] = value
    return out


@dataclass
class RelayResult:
    relay: int
    address: Optional[int]
    status: str = NOT_RUN
    pulses_sent: int = 0        # pulses given to the PLC
    pulses_ok: int = 0          # of them: ON and OFF confirmed by read-back
    error: str = ""
    mark: str = NOT_CHECKED


class EjectorCheck:
    def __init__(self, relay_output, log_dir=None, safe_to_run: Optional[Callable[[], str]] = None, wait=None):
        """safe_to_run() returns "" when the check may run, otherwise the reason (machine RUNNING, conveyor ON).
        wait(seconds) -> True when the check was stopped meanwhile (tests give their own)."""
        self.out = relay_output
        self.log_dir = log_dir
        self.safe_to_run = safe_to_run or (lambda: "")
        self._stop = threading.Event()
        self._wait = wait or self._stop.wait
        self._lock = threading.Lock()
        self._thread = None
        self.phase, self.message = IDLE, ""
        self.current_relay, self.current_pulse, self.pulses, self.countdown = None, 0, DEFAULTS["pulses"], 0
        self.results: Dict[int, RelayResult] = {r: RelayResult(r, relay_output.address_of(r)) for r in range(ro.FIRST_EJECTOR, ro.LAST_EJECTOR + 1)}
        self.last_summary = ("NOT RUN", "idle")     # (text for the top bar pill, level: good / warn / bad / idle)

    # ------------------------------------------------------------------ control
    def busy(self) -> bool:
        return self.phase in (COUNTDOWN, RUNNING)

    def start(self, relays: List[int], params: Optional[dict] = None, repeat: int = 1) -> str:
        """Start the check in its own thread. Returns "" or the reason why it did not start."""
        if self.busy():
            return "The check is already running."
        reason = self.safe_to_run()
        if reason:
            return reason
        try:
            params = clean_params(params)
            relays = [r for r in relays if ro.is_ejector(r)]
            if not relays or not 1 <= int(repeat) <= 100:
                raise ValueError("choose at least one relay (1-40) and repeat 1 to 100")
        except ValueError as e:
            return str(e)
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, args=(relays * int(repeat), params), daemon=True, name="EjectorCheck")
        self._thread.start()
        return ""

    def stop(self):
        """STOP / ALL RELAYS OFF NOW: ends the check and switches relays 1-40 off. Always allowed."""
        self._stop.set()
        done, error = self.out.all_ejectors_off()
        with self._lock:
            self._off_note = f"ALL RELAYS OFF: {done} of 40 written OFF." + (f" {error}" if error else "")
            if self.phase in (COUNTDOWN, RUNNING):
                self.phase = STOPPED
            self.message = f"STOPPED. {self._off_note}"
        return done, error

    def mark(self, relay: int, mark: str):
        """Operator's judgement of one relay (AIR OK / NO AIR / WRONG VALVE). Logged as its own line."""
        if relay in self.results and mark in MARKS:
            with self._lock:
                self.results[relay].mark = mark
            self._log(self.results[relay])
            self._summarise()

    # ------------------------------------------------------------------ the check itself (blocking)
    def run(self, relays: List[int], params: Optional[dict] = None):
        p = clean_params(params)
        with self._lock:
            self.phase, self.pulses = COUNTDOWN, p["pulses"]
            self.message = "Stand clear of the nozzles."
            for relay in set(relays):
                self.results[relay] = RelayResult(relay, self.out.address_of(relay))
        try:
            for left in range(p["countdown_s"], 0, -1):
                self.countdown = left
                if self._wait(1.0):
                    return self._end(STOPPED, "Stopped during the countdown.")
            self.countdown = 0
            self.phase = RUNNING
            for n, relay in enumerate(relays):
                reason = self.safe_to_run()
                if reason:
                    return self._abort(f"Stopped before relay {relay}: {reason}")
                if self._stop.is_set():
                    return self._end(STOPPED, f"Stopped before relay {relay}.")
                fault = self._check_relay(relay, p)
                if fault:
                    return self._abort(fault)
                if self._stop.is_set():
                    return self._end(STOPPED, f"Stopped at relay {relay}.")
                if n < len(relays) - 1 and self._wait(p["relay_gap_ms"] / 1000.0):
                    return self._end(STOPPED, f"Stopped after relay {relay}.")
            self._end(DONE, "Check finished.")
        except Exception as e:      # never leave a relay on because of a bug here
            log.exception("Ejector check failed")
            self._abort(f"Internal error: {e}")

    def _check_relay(self, relay: int, p: dict) -> str:
        """All pulses of one relay. Returns "" or the text of a fault that must end the whole check."""
        res = RelayResult(relay, self.out.address_of(relay), status=RUNNING)     # a new run is judged again: mark reset
        with self._lock:
            self.results[relay], self.current_relay, self.current_pulse = res, relay, 0
        fault, attempted = "", 0
        for i in range(1, p["pulses"] + 1):
            self.current_pulse = i
            result, error = self.out.check_pulse(relay, p["pulse_ms"], wait=self._wait_plain)
            attempted = i
            if result == ro.SENT:
                res.pulses_sent += 1
                res.pulses_ok += 1
            else:
                if result in (ro.WRITE_FAIL, ro.READBACK_FAIL):
                    res.pulses_sent += 1
                res.error = res.error or f"pulse {i}: {error or result}"
                if result == ro.PLC_OFFLINE:
                    fault = f"PLC disconnected at relay {relay}, pulse {i}."
                    break
                if result == ro.NOT_STOPPED:
                    fault = f"The machine was started at ejector {relay}, pulse {i}."
                    break
                if result == ro.REFUSED:
                    break
            if self._stop.is_set():
                break
            if i < p["pulses"] and self._wait(p["gap_ms"] / 1000.0):
                break
        if res.error:
            res.status = PLC_FAIL
        elif attempted < p["pulses"]:
            res.status = UNFINISHED
        else:
            res.status = PLC_OK
        self._log(res)
        return fault

    def _wait_plain(self, seconds):
        """Pulse time: not cut short by STOP (the OFF write follows at once anyway)."""
        self._wait(seconds)

    def _abort(self, text):
        done, error = self.out.all_ejectors_off()
        self._end(ERROR, f"{text} Check stopped; {done} of 40 relays written OFF." + (f" {error}" if error else ""))

    def _end(self, phase, text):
        with self._lock:
            note = getattr(self, "_off_note", "") if phase == STOPPED else ""
            self.phase, self.message, self.countdown = phase, f"{text} {note}".strip(), 0
            self._off_note = ""
        self._summarise()
        log.info("Ejector check %s: %s", phase, text)

    # ------------------------------------------------------------------ result for the top bar, log file
    def _summarise(self):
        with self._lock:
            ran = [r for r in self.results.values() if r.status != NOT_RUN]
            failed = [r.relay for r in ran if r.status == PLC_FAIL]
            bad_air = [r.relay for r in ran if r.mark in (NO_AIR, WRONG_VALVE)]
            when = time.strftime("%H:%M")
            short = lambda relays: ",".join(f"R{r}" for r in relays[:4]) + ("..." if len(relays) > 4 else "")
            if self.phase == ERROR:
                self.last_summary = (f"ERROR ({when})", "bad")
            elif failed:
                self.last_summary = (f"FAILED {short(failed)} ({when})", "bad")
            elif bad_air:
                self.last_summary = (f"AIR FAULT {short(bad_air)} ({when})", "bad")
            elif self.phase == STOPPED:
                self.last_summary = (f"STOPPED ({when})", "warn")
            elif ran:
                self.last_summary = (f"{'ALL' if len(ran) == 40 else len(ran)} OK ({when})", "good")

    def _log(self, res: RelayResult):
        if not self.log_dir:
            return
        try:
            os.makedirs(str(self.log_dir), exist_ok=True)
            path = os.path.join(str(self.log_dir), f"ejector_check_{time.strftime('%Y%m%d')}.csv")
            new = not os.path.exists(path)
            with self._lock, open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if new:
                    writer.writerow(LOG_FIELDS)
                writer.writerow([time.strftime("%H:%M:%S"), res.relay, f"M{res.address}" if res.address is not None else "",
                                 res.pulses_sent, res.status, res.mark, res.error])
        except Exception:
            log.exception("Could not write the ejector check log")
