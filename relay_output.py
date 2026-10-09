"""The only door through which the grader drives ejectors.

Rules enforced here (innovance_plc.py is left untouched):
  * Only ejector relays 1-40 can be driven. Relay 41 (conveyor, M140), 42 (feeder, M141) and anything above
    are refused, and so is an ejector whose address is M140 / M141. The PLC controller's fire_all() /
    sequence_all() / all_off() are never called (they touch 48 coils, conveyor and feeder included).
  * START / STOP is the only control (owner, 2026-10-10; there is no DRY-RUN and no MASTER EJECTOR any more):
        RUNNING  grading fires the valves; manual buttons are refused
        STOPPED  nothing fires from grading (pulses still waiting are cancelled); manual buttons work
  * Every pulse is written the way the old diagnostics tool does it, which is what fires on this PLC: the coil
    is written ON, written ON again while the pulse lasts, then written OFF.
  * Every manual fire is logged in logs/manual_fire_YYYYMMDD.csv.
"""
import csv
import heapq
import itertools
import logging
import os
import threading
import time
from collections import deque

log = logging.getLogger("relay")

FIRST_EJECTOR, LAST_EJECTOR = 1, 40
RELAYS_PER_ROI = 20
REJECT_RELAY = {1: 1, 2: 21}        # ROI -> reject relay (double, undefined, REJECTION, classes without an ejector)
CONVEYOR_RELAY, FEEDER_RELAY = 41, 42               # M140, M141: never driven from here

# results
SENT, PLC_OFFLINE, REFUSED, WRITE_FAIL, READBACK_FAIL = "SENT", "PLC_OFFLINE", "REFUSED", "WRITE_FAIL", "READBACK_FAIL"
NOT_RUNNING, NOT_STOPPED, CANCELLED, SCHEDULED = "NOT_RUNNING", "NOT_STOPPED", "CANCELLED", "SCHEDULED"
RESULT_TEXT = {SENT: "OK", SCHEDULED: "firing", PLC_OFFLINE: "PLC error: not connected", REFUSED: "refused (not an ejector)",
               WRITE_FAIL: "PLC error: write failed", READBACK_FAIL: "PLC error: read-back wrong", NOT_RUNNING: "not fired: machine STOPPED",
               NOT_STOPPED: "refused: machine is RUNNING", CANCELLED: "cancelled (STOP)"}
MANUAL_LOG_FIELDS = ["time", "action", "relay", "m_address", "pulse_ms", "result", "sent_to_plc", "plc_connected", "error"]
REFRESH_S = 0.05        # the coil is written ON again this often while the pulse lasts


def is_ejector(relay) -> bool:
    return isinstance(relay, int) and not isinstance(relay, bool) and FIRST_EJECTOR <= relay <= LAST_EJECTOR


def roi_of_relay(relay) -> int:
    return 1 if relay <= RELAYS_PER_ROI else 2


class RelayOutput:
    def __init__(self, plc, is_running=None, clock=time.time, start_thread=True):
        """is_running(): True while the machine is RUNNING (set by the app). clock / start_thread are for tests,
        which call step() themselves."""
        self._plc = plc
        self.is_running = is_running or (lambda: False)
        self._clock = clock
        self._lock = threading.RLock()
        self._queue = []                    # heap of (start time, n, pulse dict): pulses waiting for their time
        self._active = {}                   # relay -> pulse dict of the pulse that is ON now
        self._count = itertools.count()
        self.wanted = {r: 0 for r in range(FIRST_EJECTOR, LAST_EJECTOR + 1)}   # pulses asked for this ejector
        self.sent = {r: 0 for r in range(FIRST_EJECTOR, LAST_EJECTOR + 1)}     # pulses written to the PLC without an error
        self.not_sent = {PLC_OFFLINE: 0, REFUSED: 0, WRITE_FAIL: 0, READBACK_FAIL: 0, NOT_RUNNING: 0, NOT_STOPPED: 0, CANCELLED: 0}
        self.events = deque(maxlen=500)     # (time, relay, delay_ms, pulse_ms, result, source), newest first
        self.last_result = {}               # relay -> (time, result, error text)
        self.manual_log_dir = None          # folder for manual_fire_YYYYMMDD.csv (None = no file)
        self._stop = threading.Event()
        self._thread = None
        if start_thread:
            self._thread = threading.Thread(target=self._run, daemon=True, name="RelayOutput")
            self._thread.start()

    # ------------------------------------------------------------------ status
    def is_connected(self) -> bool:
        try:
            return bool(self._plc.is_connected())
        except Exception:
            return False

    def address_of(self, relay):
        """Modbus address of a relay as the PLC controller is configured, or None if it cannot tell."""
        getter = getattr(self._plc, "get_relay_address", None)
        try:
            return int(getter(relay)) if getter else None
        except Exception:
            return None

    def describe(self, relay) -> str:
        address = self.address_of(relay)
        return f"Ejector {relay}" + (f" -> M{address} (Modbus address {address})" if address is not None else "")

    def counts(self):
        with self._lock:
            return dict(self.wanted), dict(self.sent), dict(self.not_sent)

    def reset_counts(self):
        with self._lock:
            for table in (self.wanted, self.sent, self.not_sent):
                for key in table:
                    table[key] = 0

    def recent_events(self):
        with self._lock:
            return list(self.events)

    def pending(self):
        """Grading pulses whose valve has not opened yet: [(start time, relay, source)]."""
        with self._lock:
            return sorted((start, p["relay"], p["source"]) for start, _, p in self._queue)

    def busy(self) -> bool:
        with self._lock:
            return bool(self._queue or self._active)

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------------ what may be driven
    def _protected_addresses(self):
        """PLC addresses of conveyor and feeder as configured, plus M140 / M141: an ejector must never have them."""
        out = {140, 141}
        for relay in (CONVEYOR_RELAY, FEEDER_RELAY):
            address = self.address_of(relay)
            if address is not None:
                out.add(address)
        return out

    def _refusal(self, relay):
        """(result, error) when this relay may not be driven at all, else None."""
        if not is_ejector(relay):
            log.error("Refused to drive relay %r: only relays %d-%d are ejectors.", relay, FIRST_EJECTOR, LAST_EJECTOR)
            return REFUSED, "not an ejector relay (only 1-40)"
        if self.address_of(relay) in self._protected_addresses():
            log.error("Refused to drive ejector %s: it is mapped to the conveyor / feeder address.", relay)
            return REFUSED, f"ejector {relay} is mapped to the conveyor / feeder address M{self.address_of(relay)}"
        if not self.is_connected():
            return PLC_OFFLINE, "PLC not connected"
        return None

    def _write(self, relay, state) -> bool:
        """One coil write, at once and waited for (the PLC controller's own single-relay write)."""
        try:
            return bool(self._plc._write_relay(relay, state))
        except Exception:
            log.exception("PLC write of ejector %s failed", relay)
            return False

    # ------------------------------------------------------------------ firing
    def _record(self, relay, delay_ms, pulse_ms, result, source, error=""):
        now = self._clock()
        with self._lock:
            if result == SENT and is_ejector(relay):
                self.sent[relay] += 1
            elif result in self.not_sent:
                self.not_sent[result] += 1
            self.events.appendleft((now, relay, delay_ms, pulse_ms, result, source))
            self.last_result[relay] = (now, result, error)
        if result == SENT:
            log.info("[FIRE] ejector %s for %.0f ms: OK (%s)", relay, pulse_ms, source)
        elif result not in (CANCELLED, NOT_RUNNING, NOT_STOPPED):
            log.warning("[FIRE] ejector %s NOT fired: %s %s (%s)", relay, result, error, source)

    def _schedule(self, relay, delay_ms, pulse_ms, source, manual, action, on_done):
        refusal = self._refusal(relay)
        if is_ejector(relay):
            with self._lock:
                self.wanted[relay] += 1
        if refusal:
            self._record(relay, delay_ms, pulse_ms, refusal[0], source, refusal[1])
            if manual:
                self.log_manual(action, relay, pulse_ms, refusal[0], refusal[1])
            if on_done:
                on_done(relay, refusal[0], refusal[1])
            return refusal[0]
        pulse = {"relay": relay, "delay_ms": max(0.0, float(delay_ms)), "pulse_s": max(0.005, float(pulse_ms) / 1000.0), "pulse_ms": pulse_ms,
                 "source": source, "manual": manual, "action": action, "on_done": on_done, "ok": True, "error": ""}
        with self._lock:
            heapq.heappush(self._queue, (self._clock() + pulse["delay_ms"] / 1000.0, next(self._count), pulse))
        return SCHEDULED

    def fire(self, relay, delay_ms, pulse_ms, source="grading"):
        """Grading: fire one ejector `delay_ms` from now. Only while RUNNING; a STOP cancels what is still waiting."""
        if not self.is_running():
            with self._lock:
                self.not_sent[NOT_RUNNING] += 1
            return NOT_RUNNING
        return self._schedule(relay, delay_ms, pulse_ms, source, manual=False, action="", on_done=None)

    def manual_pulse(self, relay, pulse_ms, action="PULSE", on_done=None):
        """Manual buttons: fire one ejector now. Only while STOPPED. Returns SCHEDULED (the final result goes to
        last_result, the log and on_done(relay, result, error)), or the reason why nothing is fired."""
        if self.is_running():
            with self._lock:
                self.not_sent[NOT_STOPPED] += 1
            self.log_manual(action, relay, pulse_ms, NOT_STOPPED, "machine is RUNNING")
            return NOT_STOPPED
        return self._schedule(relay, 0, pulse_ms, f"manual: {action}", manual=True, action=action, on_done=on_done)

    def fire_all_now(self, pulse_ms, on_done=None):
        """FIRE ALL: ejectors 1-40 at once, each as its own pulse (never the PLC's fire_all)."""
        return {relay: self.manual_pulse(relay, pulse_ms, "FIRE ALL", on_done) for relay in range(FIRST_EJECTOR, LAST_EJECTOR + 1)}

    def cancel_pending(self, switch_off=True):
        """STOP: forget every pulse that is still waiting and switch off what is ON. Returns the number cancelled."""
        with self._lock:
            waiting, self._queue = [p for _, _, p in self._queue], []
            active, self._active = list(self._active.values()), {}
        for pulse in waiting:
            self._record(pulse["relay"], pulse["delay_ms"], pulse["pulse_ms"], CANCELLED, pulse["source"])
        if switch_off:
            for pulse in active:
                self._write(pulse["relay"], False)
                self._record(pulse["relay"], pulse["delay_ms"], pulse["pulse_ms"], CANCELLED, pulse["source"])
        return len(waiting) + len(active)

    def all_ejectors_off(self):
        """ALL RELAYS OFF NOW: cancels every pulse and writes OFF to ejectors 1-40, one by one (never the PLC's
        all_off(), which also switches conveyor and feeder). Returns (number switched off, error text)."""
        self.cancel_pending(switch_off=False)
        if not self.is_connected():
            self.log_manual("ALL RELAYS OFF", "1-40", "", PLC_OFFLINE, "PLC not connected")
            return 0, "PLC not connected: nothing could be written"
        done, failed = 0, []
        for relay in range(FIRST_EJECTOR, LAST_EJECTOR + 1):
            if self.address_of(relay) in self._protected_addresses():
                failed.append(relay)
            elif self._write(relay, False):
                done += 1
            else:
                failed.append(relay)
        log.warning("ALL RELAYS OFF: %d of 40 ejectors written OFF%s", done, f", failed: {failed}" if failed else "")
        error = f"OFF could not be written for ejectors {failed}" if failed else ""
        self.log_manual("ALL RELAYS OFF", "1-40", "", SENT if not failed else WRITE_FAIL, error)
        return done, error

    # ------------------------------------------------------------------ the pulse engine
    def step(self):
        """Start pulses whose time has come, keep active coils ON, switch finished ones OFF. Called by the thread
        (every 2 ms) or by a test."""
        now = self._clock()
        with self._lock:
            due = []
            while self._queue and self._queue[0][0] <= now:
                due.append(heapq.heappop(self._queue)[2])
        for pulse in due:
            if not pulse["manual"] and not self.is_running():      # STOP came while it was waiting
                self._record(pulse["relay"], pulse["delay_ms"], pulse["pulse_ms"], CANCELLED, pulse["source"])
                continue
            relay = pulse["relay"]
            pulse["ok"] = self._write(relay, True)
            pulse["error"] = "" if pulse["ok"] else "write ON failed"
            pulse["t_off"], pulse["written"] = now + pulse["pulse_s"], now
            with self._lock:
                running = self._active.get(relay)
                if running is not None:                 # two pulses on one ejector: one longer pulse, both reported
                    pulse["t_off"] = max(pulse["t_off"], running["t_off"])
                    pulse.setdefault("also", []).append(running)
                self._active[relay] = pulse
        with self._lock:
            active = list(self._active.items())
        for relay, pulse in active:
            if now >= pulse["t_off"]:
                if not self._write(relay, False):
                    pulse["ok"], pulse["error"] = False, (pulse["error"] + "; " if pulse["error"] else "") + "write OFF failed"
                with self._lock:
                    if self._active.get(relay) is pulse:
                        del self._active[relay]
                for p in [pulse] + pulse.get("also", []):
                    self._finish(p, pulse["ok"], pulse["error"])
            elif now - pulse["written"] >= REFRESH_S:   # hold it ON, as the old diagnostics tool does
                pulse["written"] = now
                if not self._write(relay, True):
                    pulse["ok"], pulse["error"] = False, "write ON failed"

    def _finish(self, pulse, ok, error):
        result = SENT if ok else (PLC_OFFLINE if not self.is_connected() else WRITE_FAIL)
        self._record(pulse["relay"], pulse["delay_ms"], pulse["pulse_ms"], result, pulse["source"], error)
        if pulse["manual"]:
            self.log_manual(pulse["action"], pulse["relay"], pulse["pulse_ms"], result, error)
        if pulse["on_done"]:
            try:
                pulse["on_done"](pulse["relay"], result, error)
            except Exception:
                log.exception("on_done of a pulse failed")

    def _run(self):
        while not self._stop.is_set():
            try:
                self.step()
            except Exception:
                log.exception("Pulse engine step failed")
            time.sleep(0.002)

    # ------------------------------------------------------------------ ejector check: pulse with read-back (blocking)
    def read_back(self, relay):
        """State of the relay's coil as the PLC reports it (Modbus read): True / False, or None when it cannot be read."""
        address = self.address_of(relay)
        if address is None:
            return None
        try:
            driver = self._plc.driver
            if "Coil" in str(getattr(self._plc, "register_type", "Coil")):
                values = driver.read_coils(address, 1)
            else:
                values = driver.read_registers(address, 1)
            return None if not values else bool(values[0])
        except Exception:
            log.exception("Read-back of ejector %s failed", relay)
            return None

    def check_pulse(self, relay, pulse_ms, wait=time.sleep, action="EJECTOR CHECK"):
        """One checked pulse (blocking; only while STOPPED): write ON, read the coil back, keep it ON for the pulse,
        write OFF, read it back. Returns (result, error text): SENT, NOT_STOPPED, PLC_OFFLINE, REFUSED, WRITE_FAIL or READBACK_FAIL."""
        error = ""
        if self.is_running():
            result, error = NOT_STOPPED, "machine is RUNNING"
        else:
            refusal = self._refusal(relay)
            if refusal:
                result, error = refusal
            else:
                result = SENT
                try:
                    if not self._write(relay, True):
                        result, error = WRITE_FAIL, "write ON failed"
                    else:
                        seen = self.read_back(relay)
                        if seen is not True:
                            result = READBACK_FAIL
                            error = "no answer when reading the coil after ON" if seen is None else "coil reads OFF after ON was written"
                        left = pulse_ms / 1000.0
                        while left > 0:                 # hold it ON for the whole pulse
                            part = min(REFRESH_S, left)
                            wait(part)
                            left -= part
                            if left > 0:
                                self._write(relay, True)
                finally:
                    # Whatever happened above, the ejector is switched OFF again.
                    if not self._write(relay, False):
                        result, error = WRITE_FAIL, (error + "; " if error else "") + "write OFF failed"
                    else:
                        seen = self.read_back(relay)
                        if seen is not False and result == SENT:
                            result = READBACK_FAIL
                            error = "no answer when reading the coil after OFF" if seen is None else "coil still reads ON after OFF was written"
                if not self.is_connected():
                    result, error = PLC_OFFLINE, "PLC connection lost during the pulse"
        if is_ejector(relay):
            with self._lock:
                self.wanted[relay] += 1
        self._record(relay, 0, pulse_ms, result, f"manual: {action}", error)
        self.log_manual(action, relay, pulse_ms, result, error)
        return result, error

    # ------------------------------------------------------------------ manual fire log
    def log_manual(self, action, relay, pulse_ms, result, error=""):
        """One line per manual fire in logs/manual_fire_YYYYMMDD.csv (never raises)."""
        if not self.manual_log_dir:
            return
        try:
            os.makedirs(str(self.manual_log_dir), exist_ok=True)
            path = os.path.join(str(self.manual_log_dir), f"manual_fire_{time.strftime('%Y%m%d')}.csv")
            header = ",".join(MANUAL_LOG_FIELDS)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    if f.readline().strip() != header:     # columns of an older version: keep that file, start a new one
                        os.replace(path, path[:-4] + f"_old_{time.strftime('%H%M%S')}.csv")
            new = not os.path.exists(path)
            address = self.address_of(relay) if is_ejector(relay) else None
            with self._lock, open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if new:
                    writer.writerow(MANUAL_LOG_FIELDS)
                now = time.time()
                writer.writerow([time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now * 1000) % 1000:03d}", action, relay,
                                 f"M{address}" if address is not None else "", pulse_ms, result, "YES" if result == SENT else "NO",
                                 "YES" if self.is_connected() else "NO", error])
        except Exception:
            log.exception("Could not write the manual fire log")
