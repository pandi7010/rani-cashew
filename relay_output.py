"""The only door through which the grader fires relays.

Rules enforced here (not in innovance_plc.py, which is left untouched):
  * Only ejector relays 1-40 can be fired. Relay 41 (conveyor, M140) and 42 (feeder, M141)
    and anything above are refused.
  * The PLC controller's fire_all() / sequence_all() are never called: they pulse all 48 coils,
    conveyor and feeder included. "Fire all" here pulses relays 1-40 one by one.
  * DRY-RUN (on at every start): nothing is sent to the PLC; what would have fired is counted and logged.
  * MASTER EJECTOR (off at every start): blocks firing that comes from grading.
    A manual test fire of one relay ignores it (the operator asked for that relay), but still obeys DRY-RUN.
  * EVERY manual fire (row test, single relay test, FIRE 1-40, SEQUENCE 1-40, relay check) is the same call,
    test_fire(), so they all obey DRY-RUN in the same way, and each one is written to
    logs/manual_fire_YYYYMMDD.csv: time, action, relay, M address, pulse, result, sent to the PLC or not.
"""
import csv
import logging
import os
import threading
import time
from collections import deque

log = logging.getLogger("relay")

FIRST_EJECTOR, LAST_EJECTOR = 1, 40
RELAYS_PER_ROI = 20
REJECT_RELAY = {1: 1, 2: 21}        # ROI -> reject relay (double, undefined, good not rotated)

SENT, DRY_RUN, MASTER_OFF, PLC_OFFLINE, REFUSED = "SENT", "DRY_RUN", "MASTER_OFF", "PLC_OFFLINE", "REFUSED"
# The one wording for every button that fires by hand.
RESULT_TEXT = {SENT: "pulse sent to the PLC", DRY_RUN: "DRY-RUN: NOT sent to the PLC", PLC_OFFLINE: "PLC OFFLINE: NOT sent",
               REFUSED: "refused (not an ejector relay)", MASTER_OFF: "MASTER EJECTOR off"}
WRITE_FAIL, READBACK_FAIL = "WRITE_FAIL", "READBACK_FAIL"      # only from check_pulse(), which reads the coil back
RESULT_TEXT.update({WRITE_FAIL: "PLC write failed", READBACK_FAIL: "PLC read-back wrong"})
CONVEYOR_RELAY, FEEDER_RELAY = 41, 42               # M140, M141: never driven from here
MANUAL_LOG_FIELDS = ["time", "action", "relay", "m_address", "pulse_ms", "result", "sent_to_plc", "dry_run", "plc_connected"]


def is_ejector(relay) -> bool:
    return isinstance(relay, int) and not isinstance(relay, bool) and FIRST_EJECTOR <= relay <= LAST_EJECTOR


def roi_of_relay(relay) -> int:
    return 1 if relay <= RELAYS_PER_ROI else 2


class RelayOutput:
    def __init__(self, plc):
        self._plc = plc
        self._lock = threading.Lock()
        self.dry_run = True
        self.master_enabled = False
        self.wanted = {r: 0 for r in range(FIRST_EJECTOR, LAST_EJECTOR + 1)}   # decisions that asked for this relay
        self.sent = {r: 0 for r in range(FIRST_EJECTOR, LAST_EJECTOR + 1)}     # pulses really given to the PLC
        self.not_sent = {DRY_RUN: 0, MASTER_OFF: 0, PLC_OFFLINE: 0, REFUSED: 0, "WRITE_FAIL": 0, "READBACK_FAIL": 0}
        self.events = deque(maxlen=500)     # (time, relay, delay_ms, pulse_ms, result, source)
        self._pending = deque(maxlen=2000)  # (time the pulse starts, relay, source) of pulses given to the PLC
        self.manual_log_dir = None          # folder for manual_fire_YYYYMMDD.csv (None = no file)

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
        return f"Relay {relay}" + (f" -> M{address} (Modbus address {address})" if address is not None else "")

    def counts(self):
        with self._lock:
            return dict(self.wanted), dict(self.sent), dict(self.not_sent)

    def reset_counts(self):
        with self._lock:
            for table in (self.wanted, self.sent, self.not_sent):
                for key in table:
                    table[key] = 0

    def pending(self):
        """Pulses already given to the PLC whose valve has not opened yet: [(start time, relay, source)]."""
        now = time.time()
        with self._lock:
            while self._pending and self._pending[0][0] <= now:
                self._pending.popleft()
            return [p for p in self._pending if p[0] > now]

    def recent_events(self):
        with self._lock:
            return list(self.events)

    # ------------------------------------------------------------------ firing
    def _pulse(self, relay, delay_ms, pulse_ms, source, respect_master):
        if not is_ejector(relay):
            result = REFUSED
            log.error("Refused to fire relay %r (%s): only relays %d-%d are ejectors.", relay, source, FIRST_EJECTOR, LAST_EJECTOR)
        elif respect_master and not self.master_enabled:
            result = MASTER_OFF
        elif self.dry_run:
            result = DRY_RUN
        elif not self.is_connected():
            result = PLC_OFFLINE
        else:
            self._plc.pulse_actuator(relay, max(0, int(round(delay_ms))), max(1, int(round(pulse_ms))))
            result = SENT
        with self._lock:
            if is_ejector(relay):
                self.wanted[relay] += 1
                if result == SENT:
                    self.sent[relay] += 1
            if result != SENT:
                self.not_sent[result] += 1
            now = time.time()
            self.events.appendleft((now, relay, delay_ms, pulse_ms, result, source))
            if result == SENT and delay_ms > 0:
                self._pending.append((now + delay_ms / 1000.0, relay, source))
        if result == DRY_RUN:
            log.info("[DRY-RUN] would fire relay %s after %.0f ms for %.0f ms (%s)", relay, delay_ms, pulse_ms, source)
        elif result == SENT:
            log.info("[FIRE] relay %s after %.0f ms for %.0f ms (%s)", relay, delay_ms, pulse_ms, source)
        elif result == PLC_OFFLINE:
            log.warning("[FIRE] relay %s NOT fired: PLC offline (%s)", relay, source)
        return result

    def fire(self, relay, delay_ms, pulse_ms, source="grading"):
        """Fire one ejector for a grading decision. Returns SENT, DRY_RUN, MASTER_OFF, PLC_OFFLINE or REFUSED."""
        return self._pulse(relay, delay_ms, pulse_ms, source, respect_master=True)

    def test_fire(self, relay, pulse_ms, action="SINGLE RELAY TEST"):
        """Manual fire of one relay. The ONLY way any button fires by hand, so DRY-RUN holds for all of them."""
        result = self._pulse(relay, 0, pulse_ms, f"manual: {action}", respect_master=False)
        self.log_manual(action, relay, pulse_ms, result)
        return result

    def log_manual(self, action, relay, pulse_ms, result):
        """One line per manual fire in logs/manual_fire_YYYYMMDD.csv (never raises)."""
        if not self.manual_log_dir:
            return
        try:
            os.makedirs(str(self.manual_log_dir), exist_ok=True)
            path = os.path.join(str(self.manual_log_dir), f"manual_fire_{time.strftime('%Y%m%d')}.csv")
            new = not os.path.exists(path)
            address = self.address_of(relay) if is_ejector(relay) else None
            with self._lock, open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if new:
                    writer.writerow(MANUAL_LOG_FIELDS)
                now = time.time()
                writer.writerow([time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now * 1000) % 1000:03d}", action, relay,
                                 f"M{address}" if address is not None else "", pulse_ms, result, "YES" if result == SENT else "NO",
                                 "ON" if self.dry_run else "OFF", "YES" if self.is_connected() else "NO"])
        except Exception:
            log.exception("Could not write the manual fire log")

    # ------------------------------------------------------------------ ejector check: pulse with read-back
    def _protected_addresses(self):
        """PLC addresses of conveyor and feeder as configured, plus M140 / M141: an ejector must never have them."""
        out = {140, 141}
        for relay in (CONVEYOR_RELAY, FEEDER_RELAY):
            address = self.address_of(relay)
            if address is not None:
                out.add(address)
        return out

    def _write_now(self, relay, state) -> bool:
        """One coil write, at once and waited for (the PLC controller's own single-relay write)."""
        return bool(self._plc._write_relay(relay, state))

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
            log.exception("Read-back of relay %s failed", relay)
            return None

    def check_pulse(self, relay, pulse_ms, wait=time.sleep, action="EJECTOR CHECK"):
        """One checked pulse of one ejector: write ON, read the coil back, wait, write OFF, read it back.
        Returns (result, error text). Follows DRY-RUN like every manual fire: then nothing is sent.
        Result: SENT (both writes confirmed), DRY_RUN, PLC_OFFLINE, REFUSED, WRITE_FAIL or READBACK_FAIL."""
        error = ""
        if not is_ejector(relay):
            result, error = REFUSED, "not an ejector relay (only 1-40)"
        elif self.address_of(relay) in self._protected_addresses():
            result, error = REFUSED, f"relay {relay} is mapped to the conveyor / feeder address M{self.address_of(relay)}"
        elif self.dry_run:
            wait(pulse_ms / 1000.0)
            result = DRY_RUN
        elif not self.is_connected():
            result, error = PLC_OFFLINE, "PLC not connected"
        else:
            result = SENT
            try:
                if not self._write_now(relay, True):
                    result, error = WRITE_FAIL, "write ON failed"
                else:
                    seen = self.read_back(relay)
                    if seen is not True:
                        result = READBACK_FAIL
                        error = "no answer when reading the coil after ON" if seen is None else "coil reads OFF after ON was written"
                    wait(pulse_ms / 1000.0)
            finally:
                # Whatever happened above, the relay is switched OFF again.
                off_ok = False
                try:
                    off_ok = self._write_now(relay, False)
                except Exception:
                    log.exception("Write OFF of relay %s failed", relay)
                if not off_ok:
                    result, error = WRITE_FAIL, (error + "; " if error else "") + "write OFF failed"
                else:
                    seen = self.read_back(relay)
                    if seen is not False and result == SENT:
                        result = READBACK_FAIL
                        error = "no answer when reading the coil after OFF" if seen is None else "coil still reads ON after OFF was written"
            if not self.is_connected():
                result, error = PLC_OFFLINE, "PLC connection lost during the pulse"
        with self._lock:
            if is_ejector(relay):
                self.wanted[relay] += 1
                if result == SENT:
                    self.sent[relay] += 1
            if result in self.not_sent:
                self.not_sent[result] += 1
            self.events.appendleft((time.time(), relay, 0, pulse_ms, result, f"manual: {action}"))
        self.log_manual(action, relay, pulse_ms, result)
        return result, error

    def all_ejectors_off(self):
        """ALL RELAYS OFF NOW: writes OFF to relays 1-40, one by one (never the PLC's all_off(), which also
        switches conveyor and feeder). Works in DRY-RUN too: switching OFF cannot fire a valve.
        Returns (number switched off, error text)."""
        if not self.is_connected():
            self.log_manual("ALL EJECTORS OFF", "1-40", "", PLC_OFFLINE)
            return 0, "PLC not connected: nothing could be written"
        done, failed = 0, []
        for relay in range(FIRST_EJECTOR, LAST_EJECTOR + 1):
            try:
                if self._write_now(relay, False):
                    done += 1
                else:
                    failed.append(relay)
            except Exception:
                failed.append(relay)
        log.warning("ALL EJECTORS OFF: %d of 40 relays written OFF%s", done, f", failed: {failed}" if failed else "")
        self.log_manual("ALL EJECTORS OFF", "1-40", "", SENT if not failed else WRITE_FAIL)
        return done, (f"OFF could not be written for relays {failed}" if failed else "")

    def test_sequence(self, pulse_ms, interval_ms=120, relays=None, stop_event=None, on_step=None):
        """Manual test: relays 1-40 (or the given ejectors) one after another. Blocks; run it in a thread."""
        results = {}
        for relay in (relays if relays is not None else range(FIRST_EJECTOR, LAST_EJECTOR + 1)):
            if stop_event is not None and stop_event.is_set():
                break
            results[relay] = self.test_fire(relay, pulse_ms, action="SEQUENCE 1-40")
            if on_step:
                on_step(relay, results[relay])
            time.sleep(interval_ms / 1000.0)
        return results

    def test_all(self, pulse_ms):
        """Manual test: relays 1-40 at once, each as its own pulse (never the PLC's fire_all)."""
        return {relay: self.test_fire(relay, pulse_ms, action="FIRE 1-40") for relay in range(FIRST_EJECTOR, LAST_EJECTOR + 1)}
