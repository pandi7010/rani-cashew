"""The only door through which the grader fires relays.

Rules enforced here (not in innovance_plc.py, which is left untouched):
  * Only ejector relays 1-40 can be fired. Relay 41 (conveyor, M140) and 42 (feeder, M141)
    and anything above are refused.
  * The PLC controller's fire_all() / sequence_all() are never called: they pulse all 48 coils,
    conveyor and feeder included. "Fire all" here pulses relays 1-40 one by one.
  * DRY-RUN (on at every start): nothing is sent to the PLC; what would have fired is counted and logged.
  * MASTER EJECTOR (off at every start): blocks firing that comes from grading.
    A manual test fire of one relay ignores it (the operator asked for that relay), but still obeys DRY-RUN.
"""
import logging
import threading
import time
from collections import deque

log = logging.getLogger("relay")

FIRST_EJECTOR, LAST_EJECTOR = 1, 40
RELAYS_PER_ROI = 20
REJECT_RELAY = {1: 1, 2: 21}        # ROI -> reject relay (double, undefined, good not rotated)

SENT, DRY_RUN, MASTER_OFF, PLC_OFFLINE, REFUSED = "SENT", "DRY_RUN", "MASTER_OFF", "PLC_OFFLINE", "REFUSED"


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
        self.not_sent = {DRY_RUN: 0, MASTER_OFF: 0, PLC_OFFLINE: 0, REFUSED: 0}
        self.events = deque(maxlen=500)     # (time, relay, delay_ms, pulse_ms, result, source)
        self._pending = deque(maxlen=2000)  # (time the pulse starts, relay, source) of pulses given to the PLC

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

    def test_fire(self, relay, pulse_ms):
        """Manual test of one relay (field check that a coil drives the right valve). Obeys DRY-RUN only."""
        return self._pulse(relay, 0, pulse_ms, "manual test", respect_master=False)

    def test_sequence(self, pulse_ms, interval_ms=120, relays=None, stop_event=None, on_step=None):
        """Manual test: relays 1-40 (or the given ejectors) one after another. Blocks; run it in a thread."""
        results = {}
        for relay in (relays if relays is not None else range(FIRST_EJECTOR, LAST_EJECTOR + 1)):
            if stop_event is not None and stop_event.is_set():
                break
            results[relay] = self.test_fire(relay, pulse_ms)
            if on_step:
                on_step(relay, results[relay])
            time.sleep(interval_ms / 1000.0)
        return results

    def test_all(self, pulse_ms):
        """Manual test: relays 1-40 at once, each as its own pulse (never the PLC's fire_all)."""
        return {relay: self.test_fire(relay, pulse_ms) for relay in range(FIRST_EJECTOR, LAST_EJECTOR + 1)}
