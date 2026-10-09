import csv
import glob
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ejector_check as ec
import relay_output as ro
from relay_output import RelayOutput


class FakeDriver:
    def __init__(self, plc):
        self.plc = plc

    def read_coils(self, address, count):
        self.plc.reads.append(address)
        if not self.plc.connected_flag or address in self.plc.unreadable:
            return None
        return [self.plc.coils.get(address + i, False) for i in range(count)]


class FakeCheckPLC:
    """A PLC with real coil memory. Anything except single coil writes / reads of the wrapper is forbidden."""
    register_type = "Coil (FC01/05/15)"

    def __init__(self):
        self.connected_flag = True
        self.coils, self.writes, self.reads, self.forbidden = {}, [], [], []
        self.fail_write, self.stuck_off, self.stuck_on, self.unreadable = set(), set(), set(), set()
        self.disconnect_after_writes = None
        self.driver = FakeDriver(self)

    def is_connected(self):
        return self.connected_flag

    def get_relay_address(self, relay):
        return 100 + relay - 1

    def _write_relay(self, relay, state):
        if not self.connected_flag:
            return False
        self.writes.append((relay, state))
        if self.disconnect_after_writes is not None and len(self.writes) >= self.disconnect_after_writes:
            self.connected_flag = False
        if relay in self.fail_write:
            return False
        address = self.get_relay_address(relay)
        if (state and relay in self.stuck_off) or (not state and relay in self.stuck_on):
            return True                                  # the PLC says OK but the coil does not change
        self.coils[address] = state
        return True

    def __getattr__(self, name):                         # pulse_actuator, fire_all, sequence_all, all_off, set_relay ...
        def call(*args, **kwargs):
            self.forbidden.append((name, args))
        return call


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.plc = FakeCheckPLC()
        self.out = RelayOutput(self.plc)
        self.out.manual_log_dir = self.tmp.name
        self.waits = []
        self.safe = [""]
        self.check = ec.EjectorCheck(self.out, log_dir=self.tmp.name, safe_to_run=lambda: self.safe[0], wait=self.wait)
        self.stop_at_wait = None

    def tearDown(self):
        self.tmp.cleanup()

    def wait(self, seconds):
        self.waits.append(seconds)
        if self.stop_at_wait is not None and len(self.waits) >= self.stop_at_wait:
            self.check._stop.set()
        return self.check._stop.is_set()

    def log_rows(self):
        files = glob.glob(os.path.join(self.tmp.name, "ejector_check_*.csv"))
        if not files:
            return []
        with open(files[0], encoding="utf-8") as f:
            return list(csv.DictReader(f))


class CheckedPulseTest(Base):
    def test_refused_while_running(self):
        self.out.is_running = lambda: True
        self.assertEqual(self.out.check_pulse(5, 100, wait=lambda s: None), (ro.NOT_STOPPED, "machine is RUNNING"))
        self.assertEqual((self.plc.writes, self.plc.reads, self.plc.forbidden), ([], [], []))

    def test_on_and_off_are_written_and_read_back(self):
        self.assertEqual(self.out.check_pulse(5, 100, wait=lambda s: None), (ro.SENT, ""))
        self.assertEqual(self.plc.writes, [(5, True), (5, True), (5, False)])       # held ON during the 100 ms pulse
        self.assertEqual(self.plc.reads, [104, 104])
        self.assertFalse(self.plc.coils[104])

    def test_failures_are_named_and_the_relay_is_always_switched_off(self):
        self.plc.stuck_off.add(3)
        self.assertEqual(self.out.check_pulse(3, 100, wait=lambda s: None), (ro.READBACK_FAIL, "coil reads OFF after ON was written"))
        self.assertEqual(self.plc.writes[-1], (3, False))
        self.plc.stuck_on.add(4)
        self.assertEqual(self.out.check_pulse(4, 100, wait=lambda s: None), (ro.READBACK_FAIL, "coil still reads ON after OFF was written"))
        self.plc.fail_write.add(6)
        result, error = self.out.check_pulse(6, 100, wait=lambda s: None)
        self.assertEqual(result, ro.WRITE_FAIL)
        self.assertIn("write ON failed", error)
        self.assertEqual(self.plc.writes[-1], (6, False))
        self.plc.unreadable.add(106)
        self.assertEqual(self.out.check_pulse(7, 100, wait=lambda s: None)[0], ro.READBACK_FAIL)

    def test_conveyor_feeder_and_other_relays_are_refused(self):
        for relay in (0, 41, 42, 43, 48):
            self.assertEqual(self.out.check_pulse(relay, 100, wait=lambda s: None)[0], ro.REFUSED)
        self.plc.get_relay_address = lambda relay: 140 if relay in (7, 41) else 100 + relay - 1    # a wrong mapping
        result, error = self.out.check_pulse(7, 100, wait=lambda s: None)
        self.assertEqual(result, ro.REFUSED)
        self.assertIn("conveyor / feeder", error)
        self.assertEqual(self.plc.writes, [])

    def test_offline(self):
        self.plc.connected_flag = False
        self.assertEqual(self.out.check_pulse(2, 100, wait=lambda s: None)[0], ro.PLC_OFFLINE)

    def test_all_ejectors_off_writes_1_to_40_only(self):
        self.plc.coils[139] = self.plc.coils[140] = self.plc.coils[141] = True
        self.assertEqual(self.out.all_ejectors_off(), (40, ""))
        self.assertEqual(self.plc.writes, [(r, False) for r in range(1, 41)])
        self.assertFalse(self.plc.coils[139])
        self.assertTrue(self.plc.coils[140] and self.plc.coils[141])        # conveyor and feeder untouched
        self.assertEqual(self.plc.forbidden, [])
        self.plc.connected_flag = False
        self.assertEqual(self.out.all_ejectors_off()[0], 0)


class EjectorCheckTest(Base):
    def test_all_40_relays_five_pulses_each_in_order(self):
        self.check.run(ec.relays_for("ALL"))
        self.assertEqual(self.check.phase, ec.DONE)
        ons = [relay for relay, state in self.plc.writes if state]
        firsts = [relay for (relay, state), (_, before) in zip(self.plc.writes, [(0, False)] + self.plc.writes[:-1]) if state and not before]
        self.assertEqual(firsts, [r for r in range(1, 41) for _ in range(5)])
        self.assertEqual(len([w for w in self.plc.writes if not w[1]]), 200)
        self.assertEqual(len(self.plc.reads), 400)
        self.assertEqual({r.status for r in self.check.results.values()}, {ec.PLC_OK})
        self.assertEqual(self.check.results[40].address, 139)
        self.assertEqual(self.plc.forbidden, [])                             # never fire_all / sequence_all / all_off
        self.assertFalse(any(a in (140, 141) for a in self.plc.reads))
        self.assertFalse(any(self.plc.coils.values()))
        self.assertTrue(self.check.last_summary[0].startswith("ALL OK"))
        self.assertEqual(self.check.last_summary[1], "good")
        # 3 s countdown, then per relay 5 pulses of 0.1 s with 4 gaps of 0.4 s, 1 s between relays
        self.assertEqual(self.waits[:3], [1.0, 1.0, 1.0])
        self.assertEqual(self.waits[3:6], [0.05, 0.05, 0.4])               # the pulse is held in two parts, then the gap
        self.assertAlmostEqual(sum(self.waits[3:17]), 5 * 0.1 + 4 * 0.4)
        self.assertEqual(self.waits[17], 1.0)
        rows = self.log_rows()
        self.assertEqual(len(rows), 40)
        self.assertEqual((rows[0]["relay"], rows[0]["m_address"], rows[0]["pulses_sent"], rows[0]["plc"], rows[0]["operator_mark"], rows[0]["error"]),
                         ("1", "M100", "5", "PLC OK", "not checked", ""))

    def test_ranges_and_parameters(self):
        self.assertEqual(ec.relays_for("ROI1"), list(range(1, 21)))
        self.assertEqual(ec.relays_for("ROI2"), list(range(21, 41)))
        self.assertEqual(ec.relays_for("SINGLE", 17), [17])
        self.assertEqual(len(ec.relays_for("ALL")), 40)
        self.assertRaises(ValueError, ec.relays_for, "SINGLE", 41)
        self.assertEqual(ec.clean_params(None), ec.DEFAULTS)
        self.assertEqual(ec.DEFAULTS, {"pulses": 5, "pulse_ms": 100, "gap_ms": 400, "relay_gap_ms": 1000, "countdown_s": 3})
        self.assertRaises(ValueError, ec.clean_params, {"pulse_ms": 5000})
        self.check.run(ec.relays_for("SINGLE", 21) * 3, {"pulses": 2, "countdown_s": 0})      # one relay, repeated
        self.assertEqual({w for w in self.plc.writes}, {(21, True), (21, False)})
        self.assertEqual(len([w for w in self.plc.writes if not w[1]]), 6)

    def test_plc_fail_is_reported_per_relay_and_the_check_goes_on(self):
        self.plc.stuck_off.add(2)
        self.plc.fail_write.add(4)
        self.check.run([1, 2, 3, 4, 5], {"countdown_s": 0})
        status = {r: self.check.results[r].status for r in range(1, 6)}
        self.assertEqual(status, {1: ec.PLC_OK, 2: ec.PLC_FAIL, 3: ec.PLC_OK, 4: ec.PLC_FAIL, 5: ec.PLC_OK})
        self.assertIn("coil reads OFF after ON", self.check.results[2].error)
        self.assertIn("write ON failed", self.check.results[4].error)
        self.assertEqual(self.check.last_summary, (self.check.last_summary[0], "bad"))
        self.assertIn("FAILED R2,R4", self.check.last_summary[0])

    def test_plc_disconnect_stops_the_check(self):
        self.plc.disconnect_after_writes = 18                # in the middle of relay 2
        self.check.run(ec.relays_for("ALL"), {"countdown_s": 0})
        self.assertEqual(self.check.phase, ec.ERROR)
        self.assertIn("PLC disconnected at relay 2", self.check.message)
        self.assertEqual(self.check.results[2].status, ec.PLC_FAIL)
        self.assertEqual(self.check.results[3].status, ec.NOT_RUN)
        self.assertFalse(any(state for relay, state in self.plc.writes[18:]))    # nothing was switched on afterwards
        self.assertEqual(self.check.last_summary[1], "bad")

    def test_stop_ends_the_check_and_switches_everything_off(self):
        self.stop_at_wait = 9                                # during relay 1
        self.check.run(ec.relays_for("ALL"), {"countdown_s": 0})
        self.assertEqual(self.check.phase, ec.STOPPED)
        self.assertEqual(self.check.results[1].status, ec.UNFINISHED)
        self.assertEqual(self.check.results[2].status, ec.NOT_RUN)
        n = len(self.plc.writes)
        done, error = self.check.stop()                      # the red button
        self.assertEqual((done, error), (40, ""))
        self.assertEqual(self.plc.writes[n:], [(r, False) for r in range(1, 41)])
        self.assertFalse(any(self.plc.coils.values()))

    def test_does_not_start_or_go_on_when_running_or_conveyor_on(self):
        self.safe[0] = "The conveyor is ON."
        self.assertEqual(self.check.start(ec.relays_for("ALL")), "The conveyor is ON.")
        self.assertEqual(self.check.phase, ec.IDLE)
        self.safe[0] = ""
        calls = []

        def safe():
            calls.append(1)
            return "The machine is RUNNING." if len(calls) > 2 else ""
        self.check.safe_to_run = safe
        self.check.run([1, 2, 3, 4], {"countdown_s": 0})
        self.assertEqual(self.check.phase, ec.ERROR)
        self.assertIn("Stopped before relay 3: The machine is RUNNING.", self.check.message)
        self.assertEqual(sorted({relay for relay, state in self.plc.writes if state}), [1, 2])

    def test_operator_marks(self):
        self.check.run([1, 2, 3], {"countdown_s": 0})
        self.check.mark(1, ec.AIR_OK)
        self.check.mark(2, ec.NO_AIR)
        self.check.mark(3, "nonsense")
        self.assertEqual([self.check.results[r].mark for r in (1, 2, 3)], [ec.AIR_OK, ec.NO_AIR, ec.NOT_CHECKED])
        self.assertIn("AIR FAULT R2", self.check.last_summary[0])
        self.check.mark(2, ec.WRONG_VALVE)
        rows = self.log_rows()
        self.assertEqual([(r["relay"], r["operator_mark"]) for r in rows[3:]], [("1", "AIR OK"), ("2", "NO AIR"), ("2", "WRONG VALVE")])
        self.check.run([2], {"countdown_s": 0})              # checking a relay again: it is judged again
        self.assertEqual(self.check.results[2].mark, ec.NOT_CHECKED)

    def test_start_runs_in_a_thread_with_a_real_stop(self):
        check = ec.EjectorCheck(self.out, log_dir=self.tmp.name)
        self.assertEqual(check.start(ec.relays_for("ALL"), {"countdown_s": 0, "pulse_ms": 10, "gap_ms": 50, "relay_gap_ms": 0}), "")
        self.assertEqual(check.start([1]), "The check is already running.")
        check.stop()
        check._thread.join(2.0)
        self.assertFalse(check._thread.is_alive())
        self.assertEqual(check.phase, ec.STOPPED)
        self.assertEqual({relay for relay, state in self.plc.writes if not state}, set(range(1, 41)))    # STOP wrote OFF to every ejector
        self.assertFalse(any(self.plc.coils.values()))


if __name__ == "__main__":
    unittest.main()
