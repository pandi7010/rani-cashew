import csv
import glob
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import relay_output as ro
from relay_output import RelayOutput
from test_ejector_check import FakeCheckPLC


class FakePLC(FakeCheckPLC):
    """PLC with coil memory; `pulses` lists the ejectors that were switched ON, in order (used by the window tests)."""

    @property
    def pulses(self):
        out, on = [], set()
        for relay, state in self.writes:
            if state and relay not in on:
                out.append(relay)
                on.add(relay)
            elif not state:
                on.discard(relay)
        return out


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.plc = FakePLC()
        self.now = [1000.0]
        self.running = [False]
        self.out = RelayOutput(self.plc, is_running=lambda: self.running[0], clock=lambda: self.now[0], start_thread=False)
        self.out.manual_log_dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def at(self, seconds):
        """Move the clock to start + seconds and let the engine work."""
        self.now[0] = 1000.0 + seconds
        self.out.step()

    def log_rows(self):
        files = glob.glob(os.path.join(self.tmp.name, "manual_fire_*.csv"))
        if not files:
            return []
        with open(files[0], encoding="utf-8") as f:
            return list(csv.DictReader(f))


class StartStopIsTheOnlyControl(Base):
    def test_there_is_no_dry_run_and_no_master_switch(self):
        for name in ("dry_run", "master_enabled", "test_fire", "test_all"):
            self.assertFalse(hasattr(self.out, name), name)
        for name in ("DRY_RUN", "MASTER_OFF"):
            self.assertFalse(hasattr(ro, name), name)

    def test_stopped_nothing_fires_from_grading(self):
        self.assertEqual(self.out.fire(3, 200, 100), ro.NOT_RUNNING)
        self.at(1.0)
        self.assertEqual(self.plc.writes, [])
        self.assertEqual(self.out.counts()[2][ro.NOT_RUNNING], 1)

    def test_running_grading_fires_after_its_delay_and_holds_the_coil_on(self):
        self.running[0] = True
        self.assertEqual(self.out.fire(3, 500, 100, source="R1-000007 TESTA"), ro.SCHEDULED)
        self.at(0.49)
        self.assertEqual(self.plc.writes, [])                      # not before its time
        self.at(0.50)
        self.assertEqual(self.plc.writes, [(3, True)])
        self.assertTrue(self.plc.coils[102])
        self.at(0.52)
        self.assertEqual(len(self.plc.writes), 1)                  # no need to write again yet
        self.at(0.56)
        self.assertEqual(self.plc.writes[-1], (3, True))           # written ON again while the pulse lasts
        self.assertEqual(len(self.plc.writes), 2)
        self.at(0.60)
        self.assertEqual(self.plc.writes[-1], (3, False))
        self.assertFalse(self.plc.coils[102])
        self.assertEqual(self.out.recent_events()[0][1:], (3, 500.0, 100, ro.SENT, "R1-000007 TESTA"))
        self.assertEqual((self.out.wanted[3], self.out.sent[3]), (1, 1))
        self.assertEqual(self.log_rows(), [])                      # grading is not a manual fire

    def test_stop_cancels_what_is_waiting_and_switches_off_what_is_on(self):
        self.running[0] = True
        self.out.fire(5, 0, 300)
        self.out.fire(9, 4000, 100)
        self.at(0.01)
        self.assertTrue(self.plc.coils[104])
        self.assertEqual([p[1] for p in self.out.pending()], [9])
        self.running[0] = False
        self.assertEqual(self.out.cancel_pending(), 2)
        self.assertFalse(self.plc.coils[104])
        self.at(5.0)
        self.assertEqual([w for w in self.plc.writes if w[0] == 9], [])       # the waiting one never fired
        self.assertEqual(self.out.counts()[2][ro.CANCELLED], 2)

    def test_a_waiting_grading_pulse_is_dropped_when_the_machine_was_stopped_meanwhile(self):
        self.running[0] = True
        self.out.fire(9, 1000, 100)
        self.running[0] = False                                    # stopped without anybody calling cancel_pending
        self.at(1.2)
        self.assertEqual(self.plc.writes, [])
        self.assertEqual(self.out.recent_events()[0][4], ro.CANCELLED)


class ManualFire(Base):
    def test_manual_pulse_fires_the_real_coil_at_once_only_when_stopped(self):
        done = []
        self.assertEqual(self.out.manual_pulse(7, 120, "ROW PULSE", on_done=lambda *a: done.append(a)), ro.SCHEDULED)
        self.at(0.0)
        self.assertEqual(self.plc.writes, [(7, True)])             # at once
        self.at(0.06)
        self.at(0.11)
        self.at(0.12)
        self.assertEqual(self.plc.writes, [(7, True), (7, True), (7, True), (7, False)])    # held ON, then OFF
        self.assertEqual(done, [(7, ro.SENT, "")])
        self.assertEqual(self.out.last_result[7][1], ro.SENT)
        row = self.log_rows()[0]
        self.assertEqual((row["action"], row["relay"], row["m_address"], row["pulse_ms"], row["result"], row["sent_to_plc"]),
                         ("ROW PULSE", "7", "M106", "120", "SENT", "YES"))
        self.running[0] = True
        self.assertEqual(self.out.manual_pulse(7, 120), ro.NOT_STOPPED)
        self.at(1.0)
        self.assertEqual(len(self.plc.writes), 4)
        self.assertEqual(self.log_rows()[-1]["result"], "NOT_STOPPED")

    def test_fire_all_is_40_single_pulses_of_ejectors_1_to_40(self):
        results = self.out.fire_all_now(100)
        self.assertEqual(set(results.values()), {ro.SCHEDULED})
        self.at(0.0)
        self.assertEqual(self.plc.pulses, list(range(1, 41)))
        self.assertEqual(sorted(a for a, on in self.plc.coils.items() if on), list(range(100, 140)))
        self.at(0.11)
        self.assertFalse(any(self.plc.coils.values()))
        self.assertEqual(self.plc.forbidden, [])                   # never fire_all / sequence_all / all_off / pulse_actuator of the PLC
        self.assertEqual({r["action"] for r in self.log_rows()}, {"FIRE ALL"})
        self.assertEqual(len(self.log_rows()), 40)

    def test_only_ejectors_1_to_40_and_never_the_conveyor_or_feeder_address(self):
        for relay in (0, 41, 42, 43, 48, -1, 7.0, "7", None, True):
            self.assertEqual(self.out.manual_pulse(relay, 100), ro.REFUSED, relay)
        self.running[0] = True
        self.assertEqual(self.out.fire(41, 0, 100), ro.REFUSED)
        self.assertEqual(self.out.fire(42, 0, 100), ro.REFUSED)
        self.running[0] = False
        self.plc.get_relay_address = lambda relay: 141 if relay in (9, 42) else 100 + relay - 1      # a wrong mapping
        self.assertEqual(self.out.manual_pulse(9, 100), ro.REFUSED)
        self.at(1.0)
        self.assertEqual(self.plc.writes, [])
        self.assertIn("conveyor / feeder", self.out.last_result[9][2])

    def test_plc_errors_are_reported_not_hidden(self):
        self.plc.connected_flag = False
        self.assertEqual(self.out.manual_pulse(3, 100), ro.PLC_OFFLINE)
        self.plc.connected_flag = True
        self.plc.fail_write.add(4)
        done = []
        self.out.manual_pulse(4, 100, on_done=lambda *a: done.append(a))
        self.at(0.0)
        self.at(0.2)
        self.assertEqual(done[0][1], ro.WRITE_FAIL)
        self.assertIn("write ON failed", done[0][2])
        self.assertTrue(ro.RESULT_TEXT[ro.WRITE_FAIL].startswith("PLC error"))
        self.assertEqual(ro.RESULT_TEXT[ro.SENT], "OK")
        self.assertEqual([r["sent_to_plc"] for r in self.log_rows()], ["NO", "NO"])

    def test_all_relays_off_cancels_everything_and_leaves_conveyor_and_feeder(self):
        self.plc.coils[140] = self.plc.coils[141] = True
        self.out.manual_pulse(2, 500)
        self.at(0.0)
        self.assertTrue(self.plc.coils[101])
        self.assertEqual(self.out.all_ejectors_off(), (40, ""))
        self.assertEqual(self.plc.writes[-40:], [(r, False) for r in range(1, 41)])
        self.assertFalse(self.plc.coils[101])
        self.assertTrue(self.plc.coils[140] and self.plc.coils[141])
        self.assertFalse(self.out.busy())
        self.assertEqual(self.plc.forbidden, [])

    def test_two_pulses_on_one_ejector_become_one_longer_pulse_and_both_are_reported(self):
        self.running[0] = True
        self.out.fire(6, 0, 100, source="a")
        self.out.fire(6, 60, 100, source="b")
        self.at(0.0)
        self.at(0.06)
        self.at(0.11)
        self.assertTrue(self.plc.coils[105])                       # still ON for the second cashew
        self.at(0.16)
        self.assertFalse(self.plc.coils[105])
        self.assertEqual(sorted(e[5] for e in self.out.recent_events()), ["a", "b"])
        self.assertEqual(self.out.sent[6], 2)

    def test_the_engine_thread_fires_a_real_time_pulse(self):
        plc = FakePLC()
        out = RelayOutput(plc)                                     # its own thread and the real clock
        try:
            self.assertEqual(out.manual_pulse(12, 30), ro.SCHEDULED)
            end = time.time() + 2.0
            while 12 not in out.last_result and time.time() < end:
                time.sleep(0.005)
            self.assertEqual(out.last_result[12][1], ro.SENT)
            self.assertEqual((plc.writes[0], plc.writes[-1]), ((12, True), (12, False)))
        finally:
            out.stop()

    def test_describe_shows_the_coil(self):
        self.assertEqual(self.out.describe(21), "Ejector 21 -> M120 (Modbus address 120)")


if __name__ == "__main__":
    unittest.main()
