import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import relay_output as ro
from relay_output import RelayOutput


class FakePLC:
    """Records every call. fire_all / sequence_all / set_relay / conveyor / feeder must never be reached."""

    def __init__(self, connected=True):
        self.connected_flag = connected
        self.pulses = []
        self.forbidden = []

    def is_connected(self):
        return self.connected_flag

    def get_relay_address(self, relay):
        return 100 + relay - 1

    def pulse_actuator(self, idx, delay_ms, pulse_ms):
        self.pulses.append((idx, delay_ms, pulse_ms))

    def __getattr__(self, name):            # anything else the wrapper might call is recorded as forbidden
        def call(*args, **kwargs):
            self.forbidden.append((name, args))
        return call


class RelayOutputTest(unittest.TestCase):
    def live(self, connected=True):
        plc = FakePLC(connected)
        out = RelayOutput(plc)
        out.dry_run, out.master_enabled = False, True
        return plc, out

    def test_safe_at_start(self):
        plc = FakePLC()
        out = RelayOutput(plc)
        self.assertTrue(out.dry_run)
        self.assertFalse(out.master_enabled)
        self.assertEqual(out.fire(5, 100, 50), ro.MASTER_OFF)
        self.assertEqual(out.test_fire(5, 50), ro.DRY_RUN)
        out.master_enabled = True
        self.assertEqual(out.fire(5, 100, 50), ro.DRY_RUN)
        self.assertEqual(plc.pulses, [])

    def test_fires_when_live_and_counts(self):
        plc, out = self.live()
        self.assertEqual(out.fire(7, 1234.4, 50), ro.SENT)
        self.assertEqual(plc.pulses, [(7, 1234, 50)])
        wanted, sent, not_sent = out.counts()
        self.assertEqual((wanted[7], sent[7]), (1, 1))
        out.dry_run = True
        out.fire(7, 10, 50)
        wanted, sent, not_sent = out.counts()
        self.assertEqual((wanted[7], sent[7], not_sent[ro.DRY_RUN]), (2, 1, 1))
        self.assertEqual(out.recent_events()[0][4], ro.DRY_RUN)

    def test_only_relays_1_to_40(self):
        plc, out = self.live()
        for relay in (0, 41, 42, 43, 48, -1, 7.0, "7", None, True):
            self.assertEqual(out.fire(relay, 0, 50), ro.REFUSED, relay)
            self.assertEqual(out.test_fire(relay, 50), ro.REFUSED, relay)
        self.assertEqual(plc.pulses, [])
        self.assertEqual(out.fire(1, 0, 50), ro.SENT)
        self.assertEqual(out.fire(40, 0, 50), ro.SENT)

    def test_all_and_sequence_use_single_pulses_of_1_to_40_only(self):
        plc, out = self.live()
        out.test_all(60)
        self.assertEqual([p[0] for p in plc.pulses], list(range(1, 41)))
        plc.pulses.clear()
        out.test_sequence(60, interval_ms=0)
        self.assertEqual([p[0] for p in plc.pulses], list(range(1, 41)))
        self.assertEqual(plc.forbidden, [])             # never fire_all, sequence_all, set_relay, set_conveyor, set_feeder

    def test_conveyor_and_feeder_coils_are_never_addressed(self):
        plc, out = self.live()
        out.test_all(60); out.test_sequence(60, interval_ms=0)
        for relay in range(1, 41):
            out.fire(relay, 0, 50)
        addresses = {plc.get_relay_address(p[0]) for p in plc.pulses}
        self.assertEqual(addresses, set(range(100, 140)))
        self.assertNotIn(140, addresses)
        self.assertNotIn(141, addresses)

    def test_plc_offline_is_reported_not_hidden(self):
        plc, out = self.live(connected=False)
        self.assertEqual(out.fire(3, 0, 50), ro.PLC_OFFLINE)
        self.assertEqual(out.counts()[2][ro.PLC_OFFLINE], 1)
        self.assertEqual(plc.pulses, [])

    def test_master_blocks_grading_but_not_a_manual_test(self):
        plc, out = self.live()
        out.master_enabled = False
        self.assertEqual(out.fire(9, 0, 50), ro.MASTER_OFF)
        self.assertEqual(out.test_fire(9, 50), ro.SENT)
        self.assertEqual(plc.pulses, [(9, 0, 50)])

    def test_describe_shows_the_coil(self):
        _, out = self.live()
        self.assertEqual(out.describe(1), "Relay 1 -> M100 (Modbus address 100)")
        self.assertEqual(out.describe(21), "Relay 21 -> M120 (Modbus address 120)")
        self.assertEqual(ro.REJECT_RELAY, {1: 1, 2: 21})



class ManualFireRuleTest(unittest.TestCase):
    """Every manual fire button is test_fire(): one rule, one wording, one log."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.plc = FakePLC()
        self.out = RelayOutput(self.plc)
        self.out.manual_log_dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self):
        import csv
        import glob
        files = glob.glob(os.path.join(self.tmp.name, "manual_fire_*.csv"))
        if not files:
            return []
        with open(files[0], encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_dry_run_blocks_single_all_and_sequence_alike(self):
        self.assertTrue(self.out.dry_run)
        self.assertEqual(self.out.test_fire(7, 100), ro.DRY_RUN)
        self.assertEqual(set(self.out.test_all(100).values()), {ro.DRY_RUN})
        self.assertEqual(set(self.out.test_sequence(100, interval_ms=0).values()), {ro.DRY_RUN})
        self.assertEqual(self.plc.pulses, [])
        self.assertEqual(self.plc.forbidden, [])
        rows = self.rows()
        self.assertEqual(len(rows), 81)
        self.assertEqual({r["sent_to_plc"] for r in rows}, {"NO"})
        self.assertEqual({r["dry_run"] for r in rows}, {"ON"})

    def test_with_dry_run_off_all_three_send_single_pulses_of_relays_1_to_40(self):
        self.out.dry_run = False
        self.out.test_fire(7, 100)
        self.out.test_all(120)
        self.out.test_sequence(90, interval_ms=0)
        self.assertEqual(self.plc.pulses[0], (7, 0, 100))
        self.assertEqual([p[0] for p in self.plc.pulses[1:41]], list(range(1, 41)))
        self.assertEqual([p[0] for p in self.plc.pulses[41:]], list(range(1, 41)))
        self.assertEqual(self.plc.forbidden, [])             # never fire_all / sequence_all / set_relay of the PLC
        rows = self.rows()
        self.assertEqual({r["sent_to_plc"] for r in rows}, {"YES"})
        self.assertEqual((rows[0]["action"], rows[0]["relay"], rows[0]["m_address"], rows[0]["pulse_ms"], rows[0]["result"]),
                         ("SINGLE RELAY TEST", "7", "M106", "100", "SENT"))
        self.assertEqual({r["action"] for r in rows[1:41]}, {"FIRE 1-40"})
        self.assertEqual(rows[40]["m_address"], "M139")
        self.assertEqual({r["action"] for r in rows[41:]}, {"SEQUENCE 1-40"})

    def test_offline_and_refused_are_logged_as_not_sent(self):
        self.out.dry_run = False
        self.plc.connected_flag = False
        self.assertEqual(self.out.test_fire(3, 100), ro.PLC_OFFLINE)
        self.assertEqual(self.out.test_fire(41, 100), ro.REFUSED)
        rows = self.rows()
        self.assertEqual([(r["result"], r["sent_to_plc"], r["plc_connected"]) for r in rows],
                         [("PLC_OFFLINE", "NO", "NO"), ("REFUSED", "NO", "NO")])
        self.assertEqual(self.plc.pulses, [])

    def test_one_wording_for_every_result(self):
        self.assertEqual(set(ro.RESULT_TEXT), {ro.SENT, ro.DRY_RUN, ro.PLC_OFFLINE, ro.REFUSED, ro.MASTER_OFF, ro.WRITE_FAIL, ro.READBACK_FAIL})
        self.assertEqual(ro.RESULT_TEXT[ro.DRY_RUN], "DRY-RUN: NOT sent to the PLC")


if __name__ == "__main__":
    unittest.main()
