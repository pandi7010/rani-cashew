import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plc_mapping as pm


class PlcMappingTest(unittest.TestCase):
    def test_default_table_is_m100_to_m139(self):
        table = pm.default_table()
        self.assertEqual((table[1], table[21], table[40], len(table)), (100, 120, 139, 40))
        self.assertEqual(pm.problems(table), [])

    def test_conveyor_and_feeder_can_never_be_ejectors(self):
        table = pm.default_table()
        table[7], table[8] = 140, 141
        faults = pm.problems(table)
        self.assertEqual(len(faults), 2)
        self.assertIn("Ejector 7: M140 conveyor can never be an ejector", faults[0])
        self.assertIn("Ejector 8: M141 feeder", faults[1])
        table = pm.default_table()
        table[3] = 200
        self.assertEqual(pm.problems(table), [])
        self.assertIn("Ejector 3", pm.problems(table, extra_protected=[200])[0])        # where the conveyor really is

    def test_two_ejectors_on_one_coil_and_bad_values_block(self):
        table = pm.default_table()
        table[5] = table[6]
        self.assertEqual(pm.problems(table), ["M105 is given to ejectors 5, 6 (each ejector needs its own coil)"])
        table = pm.default_table()
        table[2], table[9] = "abc", 70000
        self.assertEqual(len(pm.problems(table)), 2)

    def test_changes_are_logged_with_time_old_and_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, new = pm.default_table(), pm.default_table()
            new[4], new[30] = 204, 230
            self.assertEqual(pm.log_changes(tmp, old, new), ["ejector 4: M103 -> M204", "ejector 30: M129 -> M230"])
            with open(os.path.join(tmp, pm.LOG_NAME), encoding="utf-8") as f:
                lines = f.read().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertRegex(lines[0], r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d  ejector 4: M103 -> M204$")
            self.assertEqual(pm.log_changes(tmp, new, new), [])


if __name__ == "__main__":
    unittest.main()
