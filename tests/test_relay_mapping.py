import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import relay_mapping as rm

BASE = {2: ["REJECTION"], 3: ["TESTA"], 5: ["PKW2"], 6: ["PKW1"], 10: ["DW1", "DW2"], 12: ["SW180", "W180"], 13: ["W210"], 17: ["BUDDS"]}


class RelayMappingTest(unittest.TestCase):
    def test_one_ejector_carries_several_classes(self):
        table = rm.lookup(BASE)[1]
        self.assertEqual((table["dw1"], table["dw2"]), (10, 10))
        self.assertEqual(rm.resolve(table, rm.grade_candidates("WW", "W180")), 12)
        self.assertEqual(rm.resolve(table, rm.grade_candidates("SW", "W180")), 12)      # W180 + SW180 on one ejector
        self.assertEqual(rm.label(BASE, 10), "DW1, DW2")
        self.assertEqual(rm.duplicates(BASE), {})

    def test_combining_moves_a_class_and_frees_the_old_ejector(self):
        self.assertEqual(rm.conflicts(BASE, 6, ["PKW1", "PKW2"]), {"PKW2": 5})
        self.assertRaises(ValueError, rm.assign, BASE, 6, ["PKW1", "PKW2"], False, False)     # not without "move it here"
        m = rm.assign(BASE, 6, ["PKW1", "PKW2"], move=True, mirror_roi=False)
        self.assertEqual((m[6], m[5]), (["PKW1", "PKW2"], []))
        self.assertEqual(rm.label(m, 5), "(free)")
        self.assertIn(5, rm.free_relays(m, 1))
        self.assertEqual(rm.resolve(rm.lookup(m)[1], ["PKW2"]), 6)
        self.assertEqual(rm.duplicates(m), {})
        self.assertEqual(BASE[5], ["PKW2"])                                             # the input is not changed

    def test_moving_a_class_between_ejectors(self):
        m = rm.assign(BASE, 7, ["DW2"], move=True, mirror_roi=False)
        self.assertEqual((m[10], m[7]), (["DW1"], ["DW2"]))
        m = rm.assign(m, 7, [], mirror_roi=False)                                       # empty selection frees it
        self.assertEqual(rm.label(m, 7), "(free)")
        self.assertEqual(rm.unassigned(m, ["DW1", "DW2"]), ["DW2"])

    def test_duplicate_is_found_and_names_ignore_case_and_underscores(self):
        broken = {**BASE, 8: ["testa", "TESTA_SMALL"], 9: ["TESTA SMALL"]}
        self.assertEqual(rm.duplicates(broken), {1: {"TESTA": [3, 8], "TESTA_SMALL": [8, 9]}})
        self.assertEqual(rm.lookup(broken)[1]["testa"], 3)                              # lowest ejector until it is repaired
        self.assertEqual(rm.clean({4: ["OIL", "oil", "OIL"]})[4], ["OIL"])
        both = {3: ["TESTA"], 23: ["TESTA"]}                                            # the same class in both ROIs is normal
        self.assertEqual(rm.duplicates(both), {})

    def test_unassigned_goes_to_the_reject_ejector(self):
        tables = rm.lookup(BASE)
        self.assertEqual(rm.target_of(tables[1], 1, ["UPW"]), (1, False))
        self.assertEqual(rm.target_of(tables[2], 2, ["UPW"]), (21, False))
        self.assertEqual(rm.target_of(tables[1], 1, ["TESTA"]), (3, True))
        self.assertEqual(rm.unassigned(BASE, ["UPW", "TESTA", "OIL"]), ["UPW", "OIL"])
        self.assertEqual(rm.unassigned(BASE, [], rm.GRADES_WW + rm.GRADES_SW),
                         ["W240", "W320", "W450", "SW210", "SW240", "SW320", "SW450", "SW_BUDDS"])
        with_group = rm.assign(BASE, 11, ["SW"], mirror_roi=False)                      # the group "SW" catches every SW grade
        self.assertEqual(rm.unassigned(with_group, [], rm.GRADES_SW), [])
        self.assertEqual(rm.resolve(rm.lookup(with_group)[1], rm.grade_candidates("SW", "W210")), 11)
        self.assertEqual(rm.resolve(rm.lookup(with_group)[1], rm.grade_candidates("SW", "W180")), 12)     # its own grade first

    def test_reject_ejectors_take_no_class(self):
        self.assertRaises(ValueError, rm.assign, BASE, 1, ["UPW"])
        self.assertRaises(ValueError, rm.assign, BASE, 21, ["UPW"])
        self.assertRaises(ValueError, rm.assign, BASE, 41, ["UPW"])
        stored = {1: ["UPW"], 21: ["UPW"], 4: ["OIL"]}
        self.assertEqual(rm.wrongly_on_reject(stored), {1: ["UPW"], 21: ["UPW"]})
        self.assertNotIn("upw", rm.lookup(stored)[1])                                   # ignored: UPW is not assigned
        self.assertEqual(rm.label(stored, 1), "REJECT - FIXED")

    def test_roi2_mirror_with_a_combined_row(self):
        start = {7: ["OIL"], 27: ["OIL"], 10: ["DW1"], 30: ["DW1"], 28: ["DW2"]}
        m = rm.assign(start, 10, ["DW1", "DW2", "OIL"], move=True)
        self.assertEqual(m[10], ["DW1", "DW2", "OIL"])
        self.assertEqual(m[30], ["DW1", "DW2", "OIL"])                                  # ejector + 20
        self.assertEqual((m[7], m[27], m[28]), ([], [], []))                            # moved away in both ROIs
        self.assertEqual((rm.mirror(10), rm.mirror(30), rm.roi_of(20), rm.roi_of(21)), (30, 10, 1, 2))
        tables = rm.lookup(m)
        self.assertEqual((tables[1]["dw2"], tables[2]["dw2"], tables[2]["oil"]), (10, 30, 30))
        only_roi1 = rm.assign(start, 10, ["DW1", "OIL"], move=True, mirror_roi=False)
        self.assertEqual((only_roi1[10], only_roi1[30], only_roi1[27]), (["DW1", "OIL"], ["DW1"], ["OIL"]))

    def test_summary_table(self):
        m = rm.assign(BASE, 11, ["SW"], mirror_roi=False)
        rows = dict((r[0], r[1:]) for r in rm.summary(m, ["UPW", "DW1", "DW2"], ["W180", "SW180", "SW450", "W450"]))
        self.assertEqual(rows["UPW"], ("REJECT A01", "REJECT A21"))
        self.assertEqual(rows["DW1"], ("A10", "REJECT A21"))
        self.assertEqual(rows["DW2"][0], "A10")
        self.assertEqual((rows["W180"][0], rows["SW180"][0]), ("A12", "A12"))
        self.assertEqual(rows["SW450"][0], "A11 (via SW)")
        self.assertEqual(rows["W450"][0], "REJECT A01")

    def test_any_combination_works_without_code_changes(self):
        # an arbitrary mapping with invented class names: nothing in the rules knows a class name or an ejector number
        m = rm.assign(rm.assign({}, 19, ["ALPHA", "BETA"]), 2, ["GAMMA"])
        tables = rm.lookup(m)
        self.assertEqual((tables[1]["alpha"], tables[2]["beta"], tables[2]["gamma"]), (19, 39, 22))
        self.assertEqual(rm.target_of(tables[1], 1, ["DELTA"]), (1, False))

    def test_the_stored_config_has_no_duplicate(self):
        with open(os.path.join(ROOT, "pneumatic_config.json"), encoding="utf-8") as f:
            mapping = json.load(f)["mapping"]
        self.assertEqual(rm.duplicates(mapping), {})
        self.assertEqual(rm.wrongly_on_reject(mapping), {})


if __name__ == "__main__":
    unittest.main()
