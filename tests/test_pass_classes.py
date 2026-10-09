import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import colour_size as cs
import decision as dc
import relay_mapping
from decision import FrameVote

VOTING = {"neck_min_frames": 0, "single_defect_blocks_good": False}        # the machine's start values
TABLES = relay_mapping.lookup(relay_mapping.owner_mapping())                 # the owner's mapping: TESTA SMALL = 9 / 29, W240 = 13 / 33
NO_FIRE = relay_mapping.NO_FIRE_DEFAULT


def cashew(name, n=8, conf=0.95):
    return [FrameVote(float(i), name, conf, None, 0.0, view="OTHER") for i in range(n)]


def route(votes, good_classes, roi=1, length_mm=27.0, colour_cfg=None, lab=None):
    """The flow of the grader for one cashew: decision -> (PASS / GOOD: colour -> size grade -> grade ejector) or the
    class's own ejector. Returns (shown text, ejector)."""
    d = dc.decide(votes, VOTING, dc.PRIORITY_DEFAULT, good_classes)
    if d.reject:
        return d.result, relay_mapping.REJECT_RELAY[roi]
    if d.result == dc.GOOD:
        colour, _ = cs.colour_of(lab, colour_cfg)
        grade, why = cs.grade_of(colour, length_mm, 0.0, 0.0, None)          # the start table
        if grade is None:
            return why, relay_mapping.REJECT_RELAY[roi]
        relay, _ = relay_mapping.target_of(TABLES[roi], roi, [grade, colour, "GOOD_CASHEW", "GOOD"], NO_FIRE)
        name = dc.passed_class(d.votes, good_classes)
        return (f"{name} (PASS) -> {grade}" if name else f"GOOD {grade}"), relay
    relay, _ = relay_mapping.target_of(TABLES[roi], roi, [d.result], NO_FIRE)
    return d.result, relay


class PassClassTest(unittest.TestCase):
    def test_testa_small_as_fire_goes_to_its_own_ejector_9(self):
        good = list(dc.GOOD_CLASSES_DEFAULT)
        self.assertEqual(route(cashew("TESTA SMALL"), good, roi=1), ("TESTA SMALL", 9))
        self.assertEqual(route(cashew("TESTA SMALL"), good, roi=2), ("TESTA SMALL", 29))

    def test_testa_small_as_pass_goes_to_the_ejector_of_its_size_grade(self):
        good = list(dc.GOOD_CLASSES_DEFAULT) + ["TESTA SMALL"]
        self.assertEqual(route(cashew("TESTA SMALL"), good, roi=1, length_mm=27.0), ("TESTA SMALL (PASS) -> W240", 13))
        self.assertEqual(route(cashew("TESTA SMALL"), good, roi=2, length_mm=27.0), ("TESTA SMALL (PASS) -> W240", 33))
        self.assertEqual(route(cashew("TESTA SMALL"), good, roi=1, length_mm=31.0), ("TESTA SMALL (PASS) -> W180", 11))
        # with the colour split on, a scorched one uses the SW grade and its ejector
        split = {"enabled": True, "metric": "b", "threshold": 23.0, "sw_above": False}
        self.assertEqual(route(cashew("TESTA SMALL"), good, length_mm=27.0, colour_cfg=split, lab={"L": 58.0, "a": -3.0, "b": 20.0}),
                         ("TESTA SMALL (PASS) -> SW240", 18))
        # a BUDDS-size piece is not fired
        self.assertEqual(route(cashew("TESTA SMALL"), good, length_mm=12.0), ("TESTA SMALL (PASS) -> BUDDS", None))

    def test_good_cashew_is_shown_as_good(self):
        good = list(dc.GOOD_CLASSES_DEFAULT) + ["TESTA SMALL"]
        self.assertEqual(route(cashew("GOOD_CASHEW"), good, length_mm=25.0), ("GOOD W320", 14))
        mixed = cashew("GOOD_CASHEW", 6) + cashew("TESTA SMALL", 2)
        self.assertEqual(route(mixed, good, length_mm=25.0), ("GOOD W320", 14))              # mostly seen as GOOD_CASHEW

    def test_other_defects_stay_fire_when_one_class_is_pass(self):
        good = list(dc.GOOD_CLASSES_DEFAULT) + ["TESTA SMALL"]
        self.assertEqual(route(cashew("UPW"), good), ("UPW", 2))
        self.assertEqual(route(cashew("TESTA"), good), ("TESTA", 10))

    def test_rejection_double_and_undefined_can_never_be_pass(self):
        for name in ("REJECTION", "DOUBLE", "DOUBLE KERNEL", "UNDEFINED", "EDGECUT"):
            self.assertIn("cannot be set to PASS", dc.can_change_pass(name, ["EDGECUT"]), name)
        for name in ("GOOD_CASHEW", "GOOD"):
            self.assertIn("always PASS", dc.can_change_pass(name))
        for name in ("TESTA SMALL", "UPW", "SW", "KNIEF_MARK"):
            self.assertEqual(dc.can_change_pass(name, ["EDGECUT"]), "")
        # even if REJECTION were written into the PASS list by hand, the cashew is still rejected
        forced = list(dc.GOOD_CLASSES_DEFAULT) + ["REJECTION", "DOUBLE"]
        self.assertEqual(route(cashew("REJECTION"), forced, roi=1)[1], 1)
        self.assertEqual(route(cashew("DOUBLE"), forced, roi=2)[1], 21)

    def test_passed_class_name(self):
        good = ["GOOD_CASHEW", "GOOD", "SW", "TESTA SMALL"]
        self.assertEqual(dc.passed_class({"TESTA SMALL": 7, "GOOD_CASHEW": 1}, good), "TESTA SMALL")
        self.assertEqual(dc.passed_class({"GOOD_CASHEW": 8}, good), "")
        self.assertEqual(dc.passed_class({"TESTA SMALL": 4, "GOOD_CASHEW": 4}, good), "")            # a tie is GOOD
        self.assertEqual(dc.passed_class({"UPW": 3}, good), "")
        self.assertEqual(dc.passed_class({}, good), "")


if __name__ == "__main__":
    unittest.main()
