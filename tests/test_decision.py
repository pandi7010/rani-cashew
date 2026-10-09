import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import decision as dc
from decision import FrameVote

# These tests check the rules themselves, with the NECK rule and the single-defect rule switched ON.
# The start values of the machine (both off, owner 2026-10-09) are checked in StartValueTest.
STRICT = {"neck_min_frames": 1, "single_defect_blocks_good": True}


def decide(votes, voting=None, *args, **kwargs):
    return dc.decide(votes, {**STRICT, **(voting or {})}, *args, **kwargs)


def frames(*specs):
    """spec = (cnn_class, conf) or (cnn_class, conf, view) or (cnn_class, conf, view, yolo_class, yolo_conf)."""
    out = []
    for n, spec in enumerate(specs):
        spec = tuple(spec) + ("OTHER", None, 0.0)[len(spec) - 2:]
        out.append(FrameVote(float(n), spec[0], spec[1], spec[3], spec[4], view=spec[2]))
    return out


GOOD3_NECK = [("GOOD_CASHEW", 0.9), ("GOOD_CASHEW", 0.95, "NECK"), ("GOOD_CASHEW", 0.85)]


class DecisionTest(unittest.TestCase):
    def test_good_needs_three_frames_and_the_neck_view(self):
        d = decide(frames(*GOOD3_NECK))
        self.assertEqual((d.result, d.reject), (dc.GOOD, False))

    def test_good_without_neck_is_not_rolled(self):
        d = decide(frames(("GOOD_CASHEW", 0.9, "TOP"), ("GOOD_CASHEW", 0.9, "BACK"), ("GOOD_CASHEW", 0.9, "TOP")))
        self.assertEqual((d.result, d.reason, d.reject), (dc.UNDEFINED, dc.NOT_ROLLED, True))

    def test_neck_requirement_can_be_switched_off(self):
        d = decide(frames(("GOOD_CASHEW", 0.9), ("GOOD_CASHEW", 0.9), ("GOOD_CASHEW", 0.9)), {"neck_min_frames": 0})
        self.assertEqual(d.result, dc.GOOD)

    def test_low_confidence_and_too_few_frames_are_undefined(self):
        d = decide(frames(("GOOD_CASHEW", 0.7, "NECK"), ("GOOD_CASHEW", 0.9), ("GOOD_CASHEW", 0.9)))
        self.assertEqual((d.result, d.reason), (dc.UNDEFINED, dc.LOW_CONFIDENCE))
        d = decide(frames(("GOOD_CASHEW", 0.9, "NECK"), ("GOOD_CASHEW", 0.9)))
        self.assertEqual((d.result, d.reason), (dc.UNDEFINED, dc.TOO_FEW_FRAMES))
        self.assertEqual(decide([]).reason, dc.NO_FRAMES)

    def test_cnn_sw_counts_as_no_defect(self):
        d = decide(frames(("SW", 0.9), ("GOOD_CASHEW", 0.9, "NECK"), ("SW", 0.85)))
        self.assertEqual(d.result, dc.GOOD)
        self.assertEqual(d.votes["SW"], 2)

    def test_defect_needs_two_frames_at_half_confidence(self):
        d = decide(frames(*GOOD3_NECK, ("PKW1", 0.6), ("PKW1", 0.55)))
        self.assertEqual((d.result, d.reject), ("PKW1", False))
        d = decide(frames(*GOOD3_NECK, ("PKW1", 0.6), ("PKW1", 0.45)))
        self.assertNotEqual(d.result, "PKW1")

    def test_one_defect_frame_is_neither_defect_nor_good(self):
        votes = frames(*GOOD3_NECK, ("BROWN", 0.9))
        d = decide(votes)
        self.assertEqual((d.result, d.reason, d.reject), (dc.UNDEFINED, dc.SINGLE_DEFECT_FRAME, True))
        self.assertEqual(decide(votes, {"single_defect_blocks_good": False}).result, dc.GOOD)

    def test_priority_order(self):
        d = decide(frames(("PKW1", 0.9), ("PKW1", 0.9), ("PKW2", 0.9), ("PKW2", 0.9)))
        self.assertEqual(d.result, "PKW2")
        d = decide(frames(("DW1", 0.9), ("DW1", 0.9), ("BROWN", 0.9), ("BROWN", 0.9), ("OIL", 0.9), ("OIL", 0.9)))
        self.assertEqual(d.result, "OIL")
        d = decide(frames(("WHITE_DW", 0.9), ("WHITE_DW", 0.9), ("DW2", 0.9), ("DW2", 0.9)))
        self.assertEqual(d.result, "DW2")
        d = decide(frames(("TESTA SMALL", 0.9), ("TESTA SMALL", 0.9), ("TESTA", 0.9), ("TESTA", 0.9)))
        self.assertEqual(d.result, "TESTA")

    def test_priority_order_can_be_changed(self):
        votes = frames(("PKW1", 0.9), ("PKW1", 0.9), ("OIL", 0.9), ("OIL", 0.9))
        self.assertEqual(decide(votes).result, "PKW1")
        self.assertEqual(decide(votes, priority=["OIL", "PKW1"]).result, "OIL")

    def test_skin_alone_stays_testa(self):
        self.assertEqual(decide(frames(*GOOD3_NECK, ("TESTA", 0.8), ("TESTA", 0.8))).result, "TESTA")
        self.assertEqual(decide(frames(*GOOD3_NECK, ("TESTA_SMALL", 0.8), ("testa small", 0.8))).result, "TESTA SMALL")

    def test_skin_with_any_other_defect_is_upw(self):
        for other in ("PKW1", "OIL", "BROWN", "DW1", "WHITE_DW"):
            d = decide(frames(("TESTA SMALL", 0.8), ("TESTA SMALL", 0.8), (other, 0.8), (other, 0.8)))
            self.assertEqual((d.result, d.reason), ("UPW", "skin_with_other_defect"), other)
        self.assertEqual(decide(frames(("UPW", 0.8), ("UPW", 0.8), ("PKW2", 0.9), ("PKW2", 0.9))).result, "UPW")

    def test_neck_view_gives_skin_extra_weight(self):
        d = decide(frames(*GOOD3_NECK, ("TESTA", 0.7, "NECK")))
        self.assertEqual(d.result, "TESTA")                     # one NECK frame counts as two
        d = decide(frames(*GOOD3_NECK, ("TESTA", 0.7, "TOP")))
        self.assertEqual(d.reason, dc.SINGLE_DEFECT_FRAME)
        d = decide(frames(*GOOD3_NECK, ("PKW1", 0.7, "NECK")))
        self.assertEqual(d.reason, dc.SINGLE_DEFECT_FRAME)      # the extra weight is for skin only
        d = decide(frames(*GOOD3_NECK, ("TESTA", 0.7, "NECK")), {"neck_weight": 1})
        self.assertEqual(d.reason, dc.SINGLE_DEFECT_FRAME)

    def test_yolo_rejection_wins_over_everything(self):
        votes = frames(("GOOD_CASHEW", 0.99, "NECK", "REJECTION", 0.6), ("GOOD_CASHEW", 0.99, "TOP", "REJECTION", 0.7),
                       ("UPW", 0.9), ("UPW", 0.9), ("GOOD_CASHEW", 0.99))
        d = decide(votes)
        self.assertEqual((d.result, d.reject), (dc.REJECTION, True))

    def test_yolo_rejection_needs_two_frames(self):
        d = decide(frames(("GOOD_CASHEW", 0.99, "NECK", "REJECTION", 0.9), ("GOOD_CASHEW", 0.99), ("GOOD_CASHEW", 0.99)))
        self.assertEqual(d.result, dc.GOOD)
        d = decide(frames(("GOOD_CASHEW", 0.9, "NECK", "REJECTION", 0.4), ("GOOD_CASHEW", 0.9, "TOP", "REJECTION", 0.4), ("GOOD_CASHEW", 0.9)))
        self.assertEqual(d.result, dc.GOOD)                     # below the rejection confidence

    def test_cnn_rejection_counts_once_the_cnn_has_that_class(self):
        self.assertEqual(decide(frames(("REJECTION", 0.8), ("REJECTION", 0.6))).result, dc.REJECTION)

    def test_yolo_class_is_used_when_the_cnn_gave_none(self):
        d = decide(frames((None, 0.0, "OTHER", "DW1", 0.7), (None, 0.0, "OTHER", "DW1", 0.8)))
        self.assertEqual(d.result, "DW1")

    def test_double_kernel(self):
        d = decide(frames(*GOOD3_NECK), double=True)
        self.assertEqual((d.result, d.reject), (dc.DOUBLE, True))

    def test_edge_cut_frames_do_not_vote(self):
        votes = frames(*GOOD3_NECK, ("PKW1", 0.9), ("PKW1", 0.9))
        votes[3].usable = votes[4].usable = False
        d = decide(votes)
        self.assertEqual((d.result, d.frames_total, d.frames_used), (dc.GOOD, 5, 3))

    def test_class_outside_the_priority_list_is_undefined(self):
        d = decide(frames(("EDGECUT", 0.9), ("EDGECUT", 0.9)))
        self.assertEqual((d.result, d.reason, d.reject), (dc.UNDEFINED, dc.NOT_IN_PRIORITY, True))

    def test_back_cut_oil_is_graded_as_oil_and_edgecut_is_rejected(self):
        d = decide(frames(("BACK_CUT_OIL", 0.9), ("OIL_BACKCUT", 0.9)))
        self.assertEqual((d.result, d.reject), ("OIL", False))
        self.assertEqual(d.votes, {"OIL": 2})
        d = decide(frames(("TESTA", 0.9), ("TESTA", 0.9), ("BACK_CUT_OIL", 0.9), ("OIL", 0.9)))
        self.assertEqual(d.result, "UPW")                       # skin + OIL
        d = decide(frames(("EDGECUT", 0.9), ("EDGECUT", 0.9), ("GOOD_CASHEW", 0.9)))
        self.assertEqual((d.result, d.reject), (dc.UNDEFINED, True))

    def test_cnn_class_double_goes_to_the_reject_ejector(self):
        for name in ("DOUBLE", "DOUBLE_KERNELS", "double cashews"):
            d = decide(frames(*GOOD3_NECK, (name, 0.8), (name, 0.7)))
            self.assertEqual((d.result, d.reason, d.reject), (dc.DOUBLE, "double_class_seen", True), name)
        d = decide(frames(*GOOD3_NECK, ("DOUBLE", 0.8)), {"single_defect_blocks_good": False})      # one frame is not enough
        self.assertEqual(d.result, dc.GOOD)
        d = decide(frames(("DOUBLE", 0.8), ("DOUBLE", 0.8), ("UPW", 0.9), ("UPW", 0.9)))
        self.assertEqual(d.result, dc.DOUBLE)                   # before the priority list

    def test_owner_combination_rule(self):
        rules = [{"conditions": ["PKW1", "BROWN", "", ""], "final": "DW2"}]
        d = decide(frames(("PKW1", 0.9), ("PKW1", 0.9), ("BROWN", 0.9), ("BROWN", 0.9)), combination_rules=rules)
        self.assertEqual((d.result, d.reason), ("DW2", "combination_rule"))
        d = decide(frames(("PKW1", 0.9), ("PKW1", 0.9)), combination_rules=rules)
        self.assertEqual(d.result, "PKW1")


class StartValueTest(unittest.TestCase):
    def test_neck_rule_is_on_hold_but_still_noted(self):
        self.assertEqual(dc.VOTING_DEFAULTS["neck_min_frames"], 0)
        no_neck = frames(("GOOD_CASHEW", 0.9, "TOP"), ("GOOD_CASHEW", 0.9, "BACK"), ("GOOD_CASHEW", 0.9, "TOP"))
        for mode in (dc.ACTIVE, dc.SHADOW):
            d = dc.decide(no_neck, view_engine=mode)
            self.assertEqual((d.result, d.reject, d.shadow), (dc.GOOD, False, [dc.VIEW_NOT_ROLLED]), mode)
        self.assertEqual(dc.decide(no_neck, view_engine=dc.OFF).shadow, [])
        self.assertEqual(dc.decide(frames(*GOOD3_NECK)).shadow, [])

    def test_one_defect_frame_does_not_block_good(self):
        self.assertFalse(dc.VOTING_DEFAULTS["single_defect_blocks_good"])
        d = dc.decide(frames(*GOOD3_NECK, ("BROWN", 0.9)))
        self.assertEqual((d.result, d.reject), (dc.GOOD, False))
        d = dc.decide(frames(*GOOD3_NECK, ("BROWN", 0.9), ("BROWN", 0.9)))
        self.assertEqual(d.result, "BROWN")                     # two frames are still a defect


if __name__ == "__main__":
    unittest.main()
