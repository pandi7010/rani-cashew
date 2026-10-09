import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import colour_size as cs

GRADES = {
    "W180": {"l_min": 28.0, "l_max": 34.0, "w_min": 13.0, "w_max": 18.0, "a_min": 300.0, "a_max": 450.0},
    "W240": {"l_min": 24.0, "l_max": 29.0, "w_min": 11.0, "w_max": 16.0, "a_min": 220.0, "a_max": 330.0},
    "W320": {"l_min": 21.0, "l_max": 25.0, "w_min": 10.0, "w_max": 14.0, "a_min": 160.0, "a_max": 250.0},
    "SW240": {"l_min": 23.0, "l_max": 28.0},
    "BUDDS": {"l_min": 8.0, "l_max": 16.0},
}


class ColourTest(unittest.TestCase):
    def test_colour_is_the_median_over_the_frames_with_a_mask(self):
        frames = [{"L": 60, "a": -4, "b": 20}, None, {"L": 62, "a": -5, "b": 30}, {"L": 61, "a": -3, "b": 22}]
        self.assertEqual(cs.track_colour(frames), {"L": 61.0, "a": -4.0, "b": 22.0})
        self.assertIsNone(cs.track_colour([None, None]))

    def test_threshold_and_side(self):
        values = {"L": 60.0, "a": -4.0, "b": 26.0}
        self.assertEqual(cs.colour_of(values, {"metric": "b", "threshold": 24.0, "sw_above": True}), (cs.SW, 26.0))
        self.assertEqual(cs.colour_of(values, {"metric": "b", "threshold": 24.0, "sw_above": False}), (cs.WW, 26.0))
        self.assertEqual(cs.colour_of(values, {"metric": "L", "threshold": 63.0, "sw_above": False}), (cs.SW, 60.0))
        self.assertEqual(cs.colour_of(values, {"metric": "b", "threshold": None}), (None, 26.0))       # not taught
        self.assertEqual(cs.colour_of(None, {"metric": "b", "threshold": 24.0}), (None, None))         # no mask

    def test_teach_suggests_metric_threshold_and_side(self):
        rng = random.Random(1)
        ww = [{"L": rng.gauss(68, 3), "a": rng.gauss(-6, 1.0), "b": rng.gauss(22, 3)} for _ in range(60)]
        sw = [{"L": rng.gauss(58, 3), "a": rng.gauss(-3, 1.0), "b": rng.gauss(23, 3)} for _ in range(60)]
        result = cs.suggest(ww, sw)
        self.assertIn(result["best"], ("L", "a"))                          # b* does not separate these two groups
        self.assertGreater(result["metrics"]["b"]["wrong_pct"], 30)
        best = result["metrics"][result["best"]]
        self.assertLess(best["wrong_pct"], 12)
        self.assertEqual(result["metrics"]["L"]["sw_above"], False)        # scorched is darker
        self.assertEqual(result["metrics"]["a"]["sw_above"], True)         # and redder
        self.assertTrue(58 < result["metrics"]["L"]["threshold"] < 68)
        self.assertIsNone(cs.suggest(ww[:19], sw))                          # needs 20 of each


class SizeTest(unittest.TestCase):
    def test_grade_from_taught_limits_of_the_cashews_colour(self):
        self.assertEqual(cs.grade_of(cs.WW, 31.0, 15.0, 380.0, GRADES), ("W180", ""))
        self.assertEqual(cs.grade_of(cs.WW, 22.0, 12.0, 200.0, GRADES), ("W320", ""))
        self.assertEqual(cs.grade_of(cs.SW, 25.0, 12.0, 250.0, GRADES), ("SW240", ""))        # SW uses SW limits only
        self.assertEqual(cs.grade_of(cs.SW, 31.0, 15.0, 380.0, GRADES), (None, cs.OUT_OF_RANGE))

    def test_overlapping_grades_give_the_best_fit(self):
        self.assertEqual(cs.grade_of(cs.WW, 24.5, 12.0, 225.0, GRADES)[0], "W320")            # fits W240 and W320
        self.assertEqual(cs.grade_of(cs.WW, 28.2, 14.0, 300.0, GRADES)[0], "W240")            # fits W180 and W240

    def test_outside_every_grade_budds_and_not_taught(self):
        self.assertEqual(cs.grade_of(cs.WW, 40.0, 15.0, 380.0, GRADES), (None, cs.OUT_OF_RANGE))
        self.assertEqual(cs.grade_of(cs.WW, 26.0, 20.0, 280.0, GRADES), (None, cs.OUT_OF_RANGE))   # width outside
        self.assertEqual(cs.grade_of(cs.WW, 12.0, 8.0, 80.0, GRADES), ("BUDDS", ""))
        self.assertEqual(cs.grade_of(cs.SW, 12.0, 8.0, 80.0, GRADES), ("BUDDS", ""))
        self.assertEqual(cs.mapping_name(cs.SW, "BUDDS"), "SW_BUDDS")
        self.assertEqual(cs.mapping_name(cs.WW, "W240"), "W240")
        self.assertEqual(cs.grade_of(cs.SW, 25.0, 12.0, 250.0, {"W240": GRADES["W240"], "BUDDS": GRADES["BUDDS"]}), (None, cs.NOT_TAUGHT))
        self.assertEqual(cs.grade_of(cs.WW, 25.0, 12.0, 250.0, {}), (None, cs.NOT_TAUGHT))

    def test_copy_w_limits_to_sw(self):
        out, copied = cs.copy_w_to_sw(GRADES)
        self.assertEqual(copied, ["SW180", "SW240", "SW320"])
        self.assertEqual({k: out["SW180"][k] for k in cs.LIMIT_KEYS}, GRADES["W180"])
        self.assertEqual(out["SW240"]["w_max"], 16.0)                       # the old SW240 row is replaced
        self.assertNotIn("SW210", out)
        self.assertEqual(GRADES["SW240"], {"l_min": 23.0, "l_max": 28.0})   # the input is not changed

    def test_would_be_text(self):
        self.assertEqual(cs.would_be(cs.SW, "SW210", ""), "SW210")
        self.assertEqual(cs.would_be(cs.SW, "BUDDS", ""), "SW_BUDDS")
        self.assertEqual(cs.would_be(cs.WW, None, cs.OUT_OF_RANGE), "WW out of range")
        self.assertEqual(cs.would_be(None, None, cs.NOT_TAUGHT), "? not taught")


if __name__ == "__main__":
    unittest.main()
