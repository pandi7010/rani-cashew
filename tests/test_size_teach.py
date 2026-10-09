import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import size_teach


class SizeTeachTest(unittest.TestCase):
    def samples(self, n):
        return [{"l": 24.0 + (i % 3 - 1), "w": 12.0 + (i % 2) * 0.5, "a": 220.0 + (i % 5) * 4} for i in range(n)]

    def test_needs_30_cashews(self):
        self.assertIsNone(size_teach.limits(self.samples(29)))
        self.assertIsNotNone(size_teach.limits(self.samples(30)))

    def test_limits_are_mean_plus_minus_k_std(self):
        data = self.samples(60)
        mean, std = size_teach.mean_std(s["l"] for s in data)
        out = size_teach.limits(data, k=3.0)
        self.assertEqual((out["l_min"], out["l_max"]), (round(mean - 3 * std, 2), round(mean + 3 * std, 2)))
        self.assertEqual((out["n"], out["k"]), (60, 3.0))
        wide, narrow = size_teach.limits(data, k=3.0), size_teach.limits(data, k=1.0)
        self.assertLess(wide["a_min"], narrow["a_min"])
        self.assertGreater(wide["w_max"], narrow["w_max"])

    def test_doubles_and_unmeasured_cashews_do_not_teach(self):
        data = self.samples(30) + [{"l": 60.0, "w": 14.0, "a": 500.0, "double": True}, {"l": 0, "w": 0, "a": 0}]
        self.assertEqual(len(size_teach.usable(data)), 30)
        self.assertLess(size_teach.limits(data)["l_max"], 30.0)
        self.assertGreaterEqual(size_teach.limits([{"l": 1.0, "w": 1.0, "a": 1.0 + i} for i in range(40)])["a_min"], 0.0)


if __name__ == "__main__":
    unittest.main()
