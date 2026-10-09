import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import auto_learning as al
import supervisor


class AutoLearningTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "machine_config.json")
        self.logs = os.path.join(self.tmp.name, "logs")

    def tearDown(self):
        self.tmp.cleanup()

    def log(self):
        with open(os.path.join(self.logs, al.LOG_NAME), encoding="utf-8") as f:
            return f.read().splitlines()

    def test_default_is_off_with_a_target_of_200000(self):
        self.assertEqual(al.settings(self.path), {"enabled": False, "target": 200_000})

    def test_cannot_be_turned_on_before_the_target(self):
        on, message = al.set_enabled(self.path, True, 2202, self.logs)
        self.assertFalse(on)
        self.assertIn("2,202 of 200,000", message)
        self.assertIn("197,798 more", message)
        self.assertFalse(al.settings(self.path)["enabled"])
        self.assertIn("OFF -> OFF (asked for ON); samples 2202 / 200000", self.log()[0])

    def test_with_the_target_reached_on_is_only_announced(self):
        self.assertFalse(al.IMPLEMENTED)
        on, message = al.set_enabled(self.path, True, 200_000, self.logs)
        self.assertEqual((on, message), (False, "Auto learning is planned, not active in this version"))
        self.assertFalse(al.settings(self.path)["enabled"])

    def test_when_the_phase_exists_it_switches_and_persists(self):
        al.IMPLEMENTED = True
        try:
            on, _ = al.set_enabled(self.path, True, 250_000, self.logs)
            self.assertTrue(on and al.settings(self.path)["enabled"])
            off, message = al.set_enabled(self.path, False, 250_000, self.logs)
            self.assertEqual((off, message, al.settings(self.path)["enabled"]), (False, "AUTO LEARNING is OFF.", False))
            lines = self.log()
            self.assertIn("OFF -> ON", lines[0])
            self.assertIn("ON -> OFF", lines[1])
        finally:
            al.IMPLEMENTED = False

    def test_admin_can_change_the_target_and_other_settings_survive(self):
        supervisor.set_password(self.path, "super-pw")
        self.assertEqual(al.set_target(self.path, 5000, 2202, self.logs), 5000)
        self.assertEqual(al.settings(self.path)["target"], 5000)
        self.assertTrue(supervisor.check(self.path, "super-pw"))           # the other entries of the file are kept
        self.assertRaises(ValueError, al.set_target, self.path, 0, 2202, self.logs)
        self.assertIn("target 200000 -> 5000; samples 2202", self.log()[0])
        on, message = al.set_enabled(self.path, True, 4999, self.logs)
        self.assertIn("4,999 of 5,000", message)

    def test_admin_password_is_separate_and_never_stored_or_logged(self):
        supervisor.set_password(self.path, "super-pw")
        self.assertFalse(supervisor.is_set(self.path, supervisor.ADMIN_KEY))
        supervisor.set_password(self.path, "admin-secret", supervisor.ADMIN_KEY)
        self.assertTrue(supervisor.check(self.path, "admin-secret", supervisor.ADMIN_KEY))
        self.assertFalse(supervisor.check(self.path, "super-pw", supervisor.ADMIN_KEY))
        self.assertFalse(supervisor.check(self.path, "admin-secret"))      # not valid as the supervisor password
        al.set_enabled(self.path, True, 10, self.logs)
        al.set_target(self.path, 300, 10, self.logs)
        with open(self.path, encoding="utf-8") as f:
            stored = f.read()
        text = stored + "\n".join(self.log())
        self.assertNotIn("admin-secret", text)
        self.assertNotIn("super-pw", text)
        self.assertEqual(set(json.loads(stored)), {"supervisor_password", "admin_password", "auto_learning"})

    def test_samples_are_dataset_plus_test_set(self):
        for folder, count in (("dataset/GOOD_CASHEW", 3), ("dataset/UPW", 2), ("testset/GOOD_CASHEW", 1)):
            os.makedirs(os.path.join(self.tmp.name, folder))
            for n in range(count):
                open(os.path.join(self.tmp.name, folder, f"{n}.jpg"), "wb").close()
        self.assertEqual(al.sample_count(os.path.join(self.tmp.name, "dataset"), os.path.join(self.tmp.name, "testset")), 6)


if __name__ == "__main__":
    unittest.main()
