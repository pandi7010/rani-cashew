import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from class_registry import ClassRegistry, normalize_name

YOLO = {0: "GOOD_CASHEW", 1: "TESTA", 2: "UPW", 3: "DW1", 4: "DW2", 5: "WHITE_DW", 6: "SW", 7: "BROWN",
        8: "REJECTION", 9: "PKW1", 10: "PKW2", 11: "TESTA SMALL"}
# The old cashew_class_mapping.json: OIL inserted at 9, which shifted PKW1 / PKW2 / TESTA SMALL.
OLD_JSON = {0: "GOOD_CASHEW", 1: "TESTA", 2: "UPW", 3: "DW1", 4: "DW2", 5: "WHITE_DW", 6: "SW", 7: "BROWN",
            8: "REJECTION", 9: "OIL", 10: "PKW1", 11: "PKW2", 12: "TESTA SMALL"}
CNN = ['BROWN', 'DW2', 'GOOD_CASHEW', 'OIL', 'PKW1', 'PKW2', 'SW', 'TESTA', 'TESTA SMALL', 'UPW', 'WHITE_DW']


class RegistryTest(unittest.TestCase):
    def make(self):
        reg = ClassRegistry(YOLO)
        reg.add_names(OLD_JSON[k] for k in sorted(OLD_JSON))
        reg.add_names(CNN)
        return reg

    def test_yolo_ids_are_never_renumbered(self):
        reg = self.make()
        for cid, name in YOLO.items():
            self.assertEqual(reg.name(cid), name)

    def test_extra_name_goes_behind_yolo_ids(self):
        reg = self.make()
        self.assertEqual(reg.id_for("OIL"), 12)
        self.assertEqual(len(reg.snapshot()), 13)

    def test_lookup_ignores_case_space_underscore(self):
        reg = self.make()
        self.assertEqual(reg.id_for("testa_small"), 11)
        self.assertEqual(reg.id_for("Good Cashew"), 0)
        self.assertIsNone(reg.id_for("NOT_A_CLASS"))
        self.assertEqual(normalize_name("WHITE_DW"), "white dw")

    def test_every_cnn_class_maps_to_its_own_name(self):
        reg = self.make()
        for name in CNN:
            self.assertEqual(normalize_name(reg.name(reg.id_for(name))), normalize_name(name))

    def test_yolo_class_cannot_be_removed_or_duplicated(self):
        reg = self.make()
        self.assertFalse(reg.remove("PKW1"))
        self.assertTrue(reg.remove("OIL"))
        self.assertIsNone(reg.id_for("OIL"))
        self.assertFalse(reg.rename("TESTA", "UPW"))

    def test_new_yolo_model_keeps_extra_names(self):
        reg = self.make()
        reg.set_yolo_names({0: "CASHEW"})
        self.assertEqual(reg.name(0), "CASHEW")
        self.assertIsNotNone(reg.id_for("OIL"))
        self.assertEqual(len(set(reg.snapshot().values())), len(reg.snapshot()))


if __name__ == "__main__":
    unittest.main()
