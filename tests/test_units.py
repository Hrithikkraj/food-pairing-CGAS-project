"""Unit tests for reference Daily Values and unit specifications."""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fnp.utils import load_dv_table


class TestDailyValueTable(unittest.TestCase):
    def setUp(self):
        self.dv_df = load_dv_table()

    def test_required_columns(self):
        expected_cols = {"nutrient", "nutrient_id", "unit", "daily_value", "source", "checked_on"}
        self.assertTrue(expected_cols.issubset(set(self.dv_df.columns)))

    def test_nutrient_ids_and_counts(self):
        # Must contain the 11 key nutrients from section 2.2
        expected_ids = {
            1003: ("Protein", 50.0, "g"),
            1079: ("Fiber", 28.0, "g"),
            1089: ("Iron", 18.0, "mg"),
            1087: ("Calcium", 1300.0, "mg"),
            1092: ("Potassium", 4700.0, "mg"),
            1095: ("Zinc", 11.0, "mg"),
            1162: ("Vitamin C", 90.0, "mg"),
            1106: ("Vitamin A", 900.0, "µg"),
            1178: ("Vitamin B-12", 2.4, "µg"),
            1177: ("Folate", 400.0, "µg"),
            1090: ("Magnesium", 420.0, "mg")
        }

        self.assertEqual(len(self.dv_df), 11)

        for nid, (name, dv, unit) in expected_ids.items():
            row = self.dv_df[self.dv_df["nutrient_id"] == nid]
            self.assertFalse(row.empty, f"Nutrient ID {nid} ({name}) not found in dv_table.csv")
            self.assertAlmostEqual(float(row.iloc[0]["daily_value"]), dv)
            self.assertEqual(row.iloc[0]["unit"], unit)

    def test_positive_daily_values(self):
        self.assertTrue((self.dv_df["daily_value"] > 0).all())


if __name__ == "__main__":
    unittest.main()
