"""Unit tests for pure scoring functions in fnp.scores."""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fnp.scores import (
    compute_flavor_sharing,
    compute_normalized_flavor_sharing,
    compute_percent_dv,
    classify_nutrient_status,
    compute_complementarity,
    identify_complementary_nutrients
)


class TestFlavorSharing(unittest.TestCase):
    def test_empty_sets(self):
        self.assertEqual(compute_flavor_sharing(set(), set()), 0)
        self.assertEqual(compute_normalized_flavor_sharing(set(), set()), 0.0)

    def test_disjoint_sets(self):
        a = {1, 2, 3}
        b = {4, 5, 6}
        self.assertEqual(compute_flavor_sharing(a, b), 0)
        self.assertEqual(compute_normalized_flavor_sharing(a, b), 0.0)

    def test_identical_sets(self):
        a = {1, 2, 3, 4}
        b = {1, 2, 3, 4}
        self.assertEqual(compute_flavor_sharing(a, b), 4)
        self.assertAlmostEqual(compute_normalized_flavor_sharing(a, b), 1.0)

    def test_partial_overlap_and_normalization(self):
        a = {1, 2, 3}        # size 3
        b = {2, 3, 4, 5, 6}  # size 5
        # intersection is {2, 3} -> size 2
        # min size is 3 -> normalized is 2 / 3
        self.assertEqual(compute_flavor_sharing(a, b), 2)
        self.assertAlmostEqual(compute_normalized_flavor_sharing(a, b), 2.0 / 3.0)


class TestNutrientMetrics(unittest.TestCase):
    def test_percent_dv_calculation(self):
        # 9 mg Iron with 18 mg DV -> 50%
        self.assertAlmostEqual(compute_percent_dv(9.0, 18.0), 50.0)
        # Invalid DV
        with self.assertRaises(ValueError):
            compute_percent_dv(5.0, 0.0)

    def test_classification_scalar(self):
        self.assertEqual(classify_nutrient_status(4.0, low_pct=10.0, high_pct=20.0), "low")
        self.assertEqual(classify_nutrient_status(9.9, low_pct=10.0, high_pct=20.0), "low")
        self.assertEqual(classify_nutrient_status(10.0, low_pct=10.0, high_pct=20.0), "neither")
        self.assertEqual(classify_nutrient_status(19.9, low_pct=10.0, high_pct=20.0), "neither")
        self.assertEqual(classify_nutrient_status(20.0, low_pct=10.0, high_pct=20.0), "high")
        self.assertEqual(classify_nutrient_status(35.0, low_pct=10.0, high_pct=20.0), "high")

    def test_classification_vector(self):
        vals = pd.Series([4.0, 15.0, 25.0, np.nan])
        flags = classify_nutrient_status(vals, low_pct=10.0, high_pct=20.0)
        self.assertEqual(flags[0], "low")
        self.assertEqual(flags[1], "neither")
        self.assertEqual(flags[2], "high")
        self.assertTrue(pd.isna(flags[3]))

    def test_section_3_4_toy_example(self):
        """Verify the exact toy example from report Section 3.4:
        Food A: Iron 4% (low), Calcium 30% (high), Protein 25% (high)
        Food B: Iron 35% (high), Calcium 6% (low), Protein 28% (high)
        K = 3 -> Complementarity C = 2/3 = 0.67
        """
        flags_a = ["low", "high", "high"]
        flags_b = ["high", "low", "high"]
        nutrients = ["Iron", "Calcium", "Protein"]

        c_score = compute_complementarity(flags_a, flags_b, total_nutrients=3)
        self.assertAlmostEqual(c_score, 2.0 / 3.0, places=2)

        drivers = identify_complementary_nutrients(flags_a, flags_b, nutrients)
        self.assertEqual(drivers, ["Iron", "Calcium"])

    def test_complementarity_all_high_or_all_low(self):
        # Both high -> 0 gap filling
        self.assertEqual(compute_complementarity(["high", "high"], ["high", "high"], total_nutrients=2), 0.0)
        # Both low -> 0 gap filling
        self.assertEqual(compute_complementarity(["low", "low"], ["low", "low"], total_nutrients=2), 0.0)
        # Low vs Neither -> 0 gap filling
        self.assertEqual(compute_complementarity(["low", "neither"], ["neither", "neither"], total_nutrients=2), 0.0)


if __name__ == "__main__":
    unittest.main()
