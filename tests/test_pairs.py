"""Unit tests for pair generation properties and invariants in fnp.pairs."""

import unittest
from pathlib import Path
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fnp.pairs import build_pair_table
from fnp.scores import compute_complementarity, compute_flavor_sharing


class TestPairInvariants(unittest.TestCase):
    def setUp(self):
        # Create a mini set of 4 mock ingredients
        self.mock_ingredients = pd.DataFrame([
            {"ahn_id": 1, "name": "Apple", "category": "Fruit", "pct_dv_protein": 1.0, "pct_dv_fiber": 15.0},
            {"ahn_id": 2, "name": "Cheese", "category": "Dairy", "pct_dv_protein": 30.0, "pct_dv_fiber": 0.0},
            {"ahn_id": 3, "name": "Tomato", "category": "Vegetable", "pct_dv_protein": 2.0, "pct_dv_fiber": 8.0},
            {"ahn_id": 4, "name": "Basil", "category": "Herbs", "pct_dv_protein": 4.0, "pct_dv_fiber": 5.0},
        ])
        self.mock_compounds = {
            1: {10, 20, 30},
            2: {20, 40, 50, 60},
            3: {20, 30, 70},
            4: {30, 80}
        }
        self.mock_recipes = [{1, 2}, {2, 3, 4}]
        self.mock_params = {
            "low_pct": 10.0,
            "high_pct": 20.0,
            "sensitivity_cutoffs": [[5.0, 15.0], [10.0, 20.0]]
        }

    def test_pair_count_and_ordering(self):
        n = len(self.mock_ingredients)
        expected_pairs = n * (n - 1) // 2  # 4 * 3 / 2 = 6

        df_pairs = build_pair_table(
            self.mock_ingredients,
            self.mock_compounds,
            self.mock_recipes,
            {},
            self.mock_params
        )

        self.assertEqual(len(df_pairs), expected_pairs)

        # Invariant check: ahn_id_a < ahn_id_b for every row
        for _, row in df_pairs.iterrows():
            self.assertLess(row["ahn_id_a"], row["ahn_id_b"])

    def test_score_ranges(self):
        df_pairs = build_pair_table(
            self.mock_ingredients,
            self.mock_compounds,
            self.mock_recipes,
            {},
            self.mock_params
        )

        # Complementarity scores must be within [0, 1]
        self.assertTrue((df_pairs["C_main"] >= 0.0).all())
        self.assertTrue((df_pairs["C_main"] <= 1.0).all())

        # Flavor sharing must be non-negative
        self.assertTrue((df_pairs["n_shared"] >= 0).all())

    def test_score_symmetry(self):
        comp_a = {10, 20, 30}
        comp_b = {20, 30, 40}
        flags_a = ["low", "high"]
        flags_b = ["high", "neither"]

        # N_s(A, B) == N_s(B, A)
        self.assertEqual(compute_flavor_sharing(comp_a, comp_b), compute_flavor_sharing(comp_b, comp_a))

        # C(A, B) == C(B, A)
        self.assertEqual(
            compute_complementarity(flags_a, flags_b, total_nutrients=2),
            compute_complementarity(flags_b, flags_a, total_nutrients=2)
        )

    def test_matrix_recipe_counts_match_slow_loop(self):
        df_pairs = build_pair_table(
            self.mock_ingredients,
            self.mock_compounds,
            self.mock_recipes,
            {"test": self.mock_recipes},
            self.mock_params,
        )
        pair = df_pairs[(df_pairs["ahn_id_a"] == 2) & (df_pairs["ahn_id_b"] == 3)].iloc[0]
        slow_both = sum(1 for recipe in self.mock_recipes if 2 in recipe and 3 in recipe)
        slow_a = sum(1 for recipe in self.mock_recipes if 2 in recipe)
        slow_b = sum(1 for recipe in self.mock_recipes if 3 in recipe)
        self.assertEqual(pair["n_recipes_both"], slow_both)
        self.assertEqual(pair["n_a"], slow_a)
        self.assertEqual(pair["n_b"], slow_b)
        self.assertAlmostEqual(pair["lift"], slow_both * len(self.mock_recipes) / (slow_a * slow_b))

    def test_matrix_counts_above_int8_match_slow_loop(self):
        ingredients = pd.DataFrame([
            {"ahn_id": 1, "name": "a", "category": "food"},
            {"ahn_id": 2, "name": "b", "category": "food"},
        ])
        compounds = {1: set(range(200)), 2: set(range(200))}
        recipes = [{1, 2} for _ in range(202)]
        pairs = build_pair_table(ingredients, compounds, recipes, {}, self.mock_params)
        pair = pairs.iloc[0]
        self.assertEqual(pair["n_recipes_both"], sum(1 for recipe in recipes if 1 in recipe and 2 in recipe))
        self.assertEqual(pair["n_shared"], len(compounds[1] & compounds[2]))
        self.assertEqual(pair["n_recipes_both"], 202)
        self.assertEqual(pair["n_shared"], 200)


if __name__ == "__main__":
    unittest.main()
