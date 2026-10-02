"""Tests for analysis serialization and vectorized bootstrap behavior."""

import json
import unittest
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fnp.analyze import aggregate_cuisine_counts, json_default, json_safe
from fnp.baseline import bootstrap_spearman_by_ingredient, permutation_test_profile_pairs


class TestAnalyze(unittest.TestCase):
    def test_json_default_handles_numpy_values_and_nan(self):
        value = {"i": np.int64(4), "f": np.float64(1.5), "b": np.bool_(True), "a": np.array([1, 2]), "nan": np.nan}
        encoded = json.dumps(json_safe(value), default=json_default, allow_nan=False)
        self.assertEqual(json.loads(encoded)["i"], 4)
        self.assertIsNone(json.loads(encoded)["nan"])

    def test_vectorized_bootstrap_is_seed_stable(self):
        pairs = pd.DataFrame([
            {"ahn_id_a": 1, "ahn_id_b": 2, "n_shared": 1, "n_shared_norm": .1, "C_main": .2},
            {"ahn_id_a": 1, "ahn_id_b": 3, "n_shared": 2, "n_shared_norm": .2, "C_main": .4},
            {"ahn_id_a": 1, "ahn_id_b": 4, "n_shared": 3, "n_shared_norm": .3, "C_main": .6},
            {"ahn_id_a": 2, "ahn_id_b": 3, "n_shared": 4, "n_shared_norm": .4, "C_main": .8},
            {"ahn_id_a": 2, "ahn_id_b": 4, "n_shared": 5, "n_shared_norm": .5, "C_main": .1},
            {"ahn_id_a": 3, "ahn_id_b": 4, "n_shared": 6, "n_shared_norm": .6, "C_main": .3},
        ])
        first = bootstrap_spearman_by_ingredient(pairs, n_boot=5, seed=9)
        second = bootstrap_spearman_by_ingredient(pairs, n_boot=5, seed=9)
        self.assertEqual(first, second)

    def test_vectorized_bootstrap_matches_slow_reference(self):
        pairs = pd.DataFrame([
            {"ahn_id_a": 1, "ahn_id_b": 2, "n_shared": 1, "C_main": .2},
            {"ahn_id_a": 1, "ahn_id_b": 3, "n_shared": 4, "C_main": .4},
            {"ahn_id_a": 1, "ahn_id_b": 4, "n_shared": 2, "C_main": .8},
            {"ahn_id_a": 2, "ahn_id_b": 3, "n_shared": 5, "C_main": .1},
            {"ahn_id_a": 2, "ahn_id_b": 4, "n_shared": 3, "C_main": .7},
            {"ahn_id_a": 3, "ahn_id_b": 4, "n_shared": 6, "C_main": .3},
        ])
        ids = sorted(set(pairs.ahn_id_a) | set(pairs.ahn_id_b))
        lookup = {(row.ahn_id_a, row.ahn_id_b): row for row in pairs.itertuples()}
        rng = np.random.default_rng(13)
        samples = []
        for _ in range(7):
            sampled = rng.choice(ids, size=len(ids), replace=True)
            x, y = [], []
            for first in range(len(sampled)):
                for second in range(first + 1, len(sampled)):
                    if sampled[first] == sampled[second]:
                        continue
                    key = tuple(sorted((sampled[first], sampled[second])))
                    row = lookup[key]
                    x.append(row.n_shared)
                    y.append(row.C_main)
            if len(x) >= 3:
                samples.append(pd.Series(x).corr(pd.Series(y), method="spearman"))
        result = bootstrap_spearman_by_ingredient(pairs, n_boot=7, seed=13)
        expected = np.nanpercentile(samples, [2.5, 97.5])
        self.assertAlmostEqual(result["ci_low"], expected[0], places=12)
        self.assertAlmostEqual(result["ci_high"], expected[1], places=12)

    def test_cuisine_groups_sum_pair_and_recipe_counts(self):
        pairs = pd.DataFrame({
            "n_recipes_A": [1, 2],
            "n_recipes_B": [3, 4],
            "n_recipes_Jewish": [5, 6],
        })
        cuisine_counts = pd.DataFrame({
            "cuisine": ["A", "B", "Jewish"],
            "n_recipes": [10, 20, 30],
        })

        pair_counts, recipe_counts, ungrouped = aggregate_cuisine_counts(
            pairs, cuisine_counts, {"Test region": ["A", "B"]}
        )

        self.assertEqual(pair_counts["Test region"].tolist(), [4, 6])
        self.assertEqual(recipe_counts["Test region"], 30)
        self.assertEqual(ungrouped, ["Jewish"])

    def test_permutation_p_values_detect_lower_complementarity(self):
        profiles = pd.DataFrame([
            {"ahn_id": index, "category": "same", "flag_nutrient": "low" if index < 6 else "high"}
            for index in range(12)
        ])
        pairs = pd.DataFrame([
            {"ahn_id_a": index, "ahn_id_b": index + 1, "C_main": 0.0}
            for index in range(0, 12, 2)
        ])

        result = permutation_test_profile_pairs(pairs, profiles, n_perm=300, seed=7)

        self.assertLess(result["p_less"], 0.1)
        self.assertGreater(result["p_greater"], 0.9)
        self.assertEqual(result["p_value"], result["p_greater"])

    def test_within_category_permutation_keeps_profile_categories(self):
        profiles = pd.DataFrame([
            {"ahn_id": 1, "category": "A", "flag_nutrient": "A1"},
            {"ahn_id": 2, "category": "A", "flag_nutrient": "A2"},
            {"ahn_id": 3, "category": "B", "flag_nutrient": "B1"},
            {"ahn_id": 4, "category": "B", "flag_nutrient": "B2"},
        ])
        pairs = pd.DataFrame([
            {"ahn_id_a": 1, "ahn_id_b": 2, "C_main": 0.0},
            {"ahn_id_a": 3, "ahn_id_b": 4, "C_main": 0.0},
        ])
        observed_pairs = []

        def capture_flags(flags_a, flags_b, total_nutrients):
            observed_pairs.append((flags_a[0], flags_b[0]))
            return 0.0

        with patch("fnp.baseline.compute_complementarity", side_effect=capture_flags):
            permutation_test_profile_pairs(
                pairs, profiles, n_perm=10, seed=11, within_category=True
            )

        for first_marker, second_marker in observed_pairs:
            self.assertEqual(first_marker[0], second_marker[0])


if __name__ == "__main__":
    unittest.main()