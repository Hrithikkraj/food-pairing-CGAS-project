"""Tests for dependence-aware statistical baselines."""

import unittest
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fnp.baseline import bootstrap_spearman_by_ingredient


class TestBaselines(unittest.TestCase):
    def test_ingredient_bootstrap_returns_observed_statistic(self):
        pairs = pd.DataFrame([
            {"ahn_id_a": 1, "ahn_id_b": 2, "n_shared": 1, "C_main": 0.1},
            {"ahn_id_a": 1, "ahn_id_b": 3, "n_shared": 2, "C_main": 0.4},
            {"ahn_id_a": 2, "ahn_id_b": 3, "n_shared": 3, "C_main": 0.8},
        ])
        result = bootstrap_spearman_by_ingredient(pairs, n_boot=10, seed=42)
        self.assertAlmostEqual(result["rho"], 1.0)
        self.assertEqual(result["n"], 3)


if __name__ == "__main__":
    unittest.main()
