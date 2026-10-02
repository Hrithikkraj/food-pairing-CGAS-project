"""Small integration checks against the real Ahn data, when present."""

import unittest
from pathlib import Path
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fnp.pairs import load_compounds_map


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


@unittest.skipUnless(RAW.exists(), "data/raw is not available")
class TestRealData(unittest.TestCase):
    def test_s2_ids_categories_and_compounds(self):
        filtered = pd.read_csv(ROOT / "data" / "interim" / "ingredients_filtered.csv")
        self.assertFalse((filtered["category"] == "ingredient").any())
        for name in ("ginger", "cauliflower", "rosemary"):
            self.assertIn(name, set(filtered["ahn_name"]))

        raw = pd.read_csv(RAW / "ahn" / "ingr_comp" / "ingr_comp.tsv", sep="\t")
        raw.columns = ["ahn_id", "compound_id"]
        expected = {
            int(ingredient_id): set(group.loc[group["compound_id"] != 951, "compound_id"])
            for ingredient_id, group in raw.groupby("ahn_id")
        }
        used = load_compounds_map(ROOT / "data" / "interim" / "ahn_compounds.parquet", {951})
        for name in ("apple", "rice", "tomato", "cheese", "bacon"):
            ingredient_id = int(filtered.loc[filtered["ahn_name"] == name, "ahn_id"].iloc[0])
            self.assertEqual(used[ingredient_id], expected[ingredient_id], name)

    @unittest.skipUnless((ROOT / "data" / "processed" / "pair_table.csv").exists(), "pair table is not available")
    def test_real_pair_counts_and_shared_compounds(self):
        pairs = pd.read_csv(ROOT / "data" / "processed" / "pair_table.csv")
        recipes = pd.read_parquet(ROOT / "data" / "interim" / "ahn_recipes.parquet")
        recipe_sets = [set(str(value).split(",")) for value in recipes["ingredients"].dropna()]
        for column in ("n_recipes_both", "n_a", "n_b", "lift"):
            self.assertFalse((pairs[column] < 0).any(), column)

        expected_pairs = {("rice", "bean"): 202, ("tomato", "cheese"): 1241, ("butter", "milk"): 7664, ("onion", "garlic"): 9687}
        for (name_a, name_b), expected in expected_pairs.items():
            row = pairs[
                ((pairs.name_a == name_a) & (pairs.name_b == name_b)) |
                ((pairs.name_a == name_b) & (pairs.name_b == name_a))
            ].iloc[0]
            actual = sum(name_a in recipe and name_b in recipe for recipe in recipe_sets)
            self.assertEqual(actual, expected)
            self.assertEqual(row["n_recipes_both"], actual)

        compounds = load_compounds_map(ROOT / "data" / "interim" / "ahn_compounds.parquet", {951})
        rng = np.random.default_rng(42)
        sample = pairs.iloc[rng.choice(len(pairs), size=min(200, len(pairs)), replace=False)]
        for row in sample.itertuples():
            self.assertEqual(row.n_shared, len(compounds[int(row.ahn_id_a)] & compounds[int(row.ahn_id_b)]))


if __name__ == "__main__":
    unittest.main()