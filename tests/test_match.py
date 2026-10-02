"""Regression tests for human-reviewed USDA candidate matching."""

import unittest
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fnp.match import EXCLUDED_USDA_CATEGORIES, run_match, score_candidate


class TestMatching(unittest.TestCase):
    def test_prepared_food_is_ranked_below_raw_start_match(self):
        raw = score_candidate("apple", "Apple, raw")
        dessert = score_candidate("apple", "Babyfood, apple yogurt dessert")
        self.assertGreater(raw, dessert)

    def test_excluded_categories_are_explicit(self):
        self.assertIn("Baby Foods", EXCLUDED_USDA_CATEGORIES)
        self.assertIn("Restaurant Foods", EXCLUDED_USDA_CATEGORIES)
        self.assertIn("Meals, Entrees, and Side Dishes", EXCLUDED_USDA_CATEGORIES)

    def test_reviewed_row_is_preserved_on_rerun(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            interim = root / "interim"
            external = root / "external"
            interim.mkdir()
            external.mkdir()
            pd.DataFrame([{"ahn_id": 1, "ahn_name": "apple", "category": "fruit"}]).to_csv(
                interim / "ingredients_filtered.csv", index=False
            )
            pd.DataFrame([{
                "fdc_id": 123,
                "data_type": "sr legacy",
                "description": "Apple, raw",
                "food_category_id": 1,
                "publication_date": "2018",
            }]).to_parquet(interim / "usda_food.parquet", index=False)
            pd.DataFrame([{"id": 1, "description": "Fruits and Fruit Juices"}]).to_parquet(
                interim / "usda_food_category.parquet", index=False
            )
            existing = pd.DataFrame([{
                "ahn_id": 1, "ahn_name": "apple", "fdc_id": 999,
                "usda_description": "My reviewed food", "reviewer": "human",
                "note": "keep this", "match_score": 1,
            }])
            existing.to_csv(external / "matching_log.csv", index=False)
            config = root / "params.yaml"
            config.write_text(
                "paths:\n  interim: 'interim'\n  external: 'external'\n",
                encoding="utf-8",
            )
            with patch("fnp.match.get_project_root", return_value=root):
                result = run_match(str(config))
            self.assertEqual(result["status"], "success")
            actual = pd.read_csv(external / "matching_log.csv").iloc[0]
            self.assertEqual(actual["fdc_id"], 999)
            self.assertEqual(actual["usda_description"], "My reviewed food")
            self.assertEqual(actual["reviewer"], "human")
            self.assertEqual(actual["note"], "keep this")

    def test_two_filled_rows_survive_two_runs(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            interim = root / "interim"
            external = root / "external"
            interim.mkdir()
            external.mkdir()
            for name in ("ingredients_filtered.csv", "usda_food.parquet", "usda_food_category.parquet"):
                source = project_root / "data" / "interim" / name
                if source.exists():
                    import shutil
                    shutil.copy2(source, interim / name)
            source_log = project_root / "data" / "external" / "matching_log.csv"
            log = pd.read_csv(source_log)
            for column in ("fdc_id", "usda_description", "reviewer", "note"):
                log[column] = log[column].astype(object)
                log[column] = ""
            filled_ids = log["ahn_id"].head(2).tolist()
            log.loc[log["ahn_id"] == filled_ids[0], ["fdc_id", "usda_description", "reviewer", "note"]] = [168878, "Reviewed apple", "human", "keep apple"]
            log.loc[log["ahn_id"] == filled_ids[1], ["fdc_id", "usda_description", "reviewer", "note"]] = [168879, "Reviewed food", "human", "keep food"]
            log.to_csv(external / "matching_log.csv", index=False)
            before = log[log["ahn_id"].isin(filled_ids)].copy()
            config = root / "params.yaml"
            config.write_text("paths:\n  interim: 'interim'\n  external: 'external'\n", encoding="utf-8")
            with patch("fnp.match.get_project_root", return_value=root):
                run_match(str(config))
                run_match(str(config))
            actual = pd.read_csv(external / "matching_log.csv")
            self.assertEqual(len(actual), 252)
            self.assertFalse(actual["ahn_id"].isna().any())
            self.assertEqual(actual["ahn_id"].tolist(), pd.read_csv(interim / "ingredients_filtered.csv")["ahn_id"].tolist())
            self.assertEqual(int(actual["fdc_id"].notna().sum()), 2)
            after = actual[actual["ahn_id"].isin(filled_ids)].copy()
            for column in before.columns:
                if column == "fdc_id":
                    self.assertEqual(before[column].astype("Int64").tolist(), after[column].astype("Int64").tolist())
                else:
                    self.assertEqual(before[column].astype(str).tolist(), after[column].astype(str).tolist())

    def test_missing_ahn_id_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            interim = root / "interim"
            external = root / "external"
            interim.mkdir()
            external.mkdir()
            pd.DataFrame([{"ahn_id": 1, "ahn_name": "apple", "category": "fruit"}]).to_csv(interim / "ingredients_filtered.csv", index=False)
            pd.DataFrame([{"ahn_id": None, "ahn_name": "broken", "fdc_id": None}]).to_csv(external / "matching_log.csv", index=False)
            config = root / "params.yaml"
            config.write_text("paths:\n  interim: 'interim'\n  external: 'external'\n", encoding="utf-8")
            with patch("fnp.match.get_project_root", return_value=root):
                with self.assertRaisesRegex(ValueError, "missing ahn_id"):
                    run_match(str(config))

    def test_force_creates_backup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            interim = root / "interim"
            external = root / "external"
            interim.mkdir()
            external.mkdir()
            pd.DataFrame([{"ahn_id": 1, "ahn_name": "apple", "category": "fruit"}]).to_csv(interim / "ingredients_filtered.csv", index=False)
            pd.DataFrame([{"fdc_id": 123, "data_type": "sr legacy", "description": "Apple, raw", "food_category_id": 1, "publication_date": "2018"}]).to_parquet(interim / "usda_food.parquet", index=False)
            pd.DataFrame([{"id": 1, "description": "Fruits and Fruit Juices"}]).to_parquet(interim / "usda_food_category.parquet", index=False)
            original = pd.DataFrame([{"ahn_id": 1, "ahn_name": "apple", "fdc_id": 999, "reviewer": "human"}])
            original.to_csv(external / "matching_log.csv", index=False)
            config = root / "params.yaml"
            config.write_text("paths:\n  interim: 'interim'\n  external: 'external'\n", encoding="utf-8")
            with patch("fnp.match.get_project_root", return_value=root):
                run_match(str(config), force=True)
            self.assertTrue((external / "matching_log.backup.csv").exists())
            self.assertEqual(pd.read_csv(external / "matching_log.backup.csv").iloc[0]["fdc_id"], 999)

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "data" / "interim" / "usda_food.parquet").exists(), "real interim USDA data is unavailable")
    def test_spice_candidates_include_spices_category(self):
        root = Path(__file__).resolve().parents[1]
        foods = pd.read_parquet(root / "data" / "interim" / "usda_food.parquet")
        categories = pd.read_parquet(root / "data" / "interim" / "usda_food_category.parquet")
        category_map = dict(zip(categories["id"], categories["description"]))
        foods["food_category"] = foods["food_category_id"].map(category_map)
        foods = foods[~foods["food_category"].isin(EXCLUDED_USDA_CATEGORIES)]
        names = ["cumin", "cinnamon", "black_pepper", "clove", "nutmeg", "turmeric", "cardamom"]
        for name in names:
            candidates = sorted(
                ((row.description, score_candidate(name, row.description, row.food_category, True))
                 for row in foods.itertuples()),
                key=lambda item: -item[1],
            )[:10]
            print(name, [description for description, _ in candidates[:3]])
            self.assertTrue(any(description.lower().startswith("spices,") for description, _ in candidates))


if __name__ == "__main__":
    unittest.main()
