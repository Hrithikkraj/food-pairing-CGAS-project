"""Validate a human-reviewed USDA matching log without modifying it."""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


PROBLEM_WORDS = {"canned", "frozen", "cooked", "prepared", "juice", "sauce", "dessert", "babyfood"}


def words(value: object) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value).lower().replace("_", " ")))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching-log", type=Path, default=root / "data/external/matching_log.csv")
    parser.add_argument("--usda-food", type=Path, default=root / "data/interim/usda_food.parquet")
    args = parser.parse_args()

    log = pd.read_csv(args.matching_log)
    foods = pd.read_parquet(args.usda_food)
    food_by_id = foods.set_index("fdc_id")
    empty = log[log["fdc_id"].isna()]
    invalid = log[~log["fdc_id"].isna() & ~log["fdc_id"].isin(food_by_id.index)]

    print("Ingredients with empty fdc_id:")
    for name in empty["ahn_name"].tolist():
        print(name)
    print("Rows with unknown fdc_id:")
    print(invalid[["ahn_id", "ahn_name", "fdc_id"]].to_string(index=False) if not invalid.empty else "None")

    wrong = []
    flagged = []
    for row in log[log["fdc_id"].notna()].itertuples():
        if row.fdc_id not in food_by_id.index:
            continue
        description = str(food_by_id.loc[row.fdc_id, "description"])
        if not (words(row.ahn_name) & words(description)):
            wrong.append((row.ahn_id, row.ahn_name, description))
        if words(description) & PROBLEM_WORDS:
            flagged.append((row.ahn_id, row.ahn_name, description))

    print("Chosen descriptions sharing no word with ingredient:")
    print(pd.DataFrame(wrong, columns=["ahn_id", "ahn_name", "description"]).to_string(index=False) if wrong else "None")
    print("Chosen descriptions containing preparation/problem words:")
    print(pd.DataFrame(flagged, columns=["ahn_id", "ahn_name", "description"]).to_string(index=False) if flagged else "None")

    duplicates = log[log["ahn_id"].duplicated(keep=False)].sort_values("ahn_id")
    print("Duplicate ahn_id values:")
    print(duplicates[["ahn_id", "ahn_name"]].to_string(index=False) if not duplicates.empty else "None")
    return 1 if not empty.empty or not invalid.empty or not duplicates.empty else 0


if __name__ == "__main__":
    sys.exit(main())