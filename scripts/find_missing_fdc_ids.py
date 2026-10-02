"""Print filtered ingredients that still need human USDA matches."""

from pathlib import Path
import pandas as pd


root = Path(__file__).resolve().parents[1]
filtered = pd.read_csv(root / "data" / "interim" / "ingredients_filtered.csv")
log_path = root / "data" / "external" / "matching_log.csv"

if not log_path.exists():
    print("matching_log.csv does not exist")
else:
    matching = pd.read_csv(log_path)
    missing = filtered[~filtered["ahn_id"].isin(matching.loc[matching["fdc_id"].notna(), "ahn_id"])]
    print(missing[["ahn_id", "ahn_name"]].to_string(index=False))