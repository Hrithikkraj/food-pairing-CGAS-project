"""Stage S4: Nutrient profile per ingredient.

Pivots USDA food_nutrient data for matched foods, computes %DV using dv_table.csv,
and applies low/high categorization flags.

Outputs:
    data/processed/ingredient_table.csv
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from fnp.utils import get_project_root, load_params, load_dv_table, setup_logger
from fnp.scores import compute_percent_dv, classify_nutrient_status

logger = setup_logger("fnp.profile", "logs/pipeline.log")


def run_profile(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Execute Stage S4 nutrient profiling."""
    root = get_project_root()
    params = load_params(config_path)

    interim_dir = root / params["paths"]["interim"]
    external_dir = root / params["paths"]["external"]
    processed_dir = root / params["paths"]["processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    matching_log_file = external_dir / "matching_log.csv"
    ingredients_filtered_file = interim_dir / "ingredients_filtered.csv"
    food_nutrient_file = interim_dir / "usda_food_nutrient.parquet"
    out_ingredient_table = processed_dir / "ingredient_table.csv"

    if not matching_log_file.exists():
        logger.error(f"Cannot run S4: {matching_log_file} does not exist. Run S3 first.")
        return {"status": "error", "message": f"{matching_log_file} not found"}

    df_matching = pd.read_csv(matching_log_file)
    # Keep only rows that have an approved/valid fdc_id
    valid_matches = df_matching[df_matching["fdc_id"].notna()].copy()
    valid_matches["fdc_id"] = valid_matches["fdc_id"].astype(int)

    if len(valid_matches) == 0:
        logger.error("No valid fdc_id found in matching_log.csv! Human review needed.")
        return {"status": "error", "message": "No valid matches in matching_log"}

    # Load filtered ingredients metadata
    if ingredients_filtered_file.exists():
        df_filtered = pd.read_csv(ingredients_filtered_file)
    else:
        df_filtered = valid_matches

    # Load Daily Value reference
    dv_df = load_dv_table()
    nutrient_id_to_dv = dict(zip(dv_df["nutrient_id"], dv_df["daily_value"]))
    nutrient_id_to_name = dict(zip(dv_df["nutrient_id"], dv_df["nutrient"]))

    target_nutrient_ids = list(nutrient_id_to_dv.keys())

    # Load USDA food_nutrient data
    if not food_nutrient_file.exists():
        logger.error(f"Cannot run S4: {food_nutrient_file} not found. Run S1 first.")
        return {"status": "error", "message": f"{food_nutrient_file} not found"}

    df_fn = pd.read_parquet(food_nutrient_file)
    # Filter for matched foods and target nutrients
    df_fn_matched = df_fn[
        df_fn["fdc_id"].isin(valid_matches["fdc_id"]) &
        df_fn["nutrient_id"].isin(target_nutrient_ids)
    ]

    # Pivot to: one row per fdc_id, columns = nutrient_id, values = amount
    pivoted = df_fn_matched.pivot_table(
        index="fdc_id",
        columns="nutrient_id",
        values="amount",
        aggfunc="mean"
    )

    # Ensure all target nutrients exist as columns (NaN if missing, never zero silently)
    for nid in target_nutrient_ids:
        if nid not in pivoted.columns:
            pivoted[nid] = np.nan

    low_pct = params.get("low_pct", 10.0)
    high_pct = params.get("high_pct", 20.0)

    # Compute %DV and status flags
    pct_dv_df = pd.DataFrame(index=pivoted.index)
    flag_df = pd.DataFrame(index=pivoted.index)

    for nid in target_nutrient_ids:
        nname = nutrient_id_to_name[nid].lower().replace(" ", "_").replace(",", "")
        dv = nutrient_id_to_dv[nid]
        raw_amounts = pivoted[nid]

        pct_dv = compute_percent_dv(raw_amounts, dv)
        pct_dv_df[f"pct_dv_{nname}"] = pct_dv
        flag_df[f"flag_{nname}"] = classify_nutrient_status(pct_dv, low_pct, high_pct)

    # Merge with matching log and metadata
    profile_df = valid_matches.merge(pivoted, on="fdc_id", how="left")
    profile_df = profile_df.merge(pct_dv_df, on="fdc_id", how="left")
    profile_df = profile_df.merge(flag_df, on="fdc_id", how="left")

    if "compound_count" in df_filtered.columns:
        profile_df = profile_df.merge(
            df_filtered[["ahn_id", "compound_count", "recipe_count", "category"]],
            on="ahn_id",
            how="left"
        )
        profile_df.rename(columns={"compound_count": "n_compounds", "recipe_count": "n_recipes"}, inplace=True)
        profile_df["is_spice_herb"] = profile_df["category"].astype(str).str.lower().isin({"spice", "herb"})
    else:
        profile_df["n_compounds"] = 10
        profile_df["n_recipes"] = 1
        profile_df["category"] = "general"
        profile_df["is_spice_herb"] = False

    profile_df.rename(columns={"ahn_name": "name"}, inplace=True)
    profile_df.to_csv(out_ingredient_table, index=False)
    logger.info(f"S4 complete. Profiled {len(profile_df)} ingredients. Saved to {out_ingredient_table}")

    return {
        "status": "success",
        "n_ingredients": len(profile_df),
        "file": str(out_ingredient_table)
    }


if __name__ == "__main__":
    run_profile()
