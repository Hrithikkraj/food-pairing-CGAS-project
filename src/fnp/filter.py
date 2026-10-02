"""Stage S2: Filter the ingredient list.

Filters out over-represented compounds (e.g. farnesol, id 951), enforces minimum
compound threshold (min_compounds >= 10), filters out non-nutritional items
(alcohol, flowers, vague terms), and computes recipe appearance counts.

Outputs:
    data/interim/ingredients_filtered.csv
    data/interim/excluded_ingredients.csv
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
import pandas as pd
import numpy as np

from fnp.utils import get_project_root, load_params, setup_logger

logger = setup_logger("fnp.filter", "logs/pipeline.log")

# Standard non-nutritional or vague ingredient terms to exclude
DEFAULT_EXCLUSION_RULES = [
    # Alcoholic beverages
    (lambda name: bool(set(str(name).lower().split("_")) & {"wine", "beer", "whiskey", "rum", "liqueur", "brandy", "gin", "vodka", "sherry", "sake", "cognac", "champagne"}), "Alcoholic beverage / solvent"),
    # Non-food / flowers / tobacco
    (lambda name: bool(set(str(name).lower().split("_")) & {"flower", "rose", "violet", "tobacco", "geranium", "jasmine", "lavender"}), "Floral / non-nutritive decorative"),
    # Highly ambiguous / vague ingredients
    (lambda name: str(name).lower().strip() in ["root", "condiment", "essence", "extract", "coloring", "flavoring", "oil", "wood", "bark", "leaf"], "Ambiguous or non-food extract")
]


def run_filter(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Execute Stage S2 filtering on interim compound and recipe tables."""
    root = get_project_root()
    params = load_params(config_path)

    interim_dir = root / params["paths"]["interim"]
    min_compounds = params.get("min_compounds", 10)
    filter_comp_ids = set(params.get("filter_compounds", [951]))
    filter_ing_ids = set(params.get("filter_ingredients", [339]))

    comp_parquet = interim_dir / "ahn_compounds.parquet"
    rec_parquet = interim_dir / "ahn_recipes.parquet"

    if not comp_parquet.exists():
        logger.error(f"Cannot run S2: {comp_parquet} does not exist. Run S1 first.")
        return {"status": "error", "message": f"{comp_parquet} not found"}

    df_comp = pd.read_parquet(comp_parquet)
    logger.info(f"Loaded {len(df_comp)} compound-ingredient links.")

    # Standardize column naming if needed
    # Typical Ahn schema: ingredient / ahn_id, compound / compound_id
    cols = {c.lower(): c for c in df_comp.columns}
    ingr_col = cols.get("ingredient", cols.get("ahn_name", cols.get("ingredient_name", df_comp.columns[0])))
    comp_col = cols.get("compound", cols.get("compound_id", df_comp.columns[1]))

    # Filter out blacklisted compounds and ingredients (farnesol is compound 951,
    # attached to Ahn ingredient 339).
    if comp_col in df_comp.columns:
        n_before = len(df_comp)
        df_comp = df_comp[
            ~df_comp[comp_col].isin(filter_comp_ids)
            & ~df_comp["ahn_id"].isin(filter_ing_ids)
        ]
        logger.info(f"Removed {n_before - len(df_comp)} rows with compound IDs {filter_comp_ids}.")

    # Keep the original Ahn ID and category throughout the pipeline.
    comp_counts = (
        df_comp.groupby(["ahn_id", "ahn_name", "category"], dropna=False)[comp_col]
        .nunique()
        .reset_index(name="compound_count")
    )

    # Calculate recipe appearance counts if recipe table exists
    rec_counts_map = {}
    if rec_parquet.exists():
        df_rec = pd.read_parquet(rec_parquet)
        # Parse ingredients if stored as comma/tab separated or list
        if "ingredients" in df_rec.columns:
            for item_list in df_rec["ingredients"].dropna():
                if isinstance(item_list, str):
                    items = [x.strip() for x in item_list.split(",")]
                else:
                    items = list(item_list)
                for item in set(items):
                    rec_counts_map[item] = rec_counts_map.get(item, 0) + 1
        elif "ingredient" in df_rec.columns:
            for item in df_rec["ingredient"].dropna():
                rec_counts_map[item] = rec_counts_map.get(item, 0) + 1

    rec_counts_normalized = {str(k).strip().lower(): v for k, v in rec_counts_map.items()}
    comp_counts["recipe_count"] = comp_counts["ahn_name"].map(
        lambda x: rec_counts_normalized.get(str(x).strip().lower(), 0)
    )

    # Apply exclusion rules and compound threshold
    kept_rows = []
    excluded_rows = []

    for _, row in comp_counts.iterrows():
        name = str(row["ahn_name"])
        cc = row["compound_count"]
        rc = row["recipe_count"]

        # Check compound threshold
        if cc < min_compounds:
            excluded_rows.append({"ahn_name": name, "compound_count": cc, "recipe_count": rc, "reason": f"Under min_compounds threshold ({cc} < {min_compounds})"})
            continue

        # Check recipe presence (must appear in at least 1 recipe if recipes available)
        if rec_counts_map and rc < 1:
            excluded_rows.append({"ahn_name": name, "compound_count": cc, "recipe_count": rc, "reason": "Not found in any recipe"})
            continue

        # Check semantic exclusion rules
        excluded = False
        for rule_fn, reason in DEFAULT_EXCLUSION_RULES:
            if rule_fn(name):
                excluded_rows.append({"ahn_name": name, "compound_count": cc, "recipe_count": rc, "reason": reason})
                excluded = True
                break

        if not excluded:
            kept_rows.append({
                "ahn_id": int(row["ahn_id"]),
                "ahn_name": name,
                "compound_count": cc,
                "recipe_count": rc,
                "category": row["category"]
            })

    df_filtered = pd.DataFrame(kept_rows)
    df_excluded = pd.DataFrame(excluded_rows)

    out_filtered = interim_dir / "ingredients_filtered.csv"
    out_excluded = interim_dir / "excluded_ingredients.csv"

    df_filtered.to_csv(out_filtered, index=False)
    df_excluded.to_csv(out_excluded, index=False)

    logger.info(f"S2 complete. Kept {len(df_filtered)} ingredients, excluded {len(df_excluded)}.")
    logger.info(f"Saved filtered list to {out_filtered}")

    return {
        "status": "success",
        "n_kept": len(df_filtered),
        "n_excluded": len(df_excluded),
        "output_file": str(out_filtered)
    }


if __name__ == "__main__":
    run_filter()
