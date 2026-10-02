"""Stage S5: Pair table construction.

Generates all unique unordered pairs (ahn_id_a < ahn_id_b), computes flavor sharing (N_s),
normalized N_s, nutrient complementarity (C) under main and alternative cutoffs,
recipe co-occurrences, and category pairings.

Outputs:
    data/processed/pair_table.csv
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import sparse

from fnp.utils import get_project_root, load_params, load_dv_table, setup_logger
from fnp.scores import (
    compute_flavor_sharing,
    compute_normalized_flavor_sharing,
    classify_nutrient_status,
    compute_complementarity,
    identify_complementary_nutrients
)

logger = setup_logger("fnp.pairs", "logs/pipeline.log")


def _recipe_matrix(recipes: List[Set[int]], ingredient_ids: List[int]) -> sparse.csr_matrix:
    """Build a recipe-by-ingredient binary incidence matrix."""
    id_to_col = {ingredient_id: col for col, ingredient_id in enumerate(ingredient_ids)}
    rows = []
    cols = []
    for recipe_index, recipe in enumerate(recipes):
        for ingredient_id in recipe:
            if ingredient_id in id_to_col:
                rows.append(recipe_index)
                cols.append(id_to_col[ingredient_id])
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(len(recipes), len(ingredient_ids)),
    )


def load_compounds_map(compounds_file: Path, excluded_compounds: Optional[Set[Any]] = None) -> Dict[int, Set[Any]]:
    """Load original Ahn IDs and compound sets used by pair construction."""
    compounds_map: Dict[int, Set[Any]] = {}
    if not compounds_file.exists():
        return compounds_map
    df_comp = pd.read_parquet(compounds_file)
    cols = {c.lower(): c for c in df_comp.columns}
    c_ing = cols.get("ahn_id", cols.get("ingredient", df_comp.columns[0]))
    c_cmp = cols.get("compound", cols.get("compound_id", df_comp.columns[1]))
    excluded_compounds = excluded_compounds or set()
    for ing_id, group in df_comp.groupby(c_ing):
        compounds_map[int(ing_id)] = set(group.loc[~group[c_cmp].isin(excluded_compounds), c_cmp])
    return compounds_map


def build_pair_table(
    df_ingredients: pd.DataFrame,
    compounds_map: Dict[int, Set[Any]],
    recipes_list: List[Set[int]],
    recipes_per_cuisine: Dict[str, List[Set[int]]],
    params: Dict[str, Any]
) -> pd.DataFrame:
    """Construct pair table with all metrics for unique unordered pairs."""
    dv_df = load_dv_table()
    nutrient_names = [n.lower().replace(" ", "_").replace(",", "") for n in dv_df["nutrient"]]
    pct_dv_cols = [f"pct_dv_{n}" for n in nutrient_names]

    sensitivity_cutoffs = params.get("sensitivity_cutoffs", [
        [5.0, 15.0], [10.0, 20.0], [10.0, 25.0], [15.0, 30.0]
    ])
    main_low = params.get("low_pct", 10.0)
    main_high = params.get("high_pct", 20.0)

    ingredients = df_ingredients.to_dict("records")
    n = len(ingredients)
    ingredient_ids = [int(ingredient["ahn_id"]) for ingredient in ingredients]
    recipe_matrix = _recipe_matrix(recipes_list, ingredient_ids)
    recipe_cooccurrence = (recipe_matrix.T @ recipe_matrix).toarray()
    recipe_counts = np.asarray(recipe_matrix.sum(axis=0)).ravel()

    compound_values = sorted({compound for ingredient_id in ingredient_ids for compound in compounds_map.get(ingredient_id, set())})
    compound_to_row = {compound: row for row, compound in enumerate(compound_values)}
    compound_rows = []
    compound_cols = []
    for ingredient_col, ingredient_id in enumerate(ingredient_ids):
        for compound in compounds_map.get(ingredient_id, set()):
            compound_rows.append(compound_to_row[compound])
            compound_cols.append(ingredient_col)
    compound_matrix = sparse.csr_matrix(
        (np.ones(len(compound_rows), dtype=np.int32), (compound_rows, compound_cols)),
        shape=(len(compound_values), n),
    )
    compound_sharing = (compound_matrix.T @ compound_matrix).toarray()

    cuisine_matrices = {
        cuisine: _recipe_matrix(cuisine_recipes, ingredient_ids)
        for cuisine, cuisine_recipes in recipes_per_cuisine.items()
    }
    cuisine_cooccurrences = {
        cuisine: (matrix.T @ matrix).toarray()
        for cuisine, matrix in cuisine_matrices.items()
    }
    total_pairs = n * (n - 1) // 2
    logger.info(f"Generating {total_pairs} pairs from {n} ingredients...")

    rows = []
    for i in range(n):
        ing_a = ingredients[i]
        id_a = ing_a["ahn_id"]
        name_a = ing_a["name"]
        cat_a = ing_a.get("category", "General")
        ingredient_col_a = i

        # Extract %DV values for A
        pct_a = np.array([ing_a.get(col, np.nan) for col in pct_dv_cols], dtype=float)

        for j in range(i + 1, n):
            ing_b = ingredients[j]
            id_b = ing_b["ahn_id"]
            name_b = ing_b["name"]
            cat_b = ing_b.get("category", "General")
            ingredient_col_b = j

            # Flavor scores
            n_s = int(compound_sharing[ingredient_col_a, ingredient_col_b])
            comp_a = compounds_map.get(id_a, set())
            comp_b = compounds_map.get(id_b, set())
            n_s_norm = compute_normalized_flavor_sharing(comp_a, comp_b)

            pct_b = np.array([ing_b.get(col, np.nan) for col in pct_dv_cols], dtype=float)

            # Main complementarity C
            flags_a = classify_nutrient_status(pct_a, main_low, main_high)
            flags_b = classify_nutrient_status(pct_b, main_low, main_high)
            c_main = compute_complementarity(flags_a, flags_b, total_nutrients=len(nutrient_names))

            # Complementary drivers
            top_nutrients = identify_complementary_nutrients(flags_a, flags_b, nutrient_names)

            pair_dict = {
                "ahn_id_a": id_a,
                "name_a": name_a,
                "ahn_id_b": id_b,
                "name_b": name_b,
                "n_shared": n_s,
                "n_shared_norm": round(n_s_norm, 4),
                "comp_count_a": len(comp_a),
                "comp_count_b": len(comp_b),
                "comp_min": min(len(comp_a), len(comp_b)),
                "comp_max": max(len(comp_a), len(comp_b)),
                "C_main": round(c_main, 4),
                "category_pair": f"{min(str(cat_a), str(cat_b))} x {max(str(cat_a), str(cat_b))}",
                "is_spice_herb_a": bool(ing_a.get("is_spice_herb", False)),
                "is_spice_herb_b": bool(ing_b.get("is_spice_herb", False)),
                "top_nutrients": ";".join(top_nutrients)
            }

            # Sensitivity cutoffs
            for low_cut, high_cut in sensitivity_cutoffs:
                tag = f"C_{int(low_cut)}_{int(high_cut)}"
                fa_alt = classify_nutrient_status(pct_a, low_cut, high_cut)
                fb_alt = classify_nutrient_status(pct_b, low_cut, high_cut)
                c_alt = compute_complementarity(fa_alt, fb_alt, total_nutrients=len(nutrient_names))
                pair_dict[tag] = round(c_alt, 4)

            # Recipe co-occurrences
            n_recipes_both = int(recipe_cooccurrence[ingredient_col_a, ingredient_col_b])
            pair_dict["n_recipes_both"] = n_recipes_both
            pair_dict["n_a"] = int(recipe_counts[ingredient_col_a])
            pair_dict["n_b"] = int(recipe_counts[ingredient_col_b])
            pair_dict["lift"] = (
                n_recipes_both * len(recipes_list) / (recipe_counts[ingredient_col_a] * recipe_counts[ingredient_col_b])
                if recipe_counts[ingredient_col_a] and recipe_counts[ingredient_col_b] else 0.0
            )

            # Cuisine specific co-occurrences
            for cuisine, r_list in recipes_per_cuisine.items():
                c_both = int(cuisine_cooccurrences[cuisine][ingredient_col_a, ingredient_col_b])
                pair_dict[f"n_recipes_{cuisine}"] = c_both

            rows.append(pair_dict)

    df_pairs = pd.DataFrame(rows)
    return df_pairs


def run_pairs(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Execute Stage S5 pair table generation."""
    root = get_project_root()
    params = load_params(config_path)

    interim_dir = root / params["paths"]["interim"]
    processed_dir = root / params["paths"]["processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    ingredient_file = processed_dir / "ingredient_table.csv"
    compounds_file = interim_dir / "ahn_compounds.parquet"
    recipes_file = interim_dir / "ahn_recipes.parquet"
    out_pair_file = processed_dir / "pair_table.csv"

    if not ingredient_file.exists():
        logger.error(f"Cannot run S5: {ingredient_file} not found. Run S4 first.")
        return {"status": "error", "message": f"{ingredient_file} not found"}

    df_ing = pd.read_csv(ingredient_file)

    # Load compounds map
    compounds_map = load_compounds_map(
        compounds_file,
        excluded_compounds=set(params.get("filter_compounds", [951])),
    )

    # Load recipes
    recipes_list: List[Set[int]] = []
    recipes_per_cuisine: Dict[str, List[Set[int]]] = {}

    if recipes_file.exists():
        df_rec = pd.read_parquet(recipes_file)
        cuisine_col = "cuisine" if "cuisine" in df_rec.columns else None
        ingr_col = "ingredients" if "ingredients" in df_rec.columns else "ingredient"

        # Map ingredient names to ahn_ids
        name_to_id = dict(zip(df_ing["name"], df_ing["ahn_id"]))

        for _, r in df_rec.iterrows():
            items = r[ingr_col]
            if isinstance(items, str):
                items = [x.strip() for x in items.split(",")]
            item_ids = {name_to_id[x] for x in items if x in name_to_id}
            if cuisine_col and pd.notna(r[cuisine_col]):
                cuis = str(r[cuisine_col]).strip()
                if cuis not in recipes_per_cuisine:
                    recipes_per_cuisine[cuis] = []
                recipes_per_cuisine[cuis].append(item_ids)
            if item_ids:
                recipes_list.append(item_ids)

    df_pairs = build_pair_table(df_ing, compounds_map, recipes_list, recipes_per_cuisine, params)
    df_pairs.to_csv(out_pair_file, index=False)
    pd.DataFrame(
        [{"cuisine": cuisine, "n_recipes": len(recipe_list)}
         for cuisine, recipe_list in recipes_per_cuisine.items()]
    ).to_csv(processed_dir / "cuisine_counts.csv", index=False)
    logger.info(f"S5 complete. Computed metrics for {len(df_pairs)} pairs. Saved to {out_pair_file}")

    return {"status": "success", "n_pairs": len(df_pairs), "file": str(out_pair_file)}


if __name__ == "__main__":
    run_pairs()
