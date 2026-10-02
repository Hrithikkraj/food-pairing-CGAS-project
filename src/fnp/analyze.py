"""Stage S6: Statistical analysis plan (RQ1, RQ2, RQ3, sensitivity, and case studies).

Runs:
    - A1: All-pairs correlation with bootstrap CI (Spearman & Pearson)
    - A2: Controlled OLS regression with category and compound controls
    - A3: Real recipe pairs vs random pairs permutation test
    - A4: Cuisine breakdown
    - A5: Sensitivity grid across thresholds and size-controlled N_s
    - A6: Selected & randomized case studies

Outputs:
    reports/tables/*.csv
    reports/results.json
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from fnp.utils import get_project_root, load_params, compute_params_hash, setup_logger
from fnp.baseline import (
    bootstrap_spearman,
    bootstrap_spearman_by_ingredient,
    permutation_test_real_vs_random,
    permutation_test_profile_pairs,
)

logger = setup_logger("fnp.analyze", "logs/pipeline.log")


def aggregate_cuisine_counts(
    df_pairs: pd.DataFrame,
    cuisine_counts: pd.DataFrame,
    cuisine_groups: Dict[str, List[str]],
) -> tuple[Dict[str, pd.Series], Dict[str, int], List[str]]:
    """Aggregate raw cuisine pair and recipe counts into configured groups."""
    pair_labels = {
        column.removeprefix("n_recipes_")
        for column in df_pairs.columns
        if column.startswith("n_recipes_") and column != "n_recipes_both"
    }
    recipe_labels = set(cuisine_counts["cuisine"].astype(str)) if not cuisine_counts.empty else set()
    known_labels = pair_labels | recipe_labels
    grouped_labels = {label for labels in cuisine_groups.values() for label in labels}
    ungrouped_labels = sorted(known_labels - grouped_labels)

    pair_counts = {}
    recipe_counts = {}
    for group, labels in cuisine_groups.items():
        columns = [f"n_recipes_{label}" for label in labels if f"n_recipes_{label}" in df_pairs.columns]
        pair_counts[group] = (
            df_pairs[columns].fillna(0).sum(axis=1)
            if columns else pd.Series(0, index=df_pairs.index, dtype=float)
        )
        recipe_counts[group] = int(
            cuisine_counts.loc[cuisine_counts["cuisine".strip()].isin(labels), "n_recipes"].sum()
        ) if not cuisine_counts.empty else 0
    return pair_counts, recipe_counts, ungrouped_labels


def json_default(value: Any) -> Any:
    """Convert numpy values and non-finite numbers to JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def json_safe(value: Any) -> Any:
    """Recursively replace native non-finite floats before json encoding."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return json_default(value)
    return value


def run_analyze(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Execute Stage S6 statistical analysis pipeline."""
    root = get_project_root()
    params = load_params(config_path)
    seed = params.get("seed", 42)
    n_boot = params.get("n_boot", 2000)
    n_perm = params.get("n_perm", 5000)
    params_hash = compute_params_hash(params)

    processed_dir = root / params["paths"]["processed"]
    reports_dir = root / params["paths"]["reports"]
    tables_dir = root / params["paths"]["reports_tables"]
    tables_dir.mkdir(parents=True, exist_ok=True)

    pair_file = processed_dir / "pair_table.csv"
    if not pair_file.exists():
        logger.error(f"Cannot run S6: {pair_file} not found. Run S5 first.")
        return {"status": "error", "message": f"{pair_file} not found"}

    df_pairs = pd.read_csv(pair_file)
    logger.info(f"Loaded {len(df_pairs)} pairs for analysis.")
    profile_file = processed_dir / "ingredient_table.csv"
    df_profiles = pd.read_csv(profile_file) if profile_file.exists() else pd.DataFrame()

    results = {
        "params_hash": params_hash,
        "seed": seed,
        "total_pairs": len(df_pairs),
        "analyses": {}
    }

    # ----------------------------------------------------
    # A1: All-pairs correlation (RQ1)
    # ----------------------------------------------------
    logger.info("Running A1: All-pairs correlation...")
    for suffix, df_variant in [("all", df_pairs), ("no_spice_herb", df_pairs[
        ~(df_pairs["is_spice_herb_a"].astype(str).str.lower().eq("true") |
          df_pairs["is_spice_herb_b"].astype(str).str.lower().eq("true"))
    ])]:
        if len(df_variant) < 3:
            continue
        spearman_res = bootstrap_spearman_by_ingredient(df_variant, n_boot=n_boot, seed=seed)
        pearson_r, pearson_p = stats.pearsonr(df_variant["n_shared"], df_variant["C_main"])
        a1_dict = {
            "spearman_rho": round(spearman_res["rho"], 4),
            "ci_low": round(spearman_res["ci_low"], 4),
            "ci_high": round(spearman_res["ci_high"], 4),
            "naive_p_value": spearman_res["p_value"],
            "pearson_r": round(float(pearson_r), 4),
            "pearson_p": float(pearson_p),
        }
        results["analyses"][f"A1_correlation_{suffix}"] = a1_dict
        pd.DataFrame([a1_dict]).to_csv(tables_dir / f"a1_correlation_{suffix}.csv", index=False)

    # ----------------------------------------------------
    # A2: Controlled Regression (RQ1)
    # ----------------------------------------------------
    logger.info("Running A2: Controlled OLS regression...")
    try:
        # Filter top categories or group small ones to avoid rank deficiency
        top_cats = df_pairs["category_pair"].value_counts().nlargest(15).index
        df_reg = df_pairs.copy()
        df_reg["category_pair_grouped"] = df_reg["category_pair"].apply(lambda c: c if c in top_cats else "Other")

        a2_results = {}
        for x_column, suffix in [("n_shared", "n_shared"), ("n_shared_norm", "n_shared_norm")]:
            model = smf.ols(
                f"C_main ~ {x_column} + np.log(comp_min) + np.log(comp_max) + C(category_pair_grouped)",
                data=df_reg,
            )
            fit = model.fit(cov_type="HC3")
            a2_table = pd.DataFrame({
                "coef": fit.params,
                "std_err": fit.bse,
                "p_value": fit.pvalues,
                "ci_low": fit.conf_int()[0],
                "ci_high": fit.conf_int()[1]
            }).reset_index().rename(columns={"index": "term"})
            a2_table.to_csv(tables_dir / f"a2_regression_{suffix}.csv", index=False)
            a2_results[suffix] = {
                "coefficient": float(fit.params.get(x_column, 0.0)),
                "p_value": float(fit.pvalues.get(x_column, 1.0)),
                "r_squared": float(fit.rsquared),
            }
        results["analyses"]["A2_regression"] = a2_results
        results["analyses"]["A2_note"] = "Standard errors are pair-level and optimistic because pairs share ingredients."
    except Exception as e:
        logger.warning(f"Regression skipped or failed: {e}")

    # ----------------------------------------------------
    # A3: Real vs Random Pairs (RQ2)
    # ----------------------------------------------------
    logger.info("Running A3: Real vs Random Pairs Permutation Test...")
    min_pair_recipes = params.get("min_pair_recipes", 20)
    eligible_pairs = df_pairs[df_pairs["n_recipes_both"] >= min_pair_recipes].copy()
    lift_cutoff = eligible_pairs["lift"].median() if not eligible_pairs.empty else np.nan
    real_pairs = eligible_pairs[eligible_pairs["lift"] >= lift_cutoff]
    for suffix, df_variant in [("all", real_pairs), ("no_spice_herb", real_pairs[
        ~(real_pairs["is_spice_herb_a"].astype(str).str.lower().eq("true") |
          real_pairs["is_spice_herb_b"].astype(str).str.lower().eq("true"))
    ])]:
        if len(df_variant) == 0 or df_profiles.empty:
            logger.warning("No eligible A3 pairs or ingredient profiles for %s.", suffix)
            continue
        for shuffle_suffix, within_category in [("global", False), ("within_category", True)]:
            a3_res = permutation_test_profile_pairs(
                df_variant,
                df_profiles,
                n_perm=n_perm,
                seed=seed,
                within_category=within_category,
            )
            naive = permutation_test_real_vs_random(
                real_scores=df_variant["C_main"].values,
                all_scores=df_variant["C_main"].values if len(df_variant) == len(df_pairs) else df_pairs["C_main"].values,
                n_perm=n_perm,
                seed=seed,
            )
            a3_res["naive_p_value"] = naive["p_value"]
            result_suffix = f"{suffix}_{shuffle_suffix}"
            results["analyses"][f"A3_real_vs_random_{result_suffix}"] = a3_res
            pd.DataFrame([a3_res]).to_csv(tables_dir / f"a3_permutation_{result_suffix}.csv", index=False)

    # ----------------------------------------------------
    # A4: Cuisine Split (RQ3)
    # ----------------------------------------------------
    logger.info("Running A4: Cuisine Split...")
    cuisine_counts_file = processed_dir / "cuisine_counts.csv"
    cuisine_counts = pd.read_csv(cuisine_counts_file) if cuisine_counts_file.exists() else pd.DataFrame(columns=["cuisine", "n_recipes"])
    min_cuisine_recipes = params.get("min_recipes_per_cuisine", 50)
    min_cuisine_pairs = params.get("min_pair_recipes_cuisine", 5)
    cuisine_groups = params.get("cuisine_groups", {})
    grouped_pair_counts, grouped_recipe_counts, ungrouped_labels = aggregate_cuisine_counts(
        df_pairs, cuisine_counts, cuisine_groups
    )
    if ungrouped_labels:
        logger.warning("Cuisine labels not assigned to a group: %s", ", ".join(ungrouped_labels))
    cuisine_rows = []

    for group, group_pair_counts in grouped_pair_counts.items():
        n_recipes = grouped_recipe_counts[group]
        sub_real = df_pairs[group_pair_counts >= min_cuisine_pairs]
        if n_recipes >= min_cuisine_recipes and len(sub_real) >= 3:
            sp = bootstrap_spearman_by_ingredient(sub_real, n_boot=n_boot, seed=seed)
            cuisine_rows.append({
                "group": group,
                "n_recipes": n_recipes,
                "n_pairs": len(sub_real),
                "mean_C": float(np.mean(sub_real["C_main"])),
                "rho": sp["rho"],
                "ci_low": sp["ci_low"],
                "ci_high": sp["ci_high"],
                "naive_p_value": sp["p_value"]
            })

    df_cuisines = pd.DataFrame(cuisine_rows, columns=[
        "group", "n_recipes", "n_pairs", "mean_C", "rho", "ci_low", "ci_high", "naive_p_value"
    ])
    df_cuisines.to_csv(tables_dir / "a4_cuisines.csv", index=False)
    results["analyses"]["A4_cuisine_split"] = cuisine_rows

    # ----------------------------------------------------
    # A5: Sensitivity Analysis (Cutoff Grid & Norm N_s)
    # ----------------------------------------------------
    logger.info("Running A5: Sensitivity Analysis...")
    c_cols = [col for col in df_pairs.columns if col.startswith("C_")]
    sens_rows = []
    for c_metric in c_cols:
        for x_column in ("n_shared", "n_shared_norm"):
            res = bootstrap_spearman_by_ingredient(
                df_pairs, n_boot=n_boot, seed=seed, x_col=x_column, y_col=c_metric
            )
            sens_rows.append({
                "x": x_column,
                "metric": c_metric,
                "rho": res["rho"],
                "ci_low": res["ci_low"],
                "ci_high": res["ci_high"],
                "naive_p_value": res["p_value"]
            })
    df_sens = pd.DataFrame(sens_rows)
    df_sens.to_csv(tables_dir / "a5_sensitivity.csv", index=False)
    results["analyses"]["A5_sensitivity"] = sens_rows

    # ----------------------------------------------------
    # A6: Case Studies
    # ----------------------------------------------------
    logger.info("Running A6: Case Studies...")
    target_names = [
        ("rice", "bean"), ("tomato", "cheese"), ("rice", "chicken"),
        ("butter", "milk"), ("tomato", "basil")
    ]
    case_rows = []
    available_names = set(df_pairs["name_a"]) | set(df_pairs["name_b"])

    for name_a, name_b in target_names:
        missing_names = [name for name in (name_a, name_b) if name not in available_names]
        if missing_names:
            logger.warning("Skipping case study %s + %s; missing name(s): %s", name_a, name_b, ", ".join(missing_names))
            continue
        match = df_pairs[
            ((df_pairs["name_a"] == name_a) & (df_pairs["name_b"] == name_b)) |
            ((df_pairs["name_a"] == name_b) & (df_pairs["name_b"] == name_a))
        ]
        if not match.empty:
            m = match.iloc[0]
            case_rows.append({
                "pair": f"{m['name_a']} + {m['name_b']}",
                "type": "Curated",
                "n_shared": m["n_shared"],
                "C_main": m["C_main"],
                "top_nutrients": m["top_nutrients"]
            })

    # Add 3 random pairs
    rng = np.random.default_rng(seed)
    if len(df_pairs) >= 3:
        rand_idx = rng.choice(len(df_pairs), size=3, replace=False)
        for idx in rand_idx:
            m = df_pairs.iloc[idx]
            case_rows.append({
                "pair": f"{m['name_a']} + {m['name_b']}",
                "type": "Random Baseline",
                "n_shared": m["n_shared"],
                "C_main": m["C_main"],
                "top_nutrients": m["top_nutrients"]
            })

    df_cases = pd.DataFrame(case_rows)
    df_cases.to_csv(tables_dir / "a6_case_studies.csv", index=False)
    results["analyses"]["A6_case_studies"] = case_rows

    # Write master results.json
    results_json_path = reports_dir / "results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(results), f, indent=2, default=json_default, allow_nan=False)

    logger.info(f"S6 complete. Saved master results to {results_json_path}")
    return {"status": "success", "results_file": str(results_json_path)}


if __name__ == "__main__":
    run_analyze()
