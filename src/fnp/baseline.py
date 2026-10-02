"""Statistical baselines, bootstrap confidence intervals, and permutation tests."""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
from fnp.scores import compute_complementarity


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """Compute Spearman correlation with bootstrap confidence intervals."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 3:
        return {"rho": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}

    res = stats.spearmanr(x, y)
    obs_rho = float(res.statistic)
    p_val = float(res.pvalue)

    indices = np.arange(n)
    boot_rhos = []
    for _ in range(n_boot):
        sample_idx = rng.choice(indices, size=n, replace=True)
        r = stats.spearmanr(x[sample_idx], y[sample_idx]).statistic
        if not np.isnan(r):
            boot_rhos.append(r)

    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.percentile(boot_rhos, alpha * 100))
    ci_high = float(np.percentile(boot_rhos, (1.0 - alpha) * 100))

    return {
        "rho": obs_rho,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_val,
        "n": n
    }


def bootstrap_spearman_by_ingredient(
    df_pairs: pd.DataFrame,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
    x_col: str = "n_shared",
    y_col: str = "C_main",
) -> Dict[str, float]:
    """Bootstrap Spearman by resampling ingredients with vectorized pair lookup."""
    ingredient_ids = sorted(set(df_pairs["ahn_id_a"]) | set(df_pairs["ahn_id_b"]))
    if len(ingredient_ids) < 3:
        return {"rho": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    id_to_position = {ingredient_id: position for position, ingredient_id in enumerate(ingredient_ids)}
    size = len(ingredient_ids)
    x_matrix = np.full((size, size), np.nan)
    y_matrix = np.full((size, size), np.nan)
    for row in df_pairs.itertuples():
        first = id_to_position[row.ahn_id_a]
        second = id_to_position[row.ahn_id_b]
        x_matrix[first, second] = x_matrix[second, first] = getattr(row, x_col)
        y_matrix[first, second] = y_matrix[second, first] = getattr(row, y_col)
    observed = stats.spearmanr(df_pairs[x_col], df_pairs[y_col])
    rng = np.random.default_rng(seed)
    boot_rhos = []
    upper = np.triu_indices(size, k=1)
    for _ in range(n_boot):
        sampled = rng.choice(size, size=size, replace=True)
        sampled_x = x_matrix[np.ix_(sampled, sampled)][upper]
        sampled_y = y_matrix[np.ix_(sampled, sampled)][upper]
        distinct = sampled[upper[0]] != sampled[upper[1]]
        valid = distinct & ~np.isnan(sampled_x) & ~np.isnan(sampled_y)
        values_x = sampled_x[valid]
        values_y = sampled_y[valid]
        if len(values_x) >= 3:
            rho = stats.spearmanr(values_x, values_y).statistic
            if not np.isnan(rho):
                boot_rhos.append(rho)

    alpha = (1.0 - ci) / 2.0
    return {
        "rho": float(observed.statistic),
        "ci_low": float(np.percentile(boot_rhos, alpha * 100)) if boot_rhos else np.nan,
        "ci_high": float(np.percentile(boot_rhos, (1.0 - alpha) * 100)) if boot_rhos else np.nan,
        "p_value": float(observed.pvalue),
        "n": len(df_pairs),
    }


def permutation_test_profile_pairs(
    pair_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    n_perm: int = 5000,
    seed: int = 42,
    within_category: bool = False,
) -> Dict[str, Any]:
    """Test observed pair complementarity against permuted ingredient profiles."""
    flag_cols = [col for col in profile_df.columns if col.startswith("flag_")]
    profile_ids = profile_df["ahn_id"].tolist()
    profile_values = profile_df.set_index("ahn_id")[flag_cols].to_numpy(dtype=object)
    category_groups = None
    if within_category:
        if "category" not in profile_df.columns:
            raise ValueError("profile_df must contain a category column for within-category shuffling")
        category_groups = profile_df.groupby("category", dropna=False, sort=False).indices.values()
    id_to_position = {ingredient_id: index for index, ingredient_id in enumerate(profile_ids)}
    valid_rows = [
        row for row in pair_df.itertuples()
        if row.ahn_id_a in id_to_position and row.ahn_id_b in id_to_position
    ]
    pair_positions = [
        (id_to_position[row.ahn_id_a], id_to_position[row.ahn_id_b])
        for row in valid_rows
    ]
    observed_scores = np.asarray([row.C_main for row in valid_rows], dtype=float)
    observed_mean = float(np.mean(observed_scores))
    rng = np.random.default_rng(seed)
    null_means = np.empty(n_perm)
    for permutation in range(n_perm):
        if category_groups is None:
            shuffled = profile_values[rng.permutation(len(profile_values))]
        else:
            shuffled = profile_values.copy()
            for positions in category_groups:
                positions = np.asarray(positions)
                shuffled[positions] = profile_values[rng.permutation(positions)]
        scores = [
            compute_complementarity(shuffled[first], shuffled[second], total_nutrients=len(flag_cols))
            for first, second in pair_positions
        ]
        null_means[permutation] = np.mean(scores)
    null_mean = float(np.mean(null_means))
    null_std = float(np.std(null_means, ddof=1))
    greater_count = int(np.count_nonzero(null_means >= observed_mean))
    less_count = int(np.count_nonzero(null_means <= observed_mean))
    p_greater = (greater_count + 1) / (n_perm + 1)
    p_less = (less_count + 1) / (n_perm + 1)
    return {
        "observed_mean": observed_mean,
        "null_mean": null_mean,
        "null_std": null_std,
        "z": (observed_mean - null_mean) / null_std if null_std else 0.0,
        "p_greater": p_greater,
        "p_less": p_less,
        "p_two_sided": min(1.0, 2.0 * min(p_greater, p_less)),
        "p_value": p_greater,
        "n_perm": n_perm,
    }


def permutation_test_real_vs_random(
    real_scores: np.ndarray,
    all_scores: np.ndarray,
    n_perm: int = 5000,
    seed: int = 42
) -> Dict[str, Any]:
    """Perform permutation / Monte Carlo test comparing real recipe pairs to random pairs.

    Tests whether the mean complementarity of real pairs is significantly greater than
    random pairs drawn from the pair universe.
    """
    rng = np.random.default_rng(seed)
    k = len(real_scores)
    obs_mean = float(np.mean(real_scores))

    perm_means = np.zeros(n_perm)
    for i in range(n_perm):
        sim_sample = rng.choice(all_scores, size=k, replace=False)
        perm_means[i] = np.mean(sim_sample)

    p_value = float(np.mean(perm_means >= obs_mean))
    null_mean = float(np.mean(perm_means))
    null_std = float(np.std(perm_means, ddof=1))
    effect_size = (obs_mean - null_mean) / null_std if null_std > 0 else 0.0

    return {
        "observed_mean": obs_mean,
        "null_mean": null_mean,
        "null_std": null_std,
        "effect_size_z": effect_size,
        "p_value": p_value,
        "n_perm": n_perm,
        "ci_low_95": float(np.percentile(perm_means, 2.5)),
        "ci_high_95": float(np.percentile(perm_means, 97.5))
    }
