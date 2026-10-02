"""Pure scoring functions for flavor sharing (N_s) and nutrient complementarity (C).

All functions in this module are stateless and side-effect free, designed for direct
unit testing and vectorized execution.
"""

from typing import Any, Set, Sequence, Dict, Tuple, List, Union
import numpy as np
import pandas as pd


def compute_flavor_sharing(compounds_a: Set[Any], compounds_b: Set[Any]) -> int:
    """Compute the raw flavor sharing score N_s between two compound sets.

    Parameters
    ----------
    compounds_a : Set
        Set of compound IDs or names for ingredient A.
    compounds_b : Set
        Set of compound IDs or names for ingredient B.

    Returns
    -------
    int
        N_s = |compounds_a ∩ compounds_b|
    """
    if not isinstance(compounds_a, set):
        compounds_a = set(compounds_a)
    if not isinstance(compounds_b, set):
        compounds_b = set(compounds_b)
    return len(compounds_a.intersection(compounds_b))


def compute_normalized_flavor_sharing(compounds_a: Set[Any], compounds_b: Set[Any]) -> float:
    """Compute size-controlled flavor sharing score normalized by the smaller compound set.

    Parameters
    ----------
    compounds_a : Set
    compounds_b : Set

    Returns
    -------
    float
        N_s / min(|compounds_a|, |compounds_b|) or 0.0 if either set is empty.
    """
    if not isinstance(compounds_a, set):
        compounds_a = set(compounds_a)
    if not isinstance(compounds_b, set):
        compounds_b = set(compounds_b)

    min_size = min(len(compounds_a), len(compounds_b))
    if min_size == 0:
        return 0.0

    raw_shared = len(compounds_a.intersection(compounds_b))
    return float(raw_shared) / float(min_size)


def compute_percent_dv(
    amount: Union[float, np.ndarray, pd.Series],
    daily_value: float
) -> Union[float, np.ndarray, pd.Series]:
    """Convert raw nutrient amount per 100g to percentage of Daily Value (%DV).

    Parameters
    ----------
    amount : float, np.ndarray, or pd.Series
        Nutrient amount per 100g.
    daily_value : float
        Reference Daily Value in the same unit.

    Returns
    -------
    %DV = (amount / daily_value) * 100
    """
    if daily_value <= 0:
        raise ValueError(f"Daily value must be strictly positive, got {daily_value}")
    return (amount / daily_value) * 100.0


def classify_nutrient_status(
    pct_dv: Union[float, np.ndarray, pd.Series],
    low_pct: float = 10.0,
    high_pct: float = 20.0
) -> Union[str, np.ndarray, pd.Series]:
    """Classify %DV into 'low', 'high', or 'neither'.

    Parameters
    ----------
    pct_dv : float, np.ndarray, or pd.Series
    low_pct : float, default 10.0
        Threshold below which a nutrient is classified as 'low'.
    high_pct : float, default 20.0
        Threshold at or above which a nutrient is classified as 'high'.

    Returns
    -------
    'low', 'high', 'neither', or np.nan if input is NaN.
    """
    if low_pct >= high_pct:
        raise ValueError(f"low_pct ({low_pct}) must be less than high_pct ({high_pct})")

    if isinstance(pct_dv, (pd.Series, np.ndarray)):
        arr = np.array(pct_dv, dtype=float)
        result = np.full(arr.shape, "neither", dtype=object)
        nan_mask = np.isnan(arr)
        result[arr < low_pct] = "low"
        result[arr >= high_pct] = "high"
        result[nan_mask] = np.nan
        if isinstance(pct_dv, pd.Series):
            return pd.Series(result, index=pct_dv.index)
        return result

    if pd.isna(pct_dv):
        return np.nan
    if pct_dv < low_pct:
        return "low"
    elif pct_dv >= high_pct:
        return "high"
    else:
        return "neither"


def compute_complementarity(
    flags_a: Sequence[str],
    flags_b: Sequence[str],
    total_nutrients: int = 11
) -> float:
    """Compute nutrient complementarity score C(A, B) between two ingredients.

    C(A, B) = ( #{k : A low, B high} + #{k : B low, A high} ) / K

    Parameters
    ----------
    flags_a : Sequence of str
        Nutrient classifications for food A ('low', 'high', 'neither').
    flags_b : Sequence of str
        Nutrient classifications for food B ('low', 'high', 'neither').
    total_nutrients : int, default 11
        Total nutrient count K used as denominator.

    Returns
    -------
    float
        Complementarity score C in [0.0, 1.0].
    """
    if len(flags_a) != len(flags_b):
        raise ValueError("flags_a and flags_b must have the exact same length")
    if total_nutrients <= 0:
        raise ValueError(f"total_nutrients must be positive, got {total_nutrients}")

    gap_fills = 0
    for fa, fb in zip(flags_a, flags_b):
        if (fa == "low" and fb == "high") or (fa == "high" and fb == "low"):
            gap_fills += 1

    return float(gap_fills) / float(total_nutrients)


def identify_complementary_nutrients(
    flags_a: Sequence[str],
    flags_b: Sequence[str],
    nutrient_names: Sequence[str]
) -> List[str]:
    """Identify list of nutrients driving complementarity between A and B.

    Parameters
    ----------
    flags_a : Sequence of str
    flags_b : Sequence of str
    nutrient_names : Sequence of str

    Returns
    -------
    List of nutrient names where one ingredient is low and the other is high.
    """
    if not (len(flags_a) == len(flags_b) == len(nutrient_names)):
        raise ValueError("Length mismatch between flags and nutrient names")

    drivers = []
    for fa, fb, name in zip(flags_a, flags_b, nutrient_names):
        if (fa == "low" and fb == "high") or (fa == "high" and fb == "low"):
            drivers.append(name)
    return drivers
