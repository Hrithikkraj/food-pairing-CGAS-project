"""Stage S3: Match ingredients to USDA foods with RapidFuzz and Human-in-the-Loop review.

Generates candidate matches between Ahn ingredient names and USDA SR Legacy food descriptions
with category filtering and a human-review candidate ranking.

Outputs:
    data/external/matching_log.csv:
        Columns: ahn_id, ahn_name, fdc_id, usda_description, match_score, reviewer, note,
        cand1_fdc_id, cand1_description, ... cand10_fdc_id, cand10_description
"""

import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from rapidfuzz import fuzz

from fnp.utils import get_project_root, load_params, setup_logger

logger = setup_logger("fnp.match", "logs/pipeline.log")

EXCLUDED_USDA_CATEGORIES = {
    "Baby Foods",
    "Restaurant Foods",
    "Fast Foods",
    "Snacks",
    "Sweets",
    "Meals, Entrees, and Side Dishes",
    "American Indian/Alaska Native Foods",
}
PREPARATION_PENALTIES = {
    "canned", "frozen", "dried", "prepared", "cooked", "sauce",
    "juice", "shake", "shakes", "dessert", "bar", "powder", "flour",
    "noodles", "bran", "substitute", "concentrate",
}


def clean_name(name: str) -> str:
    """Normalize text for fuzzy matching."""
    return str(name).lower().replace("_", " ").strip()


def score_candidate(query: str, description: str, food_category: Optional[str] = None, is_spice_herb: bool = False) -> float:
    """Rank a USDA description without treating an interior word as an exact match."""
    query = clean_name(query)
    description = str(description).lower().strip()
    query_tokens = query.split()
    desc_tokens = re.findall(r"[a-z0-9]+", description)
    normalized_description = " ".join(desc_tokens)
    starts_description = re.sub(r"^(spices|herbs)\s+", "", normalized_description)
    starts = starts_description.startswith(query) or starts_description.startswith(query + "s") or starts_description.startswith(query + "es")
    score = fuzz.ratio(query, " ".join(re.findall(r"[a-z0-9]+", starts_description)[: max(len(query_tokens), 1)]))
    if starts:
        score += 35.0
    if "raw" in desc_tokens:
        score += 20.0
    if not (set(query_tokens) & PREPARATION_PENALTIES):
        score -= 7.0 * len(set(desc_tokens) & PREPARATION_PENALTIES)
    if is_spice_herb and food_category == "Spices and Herbs":
        score += 30.0
    return score


def run_match(
    config_path: Optional[str] = None,
    top_k: int = 10,
    auto_assign_top: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """Execute Stage S3 ingredient matching."""
    root = get_project_root()
    params = load_params(config_path)

    interim_dir = root / params["paths"]["interim"]
    external_dir = root / params["paths"]["external"]
    external_dir.mkdir(parents=True, exist_ok=True)

    filtered_file = interim_dir / "ingredients_filtered.csv"
    usda_food_file = interim_dir / "usda_food.parquet"
    matching_log_path = external_dir / "matching_log.csv"

    if not filtered_file.exists():
        logger.error(f"Cannot run S3: {filtered_file} not found. Run S2 first.")
        return {"status": "error", "message": f"{filtered_file} not found"}

    df_ingr = pd.read_csv(filtered_file)

    df_existing = None
    if matching_log_path.exists():
        df_existing = pd.read_csv(matching_log_path)
        logger.info(f"Existing matching log found with {len(df_existing)} entries.")
        if "ahn_id" not in df_existing.columns:
            raise ValueError("Existing matching log is missing the ahn_id column.")
        missing_ids = df_existing[df_existing["ahn_id"].isna()]
        if not missing_ids.empty:
            row_numbers = (missing_ids.index + 2).tolist()
            raise ValueError(f"Existing matching log has missing ahn_id in row(s): {row_numbers}.")
        df_existing["ahn_id"] = pd.to_numeric(df_existing["ahn_id"], errors="raise").astype(int)
        if force:
            backup_path = external_dir / "matching_log.backup.csv"
            df_existing.to_csv(backup_path, index=False)
            logger.warning("Force matching requested; backed up existing log to %s", backup_path)
            df_existing = None

    if not usda_food_file.exists():
        logger.warning(f"USDA food interim file {usda_food_file} not found. Cannot search USDA descriptions.")
        return {"status": "pending_usda_data", "file": str(matching_log_path)}

    df_usda = pd.read_parquet(usda_food_file)
    logger.info(f"Loaded {len(df_usda)} USDA food entries.")

    category_file = interim_dir / "usda_food_category.parquet"
    if category_file.exists() and "food_category_id" in df_usda.columns:
        df_categories = pd.read_parquet(category_file)
        category_map = dict(zip(df_categories["id"], df_categories["description"]))
        df_usda["food_category"] = df_usda["food_category_id"].map(category_map)
        df_usda = df_usda[~df_usda["food_category"].isin(EXCLUDED_USDA_CATEGORIES)].copy()
    logger.info(f"Retained {len(df_usda)} USDA entries after category filtering.")

    usda_records = df_usda.to_dict("records")

    log_rows = []
    logger.info(f"Generating fuzzy match candidates for {len(df_ingr)} ingredients...")

    def build_candidates(row: pd.Series) -> Dict[str, Any]:
        ahn_id = row["ahn_id"]
        ahn_name = row["ahn_name"]
        query = clean_name(ahn_name)
        is_spice_herb = str(row.get("category", "")).lower() in {"spice", "herb"}

        candidates = sorted(
            ((record["description"], score_candidate(
                query,
                record["description"],
                record.get("food_category"),
                is_spice_herb,
            ), index)
             for index, record in enumerate(usda_records)),
            key=lambda item: (-item[1], len(item[0]), item[0])
        )[:top_k]

        if not candidates:
            return {
                "ahn_id": ahn_id,
                "ahn_name": ahn_name,
                "match_score": 0.0,
            }

        best_desc, best_score, best_index = candidates[0]
        candidate_fields = {}
        for candidate_number, (candidate_desc, candidate_score, candidate_index) in enumerate(candidates, start=1):
            candidate_fields[f"cand{candidate_number}_fdc_id"] = usda_records[candidate_index]["fdc_id"]
            candidate_fields[f"cand{candidate_number}_description"] = candidate_desc

        return {
            "ahn_id": ahn_id,
            "ahn_name": ahn_name,
            "match_score": round(best_score, 2),
            **candidate_fields
        }

    existing_by_id = {} if df_existing is None else {
        int(row["ahn_id"]): dict(row)
        for row in df_existing.to_dict("records")
    }
    log_rows = []
    for _, row in df_ingr.iterrows():
        ahn_id = row["ahn_id"]
        if ahn_id in existing_by_id:
            existing = existing_by_id[ahn_id]
            if pd.notna(existing.get("fdc_id")):
                existing["ahn_id"] = int(existing["ahn_id"])
                log_rows.append(existing)
            else:
                refreshed = dict(existing)
                refreshed.update(build_candidates(row))
                log_rows.append(refreshed)
        else:
            candidate_row = build_candidates(row)
            candidate_row.update({
                "fdc_id": None,
                "usda_description": None,
                "reviewer": "pending_human_review",
                "note": "Select a candidate manually and fill fdc_id.",
            })
            log_rows.append(candidate_row)

    df_log = pd.DataFrame(log_rows)
    if "fdc_id" in df_log.columns:
        df_log["fdc_id"] = pd.to_numeric(df_log["fdc_id"], errors="coerce").astype("Int64")
    df_log["ahn_id"] = pd.to_numeric(df_log["ahn_id"], errors="raise").astype(int)
    df_log.to_csv(matching_log_path, index=False)
    logger.info(f"S3 complete. Matching log written to {matching_log_path}")

    return {"status": "success", "file": str(matching_log_path), "count": len(df_log)}


if __name__ == "__main__":
    run_match()
