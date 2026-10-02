"""Stage S1: Ingest and validate raw datasets (Ahn et al. and USDA FoodData Central).

Reads data/raw/*, validates schemas, converts to interim parquet/csv, and produces
reports/data_audit.md documenting table shapes, missingness, and verified column names.
"""

import sys
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from fnp.utils import get_project_root, load_params, setup_logger

logger = setup_logger("fnp.ingest", "logs/pipeline.log")

# Expected USDA SR Legacy schemas
USDA_EXPECTED_COLS = {
    "food": ["fdc_id", "data_type", "description", "food_category_id", "publication_date"],
    "nutrient": ["id", "name", "unit_name", "nutrient_nbr", "rank"],
    "food_nutrient": ["id", "fdc_id", "nutrient_id", "amount"],
    "food_category": ["id", "code", "description"]
}


def find_file(directory: Path, patterns: List[str]) -> Optional[Path]:
    """Search for a file matching any of the given patterns (shallow then recursive)."""
    for pattern in patterns:
        for m in directory.glob(pattern):
            if m.is_file():
                return m
    for pattern in patterns:
        for m in directory.rglob(pattern):
            if m.is_file():
                return m
    return None


def run_ingest(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Execute Stage S1 data ingestion and schema audit."""
    root = get_project_root()
    params = load_params(config_path)

    raw_usda_dir = root / params["paths"].get("raw_usda", "data/raw/usda")
    raw_ahn_dir = root / params["paths"].get("raw_ahn", "data/raw/ahn")
    interim_dir = root / params["paths"]["interim"]
    reports_dir = root / params["paths"]["reports"]

    interim_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    audit_lines = [
        "# Stage S1 Data Audit Report",
        "",
        "This document details the loaded schemas, row counts, ID ranges, and missing value checks.",
        "",
        "## 1. USDA SR Legacy Tables",
        ""
    ]

    logger.info("Starting S1: Ingestion and validation...")
    usda_loaded = {}

    # Ingest USDA tables
    for table_name, expected_cols in USDA_EXPECTED_COLS.items():
        candidates = [f"{table_name}.csv", f"{table_name.upper()}.csv", f"*{table_name}*.csv"]
        file_path = find_file(raw_usda_dir, candidates)

        if file_path is None or not file_path.exists():
            msg = f"USDA table '{table_name}' not found in {raw_usda_dir}. Expected pattern: {candidates}"
            logger.warning(msg)
            audit_lines.append(f"- **{table_name}**: Missing file in `{raw_usda_dir}`.")
            continue

        df = pd.read_csv(file_path, low_memory=False)
        # Check required columns
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing expected columns in {file_path.name}: {missing_cols}")

        usda_loaded[table_name] = df
        out_path = interim_dir / f"usda_{table_name}.parquet"
        df.to_parquet(out_path, index=False)

        audit_lines.append(f"### Table: `usda_{table_name}`")
        audit_lines.append(f"- **Source File**: `{file_path.name}`")
        audit_lines.append(f"- **Shape**: {df.shape[0]:,} rows x {df.shape[1]} columns")
        audit_lines.append(f"- **Columns**: {', '.join(df.columns)}")
        null_counts = df[expected_cols].isnull().sum().to_dict()
        audit_lines.append(f"- **Null counts in key columns**: `{null_counts}`")
        audit_lines.append(f"- **Saved to**: `{out_path.relative_to(root)}`")
        audit_lines.append("")

    # Ingest Ahn et al. tables
    audit_lines.append("## 2. Ahn et al. (2011) Flavor & Recipe Tables")
    audit_lines.append("")

    # Ahn compounds and ingredient names
    comp_file = find_file(raw_ahn_dir, ["*ingr_comp.tsv", "*ingr_comp*.csv", "*ingr_comp*"])
    info_file = find_file(raw_ahn_dir, ["*ingr_info.tsv", "*ingr_info*.csv", "*ingr_info*"])

    if comp_file and comp_file.exists():
        sep = "\t" if comp_file.suffix in [".tsv", ".txt"] else ","
        df_comp = pd.read_csv(comp_file, sep=sep)
        df_comp.columns = [c.lstrip("#").strip().lower().replace(" ", "_") for c in df_comp.columns]
        # Rename standard columns: ingredient_id -> ahn_id, compound_id -> compound_id
        df_comp = df_comp.rename(columns={"ingredient_id": "ahn_id", "ingredient": "ahn_id", "compound": "compound_id"})

        if info_file and info_file.exists():
            sep_info = "\t" if info_file.suffix in [".tsv", ".txt"] else ","
            df_info = pd.read_csv(info_file, sep=sep_info)
            df_info.columns = [c.lstrip("#").strip().lower().replace(" ", "_") for c in df_info.columns]
            df_info = df_info.rename(columns={"id": "ahn_id", "ingredient_name": "ahn_name"})
            # Merge to attach ingredient names and categories to compound links
            df_comp = df_comp.merge(df_info[["ahn_id", "ahn_name", "category"]], on="ahn_id", how="left")

        out_comp = interim_dir / "ahn_compounds.parquet"
        df_comp.to_parquet(out_comp, index=False)
        audit_lines.append(f"### Ahn Compounds (`{comp_file.name}`)")
        audit_lines.append(f"- **Shape**: {df_comp.shape[0]:,} rows x {df_comp.shape[1]} columns")
        audit_lines.append(f"- **Columns**: {', '.join(df_comp.columns)}")
        audit_lines.append(f"- **Saved to**: `{out_comp.relative_to(root)}`")
        audit_lines.append("")
    else:
        audit_lines.append(f"- **Ahn Compounds**: Missing in `{raw_ahn_dir}`.")

    # Ahn recipes (can be ragged tab-separated files: cuisine \t ing1 \t ing2 ...)
    recipe_files = list(raw_ahn_dir.rglob("*recipe*.txt"))
    if not recipe_files:
        recipe_files = [f for f in raw_ahn_dir.rglob("*.txt") if "recipes" in f.name.lower()]

    if recipe_files:
        recipe_rows = []
        source_names = []
        for r_file in recipe_files:
            source_names.append(r_file.name)
            with open(r_file, "r", encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    cuisine = parts[0].strip()
                    ingredients = [p.strip() for p in parts[1:] if p.strip()]
                    if ingredients:
                        recipe_rows.append({
                            "cuisine": cuisine,
                            "ingredients": ",".join(ingredients)
                        })

        df_rec = pd.DataFrame(recipe_rows)
        out_rec = interim_dir / "ahn_recipes.parquet"
        df_rec.to_parquet(out_rec, index=False)
        audit_lines.append(f"### Ahn Recipes (`{', '.join(source_names)}`)")
        audit_lines.append(f"- **Shape**: {df_rec.shape[0]:,} recipes parsed")
        audit_lines.append(f"- **Columns**: {', '.join(df_rec.columns)}")
        audit_lines.append(f"- **Saved to**: `{out_rec.relative_to(root)}`")
        audit_lines.append("")
    else:
        audit_lines.append(f"- **Ahn Recipes**: Missing in `{raw_ahn_dir}`.")

    # Save audit report
    audit_path = root / "reports" / "data_audit.md"
    audit_path.write_text("\n".join(audit_lines), encoding="utf-8")
    logger.info(f"S1 complete. Data audit saved to {audit_path}")

    return {"status": "success", "audit_report": str(audit_path)}


if __name__ == "__main__":
    run_ingest()
