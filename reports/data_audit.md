# Stage S1 Data Audit Report

This document details the loaded schemas, row counts, ID ranges, and missing value checks.

## 1. USDA SR Legacy Tables

### Table: `usda_food`
- **Source File**: `food.csv`
- **Shape**: 7,793 rows x 5 columns
- **Columns**: fdc_id, data_type, description, food_category_id, publication_date
- **Null counts in key columns**: `{'fdc_id': 0, 'data_type': 0, 'description': 0, 'food_category_id': 0, 'publication_date': 0}`
- **Saved to**: `data\interim\usda_food.parquet`

### Table: `usda_nutrient`
- **Source File**: `nutrient.csv`
- **Shape**: 474 rows x 5 columns
- **Columns**: id, name, unit_name, nutrient_nbr, rank
- **Null counts in key columns**: `{'id': 0, 'name': 0, 'unit_name': 0, 'nutrient_nbr': 12, 'rank': 11}`
- **Saved to**: `data\interim\usda_nutrient.parquet`

### Table: `usda_food_nutrient`
- **Source File**: `food_nutrient.csv`
- **Shape**: 644,125 rows x 11 columns
- **Columns**: id, fdc_id, nutrient_id, amount, data_points, derivation_id, min, max, median, footnote, min_year_acquired
- **Null counts in key columns**: `{'id': 0, 'fdc_id': 0, 'nutrient_id': 0, 'amount': 0}`
- **Saved to**: `data\interim\usda_food_nutrient.parquet`

### Table: `usda_food_category`
- **Source File**: `food_category.csv`
- **Shape**: 28 rows x 3 columns
- **Columns**: id, code, description
- **Null counts in key columns**: `{'id': 0, 'code': 0, 'description': 0}`
- **Saved to**: `data\interim\usda_food_category.parquet`

## 2. Ahn et al. (2011) Flavor & Recipe Tables

### Ahn Compounds (`ingr_comp.tsv`)
- **Shape**: 36,781 rows x 4 columns
- **Columns**: ahn_id, compound_id, ahn_name, category
- **Saved to**: `data\interim\ahn_compounds.parquet`

### Ahn Recipes (`allr_recipes.txt, epic_recipes.txt, menu_recipes.txt`)
- **Shape**: 57,691 recipes parsed
- **Columns**: cuisine, ingredients
- **Saved to**: `data\interim\ahn_recipes.parquet`
