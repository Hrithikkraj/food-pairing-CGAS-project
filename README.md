# Flavor-Nutrition Pairing Project
> **"Do Foods That Share Flavor Also Fill Each Other's Nutrient Gaps?"**  
> *Computational Gastronomy & Nutrition Data Pipeline*

This repository implements the complete end-to-end reproducible pipeline analyzing whether ingredient pairs that share flavor compounds (Ahn et al., 2011) also display high nutrient complementarity (USDA FoodData Central SR Legacy).

---

## 🔬 Research Questions
- **RQ1**: Do pairs with more shared flavor compounds also have higher nutrient complementarity? (Spearman correlation with bootstrap CI, OLS controlled regression).
- **RQ2**: Are pairs that cooks really use in recipes more complementary than random pairs? (Permutation test against random pairing baseline).
- **RQ3**: Does the flavor-nutrition relationship differ across cuisines? (Cuisine-stratified correlation with bootstrap CIs).

---

## 📁 Repository Structure
```
├── config/
│   ├── params.yaml            # Pipeline settings, seeds, thresholds, paths
│   └── dv_table.csv           # FDA Daily Values for 11 key nutrients
├── data/
│   ├── README.md              # Data source descriptions and download links
│   ├── raw/                   # READ-ONLY raw inputs (Ahn & USDA)
│   ├── interim/               # S1-S2 cleaned parquet/csv files
│   ├── external/              # S3 human-reviewed matching log
│   └── processed/             # S4 ingredient table & S5 pair table
├── notebooks/                 # Exploratory & narrative notebooks
├── src/fnp/                   # Core Python package
│   ├── ingest.py              # S1: Schema validation & data audit
│   ├── filter.py              # S2: Compound & recipe filtering
│   ├── match.py               # S3: RapidFuzz candidate matching
│   ├── profile.py             # S4: %DV & nutrient status flags
│   ├── scores.py              # Pure scoring functions (N_s, C)
│   ├── pairs.py               # S5: Unordered pair generation
│   ├── baseline.py            # Bootstrap & permutation null models
│   ├── analyze.py             # S6: Statistical analyses (A1-A6)
│   ├── visualize.py           # S7: Plots & interactive dashboard
│   └── utils.py               # Config, logging, seed, hash helpers
├── tests/                     # Automated unit tests
│   ├── test_scores.py         # Tests for pure scoring logic & toy example
│   ├── test_pairs.py          # Tests for pair invariants & symmetry
│   └── test_units.py          # Tests for FDA Daily Value reference
├── reports/
│   ├── data_audit.md          # Generated schema audit from S1
│   ├── tables/                # Analysis output CSV tables (A1-A6)
│   ├── figures/               # High-res publication plots (RQ1-RQ3)
│   ├── dashboard.html         # Interactive HTML report dashboard
│   └── results.json           # Machine-readable output summary
├── logs/                      # Execution logs with git commits
├── run_pipeline.py            # Master pipeline runner
├── Makefile                   # make data | analysis | report | test
├── requirements.txt           # Pinned dependencies
└── .gitignore
```

---

## 🚀 Quickstart

### 1. Installation
Ensure Python 3.11+ is installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Tests
Verify pure scoring functions and invariants:
```bash
python -m unittest discover -s tests -v
```

### 3. Run Pipeline Stages
You can run all stages end-to-end or individual stages:
```bash
# Run all stages
python run_pipeline.py --stage all

# Or run stage-by-stage
python run_pipeline.py --stage S1   # Ingest & audit
python run_pipeline.py --stage S2   # Filter ingredients
python run_pipeline.py --stage S3   # Match to USDA foods
python run_pipeline.py --stage S4   # Nutrient profiling
python run_pipeline.py --stage S5   # Pair table construction
python run_pipeline.py --stage S6   # Statistical analysis
python run_pipeline.py --stage S7   # Visualizations & dashboard
```
Use `python run_pipeline.py --stage S3 --force-match` to regenerate S3 candidates after backing up the existing matching log.
Or via Makefile:
```bash
make test
make data
make analysis
make report
```

---

## ⚖️ Scientific Rigor & Rules
1. **Raw data is immutable**: `data/raw/` is never modified directly.
2. **Deterministic & reproducible**: Fixed random seeds (default 42) and configuration hashing ensure identical outputs.
3. **No fabricated claims**: Results report statistical associations at the raw ingredient-pair level and do not make unsubstantiated health claims about cooked meals.
