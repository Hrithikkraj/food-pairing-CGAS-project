# LLM Context: Flavor-Nutrition Pairing Project

This file serves as a condensed source of truth for automated scripts, LLM agents, and pipeline executors.

## 1. Metric Definitions
### 1.1 Flavor Sharing Score $N_s$
- For ingredients $A$ and $B$ with compound sets $F_A$ and $F_B$:
  $$N_s(A, B) = |F_A \cap F_B|$$
- Size-controlled variant:
  $$N_{s,\text{norm}}(A, B) = \frac{N_s(A, B)}{\min(|F_A|, |F_B|)}$$
  (or z-score against null compound degree distribution).

### 1.2 Nutrient Complementarity Score $C$
- **Step 1**: Convert nutrient amounts (per 100 g) to % Daily Value:
  $$\%DV = \frac{\text{amount}}{\text{Daily Value}} \times 100$$
- **Step 2**: Categorize each nutrient for each food:
  - Low: $\%DV < 10\%$
  - High: $\%DV \ge 20\%$
  - Neither: $10\% \le \%DV < 20\%$
- **Step 3**: For a pair $(A, B)$ evaluated on $K = 11$ nutrients:
  $$C(A, B) = \frac{\#\{k : A \text{ low}, B \text{ high}\} + \#\{k : B \text{ low}, A \text{ high}\}}{K}$$
  $C \in [0, 1]$. Higher means more nutrient gap-filling.
- Sensitivity cutoffs to test: $(5, 15)$, $(10, 20)$ [main], $(10, 25)$, $(15, 30)$.

### 1.3 Target Nutrients (11) & Daily Values (FDA standard)
- Protein (ID 1003): 50 g
- Total dietary fiber (ID 1079): 28 g
- Iron, Fe (ID 1089): 18 mg
- Calcium, Ca (ID 1087): 1300 mg
- Potassium, K (ID 1092): 4700 mg
- Zinc, Zn (ID 1095): 11 mg
- Vitamin C (ID 1162): 90 mg
- Vitamin A, RAE (ID 1106): 900 µg
- Vitamin B-12 (ID 1178): 2.4 µg
- Folate, total (ID 1177): 400 µg
- Magnesium, Mg (ID 1090): 420 mg
- *(Optional case study)*: Lysine (1214) & Methionine (1215). Energy (1008) descriptive only.

## 2. Pipeline Architecture & Contracts
- **S1 Ingest**: Load raw Ahn and USDA tables; generate `reports/data_audit.md` and interim Parquet.
- **S2 Filter**: Exclude over-represented compound ID 339 (farnesol); require $\ge 10$ compounds and $\ge 1$ recipe; filter non-nutritional items. Output `data/interim/ingredients_filtered.csv`.
- **S3 Match**: Match Ahn ingredients to USDA SR Legacy descriptions with RapidFuzz token-set ratio. Human approval logged in `data/external/matching_log.csv`.
- **S4 Profile**: Pivot nutrient values, compute %DV and low/high flags. Never silently fill NaNs with zero. Output `data/processed/ingredient_table.csv`.
- **S5 Pairs**: Build all unordered pairs ($ahn\_id\_a < ahn\_id\_b$), compute $N_s$, $N_{s,\text{norm}}$, $C$, alternative cutoffs, and recipe co-occurrences. Output `data/processed/pair_table.csv`.
- **S6 Analyze**: Statistical tests (Spearman $\rho$ with 95% bootstrap CI, OLS controlled regression, permutation test for real vs random recipe pairs with $\ge 5000$ permutations, cuisine breakdown, case studies). Output `reports/tables/*.csv` and `reports/results.json`.
- **S7 Visualize**: Hexbin/density scatter ($N_s$ vs $C$), permutation distribution plot, cuisine correlation forest plot, interactive Plotly dashboard `reports/dashboard.html`.
- **S8 Report**: Summary synthesis into `reports/final_report.pdf` or markdown.

## 3. Strict Rules & Conventions
1. Never edit `data/raw/`. Never fabricate data, matches, or statistics.
2. Store ingredient pairs unordered ($ahn\_id\_a < ahn\_id\_b$) to prevent duplication.
3. Scoring functions in `src/fnp/scores.py` must be pure functions with unit tests.
4. Set fixed random seed (42) for reproducibility.
5. Claims must reflect ingredient-pair statistical associations; do not make unwarranted health claims about cooked dishes.
