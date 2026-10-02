"""Stage S7: static figures and the HTML dashboard."""

import base64
import html
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns

from fnp.utils import get_project_root, load_params, setup_logger

logger = setup_logger("fnp.visualize", "logs/pipeline.log")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

FIGURE_NAMES = [
    "rq1_scatter.png", "rq1_scatter_no_spice_herb.png", "rq2_real_vs_random.png",
    "rq3_cuisines.png", "case_studies.png", "sensitivity.png", "category_heatmap.png",
]
SIMPLE_FIGURE_NAMES = [
    "flavor_vs_nutrients.png", "real_vs_random_simple.png", "regions_simple.png",
    "example_pairs_simple.png", "sensitivity_simple.png",
]


def _read_table(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        logger.warning("S7 missing input file: %s", path)
        return None
    return pd.read_csv(path)


def _save_figure(fig: Figure, name: str, figures_dir: Path, pdf: PdfPages) -> None:
    path = figures_dir / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    pdf.savefig(fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved S7 figure to %s", path)


def _is_true(values: pd.Series) -> pd.Series:
    return values.map(str).map(str.lower).eq("true")


def _a1_label(table: Optional[pd.DataFrame]) -> str:
    if table is None or table.empty:
        return "rho/CI unavailable"
    row = table.iloc[0]
    return f"rho={row['spearman_rho']:.3f}, 95% CI [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"


def _rq1(df: pd.DataFrame, a1: Optional[pd.DataFrame], suffix: str) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(df["n_shared"], df["C_main"], gridsize=30, cmap="Blues", mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax, label="Log10(Pair Count)")
    if len(df) > 1:
        sns.regplot(data=df, x="n_shared", y="C_main", scatter=False, ax=ax,
                    color="#d95f02", line_kws={"linewidth": 2, "label": "Linear fit"})
    ax.set_title(f"RQ1: Flavor sharing vs nutrient complementarity{suffix}\n{_a1_label(a1)}", weight="bold")
    ax.set_xlabel("Flavor sharing score $N_s$ (shared compounds)")
    ax.set_ylabel("Nutrient complementarity $C$")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best")
    fig.tight_layout()
    return fig


def _rq2(tables_dir: Path) -> Optional[Figure]:
    keys = [("all", "global"), ("all", "within_category"),
            ("no_spice_herb", "global"), ("no_spice_herb", "within_category")]
    rows = []
    for variant, shuffle in keys:
        table = _read_table(tables_dir / f"a3_permutation_{variant}_{shuffle}.csv")
        rows.append((variant, shuffle, None if table is None or table.empty else table.iloc[0]))
    if not any(row is not None for _, _, row in rows):
        logger.warning("S7 could not create rq2_real_vs_random.png: all A3 inputs are missing")
        return None
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (variant, shuffle, row) in zip(axes.flat, rows):
        ax.set_title(f"{variant.replace('_', ' ').title()} / {shuffle.replace('_', ' ').title()}")
        if row is None:
            ax.text(0.5, 0.5, "Input unavailable", ha="center", va="center", transform=ax.transAxes)
            continue
        mean, std = float(row["null_mean"]), float(row["null_std"])
        observed = float(row["observed_mean"])
        x = np.linspace(mean - 4 * std, mean + 4 * std, 300)
        density = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        ax.plot(x, density, color="#4c78a8", label="Normal approximation")
        ax.axvline(observed, color="#d95f02", linestyle="--", linewidth=2, label="Observed")
        text = (f"observed={observed:.3f}\nnull mean={mean:.3f}\n"
                f"z={float(row['z']):.2f}\ntwo-sided p={float(row['p_two_sided']):.4g}")
        ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        ax.set_xlabel("Mean complementarity $C$")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    fig.suptitle("RQ2: Real pairs vs random profile-pair null", weight="bold")
    fig.tight_layout()
    return fig


def _cuisines(df: pd.DataFrame) -> Figure:
    df = df.sort_values("rho").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.45)))
    y = np.arange(len(df))
    ax.errorbar(df["rho"], y, xerr=[df["rho"] - df["ci_low"], df["ci_high"] - df["rho"]],
                fmt="o", color="#1b9e77", ecolor="#66a61e", elinewidth=2, capsize=4)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{group} (n={int(count):,})" for group, count in zip(df["group"], df["n_recipes"])])
    ax.invert_yaxis()
    ax.set_title("RQ3: Flavor-nutrient correlation by cuisine", weight="bold")
    ax.set_xlabel("Spearman rank correlation $rho$")
    fig.tight_layout()
    return fig


def _case_studies(df: pd.DataFrame) -> Figure:
    df = df.sort_values("C_main").reset_index(drop=True)
    colors = df["type"].map({"Curated": "#4c78a8", "Random Baseline": "#f58518"}).fillna("#777777")
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.45)))
    y = np.arange(len(df))
    bars = ax.barh(y, df["C_main"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(df["pair"])
    ax.set_xlabel("Complementarity $C_{main}$")
    ax.set_title("Case studies", weight="bold")
    for bar, count in zip(bars, df["n_shared"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"n_shared={int(count)}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def _sensitivity(df: pd.DataFrame) -> Figure:
    metric_values = df["metric"].map(str)
    df = df[metric_values.map(lambda value: str(value).startswith("C_"))].copy()
    df["cutoff"] = df["metric"].map(lambda value: str(value).replace("C_", "", 1))
    cutoffs = list(dict.fromkeys(df["cutoff"].tolist()))
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(cutoffs))
    for series, offset, color in [("n_shared", -0.12, "#4c78a8"), ("n_shared_norm", 0.12, "#f58518")]:
        series_df = df[df["x"] == series].set_index("cutoff").reindex(cutoffs)
        if series_df["rho"].isna().all():
            logger.warning("S7 sensitivity series missing: %s", series)
            continue
        ax.errorbar(positions + offset, series_df["rho"],
                    yerr=[series_df["rho"] - series_df["ci_low"], series_df["ci_high"] - series_df["rho"]],
                    fmt="o", capsize=4, color=color, label=series)
    ax.set_xticks(positions)
    ax.set_xticklabels(cutoffs)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_title("Sensitivity of Spearman correlation", weight="bold")
    ax.set_xlabel("Cutoff pair")
    ax.set_ylabel("Spearman $rho$ (95% CI)")
    ax.legend()
    fig.tight_layout()
    return fig


def _categories(df: pd.DataFrame) -> Figure:
    top = df["category_pair"].value_counts().nlargest(15).index
    grouped = df.assign(category_pair_grouped=df["category_pair"].where(df["category_pair"].isin(top), "Other"))
    summary = grouped.groupby("category_pair_grouped", sort=False).agg(mean_C=("C_main", "mean"), count=("C_main", "size"))
    summary = summary.sort_values("mean_C")
    fig, ax = plt.subplots(figsize=(10, max(5, len(summary) * 0.35)))
    bars = ax.barh(summary.index, summary["mean_C"], color="#72b7b2")
    for bar, count in zip(bars, summary["count"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"n={int(count):,}", va="center", fontsize=8)
    ax.set_title("Mean complementarity by grouped category pair", weight="bold")
    ax.set_xlabel("Mean $C_{main}$")
    fig.tight_layout()
    return fig


def _findings(results: Dict[str, Any]) -> str:
    analyses = results.get("analyses", {})
    findings = []
    for variant, label in [("all", "all pairs"), ("no_spice_herb", "pairs without spice/herb")]:
        row = analyses.get(f"A1_correlation_{variant}")
        if row:
            findings.append(f"For {label}, rho is {row['spearman_rho']:.3f} (95% CI {row['ci_low']:.3f} to {row['ci_high']:.3f}).")
    p_values = []
    for variant in ("all", "no_spice_herb"):
        for shuffle in ("global", "within_category"):
            row = analyses.get(f"A3_real_vs_random_{variant}_{shuffle}")
            if row:
                p_values.append(f"{variant}/{shuffle}: {row['p_two_sided']:.4g}")
    if p_values:
        findings.append("A3 two-sided p-values: " + "; ".join(p_values) + ".")
    return " ".join(findings) if findings else "Findings unavailable because results.json has no matching analysis entries."


def _simple_link_text(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 0.10:
        strength = "almost no link"
    elif magnitude <= 0.30:
        strength = "a weak link"
    elif magnitude <= 0.50:
        strength = "a medium link"
    else:
        strength = "a strong link"
    direction = "more flavor sharing goes with slightly less nutrient teamwork" if value < 0 else "more flavor sharing goes with slightly more nutrient teamwork"
    return f"{strength}; {direction}" if value != 0 else strength


def _luck_text(value: float) -> str:
    if value < 0.001:
        return "very unlikely to be luck"
    if value < 0.05:
        return "unlikely to be luck"
    return "could easily be luck"


def _number(value: Any) -> str:
    return f"{float(value):.2f}"


def _chance_number(value: Any) -> str:
    value = float(value)
    return "less than 0.01" if value < 0.01 else f"{value:.2f}"


def _load_simple_data(root: Path, processed_dir: Path, tables_dir: Path, results: Dict[str, Any]) -> Dict[str, Any]:
    data: Dict[str, Any] = {"warnings": []}
    pair_file = processed_dir / "pair_table.csv"
    data["pairs"] = pd.read_csv(pair_file) if pair_file.exists() else pd.DataFrame()
    if data["pairs"].empty:
        data["warnings"].append(f"Missing or empty file: {pair_file}")
    data["a1_all"] = _read_table(tables_dir / "a1_correlation_all.csv")
    data["a1_no_spice"] = _read_table(tables_dir / "a1_correlation_no_spice_herb.csv")
    data["a3"] = {}
    for variant in ("all", "no_spice_herb"):
        for shuffle in ("global", "within_category"):
            path = tables_dir / f"a3_permutation_{variant}_{shuffle}.csv"
            table = _read_table(path)
            data["a3"][f"{variant}_{shuffle}"] = table
    data["a4"] = _read_table(tables_dir / "a4_cuisines.csv")
    data["a5"] = _read_table(tables_dir / "a5_sensitivity.csv")
    data["a6"] = _read_table(tables_dir / "a6_case_studies.csv")
    for key in ("a1_all", "a1_no_spice", "a4", "a5", "a6"):
        if data[key] is None:
            data["warnings"].append(f"Missing table needed for the page: {key}")
    for key, table in data["a3"].items():
        if table is None:
            data["warnings"].append(f"Missing random-comparison table: {key}")
    data["results"] = results
    recipes_file = processed_dir.parent / "interim" / "ahn_recipes.parquet"
    if recipes_file.exists():
        data["recipe_count"] = len(pd.read_parquet(recipes_file))
    else:
        data["recipe_count"] = results.get("total_recipes")
        data["warnings"].append(f"Missing recipe file: {recipes_file}")
    if not data["pairs"].empty:
        data["food_count"] = pd.concat([data["pairs"]["ahn_id_a"], data["pairs"]["ahn_id_b"]]).nunique()
    else:
        data["food_count"] = None
    data["pair_count"] = len(data["pairs"]) if not data["pairs"].empty else results.get("total_pairs")
    return data


def _simple_rq1(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hexbin(df["n_shared"], df["C_main"], gridsize=30, mincnt=1, bins="log", cmap="viridis")
    ax.set_title("Foods that share more flavor do not fill each other's gaps", fontsize=15, weight="bold")
    ax.set_xlabel("Shared flavor molecules", fontsize=12)
    ax.set_ylabel("Nutrient teamwork (0 to 1)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _simple_rq2(row: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    mean, std = float(row["null_mean"]), float(row["null_std"])
    observed = float(row["observed_mean"])
    x = np.linspace(mean - 3 * std, mean + 3 * std, 200)
    ax.axvspan(mean - 2 * std, mean + 2 * std, color="#bdbdbd", alpha=0.45, label="range of random pairs")
    ax.axvline(observed, color="#0072B2", linewidth=3, label="real recipe pairs")
    ax.set_title("Real recipe pairs are lower than random pairs", fontsize=15, weight="bold")
    ax.set_xlabel("Nutrient teamwork (0 to 1)", fontsize=12)
    ax.set_yticks([])
    ax.legend(fontsize=11)
    ax.text(0.02, 0.04, "Each line shows a mean score. The gray band shows a likely random range.",
            transform=ax.transAxes, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _simple_regions(df: pd.DataFrame, path: Path) -> None:
    df = df.sort_values("rho").reset_index(drop=True)
    clear = ~((df["ci_low"] <= 0) & (df["ci_high"] >= 0))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(df))
    colors = np.where(clear, "#0072B2", "#9e9e9e")
    for index, row in enumerate(df.itertuples(index=False)):
        ax.errorbar(row.rho, index, # type: ignore
                    xerr=[[row.rho - row.ci_low], [row.ci_high - row.rho]], # type: ignore
                    fmt="none", ecolor=colors[index], elinewidth=3, capsize=4)
    ax.scatter(df["rho"], y, color=colors, s=45, label="blue = clearer pattern; gray = uncertain")
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{group} ({int(count):,} recipes)" for group, count in zip(df["group"], df["n_recipes"])], fontsize=12)
    ax.set_xlabel("Link score", fontsize=12)
    ax.set_title("The pattern differs by food region", fontsize=15, weight="bold")
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _simple_cases(df: pd.DataFrame, path: Path) -> None:
    ordered = df.sort_values("C_main").reset_index(drop=True)
    colors = ordered["type"].map({"Curated": "#0072B2", "Random Baseline": "#E69F00"}).fillna("#777777")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(ordered))
    ax.barh(y, ordered["C_main"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["pair"], fontsize=12)
    ax.set_xlabel("Nutrient teamwork (0 to 1)", fontsize=12)
    ax.set_title("Familiar food pairs do not always score highest", fontsize=15, weight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#0072B2", label="our picks"), Patch(color="#E69F00", label="random pairs")], fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _simple_sensitivity(df: pd.DataFrame, path: Path) -> None:
    metric_values = df["metric"].map(str)
    df = df[metric_values.map(lambda value: str(value).startswith("C_"))].copy()
    df["cutoff"] = df["metric"].map(lambda value: str(value).replace("C_", "", 1))
    cutoffs = list(dict.fromkeys(df["cutoff"].tolist()))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(cutoffs))
    for series, offset, label, color in [("n_shared", -0.12, "shared flavor molecules", "#0072B2"),
                                          ("n_shared_norm", 0.12, "shared molecules, scaled", "#E69F00")]:
        series_df = df[df["x"] == series].set_index("cutoff").reindex(cutoffs)
        ax.errorbar(positions + offset, series_df["rho"],
                    yerr=[series_df["rho"] - series_df["ci_low"], series_df["ci_high"] - series_df["rho"]],
                    fmt="o", capsize=4, color=color, label=label)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(cutoffs, fontsize=12)
    ax.set_xlabel("Nutrient cut-off pair", fontsize=12)
    ax.set_ylabel("Link score", fontsize=12)
    ax.set_title("The result stays similar under different nutrient cut-offs", fontsize=15, weight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _as_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _technical_table_html(table: Optional[pd.DataFrame], name: str) -> str:
    if table is None:
        return f"<div class='warning'>Missing technical table: {html.escape(name)}</div>"
    return f"<h3>{html.escape(name)}</h3>{table.to_html(index=False, classes='technical-table') }"


def _friendly_dashboard(reports_dir: Path, figures_dir: Path, simple_dir: Path, data: Dict[str, Any]) -> None:
    results = data["results"].get("analyses", {})
    a1 = data["a1_all"].iloc[0] if data["a1_all"] is not None and not data["a1_all"].empty else None
    a1_no_spice = data["a1_no_spice"].iloc[0] if data["a1_no_spice"] is not None and not data["a1_no_spice"].empty else None
    a3 = data["a3"].get("no_spice_herb_within_category")
    a3_row = a3.iloc[0] if a3 is not None and not a3.empty else None
    a4 = data["a4"]
    clear_regions = [] if a4 is None else a4[(a4["ci_low"] > 0) | (a4["ci_high"] < 0)]["group"].tolist()
    unclear_regions = [] if a4 is None else a4[(a4["ci_low"] <= 0) & (a4["ci_high"] >= 0)]["group"].tolist()
    warning_html = "".join(f"<div class='warning'>Data warning: {html.escape(message)}</div>" for message in data["warnings"])
    if a1 is None:
        a1_text = "The main link score is unavailable."
        a1_verdict = "Unavailable"
    else:
        a1_text = f"Link score: {_number(a1['spearman_rho'])} ({_simple_link_text(float(a1['spearman_rho']))})."
        a1_verdict = _simple_link_text(float(a1["spearman_rho"])).split(";")[0].title()
    no_spice_text = ""
    if a1_no_spice is not None:
        no_spice_text = f" Without spices and herbs: {_number(a1_no_spice['spearman_rho'])} ({_simple_link_text(float(a1_no_spice['spearman_rho']))})."
    if a3_row is None:
        a3_text = "The random comparison is unavailable."
    else:
        a3_text = (f"Real recipe pairs: {_number(a3_row['observed_mean'])}; random pairs: {_number(a3_row['null_mean'])}. "
                   f"The chance this is just luck is {_chance_number(a3_row['p_two_sided'])}; this is {_luck_text(float(a3_row['p_two_sided']))}.")
    region_text = "Clearer patterns appear in " + ", ".join(clear_regions) + "." if clear_regions else "No region has a clear pattern."
    if unclear_regions:
        region_text += " We cannot be sure about " + ", ".join(unclear_regions) + "."
    food_count = "unavailable" if data["food_count"] is None else f"{int(data['food_count']):,}"
    pair_count = "unavailable" if data["pair_count"] is None else f"{int(data['pair_count']):,}"
    recipe_count = "unavailable" if data["recipe_count"] is None else f"{int(data['recipe_count']):,}"
    image = lambda name, alt: f"<img src='{_as_data_uri(simple_dir / name)}' alt='{html.escape(alt)}'>"
    cases = data["a6"]
    case_rows = ""
    if cases is not None:
        for row in cases.itertuples():
            nutrients = str(row.top_nutrients).replace(";", ", ").replace("vitamin_b-12", "vitamin B12").replace("vitamin_a", "vitamin A")
            case_rows += f"<tr><td>{html.escape(str(row.pair))}</td><td>{int(row.n_shared)}</td><td>{float(row.C_main):.2f}</td><td>{html.escape(nutrients)}</td></tr>"
    technical_names = ["a1_correlation_all.csv", "a1_correlation_no_spice_herb.csv", "a2_regression_n_shared.csv",
                       "a2_regression_n_shared_norm.csv", "a3_permutation_all_global.csv", "a3_permutation_all_within_category.csv",
                       "a3_permutation_no_spice_herb_global.csv", "a3_permutation_no_spice_herb_within_category.csv",
                       "a4_cuisines.csv", "a5_sensitivity.csv", "a6_case_studies.csv"]
    technical_tables = []
    for name in technical_names:
        technical_tables.append(_technical_table_html(_read_table(reports_dir / "tables" / name), name))
    html_content = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Flavor and Nutrition Pairing</title><style>
:root{{--ink:#24323d;--accent:#0072B2;--paper:#f7faf9;--card:#fff;--line:#d7e0df}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:18px/1.55 Arial,Helvetica,sans-serif}}main{{max-width:900px;margin:auto;padding:24px}}h1{{font-size:clamp(2rem,5vw,3.4rem);line-height:1.1;margin:0 0 12px;color:var(--accent)}}h2{{margin-top:42px;font-size:1.65rem}}h3{{font-size:1.15rem}}.hero,.card,.chart,.warning,.short,.step{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:22px;margin:18px 0;box-shadow:0 2px 8px #17324d0c}}.short{{border-left:6px solid var(--accent)}}.short strong{{display:block;color:var(--accent)}}.cards,.steps{{display:grid;gap:16px}}.cards{{grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}}.steps{{grid-template-columns:repeat(auto-fit,minmax(250px,1fr));counter-reset:step}}.step::before{{counter-increment:step;content:counter(step);display:inline-grid;place-items:center;background:var(--accent);color:#fff;width:32px;height:32px;border-radius:50%;font-weight:bold;margin-right:8px}}.chart img{{display:block;width:100%;height:auto;margin:12px 0}}.label{{font-weight:bold;color:var(--accent)}}.verdict{{display:inline-block;border-radius:999px;background:#e6f2f7;color:#075985;padding:3px 10px;font-size:.85em;font-weight:bold}}table{{border-collapse:collapse;width:100%;background:#fff}}td,th{{padding:9px;border:1px solid var(--line);text-align:left;vertical-align:top}}.example-wrap{{overflow-x:auto}}.warning{{background:#fff4e5;border-color:#e0a03b;color:#704800}}details{{margin-top:38px}}summary{{cursor:pointer;font-weight:bold;color:var(--accent)}}.technical-table{{font-size:12px;display:block;overflow-x:auto;white-space:nowrap}}small{{color:#52616b}}@media(max-width:600px){{main{{padding:14px}}body{{font-size:18px}}.hero,.card,.chart,.short,.step{{padding:16px}}}}@media print{{body{{background:#fff;font-size:12pt}}.card,.chart,.short,.step{{box-shadow:none;break-inside:avoid}}details{{display:block}}details[open] summary{{display:none}}}}
</style></head><body><main><section class='hero'><h1>Do foods that taste good together also work well together for your body?</h1><p>We studied shared flavor molecules and nutrient teamwork in food pairs.</p><div class='short'><strong>Short answer</strong>Flavor sharing and nutrient teamwork show {_simple_link_text(float(a1['spearman_rho'])) if a1 is not None else 'an unavailable link'}.<br>{a3_text}<br>{region_text}</div></section>{warning_html}
<h2>What is this project?</h2><p>Some foods share flavor molecules. Cooks often say these foods taste good together.</p><p>We asked if those pairs also fill each other's missing nutrients.</p><p>Nutrient teamwork means how well two foods fill each other's missing nutrients.</p><p>We used the Ahn et al. 2011 flavor database, {recipe_count} real recipes, and the USDA food database.</p>
<h2>Two ideas in one minute</h2><div class='cards'><div class='card'><h3>Flavor sharing</h3><p>How many flavor molecules two foods have in common.</p><p><b>Example:</b> tomato and cheese share some flavor molecules.</p></div><div class='card'><h3>Nutrient teamwork</h3><p>How often one food is low where the other is high.</p><p>The score runs from 0 (no help) to 1 (helps a lot). We looked at 11 nutrients.</p><p><b>Example:</b> rice and beans can help cover different nutrient gaps.</p></div></div>
<h2>How we did it</h2><div class='steps'><div class='step'>Pick {food_count} common foods.</div><div class='step'>Match each food to USDA nutrition.</div><div class='step'>Score every pair: {pair_count} pairs.</div><div class='step'>Compare recipe pairs with random pairs.</div><div class='step'>Repeat the check for 10 world regions.</div></div>
<h2>What we found</h2><div class='cards'><div class='card'><span class='verdict'>{a1_verdict}</span><h3>Flavor sharing and nutrient teamwork</h3><p>{a1_text}{no_spice_text}</p></div><div class='card'><span class='verdict'>Slightly lower</span><h3>Real recipes versus random pairs</h3><p>{a3_text}</p></div><div class='card'><span class='verdict'>Varies by region</span><h3>Different food regions</h3><p>{region_text}</p></div></div>
<h2>Charts</h2><div class='chart'><h3>Foods that share flavor do not fill more gaps</h3><p><span class='label'>What you are looking at:</span> Each dot area represents many food pairs.</p><p><span class='label'>How to read it:</span> Look for a rising or falling cloud.</p>{image('flavor_vs_nutrients.png','Chart showing shared flavor molecules and nutrient teamwork')}<p><span class='label'>What it tells us:</span> The cloud is mostly flat, so the link is small.</p></div>
<div class='chart'><h3>Real recipe pairs versus random pairs</h3><p><span class='label'>What you are looking at:</span> One real-pair score and a gray random range.</p><p><span class='label'>How to read it:</span> Compare the blue line with the gray band.</p>{image('real_vs_random_simple.png','Chart comparing real recipe pairs with a range of random pairs')}<p><span class='label'>What it tells us:</span> The real pairs score lower in this check.</p></div>
<div class='chart'><h3>The pattern differs by region</h3><p><span class='label'>What you are looking at:</span> Region scores and their likely ranges.</p><p><span class='label'>How to read it:</span> Gray means the range includes no link.</p>{image('regions_simple.png','Region chart with recipe counts and likely ranges')}<p><span class='label'>What it tells us:</span> Some regions show a clearer pattern than others.</p></div>
<div class='chart'><h3>Example pairs</h3><p><span class='label'>What you are looking at:</span> Scores for our picks and random pairs.</p><p><span class='label'>How to read it:</span> Longer bars mean more nutrient teamwork.</p>{image('example_pairs_simple.png','Bar chart comparing example food pairs')}<p><span class='label'>What it tells us:</span> Familiar pairs are not always the highest scoring.</p></div>
<div class='chart'><h3>Spices and herbs change the picture</h3><p><span class='label'>What you are looking at:</span> The link score under different nutrient cut-offs.</p><p><span class='label'>How to read it:</span> Compare the two colored series.</p>{image('sensitivity_simple.png','Chart showing how the link score changes under different nutrient cut-offs')}<p><span class='label'>What it tells us:</span> Removing spice and herb effects matters for interpretation.</p></div>
<h2>Example pairs</h2><p>Our hand-picked pairs are not special; some random pairs score higher.</p><div class='example-wrap'><table><thead><tr><th>Pair</th><th>Shared flavor molecules</th><th>Nutrient teamwork (0-1)</th><th>What helps</th></tr></thead><tbody>{case_rows}</tbody></table></div>
<h2>Things to keep in mind</h2><ul><li>Numbers use 100 grams of raw food.</li><li>Recipes have no ingredient amounts.</li><li>We chose some food matches ourselves. For example, bean means pinto beans, and cheese means cheddar.</li><li>We left out 33 vague foods, such as fish, meat, and truffle.</li><li>The flavor data comes from one book.</li><li>Spices and herbs behave differently. This shows a pattern, not a cause.</li></ul>
<h2>Glossary</h2><dl><dt><b>Flavor molecule</b></dt><dd>A tiny chemical linked to smell and taste.</dd><dt><b>Nutrient teamwork</b></dt><dd>How well two foods fill each other's missing nutrients.</dd><dt><b>Pair</b></dt><dd>Two foods studied together.</dd><dt><b>Link score</b></dt><dd>A number describing how two measures move together.</dd><dt><b>Likely range</b></dt><dd>A range showing uncertainty around a result.</dd><dt><b>Random comparison</b></dt><dd>A check against pairs made by chance.</dd><dt><b>Chance this is luck</b></dt><dd>How easily random choices could produce the result.</dd><dt><b>Region</b></dt><dd>A group of cuisines from a part of the world.</dd></dl>
<details><summary>Technical details</summary><p>These tables keep the original analysis names and values for checking.</p>{''.join(technical_tables)}</details></main></body></html>"""
    (reports_dir / "dashboard.html").write_text(html_content, encoding="utf-8")


def _dashboard(reports_dir: Path, tables_dir: Path, results: Dict[str, Any]) -> None:
    table_names = ("a1_correlation_all.csv", "a1_correlation_no_spice_herb.csv", "a3_permutation_all_global.csv",
                   "a3_permutation_all_within_category.csv", "a3_permutation_no_spice_herb_global.csv",
                   "a3_permutation_no_spice_herb_within_category.csv", "a4_cuisines.csv", "a5_sensitivity.csv",
                   "a6_case_studies.csv", "a2_regression_n_shared.csv")
    tables = []
    for name in table_names:
        table = _read_table(tables_dir / name)
        if table is not None:
            tables.append(f"<h3>{name}</h3>{table.to_html(index=False)}")
    images = "".join(f'<section><h2>{name[:-4].replace("_", " ").title()}</h2><img src="figures/{name}" alt="{name}"></section>' for name in FIGURE_NAMES)
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Flavor-Nutrition Pairing Dashboard</title>
<style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto;background:#f5f1e8;color:#1f2933}}section{{background:white;padding:1rem;margin:1rem 0;border:1px solid #d7d0c4}}img{{max-width:100%}}table{{border-collapse:collapse;font-size:.85rem}}td,th{{padding:.35rem;border:1px solid #d7d0c4}}.findings{{background:#e6f0ed;padding:1rem}}</style></head><body>
<h1>Flavor-Nutrition Pairing Dashboard</h1><div class='findings'><strong>Plain-language findings:</strong> {_findings(results)}</div>{images}<h2>Tables</h2>{''.join(tables)}</body></html>"""
    (reports_dir / "dashboard.html").write_text(html, encoding="utf-8")


def run_visualize(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Generate all available S7 figures and the dashboard."""
    root = get_project_root()
    params = load_params(config_path)
    np.random.seed(int(params.get("seed", 42)))
    processed_dir = root / params["paths"]["processed"]
    reports_dir = root / params["paths"]["reports"]
    figures_dir = root / params["paths"]["reports_figures"]
    tables_dir = root / params["paths"]["reports_tables"]
    simple_dir = root / "reports" / "figures_simple"
    figures_dir.mkdir(parents=True, exist_ok=True)
    simple_dir.mkdir(parents=True, exist_ok=True)
    pair_file = processed_dir / "pair_table.csv"
    if not pair_file.exists():
        logger.error("Cannot run S7: %s not found. Run S5 first.", pair_file)
        return {"status": "error", "message": f"{pair_file} not found"}
    df_pairs = pd.read_csv(pair_file)
    results_file = reports_dir / "results.json"
    if results_file.exists():
        results = json.loads(results_file.read_text(encoding="utf-8"))
    else:
        logger.warning("S7 missing input file: %s", results_file)
        results = {}
    a1_all = _read_table(tables_dir / "a1_correlation_all.csv")
    a1_no_spice = _read_table(tables_dir / "a1_correlation_no_spice_herb.csv")
    with PdfPages(figures_dir / "s7_figures.pdf") as pdf:
        _save_figure(_rq1(df_pairs, a1_all, ""), "rq1_scatter.png", figures_dir, pdf)
        no_spice = df_pairs[~(_is_true(df_pairs["is_spice_herb_a"]) | _is_true(df_pairs["is_spice_herb_b"]))]
        _save_figure(_rq1(no_spice, a1_no_spice, " (without spice/herb pairs)"), "rq1_scatter_no_spice_herb.png", figures_dir, pdf)
        rq2 = _rq2(tables_dir)
        if rq2 is not None:
            _save_figure(rq2, "rq2_real_vs_random.png", figures_dir, pdf)
        cuisine = _read_table(tables_dir / "a4_cuisines.csv")
        if cuisine is not None and not cuisine.empty:
            _save_figure(_cuisines(cuisine), "rq3_cuisines.png", figures_dir, pdf)
        cases = _read_table(tables_dir / "a6_case_studies.csv")
        if cases is not None and not cases.empty:
            _save_figure(_case_studies(cases), "case_studies.png", figures_dir, pdf)
        sensitivity = _read_table(tables_dir / "a5_sensitivity.csv")
        if sensitivity is not None and not sensitivity.empty:
            _save_figure(_sensitivity(sensitivity), "sensitivity.png", figures_dir, pdf)
        if "category_pair" in df_pairs and not df_pairs.empty:
            _save_figure(_categories(df_pairs), "category_heatmap.png", figures_dir, pdf)
    simple_data = _load_simple_data(root, processed_dir, tables_dir, results)
    if not df_pairs.empty:
        _simple_rq1(df_pairs, simple_dir / "flavor_vs_nutrients.png")
    else:
        logger.warning("S7 missing simple figure input: pair_table.csv")
    simple_a3 = simple_data["a3"].get("no_spice_herb_within_category")
    if simple_a3 is not None and not simple_a3.empty:
        _simple_rq2(simple_a3.iloc[0], simple_dir / "real_vs_random_simple.png")
    else:
        logger.warning("S7 missing simple figure input: a3_permutation_no_spice_herb_within_category.csv")
    if simple_data["a4"] is not None and not simple_data["a4"].empty:
        _simple_regions(simple_data["a4"], simple_dir / "regions_simple.png")
    if simple_data["a6"] is not None and not simple_data["a6"].empty:
        _simple_cases(simple_data["a6"], simple_dir / "example_pairs_simple.png")
    if simple_data["a5"] is not None and not simple_data["a5"].empty:
        _simple_sensitivity(simple_data["a5"], simple_dir / "sensitivity_simple.png")
    _friendly_dashboard(reports_dir, figures_dir, simple_dir, simple_data)
    logger.info("S7 complete. Generated static dashboard at %s", reports_dir / "dashboard.html")
    return {"status": "success", "dashboard": str(reports_dir / "dashboard.html")}


if __name__ == "__main__":
    run_visualize()
