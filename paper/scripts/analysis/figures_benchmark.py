"""Generate exploratory benchmark analytics from existing parquet data.

Outputs:
1. Exploratory synthetic per-DGP heatmap (method x DGP precision@10)
2. Dataset characteristics table
3. Method roster table
4. Exploratory variance decomposition (LR-only, all-config ANOVA across all k)
5. Exploratory cumulative accuracy curves (LR-only, all-config trend across k)

Only the dataset inventory portion is manuscript-facing. The other outputs are
diagnostic and use LR-only or all-config views that do not define the paper
contract.

Usage:
    uv run python paper/scripts/analysis/figures_benchmark.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper.scripts.analysis.build_dataset_inventory import build_inventory_table

RESULTS = Path("paper/results")
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

DISPLAY_NAMES = {
    "cif": "CIF", "cit": "CIT", "rf": "RF", "et": "ExtraTrees",
    "lgbm": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost",
    "rfe": "RFE", "boruta": "Boruta", "pi": "PI", "cpi": "CPI",
    "ptest_mc": "MC-ptest", "ptest_rdc": "RDC-ptest",
    "ptest_pc": "PC-ptest", "ptest_dc": "DC-ptest",
    "r_ctree": "R-ctree", "r_cforest": "R-cforest",
}

CIF_BLUE = "#2563EB"
CIT_BLUE = "#60A5FA"
R_ORANGE = "#F97316"
GRAY = "#6B7280"


def _setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def _method_color(m):
    if m in ("cif", "CIF"):
        return CIF_BLUE
    if m in ("cit", "CIT"):
        return CIT_BLUE
    if "r_c" in m.lower() or "R-c" in m:
        return R_ORANGE
    return GRAY


def _dn(m):
    return DISPLAY_NAMES.get(m, m)


def generate_dgp_heatmap():
    """Method x DGP precision@10 heatmap."""
    print("[1/5] Synthetic per-DGP heatmap...")
    df = pd.read_parquet(RESULTS / "synthetic_analysis_clf.parquet")
    df["method_base"] = df["method"].str.split("__").str[0]

    agg = df.groupby(["method_base", "dataset_type"])["precision@10"].mean().reset_index()
    pivot = agg.pivot(index="method_base", columns="dataset_type", values="precision@10")

    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean", ascending=False)
    pivot = pivot.drop(columns="mean")

    pivot.columns = [c.replace("synthetic_", "").replace("_", "\n") for c in pivot.columns]
    pivot.index = [_dn(m) for m in pivot.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.4 or val > 0.85 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label="Precision@10", shrink=0.8)
    out = FIGURES / "exploratory_synthetic_dgp_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved -> {out}")


def generate_dataset_table():
    """Dataset summary generated from the packaged parquet tree."""
    print("[2/5] Dataset characteristics table...")
    ds_info = build_inventory_table()
    out = TABLES / "dataset_characteristics.csv"
    ds_info.to_csv(out, index=False)
    print(f"  Saved -> {out} ({len(ds_info)} datasets)")


def generate_method_roster():
    """Table of all methods with category and key parameters."""
    print("[3/5] Method roster table...")

    roster = [
        {"method": "CIF", "category": "Embedded (CI)", "key_params": "selector=mc, n_est=100, alpha=0.05, Bonferroni", "source": "citrees"},
        {"method": "CIT", "category": "Embedded (CI)", "key_params": "selector=mc, alpha=0.05, Bonferroni", "source": "citrees"},
        {"method": "RF", "category": "Embedded", "key_params": "n_est=100, max_features=sqrt", "source": "scikit-learn"},
        {"method": "ExtraTrees", "category": "Embedded", "key_params": "n_est=100, max_features=sqrt", "source": "scikit-learn"},
        {"method": "XGBoost", "category": "Embedded", "key_params": "n_est=100, importance=total_gain", "source": "xgboost"},
        {"method": "LightGBM", "category": "Embedded", "key_params": "n_est=100, importance=gain", "source": "lightgbm"},
        {"method": "CatBoost", "category": "Embedded", "key_params": "n_est=100, importance=PredictionValuesChange", "source": "catboost"},
        {"method": "Boruta", "category": "Wrapper", "key_params": "RF-based, max_iter=100", "source": "boruta"},
        {"method": "RFE", "category": "Wrapper", "key_params": "RF estimator, step=0.1", "source": "scikit-learn"},
        {"method": "PI", "category": "Wrapper", "key_params": "RF permutation importance, n_repeats=10", "source": "scikit-learn"},
        {"method": "CPI", "category": "Wrapper", "key_params": "Conditional permutation importance", "source": "scikit-learn"},
        {"method": "MC-ptest", "category": "Filter (perm)", "key_params": "Multiple correlation + permutation test", "source": "citrees"},
        {"method": "RDC-ptest", "category": "Filter (perm)", "key_params": "Randomized dependence coefficient + perm", "source": "citrees"},
        {"method": "R-ctree", "category": "Embedded (CI)", "key_params": "partykit::ctree, Bonferroni", "source": "R partykit"},
        {"method": "R-cforest", "category": "Embedded (CI)", "key_params": "partykit::cforest, ntree=100, varimp", "source": "R partykit"},
    ]

    roster_df = pd.DataFrame(roster)
    out = TABLES / "method_roster.csv"
    roster_df.to_csv(out, index=False)
    print(f"  Saved -> {out} ({len(roster_df)} methods)")


def generate_variance_decomposition():
    """Exploratory ANOVA on an LR-only, all-config slice."""
    print("[4/5] Exploratory variance decomposition...")
    df = pd.read_parquet(RESULTS / "clf_evaluation.parquet")
    sub = df[(df["downstream_model"] == "lr")].copy()
    sub = sub[sub["k"].isin([5, 10, 25, 50, 100])]

    grand_mean = sub["balanced_accuracy"].mean()
    total_ss = ((sub["balanced_accuracy"] - grand_mean) ** 2).sum()

    factors = {}
    for factor in ["dataset", "k", "method_base", "downstream_model", "seed"]:
        if factor not in sub.columns:
            continue
        group_means = sub.groupby(factor)["balanced_accuracy"].transform("mean")
        ss = ((group_means - grand_mean) ** 2).sum()
        factors[factor] = ss / total_ss * 100

    result = pd.DataFrame([
        {
            "analysis_scope": "exploratory_lr_only_all_configs",
            "factor": k,
            "pct_variance": v,
        }
        for k, v in
        sorted(factors.items(), key=lambda x: -x[1])
    ])
    out = TABLES / "exploratory_variance_decomposition.csv"
    result.to_csv(out, index=False)
    print(f"  Saved -> {out}")
    print(result.to_string(index=False))


def generate_cumulative_accuracy():
    """Exploratory LR-only accuracy vs k curves aggregated over all configs."""
    print("[5/5] Exploratory cumulative accuracy curves...")
    df = pd.read_parquet(RESULTS / "clf_evaluation.parquet")
    sub = df[(df["downstream_model"] == "lr")].copy()
    standard_k = [5, 10, 25, 50, 100]
    sub = sub[sub["k"].isin(standard_k)]

    agg = sub.groupby(["method_base", "k"])["balanced_accuracy"].mean().reset_index()

    top_methods = ["cif", "rf", "cat", "lgbm", "xgb", "et", "rfe", "r_cforest"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for method in top_methods:
        mdata = agg[agg["method_base"] == method].sort_values("k")
        if mdata.empty:
            continue
        color = _method_color(method)
        lw = 2.5 if method == "cif" else 1.2
        marker = "o" if method == "cif" else ("s" if "r_" in method else "")
        ax.plot(mdata["k"], mdata["balanced_accuracy"], marker=marker, markersize=4,
                label=_dn(method), color=color, linewidth=lw,
                alpha=1.0 if method in ("cif", "r_cforest") else 0.7)

    ax.set_xlabel("Number of selected features ($k$)")
    ax.set_ylabel("Balanced accuracy (LR)")
    ax.set_xticks(standard_k)
    ax.set_title("Exploratory LR-only cumulative accuracy curves", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)

    out = FIGURES / "exploratory_cumulative_accuracy_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved -> {out}")


if __name__ == "__main__":
    _setup_style()
    print("=" * 60)
    print("GENERATING EXPLORATORY BENCHMARK ANALYTICS")
    print("=" * 60)

    generate_dgp_heatmap()
    generate_dataset_table()
    generate_method_roster()
    generate_variance_decomposition()
    generate_cumulative_accuracy()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
