"""Build canonical synthetic top-k summary tables.

This script answers two related questions:
  1. How much ground-truth signal appears in the top-k ranked features?
  2. What fills the rest of the top-k: redundant proxies, correlated-noise
     confounders, noise variables, or simply missing/unreturned positions?

The analysis uses the canonical aggregated synthetic ranking parquets and
selects one best configuration per method family within task by the mean
percentage of selected features that are truly informative over the standard
synthetic budget curve
`k in {5, 10, 25, 50, 100}` with effective `k=min(k,p)`.

Outputs:
  - paper/results/tables/synthetic_topk_best_configs.csv
  - paper/results/tables/synthetic_topk_best_config_details.csv
  - paper/results/tables/synthetic_topk_composition_summary.csv
  - paper/results/tables/synthetic_topk_composition_curve_summary.csv
  - paper/results/tables/synthetic_topk_composition_by_dataset_type.csv
  - paper/results/tables/synthetic_topk_composition_curve_by_dataset_type.csv
  - paper/results/tables/synthetic_topk_composition_by_dataset.csv
  - paper/results/tables/synthetic_topk_composition_focus.csv
  - paper/results/synthetic_topk_composition.parquet
  - paper/results/figures/synthetic_topk_focus_curves.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, TypedDict

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from paper.scripts.analysis.analyze_synthetic_ground_truth import (
    dataset_type_from_config,
    load_synthetic_metadata,
)
from paper.scripts.analysis.config_resolution import resolve_method_config_details


class TopKTaskConfig(TypedDict):
    """Typed configuration for one synthetic-ranking task."""

    rankings_path: Path
    data_dir: Path


class FeatureGroups(TypedDict):
    """Ground-truth feature categories for one synthetic dataset."""

    informative: set[int]
    redundant: set[int]
    explicit_noise: set[int]
    correlated_noise: set[int]
    background_null: set[int]
    n_features_final: int


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ARXIV_FIGURES_DIR = Path(__file__).resolve().parents[2] / "arxiv" / "figures"
EXISTING_DIAGNOSTICS_PATH = RESULTS_DIR / "synthetic_topk_composition.parquet"

TASK_CONFIG: Final[dict[str, TopKTaskConfig]] = {
    "classification": {
        "rankings_path": RESULTS_DIR / "clf_rankings.parquet",
        "data_dir": Path("paper/data/classification/synthetic"),
    },
    "regression": {
        "rankings_path": RESULTS_DIR / "reg_rankings.parquet",
        "data_dir": Path("paper/data/regression/synthetic"),
    },
}

TOP_K_VALUES: Final[tuple[int, ...]] = (1, 2, 5, 10, 20, 25, 50, 100)
SELECTION_K_VALUES: Final[tuple[int, ...]] = (5, 10, 25, 50, 100)
FOCUS_PLOT_K: Final[tuple[int, ...]] = (5, 10, 25, 50, 100)
FOCUS_K_POSITIONS: Final[dict[int, int]] = {k: idx for idx, k in enumerate(FOCUS_PLOT_K)}
FOCUS_METHODS: Final[tuple[str, ...]] = ("cit", "dt", "rt", "cif", "rf", "et", "xgb", "lgbm", "cat")
FAMILY_PANELS: Final[tuple[tuple[str, str, tuple[str, ...]], ...]] = (
    ("single_tree", "Single trees", ("cit", "dt", "rt")),
    ("forest", "Forests", ("cif", "rf", "et")),
    ("boosted_tree", "Boosted trees", ("cif", "xgb", "lgbm", "cat")),
)

DISPLAY_NAMES: Final[dict[str, str]] = {
    "cif": "CIF",
    "cit": "CIT",
    "dt": "DT",
    "rt": "RT",
    "rf": "RF",
    "et": "ExtraTrees",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "cat": "CatBoost",
}

METHOD_COLORS: Final[dict[str, str]] = {
    "cif": "#2563EB",
    "cit": "#60A5FA",
    "dt": "#6B7280",
    "rt": "#F59E0B",
    "rf": "#15803D",
    "et": "#0F766E",
    "xgb": "#EA580C",
    "lgbm": "#7C3AED",
    "cat": "#B91C1C",
}

TASK_TITLES: Final[dict[str, str]] = {
    "classification": "Classification",
    "regression": "Regression",
}


def _build_feature_groups(meta: dict[str, object]) -> FeatureGroups:
    """Return mutually exclusive ground-truth feature groups for one dataset."""
    n_features_final = int(meta["n_features_final"])
    informative = set(int(i) for i in meta["informative_indices"])
    redundant = set(int(i) for i in meta.get("redundant_indices", []))
    explicit_noise = set(int(i) for i in meta.get("noise_indices", []))
    correlated_noise = set(int(i) for i in meta.get("correlated_noise_indices", []))
    background_null = set(range(n_features_final)) - informative - redundant - explicit_noise - correlated_noise

    return {
        "informative": informative,
        "redundant": redundant,
        "explicit_noise": explicit_noise,
        "correlated_noise": correlated_noise,
        "background_null": background_null,
        "n_features_final": n_features_final,
    }


def _count_hits(top_k: list[int], indices: set[int]) -> int:
    """Count how many returned top-k positions fall into a category."""
    return sum(int(feature in indices) for feature in top_k)


def _build_row_metrics(ranking: list[int], groups: FeatureGroups, k: int) -> dict[str, float | int]:
    """Compute top-k composition metrics for one ranking."""
    effective_k = min(k, groups["n_features_final"])
    top_k = [int(feature) for feature in ranking[:effective_k]]
    returned_count = len(top_k)
    missing_count = effective_k - returned_count
    dataset_size_cap_count = k - effective_k

    informative_count = _count_hits(top_k, groups["informative"])
    redundant_count = _count_hits(top_k, groups["redundant"])
    explicit_noise_count = _count_hits(top_k, groups["explicit_noise"])
    correlated_noise_count = _count_hits(top_k, groups["correlated_noise"])
    background_null_count = _count_hits(top_k, groups["background_null"])

    signal_or_redundant_count = informative_count + redundant_count
    pure_noise_count = explicit_noise_count + background_null_count

    denom = float(effective_k) if effective_k > 0 else 1.0

    return {
        "effective_k": effective_k,
        "returned_count": returned_count,
        "missing_count": missing_count,
        "dataset_size_cap_count": dataset_size_cap_count,
        "informative_count": informative_count,
        "redundant_count": redundant_count,
        "signal_or_redundant_count": signal_or_redundant_count,
        "explicit_noise_count": explicit_noise_count,
        "background_null_count": background_null_count,
        "pure_noise_count": pure_noise_count,
        "correlated_noise_count": correlated_noise_count,
        "returned_share": returned_count / denom,
        "missing_share": missing_count / denom,
        "dataset_size_cap_share": dataset_size_cap_count / k if k > 0 else 0.0,
        "informative_share": informative_count / denom,
        "redundant_share": redundant_count / denom,
        "signal_or_redundant_share": signal_or_redundant_count / denom,
        "explicit_noise_share": explicit_noise_count / denom,
        "background_null_share": background_null_count / denom,
        "pure_noise_share": pure_noise_count / denom,
        "correlated_noise_share": correlated_noise_count / denom,
    }


def _select_best_configs(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Select one synthetic configuration per method family and task."""
    perf = (
        diagnostics[diagnostics["k"].isin(SELECTION_K_VALUES)]
        .groupby(["task", "method_base", "method_id"], as_index=False)["informative_share"]
        .mean()
        .rename(columns={"informative_share": "selection_score"})
    )
    best_idx = perf.groupby(["task", "method_base"])["selection_score"].idxmax()
    best = perf.loc[best_idx].copy()
    best["config_selection_metric"] = "mean_informative_share_over_k_5_10_25_50_100"
    return best


def _compute_task_rows_from_rankings(task: str, rankings: pd.DataFrame, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return row-level metrics and best-config table for one task from rankings."""
    rankings = rankings[rankings["dataset_source"] == "synthetic"].copy()

    metadata = load_synthetic_metadata(data_dir)
    rows: list[dict[str, object]] = []

    for _, record in rankings.iterrows():
        dataset = str(record["dataset"])
        if dataset not in metadata:
            continue

        meta = metadata[dataset]
        groups = _build_feature_groups(meta)
        ranking = [int(feature) for feature in record["feature_ranking"]]

        base_row = {
            "task": task,
            "dataset": dataset,
            "dataset_type": dataset_type_from_config(meta["config"]),
            "method_base": str(record["method_base"]),
            "method_id": str(record["method_id"]),
            "seed": int(record["seed"]),
            "fold_idx": int(record["fold_idx"]),
            "n_features_final": groups["n_features_final"],
            "n_returned_total": len(ranking),
            "n_informative": len(groups["informative"]),
            "n_redundant": len(groups["redundant"]),
            "n_explicit_noise": len(groups["explicit_noise"]),
            "n_correlated_noise": len(groups["correlated_noise"]),
            "n_background_null": len(groups["background_null"]),
        }

        for k in TOP_K_VALUES:
            row = dict(base_row)
            row["k"] = k
            row.update(_build_row_metrics(ranking, groups, k))
            rows.append(row)

    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty:
        return diagnostics, pd.DataFrame()

    best = _select_best_configs(diagnostics)

    diagnostics = diagnostics.merge(
        best[["task", "method_base", "method_id"]],
        on=["task", "method_base", "method_id"],
        how="inner",
    )

    return diagnostics, best


def _compute_task_rows(task: str, rankings_path: Path, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return row-level metrics and best-config table for one task."""
    if not rankings_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    rankings = pd.read_parquet(rankings_path)
    return _compute_task_rows_from_rankings(task, rankings, data_dir)


def _summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Method-level summary across all synthetic datasets."""
    metric_cols = [
        "effective_k",
        "returned_count",
        "missing_count",
        "dataset_size_cap_count",
        "informative_count",
        "redundant_count",
        "signal_or_redundant_count",
        "explicit_noise_count",
        "background_null_count",
        "pure_noise_count",
        "correlated_noise_count",
        "returned_share",
        "missing_share",
        "dataset_size_cap_share",
        "informative_share",
        "redundant_share",
        "signal_or_redundant_share",
        "explicit_noise_share",
        "background_null_share",
        "pure_noise_share",
        "correlated_noise_share",
    ]

    summary = (
        df.groupby(["task", "method_base", "method_id", "k"], as_index=False)
        .agg(
            n_rankings=("dataset", "size"),
            n_datasets=("dataset", "nunique"),
            **{metric: (metric, "mean") for metric in metric_cols},
        )
        .sort_values(["task", "k", "informative_share"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    summary["informative_share_rank"] = summary.groupby(["task", "k"])["informative_share"].rank(
        ascending=False,
        method="average",
    )
    summary["signal_or_redundant_share_rank"] = summary.groupby(["task", "k"])[
        "signal_or_redundant_share"
    ].rank(
        ascending=False,
        method="average",
    )
    summary["pure_noise_share_rank"] = summary.groupby(["task", "k"])["pure_noise_share"].rank(
        ascending=True,
        method="average",
    )
    summary["correlated_noise_share_rank"] = summary.groupby(["task", "k"])[
        "correlated_noise_share"
    ].rank(
        ascending=True,
        method="average",
    )
    summary["missing_share_rank"] = summary.groupby(["task", "k"])["missing_share"].rank(
        ascending=True,
        method="average",
    )
    return summary


def _summarize_by_dataset_type(df: pd.DataFrame) -> pd.DataFrame:
    """Method-level summary within each synthetic dataset type."""
    metric_cols = [
        "effective_k",
        "returned_count",
        "missing_count",
        "dataset_size_cap_count",
        "informative_count",
        "redundant_count",
        "signal_or_redundant_count",
        "explicit_noise_count",
        "background_null_count",
        "pure_noise_count",
        "correlated_noise_count",
        "returned_share",
        "missing_share",
        "dataset_size_cap_share",
        "informative_share",
        "redundant_share",
        "signal_or_redundant_share",
        "explicit_noise_share",
        "background_null_share",
        "pure_noise_share",
        "correlated_noise_share",
    ]

    summary = (
        df.groupby(["task", "dataset_type", "method_base", "method_id", "k"], as_index=False)
        .agg(
            n_rankings=("dataset", "size"),
            n_datasets=("dataset", "nunique"),
            **{metric: (metric, "mean") for metric in metric_cols},
        )
        .sort_values(["task", "dataset_type", "k", "informative_share"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )
    return summary


def _summarize_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Method-level summary within each synthetic dataset."""
    metric_cols = [
        "effective_k",
        "returned_count",
        "missing_count",
        "dataset_size_cap_count",
        "informative_count",
        "redundant_count",
        "signal_or_redundant_count",
        "explicit_noise_count",
        "background_null_count",
        "pure_noise_count",
        "correlated_noise_count",
        "returned_share",
        "missing_share",
        "dataset_size_cap_share",
        "informative_share",
        "redundant_share",
        "signal_or_redundant_share",
        "explicit_noise_share",
        "background_null_share",
        "pure_noise_share",
        "correlated_noise_share",
    ]

    summary = (
        df.groupby(["task", "dataset", "dataset_type", "method_base", "method_id", "k"], as_index=False)
        .agg(
            n_rankings=("dataset", "size"),
            n_datasets=("dataset", "nunique"),
            **{metric: (metric, "mean") for metric in metric_cols},
        )
        .sort_values(
            ["task", "dataset", "k", "informative_share"],
            ascending=[True, True, True, False],
        )
        .reset_index(drop=True)
    )
    return summary


def _summarize_curve_over_k(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Collapse the standard synthetic budget curve into trend summaries."""
    curve = df[df["k"].isin(SELECTION_K_VALUES)].copy()
    summary = (
        curve.groupby(group_cols, as_index=False)
        .agg(
            n_rankings=("dataset", "size"),
            n_datasets=("dataset", "nunique"),
            mean_informative_share_over_standard_k=("informative_share", "mean"),
            mean_signal_or_redundant_share_over_standard_k=("signal_or_redundant_share", "mean"),
            mean_pure_noise_share_over_standard_k=("pure_noise_share", "mean"),
            mean_correlated_noise_share_over_standard_k=("correlated_noise_share", "mean"),
            mean_missing_share_over_standard_k=("missing_share", "mean"),
            mean_dataset_size_cap_share_over_standard_k=("dataset_size_cap_share", "mean"),
        )
        .reset_index(drop=True)
    )

    for k_value, suffix in ((5, "k5"), (100, "k100")):
        cols = group_cols + [
            "informative_share",
            "signal_or_redundant_share",
            "pure_noise_share",
            "correlated_noise_share",
        ]
        snapshot = (
            curve[curve["k"] == k_value][cols]
            .groupby(group_cols, as_index=False)
            .mean()
            .rename(
                columns={
                    "informative_share": f"informative_share_{suffix}",
                    "signal_or_redundant_share": f"signal_or_redundant_share_{suffix}",
                    "pure_noise_share": f"pure_noise_share_{suffix}",
                    "correlated_noise_share": f"correlated_noise_share_{suffix}",
                }
            )
        )
        summary = summary.merge(snapshot, on=group_cols, how="left")

    summary["informative_share_delta_k100_minus_k5"] = (
        summary["informative_share_k100"] - summary["informative_share_k5"]
    )
    summary["signal_or_redundant_share_delta_k100_minus_k5"] = (
        summary["signal_or_redundant_share_k100"] - summary["signal_or_redundant_share_k5"]
    )
    summary["pure_noise_share_delta_k100_minus_k5"] = (
        summary["pure_noise_share_k100"] - summary["pure_noise_share_k5"]
    )
    summary["correlated_noise_share_delta_k100_minus_k5"] = (
        summary["correlated_noise_share_k100"] - summary["correlated_noise_share_k5"]
    )

    if "method_base" in summary.columns:
        summary["mean_informative_share_rank"] = summary.groupby(
            [col for col in group_cols if col != "method_base" and col != "method_id"],
            dropna=False,
        )["mean_informative_share_over_standard_k"].rank(
            ascending=False,
            method="average",
        )
        summary["mean_pure_noise_share_rank"] = summary.groupby(
            [col for col in group_cols if col != "method_base" and col != "method_id"],
            dropna=False,
        )["mean_pure_noise_share_over_standard_k"].rank(
            ascending=True,
            method="average",
        )

    return summary


def _build_focus_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Filter the overall summary to method-family comparisons used in Figure 4."""
    focus = summary[summary["method_base"].isin(FOCUS_METHODS)].copy()
    return focus.sort_values(["task", "k", "informative_share"], ascending=[True, True, False]).reset_index(
        drop=True
    )


def _setup_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def _plot_family(
    ax: plt.Axes,
    df: pd.DataFrame,
    task: str,
    methods: tuple[str, ...],
    title: str,
) -> None:
    """Plot top-k recovery curves for one method family within a task."""
    task_df = df[(df["task"] == task) & (df["k"].isin(FOCUS_PLOT_K))].copy()

    for method_base in methods:
        method_df = task_df[task_df["method_base"] == method_base].sort_values("k")
        if method_df.empty:
            continue

        is_focus = method_base in {"cit", "cif"}
        x_positions = method_df["k"].map(FOCUS_K_POSITIONS)
        ax.plot(
            x_positions,
            method_df["informative_share"],
            color=METHOD_COLORS[method_base],
            linewidth=2.6 if is_focus else 1.8,
            alpha=1.0 if is_focus else 0.9,
            marker="o" if method_base in {"cif", "cit"} else "s",
            markersize=6.5 if is_focus else 5.0,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=DISPLAY_NAMES[method_base],
            zorder=10 if is_focus else 5,
        )

    ax.set_title(title, pad=6)
    ax.set_xticks(range(len(FOCUS_PLOT_K)))
    ax.set_xticklabels([str(k) for k in FOCUS_PLOT_K])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, axis="both")
    ax.legend(loc="upper right", frameon=False, fontsize=8, handlelength=1.6)


def _save_focus_figure(focus: pd.DataFrame) -> None:
    """Render synthetic recovery curves by method family."""
    _setup_style()

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.2), sharex=True, sharey=True)
    for row_idx, task in enumerate(("classification", "regression")):
        for col_idx, (_, family_title, methods) in enumerate(FAMILY_PANELS):
            title = family_title if row_idx == 0 else ""
            _plot_family(axes[row_idx, col_idx], focus, task=task, methods=methods, title=title)
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(f"{TASK_TITLES[task]}\n\\% informative\namong selected")
            if row_idx == 1:
                axes[row_idx, col_idx].set_xlabel(r"Number of selected features ($k$)")

    fig.subplots_adjust(top=0.92, hspace=0.18, wspace=0.16, bottom=0.12, left=0.12)
    for out_dir in (FIGURES_DIR, ARXIV_FIGURES_DIR):
        out_path = out_dir / "synthetic_topk_focus_curves.png"
        fig.savefig(out_path)
        print(f"Saved {out_path}")
    plt.close(fig)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARXIV_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []

    rankings_available = all(config["rankings_path"].exists() for config in TASK_CONFIG.values())
    if not rankings_available and EXISTING_DIAGNOSTICS_PATH.exists():
        all_rows.append(pd.read_parquet(EXISTING_DIAGNOSTICS_PATH))
        print(f"Using joined synthetic top-k diagnostics from {EXISTING_DIAGNOSTICS_PATH}")
    else:
        for task, config in TASK_CONFIG.items():
            diagnostics, _ = _compute_task_rows(
                task=task,
                rankings_path=config["rankings_path"],
                data_dir=config["data_dir"],
            )
            if diagnostics.empty:
                continue
            all_rows.append(diagnostics)

    if not all_rows:
        raise RuntimeError("No synthetic top-k diagnostics were generated.")

    diagnostics = pd.concat(all_rows, ignore_index=True)

    best_configs = _select_best_configs(diagnostics)
    best_config_details = resolve_method_config_details(best_configs)

    summary = _summarize_overall(diagnostics)
    curve_summary = _summarize_curve_over_k(
        diagnostics,
        ["task", "method_base", "method_id"],
    )
    by_dataset_type = _summarize_by_dataset_type(diagnostics)
    curve_by_dataset_type = _summarize_curve_over_k(
        diagnostics,
        ["task", "dataset_type", "method_base", "method_id"],
    )
    by_dataset = _summarize_by_dataset(diagnostics)
    focus = _build_focus_table(summary)

    diagnostics.to_parquet(RESULTS_DIR / "synthetic_topk_composition.parquet", index=False)
    best_configs.to_csv(TABLES_DIR / "synthetic_topk_best_configs.csv", index=False)
    best_config_details.to_csv(TABLES_DIR / "synthetic_topk_best_config_details.csv", index=False)
    summary.to_csv(TABLES_DIR / "synthetic_topk_composition_summary.csv", index=False)
    curve_summary.to_csv(TABLES_DIR / "synthetic_topk_composition_curve_summary.csv", index=False)
    by_dataset_type.to_csv(TABLES_DIR / "synthetic_topk_composition_by_dataset_type.csv", index=False)
    curve_by_dataset_type.to_csv(TABLES_DIR / "synthetic_topk_composition_curve_by_dataset_type.csv", index=False)
    by_dataset.to_csv(TABLES_DIR / "synthetic_topk_composition_by_dataset.csv", index=False)
    focus.to_csv(TABLES_DIR / "synthetic_topk_composition_focus.csv", index=False)

    _save_focus_figure(focus)

    print("Saved synthetic top-k composition outputs:")
    print(f"  - {RESULTS_DIR / 'synthetic_topk_composition.parquet'}")
    print(f"  - {TABLES_DIR / 'synthetic_topk_best_configs.csv'}")
    print(f"  - {TABLES_DIR / 'synthetic_topk_best_config_details.csv'}")
    print(f"  - {TABLES_DIR / 'synthetic_topk_composition_summary.csv'}")
    print(f"  - {TABLES_DIR / 'synthetic_topk_composition_curve_summary.csv'}")
    print(f"  - {TABLES_DIR / 'synthetic_topk_composition_by_dataset_type.csv'}")
    print(f"  - {TABLES_DIR / 'synthetic_topk_composition_curve_by_dataset_type.csv'}")
    print(f"  - {TABLES_DIR / 'synthetic_topk_composition_by_dataset.csv'}")
    print(f"  - {TABLES_DIR / 'synthetic_topk_composition_focus.csv'}")
    print(f"  - {FIGURES_DIR / 'synthetic_topk_focus_curves.png'}")


if __name__ == "__main__":
    main()
