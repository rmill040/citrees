"""CIF configuration sensitivity analysis.

Archived exploratory analysis; not part of the current paper-facing rebuild path.

Quantifies how much CIF's downstream performance varies across its 4 config
variants (selector x honesty), and whether that variation matters relative to
the gap between CIF and competitor methods.

Key questions answered:
  1. How much does balanced accuracy / R2 vary across CIF configs?
  2. Is the "best" config stable across downstream models and k values?
  3. How does the CIF config gap compare to the CIF-vs-competitor gap?
  4. Is CIF config choice more or less important than downstream model choice?

Usage:
    UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/backlog/analysis/study_cif_config_sensitivity.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paper.scripts.pipeline.config import get_method_configs

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS = Path("paper/results")

# ---------------------------------------------------------------------------
# Standard k budgets used in main analysis
# ---------------------------------------------------------------------------
K_VALUES = [5, 10, 25, 50, 100]


# ---------------------------------------------------------------------------
# Config decoding
# ---------------------------------------------------------------------------
def _decode_cif_configs(task: str) -> dict[str, dict[str, Any]]:
    """Build method_id -> {selector, honesty} map for CIF configs.

    Only includes parameters that actually vary across configs.
    """
    configs = get_method_configs("cif", task)
    mapping: dict[str, dict[str, Any]] = {}
    for cfg in configs:
        params = {k: v for k, v in cfg.items() if k not in {"method", "random_state"}}
        payload = json.dumps(dict(sorted(params.items())), sort_keys=True, default=str)
        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]
        method_id = f"cif__{digest}"
        mapping[method_id] = {"selector": params["selector"], "honesty": params["honesty"]}
    return mapping


def _config_label(params: dict[str, Any]) -> str:
    """Short human-readable label for a CIF config."""
    sel = params["selector"].upper()
    hon = "honest" if params["honesty"] else "standard"
    return f"{sel}/{hon}"


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
def _load_real_data(
    path: Path,
    metric: str,
) -> pd.DataFrame:
    """Load evaluation parquet, filter to real datasets and standard k values."""
    df = pd.read_parquet(path)
    mask = (df["dataset_source"] == "real") & (df["k"].isin(K_VALUES))
    return df.loc[mask].copy()


def _mean_metric_per_cell(
    df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Collapse seeds and folds: mean metric per (method_id, downstream, k, dataset)."""
    return (
        df.groupby(["method_id", "method_base", "downstream_model", "k", "dataset"])[metric]
        .mean()
        .reset_index()
    )


def _friedman_ranks(
    cell_means: pd.DataFrame,
    metric: str,
    higher_better: bool = True,
) -> pd.DataFrame:
    """Compute within-cell ranks of each method_id within (downstream, k, dataset).

    Rank 1 = best. Ties averaged.
    """
    cell_means = cell_means.copy()
    cell_means["rank"] = cell_means.groupby(["downstream_model", "k", "dataset"])[metric].rank(
        ascending=not higher_better, method="average"
    )
    return cell_means


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def _section(title: str) -> None:
    width = 78
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _subsection(title: str) -> None:
    print(f"\n--- {title} ---")


def _print_config_table(
    cif_summary: pd.DataFrame,
    config_map: dict[str, dict[str, Any]],
    metric_col: str,
    rank_col: str,
) -> None:
    """Print a table of CIF configs with metric and rank columns."""
    rows = []
    for _, row in cif_summary.iterrows():
        mid = row["method_id"]
        params = config_map.get(mid, {})
        label = _config_label(params) if params else mid
        rows.append({
            "config": label,
            "method_id": mid,
            f"mean_{metric_col}": row[metric_col],
            "mean_rank": row[rank_col],
        })
    tbl = pd.DataFrame(rows).sort_values(f"mean_{metric_col}", ascending=False)
    print(tbl.to_string(index=False, float_format="{:.4f}".format))


# ---------------------------------------------------------------------------
# Analysis for one task
# ---------------------------------------------------------------------------
def analyze_task(
    task: str,
    path: Path,
    metric: str,
    higher_better: bool = True,
) -> None:
    """Run the full CIF config sensitivity analysis for one task."""
    task_label = task.upper()
    metric_display = metric.replace("_", " ").title()
    _section(f"{task_label}: CIF Configuration Sensitivity ({metric_display})")

    # Decode configs
    config_map = _decode_cif_configs(task)
    print(f"\nCIF configs ({len(config_map)}):")
    for mid, params in sorted(config_map.items()):
        print(f"  {mid}  ->  {_config_label(params)}")

    # Load data
    df = _load_real_data(path, metric)
    n_datasets = df["dataset"].nunique()
    print(f"\nDatasets (real): {n_datasets}")
    print(f"Downstream models: {sorted(df['downstream_model'].unique())}")
    print(f"k values: {sorted(df['k'].unique())}")

    # Mean metric per (method_id, downstream, k, dataset)
    cell_means = _mean_metric_per_cell(df, metric)

    # Within-cell ranks across ALL methods (not just CIF)
    ranked = _friedman_ranks(cell_means, metric, higher_better)

    # Total number of method_ids in the competition
    n_methods = cell_means["method_id"].nunique()
    print(f"Total method_ids in ranking pool: {n_methods}")

    # -----------------------------------------------------------------------
    # Q1: How much does metric vary across CIF configs?
    # -----------------------------------------------------------------------
    _subsection("Q1: Metric variation across CIF configs")

    cif_cells = cell_means[cell_means["method_base"] == "cif"]
    cif_ranked = ranked[ranked["method_base"] == "cif"]

    # Global summary per config (across all downstreams, k, datasets)
    global_summary = (
        cif_cells.groupby("method_id")[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: "global_mean"})
    )
    global_ranks = (
        cif_ranked.groupby("method_id")["rank"]
        .mean()
        .reset_index()
        .rename(columns={"rank": "global_mean_rank"})
    )
    global_summary = global_summary.merge(global_ranks, on="method_id")

    _print_config_table(
        global_summary.rename(columns={"global_mean": metric, "global_mean_rank": "rank"}),
        config_map,
        metric,
        "rank",
    )

    best_val = global_summary["global_mean"].max()
    worst_val = global_summary["global_mean"].min()
    config_range = best_val - worst_val
    config_std = global_summary["global_mean"].std()
    print(f"\n  Range (best - worst):  {config_range:.4f}")
    print(f"  Std across configs:    {config_std:.4f}")

    # -----------------------------------------------------------------------
    # Q2: Is the best config stable across downstreams?
    # -----------------------------------------------------------------------
    _subsection("Q2: Best CIF config per downstream model")

    for dm in sorted(cif_cells["downstream_model"].unique()):
        dm_data = cif_cells[cif_cells["downstream_model"] == dm]
        dm_ranked = cif_ranked[cif_ranked["downstream_model"] == dm]

        dm_summary = (
            dm_data.groupby("method_id")[metric]
            .mean()
            .reset_index()
        )
        dm_rank_summary = (
            dm_ranked.groupby("method_id")["rank"]
            .mean()
            .reset_index()
        )
        dm_summary = dm_summary.merge(dm_rank_summary, on="method_id")
        best_mid = dm_summary.loc[dm_summary[metric].idxmax(), "method_id"]
        best_label = _config_label(config_map[best_mid])
        best_metric = dm_summary[metric].max()
        spread = dm_summary[metric].max() - dm_summary[metric].min()
        print(f"\n  downstream={dm}:")
        _print_config_table(dm_summary, config_map, metric, "rank")
        print(f"    Best: {best_label}  ({metric}={best_metric:.4f}, spread={spread:.4f})")

    # -----------------------------------------------------------------------
    # Q3: Is the best config stable across k values?
    # -----------------------------------------------------------------------
    _subsection("Q3: Best CIF config per k value")

    # Build a pivot: rows = config, columns = k
    k_perf: dict[str, dict[int, float]] = {}
    for mid in sorted(config_map.keys()):
        mid_data = cif_cells[cif_cells["method_id"] == mid]
        k_means = mid_data.groupby("k")[metric].mean()
        k_perf[mid] = k_means.to_dict()

    # Print as table
    header = f"{'config':<18s}" + "".join(f"{'k='+str(k):>10s}" for k in K_VALUES)
    print(f"\n  {header}")
    print(f"  {'-' * len(header)}")
    for mid in sorted(config_map.keys()):
        label = _config_label(config_map[mid])
        vals = "".join(f"{k_perf[mid].get(k, float('nan')):10.4f}" for k in K_VALUES)
        print(f"  {label:<18s}{vals}")

    # Best config per k
    best_configs_by_k = []
    best_line = f"  {'Best per k:':<18s}"
    for k in K_VALUES:
        best_mid = max(config_map.keys(), key=lambda m: k_perf[m].get(k, -1))
        label = _config_label(config_map[best_mid])
        best_configs_by_k.append(label)
        best_line += f"{label:>10s}"
    print(f"\n{best_line}")

    stable = len(set(best_configs_by_k)) == 1
    print(f"\n  Config stable across k? {'YES' if stable else 'NO'} ({len(set(best_configs_by_k))} distinct)")

    # -----------------------------------------------------------------------
    # Q4: CIF config gap vs CIF-competitor gap
    # -----------------------------------------------------------------------
    _subsection("Q4: CIF config gap vs CIF-to-competitor gap")

    # Best CIF config overall
    best_cif_mid = global_summary.loc[global_summary["global_mean"].idxmax(), "method_id"]
    best_cif_val = global_summary["global_mean"].max()
    worst_cif_val = global_summary["global_mean"].min()

    # Best non-CIF method_id
    non_cif = cell_means[cell_means["method_base"] != "cif"]
    non_cif_global = non_cif.groupby(["method_id", "method_base"])[metric].mean().reset_index()
    if higher_better:
        best_competitor_row = non_cif_global.loc[non_cif_global[metric].idxmax()]
    else:
        best_competitor_row = non_cif_global.loc[non_cif_global[metric].idxmin()]
    best_competitor_mid = best_competitor_row["method_id"]
    best_competitor_base = best_competitor_row["method_base"]
    best_competitor_val = best_competitor_row[metric]

    # Also get 2nd-best non-CIF
    non_cif_sorted = non_cif_global.sort_values(metric, ascending=not higher_better)
    second_competitor_row = non_cif_sorted.iloc[1] if len(non_cif_sorted) > 1 else None

    cif_config_gap = best_cif_val - worst_cif_val
    cif_vs_top_gap = best_cif_val - best_competitor_val

    print(f"\n  Best CIF:        {_config_label(config_map[best_cif_mid]):>18s}  {metric}={best_cif_val:.4f}")
    print(f"  Worst CIF:       {_config_label(config_map[global_summary.loc[global_summary['global_mean'].idxmin(), 'method_id']]):>18s}  {metric}={worst_cif_val:.4f}")
    print(f"  Best competitor: {best_competitor_base:>18s}  {metric}={best_competitor_val:.4f}  ({best_competitor_mid})")
    if second_competitor_row is not None:
        print(f"  2nd competitor:  {second_competitor_row['method_base']:>18s}  {metric}={second_competitor_row[metric]:.4f}")
    print()
    print(f"  CIF config gap (best-worst CIF):   {cif_config_gap:+.4f}")
    print(f"  CIF vs top competitor:              {cif_vs_top_gap:+.4f}")
    ratio = cif_config_gap / abs(cif_vs_top_gap) if abs(cif_vs_top_gap) > 1e-8 else float("inf")
    print(f"  Ratio (config_gap / competitor_gap): {ratio:.2f}x")
    if cif_config_gap < abs(cif_vs_top_gap):
        print("  => CIF config choice matters LESS than method choice.")
    else:
        print("  => CIF config choice matters MORE than method choice.")

    # -----------------------------------------------------------------------
    # Q5: CIF config choice vs downstream model choice
    # -----------------------------------------------------------------------
    _subsection("Q5: Config choice vs downstream model choice")

    # Variation due to CIF config (fix downstream, vary config)
    config_var_by_dm: dict[str, float] = {}
    for dm in sorted(cif_cells["downstream_model"].unique()):
        dm_means = (
            cif_cells[cif_cells["downstream_model"] == dm]
            .groupby("method_id")[metric]
            .mean()
        )
        config_var_by_dm[dm] = dm_means.max() - dm_means.min()

    # Variation due to downstream model (fix config, vary downstream)
    dm_var_by_config: dict[str, float] = {}
    for mid in sorted(config_map.keys()):
        mid_means = (
            cif_cells[cif_cells["method_id"] == mid]
            .groupby("downstream_model")[metric]
            .mean()
        )
        dm_var_by_config[mid] = mid_means.max() - mid_means.min()

    avg_config_var = np.mean(list(config_var_by_dm.values()))
    avg_dm_var = np.mean(list(dm_var_by_config.values()))

    print(f"\n  Variation from CIF config choice (avg spread across downstreams): {avg_config_var:.4f}")
    for dm, v in sorted(config_var_by_dm.items()):
        print(f"    downstream={dm}: config spread = {v:.4f}")

    print(f"\n  Variation from downstream model choice (avg spread across configs): {avg_dm_var:.4f}")
    for mid, v in sorted(dm_var_by_config.items()):
        label = _config_label(config_map[mid])
        print(f"    config={label}: downstream spread = {v:.4f}")

    print(f"\n  Ratio (downstream_var / config_var): {avg_dm_var / avg_config_var:.2f}x")
    if avg_dm_var > avg_config_var:
        print("  => Downstream model choice matters MORE than CIF config choice.")
    else:
        print("  => CIF config choice matters MORE than downstream model choice.")

    # -----------------------------------------------------------------------
    # Summary: full breakdown table (config x downstream)
    # -----------------------------------------------------------------------
    _subsection("Full breakdown: mean metric by (config, downstream)")

    pivot = (
        cif_cells.groupby(["method_id", "downstream_model"])[metric]
        .mean()
        .reset_index()
        .pivot(index="method_id", columns="downstream_model", values=metric)
    )
    pivot["overall"] = pivot.mean(axis=1)
    pivot.index = [_config_label(config_map[mid]) for mid in pivot.index]
    pivot = pivot.sort_values("overall", ascending=False)
    print()
    print(pivot.to_string(float_format="{:.4f}".format))

    # -----------------------------------------------------------------------
    # Summary: full breakdown table (config x k)
    # -----------------------------------------------------------------------
    _subsection("Full breakdown: mean metric by (config, k)")

    pivot_k = (
        cif_cells.groupby(["method_id", "k"])[metric]
        .mean()
        .reset_index()
        .pivot(index="method_id", columns="k", values=metric)
    )
    pivot_k["overall"] = pivot_k.mean(axis=1)
    pivot_k.index = [_config_label(config_map[mid]) for mid in pivot_k.index]
    pivot_k = pivot_k.sort_values("overall", ascending=False)
    print()
    print(pivot_k.to_string(float_format="{:.4f}".format))

    # -----------------------------------------------------------------------
    # Summary: Friedman-style mean rank by (config, downstream)
    # -----------------------------------------------------------------------
    _subsection(f"Friedman-style mean rank by (config, downstream)  [1=best, {n_methods}=worst]")

    rank_pivot = (
        cif_ranked.groupby(["method_id", "downstream_model"])["rank"]
        .mean()
        .reset_index()
        .pivot(index="method_id", columns="downstream_model", values="rank")
    )
    rank_pivot["overall"] = rank_pivot.mean(axis=1)
    rank_pivot.index = [_config_label(config_map[mid]) for mid in rank_pivot.index]
    rank_pivot = rank_pivot.sort_values("overall")
    print()
    print(rank_pivot.to_string(float_format="{:.2f}".format))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("CIF Configuration Sensitivity Analysis")
    print("=" * 78)

    analyze_task(
        task="classification",
        path=RESULTS / "clf_evaluation.parquet",
        metric="balanced_accuracy",
        higher_better=True,
    )

    analyze_task(
        task="regression",
        path=RESULTS / "reg_evaluation.parquet",
        metric="r2",
        higher_better=True,
    )

    # -----------------------------------------------------------------------
    # Cross-task summary
    # -----------------------------------------------------------------------
    _section("CROSS-TASK SUMMARY")
    print("""
    Both tasks have 4 CIF configs each: 2 selectors x 2 honesty settings.

    CLF selectors: MC, RDC    |  REG selectors: PC, RDC
    Honesty: True / False     |  Honesty: True / False

    See tables above for detailed per-downstream, per-k breakdowns.
    """)


if __name__ == "__main__":
    main()
