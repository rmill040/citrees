"""Audit dataset-level trajectories across supported k values.

Operational audit helper; not part of the current paper-facing rebuild path.

This script converts the cleaned aggregate parquets into per-dataset trajectory
summaries. It avoids single-k endpoints by summarizing each dataset over every
supported standard k value.

Outputs:
  - paper/results/tables/dataset_trajectory_summary.csv
  - paper/results/tables/cif_pairwise_trajectory_summary.csv

Usage:
  uv run python paper/scripts/maintenance/audit_dataset_k_trajectories.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TABLES_DIR = RESULTS_DIR / "tables"
STANDARD_K = [5, 10, 25, 50, 100]


TASK_CONFIG = {
    "classification": {
        "path": RESULTS_DIR / "clf_evaluation.parquet",
        "metric": "balanced_accuracy",
        "downstreams": ["lr", "svm", "knn"],
        "focus_method": "cif",
    },
    "regression": {
        "path": RESULTS_DIR / "reg_evaluation.parquet",
        "metric": "r2",
        "downstreams": ["ridge", "svr", "knn"],
        "focus_method": "cif",
    },
}


def _select_best_config(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Keep the single best global config per method_base."""
    perf = df.groupby(["method_base", "method_id"])[metric].mean().reset_index()
    best_idx = perf.groupby("method_base")[metric].idxmax()
    best = perf.loc[best_idx, ["method_base", "method_id"]]
    return df.merge(best, on=["method_base", "method_id"])


def _dataset_trajectory_rows(
    df: pd.DataFrame,
    task: Literal["classification", "regression"],
    downstream_model: str,
    metric: str,
) -> list[dict[str, object]]:
    """Build per-dataset, per-method trajectory summaries."""
    rows: list[dict[str, object]] = []

    agg = (
        df.groupby(["dataset", "method_base", "k"])[metric]
        .mean()
        .reset_index()
        .sort_values(["dataset", "k", "method_base"])
    )

    for dataset, ds_df in agg.groupby("dataset"):
        ds_ranked = ds_df.copy()
        ds_ranked["rank"] = ds_ranked.groupby("k")[metric].rank(ascending=False, method="average")

        for method_base, method_df in ds_ranked.groupby("method_base"):
            method_df = method_df.sort_values("k")
            supported_k = method_df["k"].tolist()
            rank_series = method_df["rank"]
            score_series = method_df[metric]

            rows.append(
                {
                    "task": task,
                    "downstream_model": downstream_model,
                    "dataset": dataset,
                    "method_base": method_base,
                    "n_k": len(supported_k),
                    "supported_k": ",".join(str(k) for k in supported_k),
                    "start_k": int(supported_k[0]),
                    "end_k": int(supported_k[-1]),
                    "start_rank": float(rank_series.iloc[0]),
                    "end_rank": float(rank_series.iloc[-1]),
                    "delta_rank": float(rank_series.iloc[-1] - rank_series.iloc[0]),
                    "mean_rank": float(rank_series.mean()),
                    "best_rank": float(rank_series.min()),
                    "worst_rank": float(rank_series.max()),
                    "start_score": float(score_series.iloc[0]),
                    "end_score": float(score_series.iloc[-1]),
                    "delta_score": float(score_series.iloc[-1] - score_series.iloc[0]),
                    "mean_score": float(score_series.mean()),
                }
            )

    return rows


def _pairwise_rows(
    df: pd.DataFrame,
    task: Literal["classification", "regression"],
    downstream_model: str,
    metric: str,
    focus_method: str,
) -> list[dict[str, object]]:
    """Build CIF-vs-baseline summaries over each dataset's supported k values."""
    rows: list[dict[str, object]] = []
    agg = (
        df.groupby(["dataset", "method_base", "k"])[metric]
        .mean()
        .reset_index()
        .sort_values(["dataset", "k", "method_base"])
    )

    for dataset, ds_df in agg.groupby("dataset"):
        pivot = ds_df.pivot(index="k", columns="method_base", values=metric).sort_index()
        if focus_method not in pivot.columns:
            continue

        for baseline in sorted(col for col in pivot.columns if col != focus_method):
            common = pivot[[focus_method, baseline]].dropna()
            if common.empty:
                continue
            delta = common[focus_method] - common[baseline]
            rows.append(
                {
                    "task": task,
                    "downstream_model": downstream_model,
                    "dataset": dataset,
                    "focus_method": focus_method,
                    "baseline": baseline,
                    "n_k": len(common),
                    "supported_k": ",".join(str(k) for k in common.index.tolist()),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(delta.median()),
                    "start_delta": float(delta.iloc[0]),
                    "end_delta": float(delta.iloc[-1]),
                    "delta_change": float(delta.iloc[-1] - delta.iloc[0]),
                    "win_share": float((delta > 0).mean()),
                    "nonloss_share": float((delta >= 0).mean()),
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit dataset-level trajectories across supported k values"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TABLES_DIR,
        help="Directory for trajectory audit outputs",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []

    for task, config in TASK_CONFIG.items():
        raw = pd.read_parquet(config["path"])
        raw = raw[(raw["dataset_source"] == "real") & (raw["k"].isin(STANDARD_K))]

        metric = config["metric"]
        for downstream_model in config["downstreams"]:
            subset = raw[raw["downstream_model"] == downstream_model].copy()
            subset = _select_best_config(subset, metric)
            trajectory_rows.extend(
                _dataset_trajectory_rows(subset, task, downstream_model, metric)  # type: ignore[arg-type]
            )
            pairwise_rows.extend(
                _pairwise_rows(subset, task, downstream_model, metric, config["focus_method"])  # type: ignore[arg-type]
            )

    trajectory_out = output_dir / "dataset_trajectory_summary.csv"
    pairwise_out = output_dir / "cif_pairwise_trajectory_summary.csv"

    pd.DataFrame(trajectory_rows).sort_values(
        ["task", "downstream_model", "dataset", "method_base"]
    ).to_csv(trajectory_out, index=False)
    pd.DataFrame(pairwise_rows).sort_values(
        ["task", "downstream_model", "dataset", "baseline"]
    ).to_csv(pairwise_out, index=False)

    print(f"Saved {trajectory_out}")
    print(f"Saved {pairwise_out}")


if __name__ == "__main__":
    main()
