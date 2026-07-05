"""Build leave-one-dataset-out config-selection sensitivity tables.

This analysis answers a specific reviewer concern: how much the benchmark
ordering depends on selecting one global config per method family on the same
surface later summarized.

Contract:
  - real datasets only,
  - standard k values only,
  - 14-dataset benchmark datasets only,
  - for each held-out dataset, select one config per family using the
    remaining 14-dataset benchmark datasets,
  - evaluate the selected configs on the held-out dataset only.

Outputs:
  - paper/results/tables/paper_benchmark_lodo_selected_configs.csv
  - paper/results/tables/paper_benchmark_lodo_aggregate.csv
  - paper/results/tables/paper_benchmark_lodo_config_stability.csv

Usage:
  uv run python paper/analysis/build_lodo_config_sensitivity_tables.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.analysis.benchmark_common import (  # noqa: E402
    STANDARD_K,
    TABLES_DIR,
    TASK_CONFIG,
    load_real_task_frame,
)

BEST_CONFIGS_PATH: Final[Path] = TABLES_DIR / "paper_benchmark_best_configs.csv"
FIXED_PANEL_MEMBERSHIP_PATH: Final[Path] = TABLES_DIR / "paper_benchmark_fixed_panel_membership.csv"
SELECTED_OUT_PATH: Final[Path] = TABLES_DIR / "paper_benchmark_lodo_selected_configs.csv"
AGGREGATE_OUT_PATH: Final[Path] = TABLES_DIR / "paper_benchmark_lodo_aggregate.csv"
STABILITY_OUT_PATH: Final[Path] = TABLES_DIR / "paper_benchmark_lodo_config_stability.csv"


def _load_fixed_panel_datasets(task: str) -> list[str]:
    membership = pd.read_csv(FIXED_PANEL_MEMBERSHIP_PATH)
    fixed = membership[(membership["task"] == task) & (membership["is_fixed_panel"])]
    return sorted(fixed["dataset"].tolist())


def _global_best_map(task: str) -> dict[str, str]:
    best = pd.read_csv(BEST_CONFIGS_PATH)
    sub = best[best["task"] == task][["method_base", "method_id"]]
    return dict(zip(sub["method_base"], sub["method_id"], strict=False))


def _task_cell_scores(task: str) -> tuple[pd.DataFrame, str]:
    task_cfg = TASK_CONFIG[task]
    frame = load_real_task_frame(task=task)
    frame = frame[frame["k"].isin(STANDARD_K)].copy()
    metric = task_cfg["metric"]
    cell = (
        frame.groupby(
            ["dataset", "downstream_model", "k", "method_base", "method_id"], as_index=False
        )[metric]
        .mean()
        .rename(columns={metric: "dataset_mean_score"})
    )
    return cell, metric


def build_lodo_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build leave-one-dataset-out sensitivity outputs for both tasks."""
    selected_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []

    for task in TASK_CONFIG:
        cell_scores, _metric = _task_cell_scores(task)
        fixed_panel_datasets = _load_fixed_panel_datasets(task)
        cell_scores = cell_scores[cell_scores["dataset"].isin(fixed_panel_datasets)].copy()
        global_best = _global_best_map(task)

        heldout_evals: list[pd.DataFrame] = []

        for heldout_dataset in fixed_panel_datasets:
            train = cell_scores[cell_scores["dataset"] != heldout_dataset]
            test = cell_scores[cell_scores["dataset"] == heldout_dataset]

            perf = (
                train.groupby(["method_base", "method_id"], as_index=False)["dataset_mean_score"]
                .mean()
                .rename(columns={"dataset_mean_score": "task_global_mean_metric"})
            )
            best_idx = perf.groupby("method_base")["task_global_mean_metric"].idxmax()
            selected = perf.loc[best_idx].sort_values("method_base").reset_index(drop=True)
            selected["task"] = task
            selected["heldout_dataset"] = heldout_dataset
            selected_rows.extend(selected.to_dict(orient="records"))

            eval_df = test.merge(
                selected[["method_base", "method_id"]], on=["method_base", "method_id"], how="inner"
            )
            dataset_scores = (
                eval_df.groupby(["method_base", "method_id"], as_index=False)["dataset_mean_score"]
                .mean()
                .sort_values("method_base")
                .reset_index(drop=True)
            )
            dataset_scores["task"] = task
            dataset_scores["heldout_dataset"] = heldout_dataset
            dataset_scores["rank"] = dataset_scores["dataset_mean_score"].rank(
                ascending=False, method="average"
            )
            heldout_evals.append(dataset_scores)

        selected_df = pd.DataFrame(selected_rows)
        task_selected = selected_df[selected_df["task"] == task].copy()
        held_df = pd.concat(heldout_evals, ignore_index=True)

        aggregate = (
            held_df.groupby(["task", "method_base"], as_index=False)
            .agg(
                n_heldout_datasets=("heldout_dataset", "nunique"),
                mean_rank=("rank", "mean"),
                median_rank=("rank", "median"),
                mean_score=("dataset_mean_score", "mean"),
            )
            .sort_values(["task", "mean_rank", "method_base"])
            .reset_index(drop=True)
        )
        aggregate["rank_position"] = aggregate.groupby("task")["mean_rank"].rank(
            ascending=True, method="average"
        )
        aggregate_rows.extend(aggregate.to_dict(orient="records"))

        stability = (
            task_selected.groupby(["method_base", "method_id"], as_index=False)
            .agg(
                n_selected=("heldout_dataset", "nunique"),
                mean_train_score=("task_global_mean_metric", "mean"),
            )
            .sort_values(
                ["method_base", "n_selected", "mean_train_score"], ascending=[True, False, False]
            )
            .reset_index(drop=True)
        )
        per_method = (
            task_selected.groupby("method_base", as_index=False)
            .agg(
                n_heldout_datasets=("heldout_dataset", "nunique"),
                n_unique_selected_configs=("method_id", "nunique"),
            )
            .sort_values("method_base")
            .reset_index(drop=True)
        )

        top_choice = stability.groupby("method_base", as_index=False).first()
        top_choice["global_method_id"] = top_choice["method_base"].map(global_best)
        global_counts = (
            task_selected.assign(global_method_id=task_selected["method_base"].map(global_best))
            .query("method_id == global_method_id")
            .groupby("method_base", as_index=False)["heldout_dataset"]
            .nunique()
            .rename(columns={"heldout_dataset": "global_selected_count"})
        )
        top_choice = top_choice.merge(global_counts, on="method_base", how="left")
        top_choice["global_selected_count"] = (
            top_choice["global_selected_count"].fillna(0).astype(int)
        )
        top_choice = top_choice.rename(
            columns={
                "method_id": "most_frequent_method_id",
                "n_selected": "most_frequent_selection_count",
                "mean_train_score": "most_frequent_train_score",
            }
        )
        stability_summary = per_method.merge(top_choice, on="method_base", how="left")
        stability_summary.insert(0, "task", task)
        stability_rows.extend(stability_summary.to_dict(orient="records"))

    return pd.DataFrame(selected_rows), pd.DataFrame(aggregate_rows), pd.DataFrame(stability_rows)


def main() -> None:
    """Build and save leave-one-dataset-out sensitivity tables."""
    parser = argparse.ArgumentParser(
        description="Build leave-one-dataset-out config-selection sensitivity tables"
    )
    parser.add_argument("--selected-output", type=Path, default=SELECTED_OUT_PATH)
    parser.add_argument("--aggregate-output", type=Path, default=AGGREGATE_OUT_PATH)
    parser.add_argument("--stability-output", type=Path, default=STABILITY_OUT_PATH)
    args = parser.parse_args()

    selected, aggregate, stability = build_lodo_tables()
    args.selected_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.selected_output, index=False)
    aggregate.to_csv(args.aggregate_output, index=False)
    stability.to_csv(args.stability_output, index=False)


if __name__ == "__main__":
    main()
