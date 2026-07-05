"""Shared helpers for paper-facing benchmark analyses.

These utilities define the current benchmark contract used by the cleaned
analysis layer:

1. real datasets only,
2. one best global config per method family within task,
3. explicit standard values of k for cross-dataset comparisons, and
4. complete-case ranks for multi-method summaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, TypedDict

import pandas as pd


class BenchmarkTaskConfig(TypedDict):
    """Typed configuration for one benchmark task."""

    metric: str
    focus_method: str


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TABLES_DIR = RESULTS_DIR / "tables"
STANDARD_K: Final[tuple[int, ...]] = (5, 10, 25, 50, 100)
REAL_EVALUATION_SURFACE: Final[Path] = RESULTS_DIR / "paper_real_evaluation.parquet"

TASK_CONFIG: Final[dict[str, BenchmarkTaskConfig]] = {
    "classification": {
        "metric": "balanced_accuracy",
        "focus_method": "cif",
    },
    "regression": {
        "metric": "r2",
        "focus_method": "cif",
    },
}


def load_real_task_frame(
    *, task: str | None = None, path: Path = REAL_EVALUATION_SURFACE
) -> pd.DataFrame:
    """Load the canonical joined real-data evaluation surface."""
    df = pd.read_parquet(path)
    df = df[df["dataset_source"] == "real"].copy()
    if task is not None and "task" in df.columns:
        df = df[df["task"] == task].copy()
    return df


def task_global_config_scores(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Score each config by the paper-facing task-global cell-mean contract."""
    cell_scores = (
        df.groupby(
            ["method_base", "method_id", "dataset", "downstream_model", "k"], as_index=False
        )[metric]
        .mean()
        .rename(columns={metric: "dataset_mean_score"})
    )
    return (
        cell_scores.groupby(["method_base", "method_id"], as_index=False)["dataset_mean_score"]
        .mean()
        .rename(columns={"dataset_mean_score": "task_global_mean_metric"})
    )


def select_best_task_configs(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one best global config per method family for a task.

    The selection contract is a mean over `(dataset, downstream_model, k)`
    cells after folds and seeds have already been collapsed within each cell.
    This matches the paper-facing benchmark description and avoids silently
    reweighting configs when raw-row coverage differs across cells.
    """
    perf = task_global_config_scores(df, metric)
    best_idx = perf.groupby("method_base")["task_global_mean_metric"].idxmax()
    best = perf.loc[best_idx].sort_values("method_base").reset_index(drop=True)
    filtered = df.merge(
        best[["method_base", "method_id"]], on=["method_base", "method_id"], how="inner"
    )
    return filtered, best


def dataset_scores(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average seeds and folds into one score per dataset, method, and k cell."""
    return (
        df.groupby(
            ["downstream_model", "dataset", "k", "method_base", "method_id"], as_index=False
        )[metric]
        .mean()
        .rename(columns={metric: "dataset_mean_score"})
    )


def complete_case_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Keep dataset cells where every task method family is present."""
    n_methods = df["method_base"].nunique()
    counts = (
        df.groupby(["downstream_model", "k", "dataset"])["method_base"]
        .nunique()
        .reset_index(name="n_methods_present")
    )
    complete = counts[counts["n_methods_present"] == n_methods][
        ["downstream_model", "k", "dataset"]
    ]
    return df.merge(complete, on=["downstream_model", "k", "dataset"], how="inner")


def rank_complete_case_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add within-dataset Friedman-style ranks to complete-case scores."""
    ranked = df.copy()
    ranked["rank"] = ranked.groupby(["downstream_model", "k", "dataset"])[
        "dataset_mean_score"
    ].rank(
        ascending=False,
        method="average",
    )
    return ranked
