"""Build paper-facing summary tables for the completed threshold-search ablation.

This script summarizes the targeted threshold-search study relative to the
executed practical default, `histogram_256`. The resulting tables are meant to
support knob-level statements without relaxing the main benchmark contract.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.scripts.analysis.benchmark_common import TABLES_DIR  # noqa: E402

THRESHOLD_ABLATION_PATH: Final[Path] = TABLES_DIR / "threshold_search_ablation.csv"
DEFAULT_VARIANT: Final[str] = "histogram_256"
REAL_SCORE_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "clf": ("lr_ba_mean", "svm_ba_mean", "knn_ba_mean"),
    "reg": ("ridge_r2_mean", "svr_r2_mean", "knn_r2_mean"),
}
TASK_LABELS: Final[dict[str, str]] = {
    "clf": "classification",
    "reg": "regression",
}
STANDARD_PRECISION_COLUMNS: Final[tuple[str, ...]] = (
    "precision_at_5_mean",
    "precision_at_10_mean",
    "precision_at_25_mean",
    "precision_at_50_mean",
    "precision_at_100_mean",
)
SYNTHETIC_SCORE_COLUMNS: Final[tuple[str, ...]] = (
    "precision_at_10_mean",
    "f1_at_10_mean",
    "spread_at_10_mean",
)


def dataset_group_label(dataset_type: str) -> str:
    """Collapse threshold-study dataset types into real versus synthetic groups."""
    return "real" if dataset_type.startswith("real_") else "synthetic"


def mean_available_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Average only the score columns present in the current frame."""
    available = [column for column in columns if column in df.columns]
    return df[available].mean(axis=1, skipna=True)


def build_threshold_ablation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate threshold-search variants by task and dataset group."""
    frames: list[pd.DataFrame] = []

    for task in sorted(df["task"].unique()):
        task_frame = df[df["task"] == task].copy()
        task_frame["dataset_group"] = task_frame["dataset_type"].map(dataset_group_label)
        task_frame["real_downstream_mean"] = mean_available_columns(
            task_frame, REAL_SCORE_COLUMNS[task]
        )
        task_frame["synthetic_quality_mean"] = mean_available_columns(
            task_frame, SYNTHETIC_SCORE_COLUMNS
        )
        task_frame["precision_curve_mean"] = mean_available_columns(
            task_frame, STANDARD_PRECISION_COLUMNS
        )

        summary = (
            task_frame.groupby(["task", "dataset_group", "variant"], as_index=False)
            .agg(
                n_dataset_types=("dataset_type", "nunique"),
                mean_elapsed_seconds=("elapsed_seconds_mean", "mean"),
                mean_depth=("mean_depth_mean", "mean"),
                mean_max_depth=("max_depth_mean", "mean"),
                mean_features_used=("mean_features_used_mean", "mean"),
                mean_real_downstream_score=("real_downstream_mean", "mean"),
                mean_synthetic_quality=("synthetic_quality_mean", "mean"),
                mean_precision_over_standard_k=("precision_curve_mean", "mean"),
                mean_precision_at_10=("precision_at_10_mean", "mean"),
                mean_confounder_rate_at_10=("confounder_rate_at_10_mean", "mean"),
            )
            .sort_values(["dataset_group", "variant"])
            .reset_index(drop=True)
        )
        summary["task"] = summary["task"].map(TASK_LABELS)

        defaults = summary[summary["variant"] == DEFAULT_VARIANT][
            [
                "task",
                "dataset_group",
                "mean_elapsed_seconds",
                "mean_depth",
                "mean_features_used",
                "mean_real_downstream_score",
                "mean_synthetic_quality",
                "mean_precision_over_standard_k",
                "mean_precision_at_10",
                "mean_confounder_rate_at_10",
            ]
        ].rename(
            columns={
                "mean_elapsed_seconds": "default_elapsed_seconds",
                "mean_depth": "default_depth",
                "mean_features_used": "default_features_used",
                "mean_real_downstream_score": "default_real_downstream_score",
                "mean_synthetic_quality": "default_synthetic_quality",
                "mean_precision_over_standard_k": "default_precision_over_standard_k",
                "mean_precision_at_10": "default_precision_at_10",
                "mean_confounder_rate_at_10": "default_confounder_rate_at_10",
            }
        )

        summary = summary.merge(defaults, on=["task", "dataset_group"], how="left")
        summary["elapsed_seconds_ratio_vs_default"] = (
            summary["mean_elapsed_seconds"] / summary["default_elapsed_seconds"]
        )
        summary["delta_depth_vs_default"] = summary["mean_depth"] - summary["default_depth"]
        summary["delta_features_used_vs_default"] = (
            summary["mean_features_used"] - summary["default_features_used"]
        )
        summary["delta_real_downstream_vs_default"] = (
            summary["mean_real_downstream_score"] - summary["default_real_downstream_score"]
        )
        summary["delta_synthetic_quality_vs_default"] = (
            summary["mean_synthetic_quality"] - summary["default_synthetic_quality"]
        )
        summary["delta_precision_over_standard_k_vs_default"] = (
            summary["mean_precision_over_standard_k"] - summary["default_precision_over_standard_k"]
        )
        summary["delta_precision_at_10_vs_default"] = (
            summary["mean_precision_at_10"] - summary["default_precision_at_10"]
        )
        summary["delta_confounder_rate_at_10_vs_default"] = (
            summary["mean_confounder_rate_at_10"] - summary["default_confounder_rate_at_10"]
        )
        frames.append(summary)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    """Build and save the paper-facing threshold-ablation summary table."""
    parser = argparse.ArgumentParser(description="Build threshold-search ablation summary tables")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TABLES_DIR,
        help="Directory for summary outputs",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = pd.read_csv(THRESHOLD_ABLATION_PATH)
    summary = build_threshold_ablation_summary(threshold)
    summary.to_csv(output_dir / "paper_threshold_ablation_summary.csv", index=False)


if __name__ == "__main__":
    main()
