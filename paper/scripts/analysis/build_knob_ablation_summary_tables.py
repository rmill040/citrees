"""Build paper-facing summary tables for the mirrored CIF knob ablation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Final

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.scripts.analysis.benchmark_common import TABLES_DIR

ABLATION_PATH: Final[Path] = TABLES_DIR / "mirrored_knob_ablation.csv"
ABLATION_DOWNSTREAM_PREFIXES: Final[dict[str, tuple[str, ...]]] = {
    "clf": ("lr", "svm", "knn"),
    "reg": ("ridge", "svr", "knn"),
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


def dataset_group_label(dataset_type: str) -> str:
    """Collapse dataset types into real versus synthetic groups."""
    return "real" if dataset_type.startswith("real_") else "synthetic"


def select_downstream_columns(df: pd.DataFrame, task: str) -> list[str]:
    """Select task-specific downstream score columns from the ablation file."""
    prefixes = ABLATION_DOWNSTREAM_PREFIXES[task]
    return [
        column
        for column in df.columns
        if column.startswith("ds_")
        and column.endswith("_mean")
        and any(f"_{prefix}_" in column for prefix in prefixes)
    ]


def mean_available_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Average only the columns present in the current ablation frame."""
    available = [column for column in columns if column in df.columns]
    return df[available].mean(axis=1, skipna=True)


def build_knob_ablation_summary(ablation: pd.DataFrame) -> pd.DataFrame:
    """Summarize runtime and quality deltas for each mirrored knob variant."""
    frames: list[pd.DataFrame] = []

    for task in sorted(ablation["task"].unique()):
        task_frame = ablation[ablation["task"] == task].copy()
        task_frame["dataset_group"] = task_frame["dataset_type"].map(dataset_group_label)
        downstream_cols = select_downstream_columns(task_frame, task)
        task_frame["downstream_mean"] = task_frame[downstream_cols].mean(axis=1, skipna=True)
        task_frame["precision_curve_mean"] = mean_available_columns(task_frame, STANDARD_PRECISION_COLUMNS)

        summary = (
            task_frame.groupby(["task", "dataset_group", "variant"], as_index=False)
            .agg(
                n_dataset_types=("dataset_type", "nunique"),
                mean_elapsed_seconds=("elapsed_seconds_mean", "mean"),
                mean_downstream_score=("downstream_mean", "mean"),
                mean_precision_over_standard_k=("precision_curve_mean", "mean"),
                mean_precision_at_10=("precision_at_10_mean", "mean"),
                mean_max_depth=("max_depth_mean", "mean"),
                mean_features_used=("mean_features_used_mean", "mean"),
                mean_estimators_actual=("n_estimators_actual_mean", "mean"),
            )
            .sort_values(["dataset_group", "variant"])
            .reset_index(drop=True)
        )
        summary["task"] = summary["task"].map(TASK_LABELS)

        defaults = summary[summary["variant"] == "cif_default"][
            [
                "task",
                "dataset_group",
                "mean_elapsed_seconds",
                "mean_downstream_score",
                "mean_precision_over_standard_k",
                "mean_precision_at_10",
                "mean_max_depth",
                "mean_features_used",
            ]
        ].rename(
            columns={
                "mean_elapsed_seconds": "default_elapsed_seconds",
                "mean_downstream_score": "default_downstream_score",
                "mean_precision_over_standard_k": "default_precision_over_standard_k",
                "mean_precision_at_10": "default_precision_at_10",
                "mean_max_depth": "default_max_depth",
                "mean_features_used": "default_features_used",
            }
        )

        summary = summary.merge(defaults, on=["task", "dataset_group"], how="left")
        summary["delta_elapsed_seconds"] = summary["mean_elapsed_seconds"] - summary["default_elapsed_seconds"]
        summary["elapsed_seconds_ratio"] = summary["mean_elapsed_seconds"] / summary["default_elapsed_seconds"]
        summary["delta_downstream_score"] = summary["mean_downstream_score"] - summary["default_downstream_score"]
        summary["delta_precision_over_standard_k"] = (
            summary["mean_precision_over_standard_k"] - summary["default_precision_over_standard_k"]
        )
        summary["delta_precision_at_10"] = summary["mean_precision_at_10"] - summary["default_precision_at_10"]
        summary["delta_max_depth"] = summary["mean_max_depth"] - summary["default_max_depth"]
        summary["delta_features_used"] = summary["mean_features_used"] - summary["default_features_used"]
        frames.append(summary)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    """Build and save the mirrored-knob ablation summary table."""
    parser = argparse.ArgumentParser(description="Build mirrored knob-ablation summary tables")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TABLES_DIR,
        help="Directory for summary outputs",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation = pd.read_csv(ABLATION_PATH)
    summary = build_knob_ablation_summary(ablation)
    summary.to_csv(output_dir / "paper_mirrored_knob_ablation_summary.csv", index=False)


if __name__ == "__main__":
    main()
