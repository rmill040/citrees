"""Rank the max-type CIT/CIF configurations among the benchmark methods.

Inputs are the canonical real-data evaluation surface (Bonferroni Stage B for
every method) and the max-type extension campaign's downstream metrics for
``cit_maxt`` and ``cif_maxt``. The extension rows join the surface under the
same contract as every other method: one best configuration per method family
selected on the standard k grid, dataset-level means over folds and seeds,
complete-case ranks, and mean rank across datasets. A second table pairs each
max-type family with its Bonferroni counterpart on the cells both completed.

Usage:
    uv run python paper/analysis/build_maxt_extension_tables.py \
        --extension-dir ../data/maxt-extension/aggregated
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import pandas as pd

from paper.analysis.benchmark_common import (
    STANDARD_K,
    TABLES_DIR,
    TASK_CONFIG,
    complete_case_scores,
    dataset_scores,
    load_real_task_frame,
    select_best_task_configs,
)
from paper.analysis.build_benchmark_package_tables import DISPLAY_NAMES, _build_method_aggregate

EXTENSION_FILES: Final[dict[str, str]] = {
    "classification": "clf_evaluation.parquet",
    "regression": "reg_evaluation.parquet",
}
PAIRS: Final[tuple[tuple[str, str], ...]] = (("cit", "cit_maxt"), ("cif", "cif_maxt"))
EXTENSION_DISPLAY: Final[dict[str, str]] = {
    **DISPLAY_NAMES,
    "cit_maxt": "CIT (max-type)",
    "cif_maxt": "CIF (max-type)",
}


def load_extension_frame(extension_dir: Path, task: str, columns: list[str]) -> pd.DataFrame:
    """Load the real-data extension rows for one task with the surface's columns."""
    frame = pd.read_parquet(extension_dir / EXTENSION_FILES[task])
    frame = frame[frame["dataset_source"] == "real"].copy()
    if "task" not in frame.columns:
        frame["task"] = task
    metric = TASK_CONFIG[task]["metric"]
    if metric not in frame.columns:
        raise ValueError(f"extension rows lack the {task} metric column {metric!r}")
    # The surface carries both tasks' metric columns; the other task's are absent here.
    return frame.reindex(columns=columns)


def extended_method_aggregate(raw: pd.DataFrame, task: str, metric: str) -> pd.DataFrame:
    """Rank every method family, including the max-type aliases, on complete cases."""
    standard = raw[raw["k"].isin(STANDARD_K)].copy()
    filtered, _ = select_best_task_configs(standard, metric)
    scores = complete_case_scores(dataset_scores(filtered, metric))
    aggregate = _build_method_aggregate(scores, task, metric)
    aggregate.insert(3, "display_name", aggregate["method_base"].map(EXTENSION_DISPLAY))
    return aggregate


def pairwise_comparison(raw: pd.DataFrame, task: str, metric: str) -> pd.DataFrame:
    """Compare each max-type family with its Bonferroni counterpart on shared cells."""
    standard = raw[raw["k"].isin(STANDARD_K)].copy()
    filtered, best = select_best_task_configs(standard, metric)
    scores = dataset_scores(filtered, metric)
    rows: list[dict[str, object]] = []
    for base_name, alias in PAIRS:
        base_scores = scores[scores["method_base"] == base_name]
        alias_scores = scores[scores["method_base"] == alias]
        merged = base_scores.merge(
            alias_scores,
            on=["downstream_model", "dataset", "k"],
            suffixes=("_base", "_alias"),
        )
        if merged.empty:
            continue
        merged["delta"] = merged["dataset_mean_score_alias"] - merged["dataset_mean_score_base"]
        by_dataset = merged.groupby("dataset", as_index=False).agg(
            n_cells=("delta", "size"),
            mean_delta=("delta", "mean"),
        )
        rows.append(
            {
                "task": task,
                "metric": metric,
                "bonferroni_method": base_name,
                "bonferroni_method_id": best.loc[
                    best["method_base"] == base_name, "method_id"
                ].item(),
                "maxt_method": alias,
                "maxt_method_id": best.loc[best["method_base"] == alias, "method_id"].item(),
                "n_datasets": int(by_dataset["dataset"].nunique()),
                "n_cells": int(len(merged)),
                "mean_delta": float(by_dataset["mean_delta"].mean()),
                "median_delta": float(by_dataset["mean_delta"].median()),
                "share_datasets_maxt_higher": float((by_dataset["mean_delta"] > 0).mean()),
                "share_datasets_maxt_lower": float((by_dataset["mean_delta"] < 0).mean()),
                "mean_score_bonferroni": float(merged["dataset_mean_score_base"].mean()),
                "mean_score_maxt": float(merged["dataset_mean_score_alias"].mean()),
            }
        )
    return pd.DataFrame(rows)


def swap_method_aggregate(raw: pd.DataFrame, task: str, metric: str) -> pd.DataFrame:
    """Rank the original method set with each max-type family in place of its counterpart.

    This is the view a reader compares with the main ranking table: the same
    17 or 18 families, with the Bonferroni CIT and CIF replaced by their
    max-type versions, so positions are directly comparable.
    """
    swapped = raw[~raw["method_base"].isin([base for base, _ in PAIRS])].copy()
    aggregate = extended_method_aggregate(swapped, task, metric)
    aggregate["support_type"] = "maxt_replaces_bonferroni_counterparts"
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Max-type extension ranking tables")
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aggregates: list[pd.DataFrame] = []
    pairs: list[pd.DataFrame] = []
    swaps: list[pd.DataFrame] = []
    for task, config in TASK_CONFIG.items():
        base = load_real_task_frame(task=task)
        extension = load_extension_frame(args.extension_dir, task, list(base.columns))
        raw = pd.concat([base, extension], ignore_index=True)
        metric = config["metric"]
        aggregates.append(extended_method_aggregate(raw, task, metric))
        pairs.append(pairwise_comparison(raw, task, metric))
        swaps.append(swap_method_aggregate(raw, task, metric))
    aggregate = pd.concat(aggregates, ignore_index=True)
    pairwise = pd.concat(pairs, ignore_index=True)
    aggregate.to_csv(args.output_dir / "paper_maxt_extension_method_aggregate.csv", index=False)
    pairwise.to_csv(args.output_dir / "paper_maxt_extension_pairwise.csv", index=False)
    swap = pd.concat(swaps, ignore_index=True)
    swap.to_csv(args.output_dir / "paper_maxt_extension_swap_aggregate.csv", index=False)
    print(
        swap[swap["method_base"].isin(["cit_maxt", "cif_maxt"])][
            ["task", "method_base", "n_datasets", "mean_rank", "rank_position"]
        ].to_string(index=False)
    )
    print(
        aggregate[["task", "method_base", "n_datasets", "mean_rank", "rank_position"]].to_string(
            index=False
        )
    )
    print(pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
