#!/usr/bin/env python3
"""Rank the random forest with out-of-bag permutation importance among the benchmark methods.

Inputs are the canonical real-data evaluation surface and the per-dataset
parquet files of ``paper/benchmark/experiments/rf_oobperm_benchmark.py``
(``method_base = "rf_oobperm"``). The new rows join the surface under the same
contract as every other method: one configuration per family, dataset-level
means over folds and seeds, complete-case ranks, and mean rank across datasets.
Two tables come out: the extended method aggregate (18 classification and 19
regression families) and pairwise comparisons of the new ranker with CIF and
with the impurity-ranked random forest on the datasets both completed.

Usage:
    uv run python paper/analysis/build_rf_oobperm_tables.py \
        --extension-dir ../data/review5/rfoob
"""

from __future__ import annotations

import argparse
import glob
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

NEW: Final[str] = "rf_oobperm"
DISPLAY: Final[dict[str, str]] = {**DISPLAY_NAMES, NEW: "RF (OOB permutation)"}
COMPARATORS: Final[tuple[str, ...]] = ("cif", "rf", "r_cforest")


def load_extension_frame(extension_dir: Path, task: str, columns: list[str]) -> pd.DataFrame:
    files = sorted(glob.glob(str(extension_dir / "**" / f"{task}__*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no {task} rf_oobperm parquet files under {extension_dir}")
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    frame = frame.drop_duplicates(subset=["dataset", "seed", "fold_idx", "downstream_model", "k"])
    metric = TASK_CONFIG[task]["metric"]
    if metric not in frame.columns:
        raise ValueError(f"extension rows lack the {task} metric column {metric!r}")
    return frame.reindex(columns=columns)


def extended_aggregate(raw: pd.DataFrame, task: str, metric: str) -> pd.DataFrame:
    standard = raw[raw["k"].isin(STANDARD_K)].copy()
    filtered, _ = select_best_task_configs(standard, metric)
    scores = complete_case_scores(dataset_scores(filtered, metric))
    aggregate = _build_method_aggregate(scores, task, metric)
    aggregate.insert(3, "display_name", aggregate["method_base"].map(DISPLAY))
    return aggregate


def pairwise(raw: pd.DataFrame, task: str, metric: str) -> pd.DataFrame:
    """Dataset-mean differences, new ranker minus comparator, on shared datasets."""
    standard = raw[raw["k"].isin(STANDARD_K)].copy()
    filtered, _ = select_best_task_configs(standard, metric)
    per_dataset = dataset_scores(filtered, metric)
    new = per_dataset[per_dataset["method_base"] == NEW].set_index("dataset")["mean_score"]
    rows = []
    for other in COMPARATORS:
        comp = per_dataset[per_dataset["method_base"] == other].set_index("dataset")["mean_score"]
        shared = new.index.intersection(comp.index)
        delta = (new.loc[shared] - comp.loc[shared]).astype(float)
        rows.append(
            {
                "task": task,
                "metric": metric,
                "new_method": NEW,
                "comparator": other,
                "n_datasets": int(len(shared)),
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "wins": int((delta > 0).sum()),
                "losses": int((delta < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="RF out-of-bag permutation comparator tables")
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregates, pairs = [], []
    for task, config in TASK_CONFIG.items():
        base = load_real_task_frame(task=task)
        extension = load_extension_frame(args.extension_dir, task, list(base.columns))
        raw = pd.concat([base, extension], ignore_index=True)
        metric = config["metric"]
        aggregates.append(extended_aggregate(raw, task, metric))
        pairs.append(pairwise(raw, task, metric))
    aggregate = pd.concat(aggregates, ignore_index=True)
    pairwise_table = pd.concat(pairs, ignore_index=True)
    aggregate.to_csv(args.output_dir / "paper_rf_oobperm_method_aggregate.csv", index=False)
    pairwise_table.to_csv(args.output_dir / "paper_rf_oobperm_pairwise.csv", index=False)
    print(
        aggregate[["task", "method_base", "n_datasets", "mean_rank", "rank_position"]].to_string(
            index=False
        )
    )
    print(pairwise_table.to_string(index=False))


if __name__ == "__main__":
    main()
