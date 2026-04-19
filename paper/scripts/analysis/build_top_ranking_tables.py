"""Build top-ranking summary tables for synthetic feature-recovery experiments.

This script answers the question:
  are the very top-ranked features actually ground-truth signal?

It works from the canonical synthetic rankings parquets and synthetic dataset
metadata with known informative feature indices.

Config contract:
  - one best synthetic config per method family within task
  - selected by mean informative recovery over the standard synthetic budget
    curve `k in {5, 10, 25, 50, 100}` with effective `k=min(k,p)`

Outputs:
  - paper/results/tables/top_ranking_best_configs.csv
  - paper/results/tables/top_ranking_best_config_details.csv
  - paper/results/tables/top_ranking_curve_summary.csv
  - paper/results/tables/top_ranking_summary.csv
  - paper/results/tables/top_ranking_by_dataset_type.csv
  - paper/results/tables/top_ranking_by_dataset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, TypedDict

import pandas as pd

from paper.scripts.analysis.analyze_synthetic_ground_truth import (
    dataset_type_from_config,
    load_synthetic_metadata,
)
from paper.scripts.analysis.config_resolution import resolve_method_config_details
from paper.scripts.utils.metrics import f1_at_k, precision_at_k, recall_at_k


class TopRankingTaskConfig(TypedDict):
    """Typed configuration for one synthetic-ranking task."""

    rankings_path: Path
    data_dir: Path


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TABLES_DIR = RESULTS_DIR / "tables"

TASK_CONFIG: Final[dict[str, TopRankingTaskConfig]] = {
    "classification": {
        "rankings_path": RESULTS_DIR / "clf_rankings.parquet",
        "data_dir": Path("paper/data/classification/synthetic"),
    },
    "regression": {
        "rankings_path": RESULTS_DIR / "reg_rankings.parquet",
        "data_dir": Path("paper/data/regression/synthetic"),
    },
}

TOP_K: Final[tuple[int, ...]] = (1, 2, 5, 10, 25, 50, 100)
SELECTION_K: Final[tuple[int, ...]] = (5, 10, 25, 50, 100)
HEAD_K: Final[tuple[int, ...]] = (1, 2)


def _first_true_rank(ranking: list[int], true_indices: list[int]) -> int:
    """1-based rank of the first informative feature, or len(ranking)+1 if none found."""
    true_set = set(true_indices)
    for idx, feat in enumerate(ranking, start=1):
        if feat in true_set:
            return idx
    return len(ranking) + 1


def _build_row_metrics(
    ranking: list[int],
    informative: list[int],
    informative_plus_redundant: list[int],
    n_features: int,
) -> dict[str, float | int]:
    """Compute top-of-ranking diagnostics for one ranking."""
    informative_set = set(informative)
    first_true_rank = _first_true_rank(ranking, informative)
    first_ir_rank = _first_true_rank(ranking, informative_plus_redundant)

    row: dict[str, float | int] = {
        "first_true_rank": int(first_true_rank),
        "first_ir_rank": int(first_ir_rank),
        "mrr_true": 0.0 if first_true_rank > len(ranking) else 1.0 / first_true_rank,
        "mrr_ir": 0.0 if first_ir_rank > len(ranking) else 1.0 / first_ir_rank,
        "top1_hit": int(first_true_rank == 1),
        "top1_ir_hit": int(first_ir_rank == 1),
    }

    for k in TOP_K:
        effective_k = min(k, n_features)
        row[f"precision@{k}"] = precision_at_k(ranking, informative, effective_k)
        row[f"recall@{k}"] = recall_at_k(ranking, informative, effective_k)
        row[f"f1@{k}"] = f1_at_k(ranking, informative, effective_k)
        row[f"precision_ir@{k}"] = precision_at_k(ranking, informative_plus_redundant, effective_k)
        row[f"recall_ir@{k}"] = recall_at_k(ranking, informative_plus_redundant, effective_k)
        row[f"f1_ir@{k}"] = f1_at_k(ranking, informative_plus_redundant, effective_k)
        row[f"any_hit@{k}"] = int(any(feat in informative_set for feat in ranking[:effective_k]))
        row[f"all_hit@{k}"] = int(all(feat in informative_set for feat in ranking[:effective_k]))

    return row


def _compute_synthetic_diagnostics(
    task: str,
    rankings_path: Path,
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return row-level diagnostics and best-config table for one task."""
    rankings = pd.read_parquet(rankings_path)
    rankings = rankings[rankings["dataset_source"] == "synthetic"].copy()

    metadata = load_synthetic_metadata(data_dir)
    rows: list[dict[str, object]] = []

    for _, record in rankings.iterrows():
        dataset = str(record["dataset"])
        if dataset not in metadata:
            continue

        meta = metadata[dataset]
        informative = list(meta["informative_indices"])
        redundant = list(meta.get("redundant_indices", []))
        informative_plus_redundant = sorted(set(informative) | set(redundant))
        ranking = list(record["feature_ranking"])

        row = {
            "task": task,
            "dataset": dataset,
            "dataset_type": dataset_type_from_config(meta["config"]),
            "method_base": str(record["method_base"]),
            "method_id": str(record["method_id"]),
            "seed": int(record["seed"]),
            "fold_idx": int(record["fold_idx"]),
            "n_informative": len(informative),
            "n_features": int(record["n_features"]),
        }
        row.update(
            _build_row_metrics(
                ranking,
                informative,
                informative_plus_redundant,
                int(meta["n_features_final"]),
            )
        )
        rows.append(row)

    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty:
        return diagnostics, pd.DataFrame()

    selection_cols = [f"precision@{k}" for k in SELECTION_K]
    diagnostics["selection_curve_score"] = diagnostics[selection_cols].mean(axis=1)
    perf = (
        diagnostics.groupby(["task", "method_base", "method_id"], as_index=False)["selection_curve_score"]
        .mean()
        .rename(columns={"selection_curve_score": "selection_score"})
    )
    best_idx = perf.groupby(["task", "method_base"])["selection_score"].idxmax()
    best = perf.loc[best_idx].copy()
    best["config_selection_metric"] = "mean_precision_over_k_5_10_25_50_100"

    diagnostics = diagnostics.merge(
        best[["task", "method_base", "method_id"]],
        on=["task", "method_base", "method_id"],
        how="inner",
    )

    return diagnostics, best


def _summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Method-level summary across all synthetic datasets."""
    metrics = [
        "top1_hit",
        "top1_ir_hit",
        "any_hit@2",
        "precision@1",
        "precision@2",
        "precision@5",
        "precision@10",
        "recall@1",
        "recall@2",
        "f1@1",
        "f1@2",
        "mrr_true",
        "mrr_ir",
        "first_true_rank",
        "first_ir_rank",
    ]

    summary = (
        df.groupby(["task", "method_base", "method_id"], as_index=False)[metrics]
        .mean()
        .rename(
            columns={
                "top1_hit": "top1_hit_rate",
                "top1_ir_hit": "top1_ir_hit_rate",
                "any_hit@2": "any_hit_at_2_rate",
                "mrr_true": "mean_reciprocal_rank_true",
                "mrr_ir": "mean_reciprocal_rank_ir",
                "first_true_rank": "mean_first_true_rank",
                "first_ir_rank": "mean_first_ir_rank",
            }
        )
    )
    summary["top1_rank_position"] = summary.groupby("task")["top1_hit_rate"].rank(
        ascending=False,
        method="average",
    )
    summary["mrr_rank_position"] = summary.groupby("task")["mean_reciprocal_rank_true"].rank(
        ascending=False,
        method="average",
    )
    return summary.sort_values(["task", "top1_rank_position", "mrr_rank_position", "method_base"]).reset_index(
        drop=True
    )


def _summarize_curve_over_k(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize head-of-ranking quality over the small-k synthetic curve."""
    curve = df.copy()
    precision_cols = [f"precision@{k}" for k in HEAD_K]
    recall_cols = [f"recall@{k}" for k in HEAD_K]
    f1_cols = [f"f1@{k}" for k in HEAD_K]

    curve["mean_precision_over_head_k_1_2"] = curve[precision_cols].mean(axis=1)
    curve["mean_recall_over_head_k_1_2"] = curve[recall_cols].mean(axis=1)
    curve["mean_f1_over_head_k_1_2"] = curve[f1_cols].mean(axis=1)

    summary = (
        curve.groupby(["task", "method_base", "method_id"], as_index=False)
        .agg(
            n_rankings=("dataset", "size"),
            n_datasets=("dataset", "nunique"),
            mean_precision_over_head_k_1_2=("mean_precision_over_head_k_1_2", "mean"),
            mean_recall_over_head_k_1_2=("mean_recall_over_head_k_1_2", "mean"),
            mean_f1_over_head_k_1_2=("mean_f1_over_head_k_1_2", "mean"),
            top1_hit_rate=("top1_hit", "mean"),
            any_hit_at_2_rate=("any_hit@2", "mean"),
            mean_reciprocal_rank_true=("mrr_true", "mean"),
            mean_first_true_rank=("first_true_rank", "mean"),
        )
        .sort_values(["task", "mean_precision_over_head_k_1_2"], ascending=[True, False])
        .reset_index(drop=True)
    )

    summary["curve_rank_position"] = summary.groupby("task")["mean_precision_over_head_k_1_2"].rank(
        ascending=False,
        method="average",
    )
    summary["top1_rank_position"] = summary.groupby("task")["top1_hit_rate"].rank(
        ascending=False,
        method="average",
    )
    summary["mrr_rank_position"] = summary.groupby("task")["mean_reciprocal_rank_true"].rank(
        ascending=False,
        method="average",
    )
    return summary.sort_values(
        ["task", "curve_rank_position", "top1_rank_position", "mrr_rank_position", "method_base"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)


def _summarize_by_dataset_type(df: pd.DataFrame) -> pd.DataFrame:
    """Method-level summary within each synthetic dataset type."""
    metrics = [
        "top1_hit",
        "any_hit@2",
        "precision@1",
        "precision@2",
        "precision@5",
        "precision@10",
        "mrr_true",
        "first_true_rank",
    ]
    summary = (
        df.groupby(["task", "dataset_type", "method_base", "method_id"], as_index=False)[metrics]
        .mean()
        .rename(
            columns={
                "top1_hit": "top1_hit_rate",
                "any_hit@2": "any_hit_at_2_rate",
                "mrr_true": "mean_reciprocal_rank_true",
                "first_true_rank": "mean_first_true_rank",
            }
        )
    )
    return summary.sort_values(
        ["task", "dataset_type", "top1_hit_rate", "mean_reciprocal_rank_true"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def _summarize_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Method-level summary within each synthetic dataset."""
    metrics = [
        "top1_hit",
        "any_hit@2",
        "precision@1",
        "precision@2",
        "precision@5",
        "precision@10",
        "mrr_true",
        "first_true_rank",
    ]
    summary = (
        df.groupby(["task", "dataset", "dataset_type", "method_base", "method_id"], as_index=False)[metrics]
        .mean()
        .rename(
            columns={
                "top1_hit": "top1_hit_rate",
                "any_hit@2": "any_hit_at_2_rate",
                "mrr_true": "mean_reciprocal_rank_true",
                "first_true_rank": "mean_first_true_rank",
            }
        )
    )
    return summary.sort_values(
        ["task", "dataset", "top1_hit_rate", "mean_reciprocal_rank_true"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def main() -> None:
    """Build and save synthetic top-ranking diagnostic tables."""
    parser = argparse.ArgumentParser(description="Build top-ranking diagnostics for synthetic rankings")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TABLES_DIR,
        help="Directory for top-ranking diagnostic tables",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_frames: list[pd.DataFrame] = []
    best_frames: list[pd.DataFrame] = []

    for task, config in TASK_CONFIG.items():
        diagnostics, best = _compute_synthetic_diagnostics(
            task=task,
            rankings_path=config["rankings_path"],
            data_dir=config["data_dir"],
        )
        if diagnostics.empty:
            continue
        diagnostics_frames.append(diagnostics)
        best_frames.append(best)

    diagnostics_all = pd.concat(diagnostics_frames, ignore_index=True)
    best_all = pd.concat(best_frames, ignore_index=True)
    best_details = resolve_method_config_details(best_all)

    outputs = {
        "top_ranking_best_configs.csv": best_all,
        "top_ranking_best_config_details.csv": best_details,
        "top_ranking_curve_summary.csv": _summarize_curve_over_k(diagnostics_all),
        "top_ranking_summary.csv": _summarize_overall(diagnostics_all),
        "top_ranking_by_dataset_type.csv": _summarize_by_dataset_type(diagnostics_all),
        "top_ranking_by_dataset.csv": _summarize_by_dataset(diagnostics_all),
    }

    for filename, frame in outputs.items():
        out_path = output_dir / filename
        frame.to_csv(out_path, index=False)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
