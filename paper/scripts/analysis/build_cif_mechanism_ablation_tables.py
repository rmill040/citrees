#!/usr/bin/env python3
"""Build robust summary tables for the CIF component ablations.

The analysis includes two related comparisons:

* a readout ablation comparing ``split_importance`` with ``split_count`` on the
  same fitted selected-CIF forest;
* model-component ablations that disable bootstrap sampling, disable feature
  muting, or reduce CIF to one tree.

The analysis keeps the raw evidence paired at the strongest available level:
dataset x downstream learner x value of k, after averaging only repeated
fold/seed rows inside each cell. Paper-facing aggregate comparisons then average
paired cell deltas within each dataset before summarizing across datasets.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Any, Final

import boto3
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT: Final[Path] = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.scripts.config.constants import EVALUATION_K_VALUES  # noqa: E402

DEFAULT_INPUT_URI: Final[str] = str(ROOT / "scratch" / "cif_mechanism_ablation" / "metrics")
DEFAULT_OUTPUT_DIR: Final[Path] = ROOT / "paper" / "results" / "tables"
DEFAULT_REFERENCE_MODEL: Final[str] = "cif_default"
DEFAULT_REFERENCE_RANKING: Final[str] = "split_importance"
DEFAULT_EXPECTED_SEEDS: Final[int] = 5
DEFAULT_EXPECTED_FOLDS: Final[int] = 5
DEFAULT_N_BOOT: Final[int] = 20_000
BOOTSTRAP_SEED: Final[int] = 1718
SCORE_TOL: Final[float] = 1e-12

KEY_COLUMNS: Final[tuple[str, ...]] = (
    "task",
    "dataset",
    "model_variant",
    "ranking_variant",
    "downstream_model",
    "k",
)
PAIR_COLUMNS: Final[tuple[str, ...]] = ("task", "dataset", "downstream_model", "k")


def _split_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    bucket_key = uri[len("s3://") :]
    bucket, _, key = bucket_key.partition("/")
    if not bucket or not key:
        raise ValueError(f"Expected s3://bucket/key, got: {uri}")
    return bucket, key.rstrip("/")


def _s3_client() -> Any:
    return boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))


def _list_s3_parquets(uri: str) -> list[str]:
    bucket, prefix = _split_s3_uri(uri)
    client = _s3_client()
    paginator = client.get_paginator("list_objects_v2")
    out: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if key.endswith(".parquet"):
                out.append(f"s3://{bucket}/{key}")
    return out


def _read_parquet_uri(uri: str) -> pd.DataFrame:
    if uri.startswith("s3://"):
        bucket, key = _split_s3_uri(uri)
        body = _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
        return pd.read_parquet(io.BytesIO(body))
    return pd.read_parquet(uri)


def _list_metric_artifacts(input_uri: str) -> list[str]:
    if input_uri.startswith("s3://"):
        return _list_s3_parquets(input_uri)
    return [str(path) for path in sorted(Path(input_uri).rglob("*.parquet"))]


def load_metrics(input_uri: str) -> pd.DataFrame:
    paths = _list_metric_artifacts(input_uri)
    if not paths:
        raise FileNotFoundError(f"No metric parquet files found under {input_uri}")
    frames = [_read_parquet_uri(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def task_metric(task: str) -> str:
    if task == "classification":
        return "balanced_accuracy"
    if task == "regression":
        return "r2"
    raise ValueError(f"Unknown task: {task}")


def _validate_metric_columns(metrics: pd.DataFrame) -> None:
    required = set(KEY_COLUMNS) | {"seed", "fold_idx", "n_features"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"Metric table is missing required columns: {missing}")


def standard_metric_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    """Keep standard k values and add the task-specific score column."""
    _validate_metric_columns(metrics)
    df = metrics.copy()
    df["k"] = pd.to_numeric(df["k"], errors="raise").astype(int)
    df = df[df["k"].isin(EVALUATION_K_VALUES)].copy()
    if df.empty:
        raise ValueError("No rows remain after filtering to standard k values")

    df["score"] = np.nan
    for task in sorted(df["task"].dropna().unique()):
        metric = task_metric(str(task))
        if metric not in df.columns:
            raise ValueError(f"Metric table is missing {metric!r} for task={task!r}")
        mask = df["task"].eq(task)
        df.loc[mask, "score"] = pd.to_numeric(df.loc[mask, metric], errors="coerce")
    return df


def build_completeness_table(
    metrics: pd.DataFrame,
    *,
    expected_seeds: int,
    expected_folds: int,
) -> pd.DataFrame:
    """Audit fold/seed support and missing scores for every paired cell."""
    df = standard_metric_rows(metrics)
    expected_rows = int(expected_seeds * expected_folds)
    out = (
        df.groupby(list(KEY_COLUMNS), as_index=False)
        .agg(
            n_rows=("score", "size"),
            n_nonnull_score_rows=("score", "count"),
            n_seeds=("seed", "nunique"),
            n_folds=("fold_idx", "nunique"),
            min_seed=("seed", "min"),
            max_seed=("seed", "max"),
            min_fold_idx=("fold_idx", "min"),
            max_fold_idx=("fold_idx", "max"),
            n_features=("n_features", "first"),
        )
        .reset_index(drop=True)
    )
    out["expected_seed_count"] = int(expected_seeds)
    out["expected_fold_count"] = int(expected_folds)
    out["expected_fold_seed_rows"] = expected_rows
    out["score_missing_rows"] = out["n_rows"] - out["n_nonnull_score_rows"]
    out["fold_seed_complete"] = (
        out["n_rows"].eq(expected_rows)
        & out["n_seeds"].eq(expected_seeds)
        & out["n_folds"].eq(expected_folds)
    )
    out["scores_complete"] = out["score_missing_rows"].eq(0)
    out["is_complete"] = out["fold_seed_complete"] & out["scores_complete"]
    return out.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def build_cell_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average repeated folds/seeds inside each dataset x learner x k cell."""
    df = standard_metric_rows(metrics)
    out = (
        df.groupby(list(KEY_COLUMNS), as_index=False)
        .agg(
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            std_score=("score", "std"),
            n_fold_seed_rows=("score", "size"),
            n_nonnull_score_rows=("score", "count"),
            n_seeds=("seed", "nunique"),
            n_folds=("fold_idx", "nunique"),
            n_features=("n_features", "first"),
        )
        .reset_index(drop=True)
    )
    out["score_missing_rows"] = out["n_fold_seed_rows"] - out["n_nonnull_score_rows"]
    out["scores_complete"] = out["score_missing_rows"].eq(0)
    return out.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def build_dataset_scores(cells: pd.DataFrame) -> pd.DataFrame:
    """Average paired downstream x k cells within each dataset."""
    dataset_cols = ["task", "dataset", "model_variant", "ranking_variant"]
    out = (
        cells.groupby(dataset_cols, as_index=False)
        .agg(
            dataset_score=("mean_score", "mean"),
            n_cells=("mean_score", "size"),
            n_downstream_models=("downstream_model", "nunique"),
            n_k_values=("k", "nunique"),
            min_fold_seed_rows=("n_fold_seed_rows", "min"),
            min_nonnull_score_rows=("n_nonnull_score_rows", "min"),
            score_missing_rows=("score_missing_rows", "sum"),
        )
        .reset_index(drop=True)
    )
    out["support_type"] = "dataset_mean_over_supported_downstream_standard_k"
    out["scores_complete"] = out["score_missing_rows"].eq(0)
    return out.sort_values(dataset_cols).reset_index(drop=True)


def build_method_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize each readout over datasets."""
    rows = []
    for task, sub in scores.groupby("task"):
        means = (
            sub.groupby(["model_variant", "ranking_variant"], as_index=False)
            .agg(
                n_datasets=("dataset", "nunique"),
                mean_dataset_score=("dataset_score", "mean"),
                median_dataset_score=("dataset_score", "median"),
                mean_dataset_cells=("n_cells", "mean"),
                min_fold_seed_rows=("min_fold_seed_rows", "min"),
                min_nonnull_score_rows=("min_nonnull_score_rows", "min"),
                total_score_missing_rows=("score_missing_rows", "sum"),
            )
            .sort_values("mean_dataset_score", ascending=False)
            .reset_index(drop=True)
        )
        means.insert(0, "task", task)
        means["rank_position"] = range(1, len(means) + 1)
        rows.append(means)
    return pd.concat(rows, ignore_index=True)


def build_paired_cell_deltas(
    cells: pd.DataFrame,
    *,
    reference_model: str = DEFAULT_REFERENCE_MODEL,
    reference_ranking: str = DEFAULT_REFERENCE_RANKING,
) -> pd.DataFrame:
    """Pair every candidate readout against the reference in each learner x k cell."""
    ref = cells[
        (cells["model_variant"] == reference_model)
        & (cells["ranking_variant"] == reference_ranking)
    ][
        [
            *PAIR_COLUMNS,
            "mean_score",
            "n_fold_seed_rows",
            "n_nonnull_score_rows",
            "score_missing_rows",
        ]
    ].rename(
        columns={
            "mean_score": "reference_score",
            "n_fold_seed_rows": "reference_fold_seed_rows",
            "n_nonnull_score_rows": "reference_nonnull_score_rows",
            "score_missing_rows": "reference_score_missing_rows",
        }
    )

    candidates = cells[
        ~(
            (cells["model_variant"] == reference_model)
            & (cells["ranking_variant"] == reference_ranking)
        )
    ][
        [
            *PAIR_COLUMNS,
            "model_variant",
            "ranking_variant",
            "mean_score",
            "n_fold_seed_rows",
            "n_nonnull_score_rows",
            "score_missing_rows",
        ]
    ].rename(
        columns={
            "mean_score": "candidate_score",
            "n_fold_seed_rows": "candidate_fold_seed_rows",
            "n_nonnull_score_rows": "candidate_nonnull_score_rows",
            "score_missing_rows": "candidate_score_missing_rows",
        }
    )

    paired = candidates.merge(ref, on=list(PAIR_COLUMNS), how="inner")
    paired["reference_model_variant"] = reference_model
    paired["reference_ranking_variant"] = reference_ranking
    paired["delta_vs_reference"] = paired["candidate_score"] - paired["reference_score"]
    paired["candidate_complete"] = paired["candidate_score_missing_rows"].eq(0)
    paired["reference_complete"] = paired["reference_score_missing_rows"].eq(0)
    return paired.sort_values(
        ["task", "dataset", "downstream_model", "k", "model_variant", "ranking_variant"]
    ).reset_index(drop=True)


def build_paired_dataset_deltas(
    scores: pd.DataFrame,
    *,
    reference_model: str = DEFAULT_REFERENCE_MODEL,
    reference_ranking: str = DEFAULT_REFERENCE_RANKING,
) -> pd.DataFrame:
    """Pair candidate and reference dataset-level means."""
    ref = scores[
        (scores["model_variant"] == reference_model)
        & (scores["ranking_variant"] == reference_ranking)
    ][["task", "dataset", "dataset_score", "n_cells", "score_missing_rows"]].rename(
        columns={
            "dataset_score": "reference_dataset_score",
            "n_cells": "reference_n_cells",
            "score_missing_rows": "reference_score_missing_rows",
        }
    )
    candidates = scores[
        ~(
            (scores["model_variant"] == reference_model)
            & (scores["ranking_variant"] == reference_ranking)
        )
    ][
        [
            "task",
            "dataset",
            "model_variant",
            "ranking_variant",
            "dataset_score",
            "n_cells",
            "score_missing_rows",
        ]
    ].rename(
        columns={
            "dataset_score": "candidate_dataset_score",
            "n_cells": "candidate_n_cells",
            "score_missing_rows": "candidate_score_missing_rows",
        }
    )
    paired = candidates.merge(ref, on=["task", "dataset"], how="inner")
    paired["reference_model_variant"] = reference_model
    paired["reference_ranking_variant"] = reference_ranking
    paired["delta_vs_reference"] = (
        paired["candidate_dataset_score"] - paired["reference_dataset_score"]
    )
    return paired.sort_values(["task", "dataset", "model_variant", "ranking_variant"]).reset_index(
        drop=True
    )


def _bootstrap_mean_ci(
    values: np.ndarray, *, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    draws = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        draws[idx] = float(rng.choice(values, size=len(values), replace=True).mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def _delta_summary(delta: pd.Series, *, n_boot: int, rng: np.random.Generator) -> dict[str, object]:
    clean = pd.to_numeric(delta, errors="coerce").dropna().to_numpy(dtype=float)
    wins = int((clean > SCORE_TOL).sum())
    losses = int((clean < -SCORE_TOL).sum())
    ties = int((np.abs(clean) <= SCORE_TOL).sum())
    non_ties = wins + losses
    ci_lower, ci_upper = _bootstrap_mean_ci(clean, n_boot=n_boot, rng=rng)
    return {
        "mean_delta_vs_reference": float(clean.mean()) if len(clean) else np.nan,
        "median_delta_vs_reference": float(np.median(clean)) if len(clean) else np.nan,
        "ci95_mean_delta_lower": ci_lower,
        "ci95_mean_delta_upper": ci_upper,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_share": float(wins / len(clean)) if len(clean) else np.nan,
        "nonloss_share": float((wins + ties) / len(clean)) if len(clean) else np.nan,
        "sign_test_pvalue_two_sided": (
            float(binomtest(wins, non_ties, 0.5, alternative="two-sided").pvalue)
            if non_ties
            else np.nan
        ),
        "bootstrap_replicates": int(n_boot),
    }


def build_pairwise_by_downstream_k(
    cell_deltas: pd.DataFrame, *, n_boot: int = DEFAULT_N_BOOT
) -> pd.DataFrame:
    """Summarize paired deltas separately for each downstream learner and k."""
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    group_cols = ["task", "model_variant", "ranking_variant", "downstream_model", "k"]
    for keys, sub in cell_deltas.groupby(group_cols):
        task, model_variant, ranking_variant, downstream_model, k = keys
        rows.append(
            {
                "task": task,
                "model_variant": model_variant,
                "ranking_variant": ranking_variant,
                "reference_model_variant": DEFAULT_REFERENCE_MODEL,
                "reference_ranking_variant": DEFAULT_REFERENCE_RANKING,
                "downstream_model": downstream_model,
                "k": int(k),
                "n_datasets": int(sub["dataset"].nunique()),
                **_delta_summary(sub["delta_vs_reference"], n_boot=n_boot, rng=rng),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_pairwise_vs_reference(
    dataset_deltas: pd.DataFrame, *, n_boot: int = DEFAULT_N_BOOT
) -> pd.DataFrame:
    """Summarize dataset-averaged candidate-vs-reference deltas."""
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    group_cols = ["task", "model_variant", "ranking_variant"]
    for keys, sub in dataset_deltas.groupby(group_cols):
        task, model_variant, ranking_variant = keys
        rows.append(
            {
                "task": task,
                "model_variant": model_variant,
                "ranking_variant": ranking_variant,
                "reference_model_variant": DEFAULT_REFERENCE_MODEL,
                "reference_ranking_variant": DEFAULT_REFERENCE_RANKING,
                "support_type": "dataset_mean_over_supported_downstream_standard_k",
                "n_datasets": int(sub["dataset"].nunique()),
                "mean_candidate_cells": float(sub["candidate_n_cells"].mean()),
                "mean_reference_cells": float(sub["reference_n_cells"].mean()),
                "candidate_score_missing_rows": int(sub["candidate_score_missing_rows"].sum()),
                "reference_score_missing_rows": int(sub["reference_score_missing_rows"].sum()),
                **_delta_summary(sub["delta_vs_reference"], n_boot=n_boot, rng=rng),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CIF component-ablation summary tables")
    parser.add_argument(
        "--input-uri",
        default=DEFAULT_INPUT_URI,
        help="Local path or s3:// metrics prefix",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=DEFAULT_EXPECTED_SEEDS,
        help="Expected number of seeds per complete cell",
    )
    parser.add_argument(
        "--expected-folds",
        type=int,
        default=DEFAULT_EXPECTED_FOLDS,
        help="Expected number of folds per complete cell",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=DEFAULT_N_BOOT,
        help="Bootstrap replicates for mean-delta confidence intervals",
    )
    parser.add_argument(
        "--strict-complete",
        action="store_true",
        help="Fail if any standard-k cell is missing expected fold/seed support",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(args.input_uri)
    metric_sort_cols = [
        col
        for col in [
            "task",
            "dataset",
            "seed",
            "fold_idx",
            "model_variant",
            "ranking_variant",
            "downstream_model",
            "k",
        ]
        if col in metrics.columns
    ]
    metrics_flat = metrics.sort_values(metric_sort_cols).reset_index(drop=True)
    completeness = build_completeness_table(
        metrics,
        expected_seeds=args.expected_seeds,
        expected_folds=args.expected_folds,
    )
    cells = build_cell_scores(metrics)
    scores = build_dataset_scores(cells)
    method_summary = build_method_summary(scores)
    cell_deltas = build_paired_cell_deltas(cells)
    dataset_deltas = build_paired_dataset_deltas(scores)
    by_downstream_k = build_pairwise_by_downstream_k(cell_deltas, n_boot=args.n_boot)
    pairwise = build_pairwise_vs_reference(dataset_deltas, n_boot=args.n_boot)

    outputs = {
        "cif_mechanism_ablation_metrics_flat.csv": metrics_flat,
        "cif_mechanism_ablation_completeness.csv": completeness,
        "cif_mechanism_ablation_cell_scores.csv": cells,
        "cif_mechanism_ablation_dataset_scores.csv": scores,
        "cif_mechanism_ablation_method_summary.csv": method_summary,
        "cif_mechanism_ablation_paired_cell_deltas_vs_default.csv": cell_deltas,
        "cif_mechanism_ablation_paired_dataset_deltas_vs_default.csv": dataset_deltas,
        "cif_mechanism_ablation_pairwise_by_downstream_k_vs_default.csv": by_downstream_k,
        "cif_mechanism_ablation_pairwise_vs_default.csv": pairwise,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)

    incomplete = completeness[~completeness["is_complete"]]
    print(f"Loaded metric rows: {len(metrics)}")
    print(f"Standard-k cell rows: {len(cells)}")
    print(f"Dataset score rows: {len(scores)}")
    print(f"Incomplete standard-k cells: {len(incomplete)}")
    print(f"Wrote tables to: {args.output_dir}")
    if args.strict_complete and not incomplete.empty:
        raise RuntimeError(f"{len(incomplete)} standard-k cell(s) are missing expected support")


if __name__ == "__main__":
    main()
