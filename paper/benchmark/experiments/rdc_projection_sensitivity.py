#!/usr/bin/env python3
"""Run the small local RDC projection-count sensitivity study."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from paper.benchmark.adapters.data import (
    get_cv_splitter,
    get_dataset_identity,
    load_dataset,
)
from paper.benchmark.pipeline.stage1 import permutation_selector
from paper.benchmark.pipeline.stage2 import (
    evaluate_fold,
    get_requested_evaluation_k_values,
)
from paper.benchmark.utils import get_hardware_metadata, get_library_versions

ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR: Final[Path] = (
    ROOT / "paper" / "results" / "rdc-projection-sensitivity"
)
DATASETS: Final[tuple[str, ...]] = ("wine", "glass")
TASK: Final[Literal["classification"]] = "classification"
PROJECTION_COUNTS: Final[tuple[int, ...]] = (5, 10, 20, 40)
N_FOLDS: Final[int] = 5
DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
RANKING_COLUMNS: Final[tuple[str, ...]] = (
    "task",
    "dataset",
    "dataset_sha256",
    "n_samples",
    "n_features",
    "seed",
    "fold",
    "projection_count",
    "projected_columns",
    "selection_seconds",
    "feature_ranking",
)


def _parse_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(seed < 0 for seed in values) or len(set(values)) != len(values):
        raise ValueError("seeds must be a nonempty comma-separated list of unique integers")
    return values


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _completed_keys(
    rankings: pd.DataFrame,
    metrics: pd.DataFrame,
) -> set[tuple[str, int, int, int]]:
    if rankings.empty or metrics.empty:
        return set()
    expected_models = {"lr", "svm", "knn"}
    completed: set[tuple[str, int, int, int]] = set()
    for key, ranking_group in rankings.groupby(
        ["dataset", "seed", "fold", "projection_count"],
        sort=False,
    ):
        if len(ranking_group) != 1:
            continue
        dataset, seed, fold, projection_count = key
        metric_group = metrics[
            (metrics["dataset"] == dataset)
            & (metrics["seed"] == seed)
            & (metrics["fold"] == fold)
            & (metrics["projection_count"] == projection_count)
        ]
        n_features = int(ranking_group.iloc[0]["n_features"])
        expected_k = set(get_requested_evaluation_k_values(n_features))
        observed_pairs = set(
            zip(
                metric_group["k"].astype(int),
                metric_group["downstream_model"],
                strict=False,
            )
        )
        expected_pairs = {
            (selected, model)
            for selected in expected_k
            for model in expected_models
        }
        if observed_pairs == expected_pairs:
            completed.add(
                (
                    str(dataset),
                    int(seed),
                    int(fold),
                    int(projection_count),
                )
            )
    return completed


def _load_existing(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking_path = output_dir / "rankings.csv"
    metric_path = output_dir / "metrics.csv"
    rankings = (
        pd.read_csv(ranking_path)
        if ranking_path.exists()
        else pd.DataFrame(columns=RANKING_COLUMNS)
    )
    metrics = pd.read_csv(metric_path) if metric_path.exists() else pd.DataFrame()
    return rankings, metrics


def _warm_rdc() -> None:
    identity = get_dataset_identity("wine", TASK, source="real")
    X, y = load_dataset("wine", TASK, identity=identity, source="real")
    train_idx, _ = next(iter(get_cv_splitter(TASK, N_FOLDS, 0).split(X, y)))
    X_train = StandardScaler().fit_transform(X[train_idx])
    permutation_selector(
        X_train,
        y[train_idx],
        method="ptest_rdc",
        task=TASK,
        random_state=0,
        params={
            "alpha": 0.05,
            "n_resamples": "auto",
            "early_stopping": "adaptive",
            "rdc_n_projections": PROJECTION_COUNTS[0],
        },
    )


def run(
    *,
    seeds: tuple[int, ...],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run missing dataset, fold, and projection-count combinations."""
    rankings, metrics = _load_existing(output_dir)
    completed = _completed_keys(rankings, metrics)
    ranking_rows = rankings.to_dict("records")
    metric_rows = metrics.to_dict("records")

    _warm_rdc()
    for dataset in DATASETS:
        identity = get_dataset_identity(dataset, TASK, source="real")
        X, y = load_dataset(
            dataset,
            TASK,
            identity=identity,
            source="real",
        )
        k_values = get_requested_evaluation_k_values(identity.n_features)
        for seed in seeds:
            cv = get_cv_splitter(TASK, N_FOLDS, seed)
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                ran_item = False
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                random_state = seed * 1000 + fold
                for projection_count in PROJECTION_COUNTS:
                    key = (dataset, seed, fold, projection_count)
                    if key in completed:
                        continue
                    ran_item = True
                    started = time.perf_counter()
                    ranking = permutation_selector(
                        X_train,
                        y[train_idx],
                        method="ptest_rdc",
                        task=TASK,
                        random_state=random_state,
                        params={
                            "alpha": 0.05,
                            "n_resamples": "auto",
                            "early_stopping": "adaptive",
                            "rdc_n_projections": projection_count,
                        },
                    )
                    elapsed = time.perf_counter() - started
                    ranking_rows.append(
                        {
                            "task": TASK,
                            "dataset": dataset,
                            "dataset_sha256": identity.sha256,
                            "n_samples": identity.n_samples,
                            "n_features": identity.n_features,
                            "seed": seed,
                            "fold": fold,
                            "projection_count": projection_count,
                            "projected_columns": 2 * projection_count,
                            "selection_seconds": elapsed,
                            "feature_ranking": json.dumps(ranking.tolist()),
                        }
                    )
                    for row in evaluate_fold(
                        X[train_idx],
                        y[train_idx],
                        X[test_idx],
                        y[test_idx],
                        ranking,
                        TASK,
                        random_state,
                        k_values=k_values,
                        n_jobs=1,
                    ):
                        metric_rows.append(
                            {
                                "task": TASK,
                                "dataset": dataset,
                                "seed": seed,
                                "fold": fold,
                                "projection_count": projection_count,
                                "projected_columns": 2 * projection_count,
                                **row,
                            }
                        )
                rankings = pd.DataFrame(ranking_rows)
                metrics = pd.DataFrame(metric_rows)
                _atomic_csv(rankings, output_dir / "rankings.csv")
                _atomic_csv(metrics, output_dir / "metrics.csv")
                if ran_item:
                    print(f"{dataset}: seed={seed} fold={fold} complete", flush=True)
    return pd.DataFrame(ranking_rows), pd.DataFrame(metric_rows)


def build_stability(rankings: pd.DataFrame) -> pd.DataFrame:
    """Compare complete rankings and selected sets within matched folds."""
    rows: list[dict[str, Any]] = []
    for (dataset, seed, fold), group in rankings.groupby(
        ["dataset", "seed", "fold"],
        sort=True,
    ):
        by_projection = {
            int(row.projection_count): np.asarray(
                json.loads(row.feature_ranking),
                dtype=np.int64,
            )
            for row in group.itertuples()
        }
        if set(by_projection) != set(PROJECTION_COUNTS):
            continue
        n_features = len(next(iter(by_projection.values())))
        positions = {
            projection_count: np.argsort(ranking)
            for projection_count, ranking in by_projection.items()
        }
        for left, right in combinations(PROJECTION_COUNTS, 2):
            rank_correlation = float(
                spearmanr(positions[left], positions[right]).statistic
            )
            for selected in get_requested_evaluation_k_values(n_features):
                left_set = set(by_projection[left][:selected])
                right_set = set(by_projection[right][:selected])
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "fold": fold,
                        "projection_count_left": left,
                        "projection_count_right": right,
                        "selected_features": selected,
                        "overlap_fraction": len(left_set & right_set) / selected,
                        "jaccard": len(left_set & right_set)
                        / len(left_set | right_set),
                        "spearman_complete_ranking": rank_correlation,
                    }
                )
    return pd.DataFrame(rows)


def summarize(
    rankings: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    output_dir: Path,
) -> None:
    """Write compact timing, accuracy, stability, and provenance outputs."""
    stability = build_stability(rankings)
    timing = rankings.groupby(
        ["dataset", "projection_count"],
        as_index=False,
    ).agg(
        mean_seconds=("selection_seconds", "mean"),
        median_seconds=("selection_seconds", "median"),
        std_seconds=("selection_seconds", "std"),
        max_seconds=("selection_seconds", "max"),
    )
    reference_median = (
        timing[timing["projection_count"] == 10]
        .set_index("dataset")["median_seconds"]
        .to_dict()
    )
    timing["median_ratio_to_10"] = timing.apply(
        lambda row: row["median_seconds"] / reference_median[row["dataset"]],
        axis=1,
    )
    performance = metrics.groupby(
        ["dataset", "k", "downstream_model", "projection_count"],
        as_index=False,
    ).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        folds=("balanced_accuracy", "count"),
    )
    stability_summary = (
        stability.groupby(
            [
                "dataset",
                "projection_count_left",
                "projection_count_right",
                "selected_features",
            ],
            as_index=False,
        )[["overlap_fraction", "jaccard", "spearman_complete_ranking"]]
        .mean()
    )
    _atomic_csv(timing, output_dir / "timing-summary.csv")
    _atomic_csv(performance, output_dir / "performance-summary.csv")
    _atomic_csv(stability, output_dir / "stability.csv")
    _atomic_csv(stability_summary, output_dir / "stability-summary.csv")

    comparison_rows: list[dict[str, Any]] = []
    metric_index = ["dataset", "seed", "fold", "k", "downstream_model"]
    paired_metrics = metrics.pivot(
        index=metric_index,
        columns="projection_count",
        values="balanced_accuracy",
    )
    for dataset in DATASETS:
        timing_rows = timing[timing["dataset"] == dataset].set_index(
            "projection_count"
        )
        metric_rows = paired_metrics.xs(dataset, level="dataset")
        stability_rows = stability[
            (stability["dataset"] == dataset)
            & (stability["projection_count_left"] == 10)
            & (stability["projection_count_right"] == 20)
        ]
        comparison_rows.append(
            {
                "dataset": dataset,
                "projection_count_reference": 10,
                "projection_count_comparison": 20,
                "median_runtime_ratio": timing_rows.loc[20, "median_seconds"]
                / timing_rows.loc[10, "median_seconds"],
                "balanced_accuracy_mean_difference": (
                    metric_rows[20] - metric_rows[10]
                ).mean(),
                "selected_set_overlap_mean": stability_rows[
                    "overlap_fraction"
                ].mean(),
                "complete_ranking_spearman_mean": stability_rows[
                    "spearman_complete_ranking"
                ].mean(),
            }
        )
    _atomic_csv(
        pd.DataFrame(comparison_rows),
        output_dir / "comparison-summary.csv",
    )

    dataset_identities = {}
    for dataset in DATASETS:
        identity = get_dataset_identity(dataset, TASK, source="real")
        dataset_identities[dataset] = {
            "sha256": identity.sha256,
            "n_samples": identity.n_samples,
            "n_features": identity.n_features,
        }
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt = {
        "analysis": "rdc_projection_sensitivity",
        "datasets": dataset_identities,
        "folds": N_FOLDS,
        "hardware": get_hardware_metadata(),
        "library_versions": get_library_versions(),
        "metric_rows": len(metrics),
        "projected_columns": [2 * count for count in PROJECTION_COUNTS],
        "projection_counts": list(PROJECTION_COUNTS),
        "ranking_rows": len(rankings),
        "selection_seconds_total": float(rankings["selection_seconds"].sum()),
        "script_sha256": script_sha256,
        "seeds": sorted(int(seed) for seed in rankings["seed"].unique()),
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated CV seeds",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Local result directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = _parse_int_csv(args.seeds)
    rankings, metrics = run(seeds=seeds, output_dir=args.output_dir)
    summarize(
        rankings,
        metrics,
        output_dir=args.output_dir,
    )
    print(
        f"completed rankings={len(rankings)} metrics={len(metrics)} "
        f"output={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
