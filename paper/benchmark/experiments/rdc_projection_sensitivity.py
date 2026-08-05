#!/usr/bin/env python3
"""Run the small local RDC projection-count sensitivity study."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from paper.benchmark.adapters.data import (
    get_cv_splitter,
    get_dataset_file_identity,
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
DEFAULT_OUTPUT_DIR: Final[Path] = ROOT / "paper" / "results" / "rdc-projection-sensitivity"
STUDY_DATA_DIR: Final[Path] = ROOT / "paper" / "data" / "rdc-projection-sensitivity"
DATASETS: Final[tuple[str, ...]] = (
    "wine",
    "glass",
    "heart-statlog",
    "ionosphere",
    "breast-cancer",
)
SNAPSHOT_SOURCES: Final[dict[str, str]] = {
    "heart-statlog": "OpenML dataset 53, version 1 (UCI Statlog Heart)",
    "ionosphere": "OpenML dataset 59, version 1 (UCI Ionosphere)",
}
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
SUMMARY_FILENAMES: Final[tuple[str, ...]] = (
    "timing-summary.csv",
    "performance-summary.csv",
    "stability.csv",
    "stability-summary.csv",
    "comparison-summary.csv",
    "receipt.json",
)


@dataclass(frozen=True)
class StudyDataset:
    """One fixed dataset used by the projection-count sensitivity study."""

    name: str
    sha256: str
    source: str
    X: np.ndarray
    y: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


def _array_sha256(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...],
) -> str:
    """Hash normalized array contents, shapes, and feature names."""
    digest = hashlib.sha256()
    for label, array, dtype in (
        (b"X", X, "<f8"),
        (b"y", y, "<i8"),
    ):
        normalized = np.ascontiguousarray(array, dtype=np.dtype(dtype))
        digest.update(label)
        digest.update(np.asarray(normalized.shape, dtype="<i8").tobytes())
        digest.update(normalized.tobytes())
    digest.update("\0".join(feature_names).encode("utf-8"))
    return digest.hexdigest()


def _load_study_dataset(name: str) -> StudyDataset:
    """Load one fixed local or scikit-learn dataset with content identity."""
    if name in SNAPSHOT_SOURCES:
        path = STUDY_DATA_DIR / f"{name}.parquet"
        identity = get_dataset_file_identity(path)
        frame = pd.read_parquet(path)
        feature_columns = [column for column in frame.columns if column != "y"]
        X = frame[feature_columns].to_numpy(dtype=np.float64)
        y = frame["y"].to_numpy(dtype=np.int64)
        return StudyDataset(
            name=name,
            sha256=identity.sha256,
            source=SNAPSHOT_SOURCES[name],
            X=X,
            y=y,
        )

    if name == "breast-cancer":
        bunch = load_breast_cancer()
        X = np.asarray(bunch.data, dtype=np.float64)
        y = np.asarray(bunch.target, dtype=np.int64)
        feature_names = tuple(str(value) for value in bunch.feature_names)
        return StudyDataset(
            name=name,
            sha256=_array_sha256(X, y, feature_names),
            source="scikit-learn Breast Cancer Wisconsin (Diagnostic)",
            X=X,
            y=y,
        )

    identity = get_dataset_identity(name, TASK, source="real")
    X, y = load_dataset(
        name,
        TASK,
        identity=identity,
        source="real",
    )
    return StudyDataset(
        name=name,
        sha256=identity.sha256,
        source="benchmark parquet",
        X=X,
        y=y,
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


def _file_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _input_hashes(output_dir: Path) -> tuple[str | None, str | None]:
    return (
        _file_sha256(output_dir / "rankings.csv"),
        _file_sha256(output_dir / "metrics.csv"),
    )


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
        expected_pairs = {(selected, model) for selected in expected_k for model in expected_models}
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
    dataset = _load_study_dataset("wine")
    train_idx, _ = next(iter(get_cv_splitter(TASK, N_FOLDS, 0).split(dataset.X, dataset.y)))
    X_train = StandardScaler().fit_transform(dataset.X[train_idx])
    permutation_selector(
        X_train,
        dataset.y[train_idx],
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
    for dataset_name in DATASETS:
        dataset = _load_study_dataset(dataset_name)
        X, y = dataset.X, dataset.y
        k_values = get_requested_evaluation_k_values(dataset.n_features)
        for seed in seeds:
            cv = get_cv_splitter(TASK, N_FOLDS, seed)
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                ran_item = False
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                random_state = seed * 1000 + fold
                for projection_count in PROJECTION_COUNTS:
                    key = (dataset_name, seed, fold, projection_count)
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
                            "dataset": dataset_name,
                            "dataset_sha256": dataset.sha256,
                            "n_samples": dataset.n_samples,
                            "n_features": dataset.n_features,
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
                                "dataset": dataset_name,
                                "seed": seed,
                                "fold": fold,
                                "projection_count": projection_count,
                                "projected_columns": 2 * projection_count,
                                **row,
                            }
                        )
                if ran_item:
                    rankings = pd.DataFrame(ranking_rows)
                    metrics = pd.DataFrame(metric_rows)
                    _atomic_csv(rankings, output_dir / "rankings.csv")
                    _atomic_csv(metrics, output_dir / "metrics.csv")
                    print(
                        f"{dataset_name}: seed={seed} fold={fold} complete",
                        flush=True,
                    )
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
            rank_correlation = float(spearmanr(positions[left], positions[right]).statistic)
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
                        "jaccard": len(left_set & right_set) / len(left_set | right_set),
                        "spearman_complete_ranking": rank_correlation,
                    }
                )
    return pd.DataFrame(rows)


def build_comparison_summary(
    rankings: pd.DataFrame,
    metrics: pd.DataFrame,
    timing: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    """Compare 10 and 20 projections at reduced feature counts."""
    comparison_rows: list[dict[str, Any]] = []
    metric_index = ["dataset", "seed", "fold", "k", "downstream_model"]
    paired_metrics = metrics.pivot(
        index=metric_index,
        columns="projection_count",
        values="balanced_accuracy",
    )
    n_features = rankings.groupby("dataset")["n_features"].first().astype(int).to_dict()
    for dataset_name in DATASETS:
        timing_rows = timing[timing["dataset"] == dataset_name].set_index("projection_count")
        metric_rows = paired_metrics.xs(dataset_name, level="dataset")
        metric_rows = metric_rows[
            metric_rows.index.get_level_values("k") < n_features[dataset_name]
        ]
        stability_rows = stability[
            (stability["dataset"] == dataset_name)
            & (stability["projection_count_left"] == 10)
            & (stability["projection_count_right"] == 20)
            & (stability["selected_features"] < n_features[dataset_name])
        ]
        if metric_rows.empty or stability_rows.empty:
            raise RuntimeError(f"{dataset_name} has no reduced-feature comparison rows")
        comparison_rows.append(
            {
                "dataset": dataset_name,
                "projection_count_reference": 10,
                "projection_count_comparison": 20,
                "median_runtime_ratio": timing_rows.loc[20, "median_seconds"]
                / timing_rows.loc[10, "median_seconds"],
                "balanced_accuracy_mean_difference_reduced_k": (
                    metric_rows[20] - metric_rows[10]
                ).mean(),
                "selected_set_overlap_mean_reduced_k": stability_rows["overlap_fraction"].mean(),
                "complete_ranking_spearman_mean": stability_rows[
                    "spearman_complete_ranking"
                ].mean(),
            }
        )
    return pd.DataFrame(comparison_rows)


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
        timing[timing["projection_count"] == 10].set_index("dataset")["median_seconds"].to_dict()
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
    stability_summary = stability.groupby(
        [
            "dataset",
            "projection_count_left",
            "projection_count_right",
            "selected_features",
        ],
        as_index=False,
    )[["overlap_fraction", "jaccard", "spearman_complete_ranking"]].mean()
    _atomic_csv(timing, output_dir / "timing-summary.csv")
    _atomic_csv(performance, output_dir / "performance-summary.csv")
    _atomic_csv(stability, output_dir / "stability.csv")
    _atomic_csv(stability_summary, output_dir / "stability-summary.csv")

    _atomic_csv(
        build_comparison_summary(rankings, metrics, timing, stability),
        output_dir / "comparison-summary.csv",
    )

    dataset_identities = {}
    reduced_feature_counts = {}
    for dataset_name in DATASETS:
        dataset = _load_study_dataset(dataset_name)
        dataset_identities[dataset_name] = {
            "sha256": dataset.sha256,
            "n_samples": dataset.n_samples,
            "n_features": dataset.n_features,
            "source": dataset.source,
        }
        reduced_feature_counts[dataset_name] = [
            selected
            for selected in get_requested_evaluation_k_values(dataset.n_features)
            if selected < dataset.n_features
        ]
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
        "reduced_feature_counts": reduced_feature_counts,
        "selection_seconds_total": float(rankings["selection_seconds"].sum()),
        "script_sha256": script_sha256,
        "seeds": sorted(int(seed) for seed in rankings["seed"].unique()),
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def execute(
    *,
    seeds: tuple[int, ...],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run missing cells and rebuild summaries only when their inputs change."""
    hashes_before = _input_hashes(output_dir)
    rankings, metrics = run(seeds=seeds, output_dir=output_dir)
    hashes_after = _input_hashes(output_dir)
    summaries_complete = all((output_dir / filename).exists() for filename in SUMMARY_FILENAMES)
    if hashes_before != hashes_after or not summaries_complete:
        summarize(rankings, metrics, output_dir=output_dir)
    else:
        receipt = json.loads((output_dir / "receipt.json").read_text(encoding="ascii"))
        script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if receipt.get("script_sha256") != script_sha256:
            raise RuntimeError("existing results were generated by a different script revision")
    return rankings, metrics


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
    rankings, metrics = execute(seeds=seeds, output_dir=args.output_dir)
    print(
        f"completed rankings={len(rankings)} metrics={len(metrics)} output={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
