"""Run the JSS RDC projection-count sensitivity study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from paper.benchmark.adapters.data import (
    get_cv_splitter,
    get_dataset_file_identity,
    get_dataset_identity,
    load_dataset,
)
from paper.benchmark.config.constants import CLF_DOWNSTREAM_MODELS
from paper.benchmark.pipeline.stage1 import permutation_selector
from paper.benchmark.pipeline.stage2 import (
    evaluate_fold,
    get_requested_evaluation_k_values,
)
from paper.benchmark.utils import get_hardware_metadata

Profile = Literal["smoke", "quick", "full"]

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR: Final[Path] = REPO_ROOT / "paper" / "jss" / "results" / "rdc-sensitivity"
STUDY_DATA_DIR: Final[Path] = REPO_ROOT / "paper" / "data" / "rdc-projection-sensitivity"
DATASETS: Final[tuple[str, ...]] = ("wine", "glass", "heart-statlog", "parkinsons")
DATASET_EXPECTATIONS: Final[dict[str, tuple[int, int, int, str]]] = {
    "wine": (
        178,
        13,
        3,
        "f559edb359a04d51531da5e80a499c8d7fde49233ffb9be8949f00ee36336b9e",
    ),
    "glass": (
        214,
        10,
        6,
        "8f6eea990c4fdc7bcc55a2bd26743435287d88d3f49b60dd4a8dd68dc931d9de",
    ),
    "heart-statlog": (
        270,
        13,
        2,
        "8c624c057ab20b49e8ce2b4b222ae5655a7652e3991aa997d100570ef9c3f438",
    ),
    "parkinsons": (
        195,
        22,
        2,
        "f7d1985c10604b4f65c5a0444767a4b1433395016e155009892abe1cf854f91f",
    ),
}
SNAPSHOT_SOURCES: Final[dict[str, str]] = {
    "heart-statlog": "OpenML dataset 53, version 1 (UCI Statlog Heart)",
    "parkinsons": "OpenML dataset 1488, version 1 (UCI Parkinsons)",
}
TASK: Final[Literal["classification"]] = "classification"
PROJECTION_COUNTS: Final[tuple[int, ...]] = (5, 10, 20, 40)
N_FOLDS: Final[int] = 5
DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
SELECTOR_METHOD: Final[str] = "ptest_rdc"
SELECTOR_ALPHA: Final[float] = 0.05
SELECTOR_N_RESAMPLES: Final[str] = "auto"
SELECTOR_EARLY_STOPPING: Final[str] = "adaptive"
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
METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "task",
    "dataset",
    "seed",
    "fold",
    "projection_count",
    "projected_columns",
    "k",
    "n_features_selected",
    "downstream_model",
    "accuracy",
    "f1",
    "f1_macro",
    "balanced_accuracy",
    "roc_auc",
    "auc",
)
STABILITY_COLUMNS: Final[tuple[str, ...]] = (
    "dataset",
    "seed",
    "fold",
    "projection_count_left",
    "projection_count_right",
    "selected_features",
    "overlap_fraction",
    "jaccard",
)
RANKING_STABILITY_COLUMNS: Final[tuple[str, ...]] = (
    "dataset",
    "seed",
    "fold",
    "projection_count_left",
    "projection_count_right",
    "spearman_complete_ranking",
)
TIMING_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "dataset",
    "projection_count",
    "mean_seconds",
    "median_seconds",
    "std_seconds",
    "max_seconds",
    "median_ratio_to_10",
)
PERFORMANCE_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "dataset",
    "k",
    "downstream_model",
    "projection_count",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "folds",
)
STABILITY_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "dataset",
    "projection_count_left",
    "projection_count_right",
    "selected_features",
    "overlap_fraction",
    "jaccard",
)
RANKING_STABILITY_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "dataset",
    "projection_count_left",
    "projection_count_right",
    "spearman_complete_ranking_mean",
    "spearman_complete_ranking_std",
    "fold_pairs",
)
COMPARISON_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "dataset",
    "projection_count_reference",
    "projection_count_comparison",
    "fold_pairs",
    "reduced_feature_counts",
    "runtime_ratio_median",
    "runtime_ratio_mean",
    "runtime_ratio_std",
    "runtime_ratio_min",
    "runtime_ratio_max",
    "balanced_accuracy_pairs",
    "balanced_accuracy_mean_difference_reduced_k",
    "balanced_accuracy_fold_difference_std",
    "selected_set_overlap_mean_reduced_k",
    "selected_set_overlap_fold_std",
    "complete_ranking_spearman_mean",
    "complete_ranking_spearman_std",
)
RESULT_SCHEMAS: Final[dict[str, tuple[str, ...]]] = {
    "rankings": RANKING_COLUMNS,
    "metrics": METRIC_COLUMNS,
    "timing_summary": TIMING_SUMMARY_COLUMNS,
    "performance_summary": PERFORMANCE_SUMMARY_COLUMNS,
    "stability": STABILITY_COLUMNS,
    "stability_summary": STABILITY_SUMMARY_COLUMNS,
    "ranking_stability": RANKING_STABILITY_COLUMNS,
    "ranking_stability_summary": RANKING_STABILITY_SUMMARY_COLUMNS,
    "comparison_summary": COMPARISON_SUMMARY_COLUMNS,
}
RESULT_FILENAMES: Final[dict[str, str]] = {
    "rankings": "rankings.csv",
    "metrics": "metrics.csv",
    "timing_summary": "timing-summary.csv",
    "performance_summary": "performance-summary.csv",
    "stability": "stability.csv",
    "stability_summary": "stability-summary.csv",
    "ranking_stability": "ranking-stability.csv",
    "ranking_stability_summary": "ranking-stability-summary.csv",
    "comparison_summary": "comparison-summary.csv",
}
SOURCE_PATHS: Final[tuple[Path, ...]] = (
    Path(__file__).resolve(),
    REPO_ROOT / "paper" / "benchmark" / "adapters" / "data.py",
    REPO_ROOT / "paper" / "benchmark" / "pipeline" / "stage1.py",
    REPO_ROOT / "paper" / "benchmark" / "pipeline" / "stage2.py",
    REPO_ROOT / "citrees" / "_permutation.py",
    REPO_ROOT / "citrees" / "_selector.py",
    REPO_ROOT / "citrees" / "_sequential.py",
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "uv.lock",
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


@dataclass(frozen=True)
class RdcSensitivitySettings:
    """Controlled workload for one JSS replication profile."""

    datasets: tuple[str, ...]
    seeds: tuple[int, ...]
    folds: int
    projection_counts: tuple[int, ...]


def _settings(profile: Profile) -> RdcSensitivitySettings:
    if profile == "smoke":
        return RdcSensitivitySettings(
            datasets=("wine",),
            seeds=(0,),
            folds=N_FOLDS,
            projection_counts=PROJECTION_COUNTS,
        )
    if profile == "quick":
        return RdcSensitivitySettings(
            datasets=DATASETS,
            seeds=(0,),
            folds=N_FOLDS,
            projection_counts=PROJECTION_COUNTS,
        )
    if profile == "full":
        return RdcSensitivitySettings(
            datasets=DATASETS,
            seeds=DEFAULT_SEEDS,
            folds=N_FOLDS,
            projection_counts=PROJECTION_COUNTS,
        )
    raise ValueError(f"unknown RDC sensitivity profile: {profile}")


def _load_study_dataset(name: str) -> StudyDataset:
    """Load one fixed benchmark dataset with content identity."""
    if name not in DATASET_EXPECTATIONS:
        raise ValueError(f"unknown RDC sensitivity dataset: {name}")
    if name in SNAPSHOT_SOURCES:
        path = STUDY_DATA_DIR / f"{name}.parquet"
        identity = get_dataset_file_identity(path)
        frame = pd.read_parquet(path)
        feature_columns = [column for column in frame.columns if column != "y"]
        dataset = StudyDataset(
            name=name,
            sha256=identity.sha256,
            source=SNAPSHOT_SOURCES[name],
            X=frame[feature_columns].to_numpy(dtype=np.float64),
            y=frame["y"].to_numpy(dtype=np.int64),
        )
    else:
        identity = get_dataset_identity(name, TASK, source="real")
        X, y = load_dataset(
            name,
            TASK,
            identity=identity,
            source="real",
        )
        dataset = StudyDataset(
            name=name,
            sha256=identity.sha256,
            source="benchmark parquet",
            X=X,
            y=y,
        )
    expected_samples, expected_features, expected_classes, expected_sha256 = DATASET_EXPECTATIONS[
        name
    ]
    if dataset.X.shape != (expected_samples, expected_features):
        raise ValueError(
            f"{name} shape differs: "
            f"expected={(expected_samples, expected_features)}, observed={dataset.X.shape}"
        )
    if dataset.y.shape != (expected_samples,):
        raise ValueError(
            f"{name} target shape differs: expected={(expected_samples,)}, observed={dataset.y.shape}"
        )
    if dataset.sha256 != expected_sha256:
        raise ValueError(
            f"{name} hash differs: expected={expected_sha256}, observed={dataset.sha256}"
        )
    if not np.isfinite(dataset.X).all():
        raise ValueError(f"{name} contains non-finite predictors")
    if len(np.unique(dataset.y)) != expected_classes:
        raise ValueError(f"{name} class inventory differs")
    return dataset


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in SOURCE_PATHS}


def _versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(distribution)
        for package, distribution in (
            ("citrees", "citrees"),
            ("numpy", "numpy"),
            ("pandas", "pandas"),
            ("pyarrow", "pyarrow"),
            ("scikit-learn", "scikit-learn"),
            ("scipy", "scipy"),
        )
    }


def _warm_rdc(settings: RdcSensitivitySettings) -> None:
    dataset = _load_study_dataset(settings.datasets[0])
    seed = settings.seeds[0]
    train_idx, _ = next(
        iter(get_cv_splitter(TASK, settings.folds, seed).split(dataset.X, dataset.y))
    )
    X_train = StandardScaler().fit_transform(dataset.X[train_idx])
    permutation_selector(
        X_train,
        dataset.y[train_idx],
        method=SELECTOR_METHOD,
        task=TASK,
        random_state=seed * 1000,
        params={
            "alpha": SELECTOR_ALPHA,
            "n_resamples": SELECTOR_N_RESAMPLES,
            "early_stopping": SELECTOR_EARLY_STOPPING,
            "rdc_n_projections": settings.projection_counts[0],
        },
    )


def run_sensitivity(
    settings: RdcSensitivitySettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every dataset, fold, seed, and projection-count cell."""
    ranking_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    _warm_rdc(settings)
    for dataset_name in settings.datasets:
        dataset = _load_study_dataset(dataset_name)
        X, y = dataset.X, dataset.y
        k_values = get_requested_evaluation_k_values(dataset.n_features)
        for seed in settings.seeds:
            cv = get_cv_splitter(TASK, settings.folds, seed)
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                random_state = seed * 1000 + fold
                for projection_count in settings.projection_counts:
                    started = time.perf_counter()
                    ranking = permutation_selector(
                        X_train,
                        y[train_idx],
                        method=SELECTOR_METHOD,
                        task=TASK,
                        random_state=random_state,
                        params={
                            "alpha": SELECTOR_ALPHA,
                            "n_resamples": SELECTOR_N_RESAMPLES,
                            "early_stopping": SELECTOR_EARLY_STOPPING,
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
    rankings = pd.DataFrame(ranking_rows, columns=RANKING_COLUMNS)
    metrics = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS)
    return rankings, metrics


def build_stability(rankings: pd.DataFrame) -> pd.DataFrame:
    """Compare selected sets within matched folds."""
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
        if len(group) != len(PROJECTION_COUNTS) or set(by_projection) != set(PROJECTION_COUNTS):
            raise ValueError(
                f"incomplete RDC projection inventory for {dataset}, seed={seed}, fold={fold}"
            )
        n_features = len(next(iter(by_projection.values())))
        for left, right in combinations(PROJECTION_COUNTS, 2):
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
                    }
                )
    return pd.DataFrame(rows, columns=STABILITY_COLUMNS).sort_values(
        [
            "dataset",
            "seed",
            "fold",
            "projection_count_left",
            "projection_count_right",
            "selected_features",
        ],
        kind="stable",
        ignore_index=True,
    )


def build_ranking_stability(rankings: pd.DataFrame) -> pd.DataFrame:
    """Compare complete ranking positions once per matched fold and pair."""
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
        if len(group) != len(PROJECTION_COUNTS) or set(by_projection) != set(PROJECTION_COUNTS):
            raise ValueError(
                f"incomplete RDC projection inventory for {dataset}, seed={seed}, fold={fold}"
            )
        positions = {
            projection_count: np.argsort(ranking)
            for projection_count, ranking in by_projection.items()
        }
        for left, right in combinations(PROJECTION_COUNTS, 2):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "fold": fold,
                    "projection_count_left": left,
                    "projection_count_right": right,
                    "spearman_complete_ranking": float(
                        spearmanr(positions[left], positions[right]).statistic
                    ),
                }
            )
    return pd.DataFrame(rows, columns=RANKING_STABILITY_COLUMNS).sort_values(
        [
            "dataset",
            "seed",
            "fold",
            "projection_count_left",
            "projection_count_right",
        ],
        kind="stable",
        ignore_index=True,
    )


def build_comparison_summary(
    rankings: pd.DataFrame,
    metrics: pd.DataFrame,
    stability: pd.DataFrame,
    ranking_stability: pd.DataFrame,
    *,
    datasets: tuple[str, ...],
) -> pd.DataFrame:
    """Compare 10 and 20 projections with paired descriptive summaries."""
    comparison_rows: list[dict[str, Any]] = []
    metric_index = ["dataset", "seed", "fold", "k", "downstream_model"]
    paired_metrics = metrics.pivot(
        index=metric_index,
        columns="projection_count",
        values="balanced_accuracy",
    )
    n_features = rankings.groupby("dataset")["n_features"].first().astype(int).to_dict()
    for dataset_name in datasets:
        selection_times = rankings[
            (rankings["dataset"] == dataset_name) & (rankings["projection_count"].isin((10, 20)))
        ].pivot(
            index=["seed", "fold"],
            columns="projection_count",
            values="selection_seconds",
        )
        runtime_ratios = selection_times[20] / selection_times[10]
        metric_rows = paired_metrics.xs(dataset_name, level="dataset")
        metric_rows = metric_rows[
            metric_rows.index.get_level_values("k") < n_features[dataset_name]
        ]
        metric_differences = metric_rows[20] - metric_rows[10]
        fold_metric_differences = metric_differences.groupby(["seed", "fold"]).mean()
        stability_rows = stability[
            (stability["dataset"] == dataset_name)
            & (stability["projection_count_left"] == 10)
            & (stability["projection_count_right"] == 20)
            & (stability["selected_features"] < n_features[dataset_name])
        ]
        fold_overlaps = stability_rows.groupby(["seed", "fold"])["overlap_fraction"].mean()
        ranking_rows = ranking_stability[
            (ranking_stability["dataset"] == dataset_name)
            & (ranking_stability["projection_count_left"] == 10)
            & (ranking_stability["projection_count_right"] == 20)
        ]["spearman_complete_ranking"]
        reduced_feature_counts = sorted(
            int(value) for value in metric_rows.index.get_level_values("k").unique()
        )
        if (
            metric_rows.empty
            or stability_rows.empty
            or ranking_rows.empty
            or len(runtime_ratios) != len(fold_metric_differences)
            or len(runtime_ratios) != len(fold_overlaps)
            or len(runtime_ratios) != len(ranking_rows)
        ):
            raise RuntimeError(f"{dataset_name} has no reduced-feature comparison rows")
        comparison_rows.append(
            {
                "dataset": dataset_name,
                "projection_count_reference": 10,
                "projection_count_comparison": 20,
                "fold_pairs": len(runtime_ratios),
                "reduced_feature_counts": ";".join(map(str, reduced_feature_counts)),
                "runtime_ratio_median": runtime_ratios.median(),
                "runtime_ratio_mean": runtime_ratios.mean(),
                "runtime_ratio_std": runtime_ratios.std(),
                "runtime_ratio_min": runtime_ratios.min(),
                "runtime_ratio_max": runtime_ratios.max(),
                "balanced_accuracy_pairs": len(metric_differences),
                "balanced_accuracy_mean_difference_reduced_k": metric_differences.mean(),
                "balanced_accuracy_fold_difference_std": fold_metric_differences.std(),
                "selected_set_overlap_mean_reduced_k": stability_rows["overlap_fraction"].mean(),
                "selected_set_overlap_fold_std": fold_overlaps.std(),
                "complete_ranking_spearman_mean": ranking_rows.mean(),
                "complete_ranking_spearman_std": ranking_rows.std(),
            }
        )
    return pd.DataFrame(comparison_rows, columns=COMPARISON_SUMMARY_COLUMNS)


def build_results(
    rankings: pd.DataFrame,
    metrics: pd.DataFrame,
    settings: RdcSensitivitySettings,
) -> dict[str, pd.DataFrame]:
    """Build every raw and aggregate table for one complete study."""
    stability = build_stability(rankings)
    ranking_stability = build_ranking_stability(rankings)
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
    )[["overlap_fraction", "jaccard"]].mean()
    ranking_stability_summary = ranking_stability.groupby(
        [
            "dataset",
            "projection_count_left",
            "projection_count_right",
        ],
        as_index=False,
    ).agg(
        spearman_complete_ranking_mean=("spearman_complete_ranking", "mean"),
        spearman_complete_ranking_std=("spearman_complete_ranking", "std"),
        fold_pairs=("spearman_complete_ranking", "count"),
    )
    results = {
        "rankings": rankings.reset_index(drop=True),
        "metrics": metrics.reset_index(drop=True),
        "timing_summary": timing.loc[:, TIMING_SUMMARY_COLUMNS].sort_values(
            ["dataset", "projection_count"],
            kind="stable",
            ignore_index=True,
        ),
        "performance_summary": performance.loc[:, PERFORMANCE_SUMMARY_COLUMNS].sort_values(
            ["dataset", "k", "downstream_model", "projection_count"],
            kind="stable",
            ignore_index=True,
        ),
        "stability": stability,
        "stability_summary": stability_summary.loc[:, STABILITY_SUMMARY_COLUMNS].sort_values(
            [
                "dataset",
                "projection_count_left",
                "projection_count_right",
                "selected_features",
            ],
            kind="stable",
            ignore_index=True,
        ),
        "ranking_stability": ranking_stability,
        "ranking_stability_summary": ranking_stability_summary.loc[
            :, RANKING_STABILITY_SUMMARY_COLUMNS
        ].sort_values(
            ["dataset", "projection_count_left", "projection_count_right"],
            kind="stable",
            ignore_index=True,
        ),
        "comparison_summary": build_comparison_summary(
            rankings,
            metrics,
            stability,
            ranking_stability,
            datasets=settings.datasets,
        ),
    }
    return results


def _require_schema(name: str, frame: pd.DataFrame) -> None:
    expected = RESULT_SCHEMAS[name]
    observed = tuple(frame.columns)
    if observed != expected:
        raise ValueError(
            f"RDC sensitivity {name} schema differs: expected={expected}, observed={observed}"
        )


def _dataset_inventory(
    settings: RdcSensitivitySettings,
) -> dict[str, StudyDataset]:
    return {dataset_name: _load_study_dataset(dataset_name) for dataset_name in settings.datasets}


def _validate_primary_results(
    rankings: pd.DataFrame,
    metrics: pd.DataFrame,
    settings: RdcSensitivitySettings,
) -> None:
    datasets = _dataset_inventory(settings)
    expected_ranking_keys = {
        (dataset, seed, fold, projection_count)
        for dataset in settings.datasets
        for seed in settings.seeds
        for fold in range(settings.folds)
        for projection_count in settings.projection_counts
    }
    observed_ranking_keys = {
        (
            str(row.dataset),
            int(row.seed),
            int(row.fold),
            int(row.projection_count),
        )
        for row in rankings.itertuples(index=False)
    }
    if observed_ranking_keys != expected_ranking_keys or len(rankings) != len(
        expected_ranking_keys
    ):
        missing = sorted(expected_ranking_keys.difference(observed_ranking_keys))
        extra = sorted(observed_ranking_keys.difference(expected_ranking_keys))
        raise ValueError(
            f"RDC sensitivity ranking inventory differs: missing={missing}, extra={extra}"
        )
    if rankings.duplicated(["dataset", "seed", "fold", "projection_count"]).any():
        raise ValueError("RDC sensitivity rankings contain duplicate cells")
    for row in rankings.itertuples(index=False):
        dataset = datasets[str(row.dataset)]
        if row.task != TASK:
            raise ValueError("RDC sensitivity ranking task differs")
        if row.dataset_sha256 != dataset.sha256:
            raise ValueError(f"RDC sensitivity dataset hash differs for {row.dataset}")
        if int(row.n_samples) != dataset.n_samples or int(row.n_features) != dataset.n_features:
            raise ValueError(f"RDC sensitivity dataset dimensions differ for {row.dataset}")
        if int(row.projected_columns) != 2 * int(row.projection_count):
            raise ValueError("RDC sensitivity projected-column count differs")
        if not np.isfinite(float(row.selection_seconds)) or float(row.selection_seconds) <= 0.0:
            raise ValueError("RDC sensitivity selection times must be finite and positive")
        ranking = json.loads(str(row.feature_ranking))
        if (
            not isinstance(ranking, list)
            or len(ranking) != dataset.n_features
            or any(not isinstance(value, int) or isinstance(value, bool) for value in ranking)
            or set(ranking) != set(range(dataset.n_features))
        ):
            raise ValueError("RDC sensitivity feature ranking is not a complete permutation")

    expected_metric_keys = {
        (dataset_name, seed, fold, projection_count, selected, model)
        for dataset_name, dataset in datasets.items()
        for seed in settings.seeds
        for fold in range(settings.folds)
        for projection_count in settings.projection_counts
        for selected in get_requested_evaluation_k_values(dataset.n_features)
        for model in CLF_DOWNSTREAM_MODELS
    }
    observed_metric_keys = {
        (
            str(row.dataset),
            int(row.seed),
            int(row.fold),
            int(row.projection_count),
            int(row.k),
            str(row.downstream_model),
        )
        for row in metrics.itertuples(index=False)
    }
    if observed_metric_keys != expected_metric_keys or len(metrics) != len(expected_metric_keys):
        missing_metrics = sorted(expected_metric_keys.difference(observed_metric_keys))
        extra_metrics = sorted(observed_metric_keys.difference(expected_metric_keys))
        raise ValueError(
            "RDC sensitivity metric inventory differs: "
            f"missing={missing_metrics}, extra={extra_metrics}"
        )
    if metrics.duplicated(
        [
            "dataset",
            "seed",
            "fold",
            "projection_count",
            "k",
            "downstream_model",
        ]
    ).any():
        raise ValueError("RDC sensitivity metrics contain duplicate cells")
    if not metrics["task"].eq(TASK).all():
        raise ValueError("RDC sensitivity metric task differs")
    if not metrics["projected_columns"].eq(2 * metrics["projection_count"]).all():
        raise ValueError("RDC sensitivity metric projected-column count differs")
    if not metrics["n_features_selected"].eq(metrics["k"]).all():
        raise ValueError("RDC sensitivity selected-feature count differs from k")
    score_columns = (
        "accuracy",
        "f1",
        "f1_macro",
        "balanced_accuracy",
        "roc_auc",
        "auc",
    )
    scores = metrics.loc[:, score_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all() or not ((scores >= 0.0) & (scores <= 1.0)).all():
        raise ValueError("RDC sensitivity metric scores must be finite probabilities")
    if not np.allclose(
        metrics["roc_auc"].to_numpy(dtype=np.float64),
        metrics["auc"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("RDC sensitivity AUC aliases differ")


def _require_frame_equal(
    name: str,
    observed: pd.DataFrame,
    expected: pd.DataFrame,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise ValueError(f"RDC sensitivity {name} differs from its raw-cell aggregates") from error


def validate_results(
    results: Mapping[str, pd.DataFrame],
    settings: RdcSensitivitySettings,
) -> None:
    """Validate schemas, complete cell inventories, and all derived summaries."""
    if set(results) != set(RESULT_SCHEMAS):
        missing = sorted(set(RESULT_SCHEMAS).difference(results))
        extra = sorted(set(results).difference(RESULT_SCHEMAS))
        raise ValueError(
            f"RDC sensitivity table inventory differs: missing={missing}, extra={extra}"
        )
    for name, frame in results.items():
        _require_schema(name, frame)
    rankings = results["rankings"]
    metrics = results["metrics"]
    _validate_primary_results(rankings, metrics, settings)
    expected = build_results(rankings, metrics, settings)
    for name in RESULT_SCHEMAS:
        if name in {"rankings", "metrics"}:
            continue
        _require_frame_equal(name, results[name], expected[name])


def _dataset_metadata(
    settings: RdcSensitivitySettings,
) -> dict[str, dict[str, object]]:
    return {
        name: {
            "sha256": dataset.sha256,
            "n_samples": dataset.n_samples,
            "n_features": dataset.n_features,
            "source": dataset.source,
        }
        for name, dataset in _dataset_inventory(settings).items()
    }


def _read_result_table(name: str, path: Path) -> pd.DataFrame:
    if name == "comparison_summary":
        return pd.read_csv(path, dtype={"reduced_feature_counts": "string"})
    return pd.read_csv(path)


def write_results(
    results: Mapping[str, pd.DataFrame],
    output_dir: Path,
    *,
    profile: Profile,
    base_seed: int,
    elapsed_seconds: float,
    git_sha: str,
    git_dirty: bool,
    source_sha256: Mapping[str, str],
) -> Path:
    """Atomically publish one complete RDC sensitivity result tree."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"RDC sensitivity output already exists: {output_dir}")
    if profile == "full" and git_dirty:
        raise RuntimeError("The full RDC sensitivity profile requires a clean source tree")
    settings = _settings(profile)
    validate_results(results, settings)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-",
        dir=output_dir.parent,
    ) as temporary:
        staging = Path(temporary) / output_dir.name
        staging.mkdir()
        artifacts: list[Path] = []
        for name, filename in RESULT_FILENAMES.items():
            path = staging / filename
            results[name].to_csv(path, index=False)
            artifacts.append(path)
        serialized = {
            name: _read_result_table(name, staging / filename)
            for name, filename in RESULT_FILENAMES.items()
        }
        validate_results(serialized, settings)
        if profile == "full" and (
            _git_sha() != git_sha or _git_dirty() or _source_hashes() != dict(source_sha256)
        ):
            raise RuntimeError("Source tree changed during the full RDC sensitivity run")
        dataset_metadata = _dataset_metadata(settings)
        receipt = {
            "analysis": "rdc_sensitivity",
            "profile": profile,
            "base_seed": base_seed,
            "settings": asdict(settings),
            "datasets": dataset_metadata,
            "created_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "git_sha": git_sha,
            "git_dirty": git_dirty,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hardware": get_hardware_metadata(),
            "source_sha256": dict(source_sha256),
            "input_sha256": {
                name: metadata["sha256"] for name, metadata in dataset_metadata.items()
            },
            "versions": _versions(),
            "selector": {
                "alpha": SELECTOR_ALPHA,
                "early_stopping": SELECTOR_EARLY_STOPPING,
                "method": SELECTOR_METHOD,
                "n_resamples": SELECTOR_N_RESAMPLES,
                "projection_counts": list(settings.projection_counts),
                "random_state": "seed * 1000 + fold",
            },
            "preprocessing": "StandardScaler fit on each training fold",
            "downstream_models": list(CLF_DOWNSTREAM_MODELS),
            "timing": {
                "clock": "time.perf_counter",
                "projection_order": list(settings.projection_counts),
                "scope": "permutation_selector call",
            },
            "tables": {
                name: {
                    "artifact": RESULT_FILENAMES[name],
                    "rows": len(results[name]),
                    "columns": list(results[name].columns),
                }
                for name in RESULT_FILENAMES
            },
            "selection_seconds_total": float(results["rankings"]["selection_seconds"].sum()),
            "artifacts": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in sorted(artifacts)
            },
        }
        (staging / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        staging.rename(output_dir)
    return output_dir


def execute(
    profile: Profile,
    output_dir: Path,
    *,
    base_seed: int,
) -> Path:
    """Run, validate, and atomically publish one fixed replication profile."""
    settings = _settings(profile)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"RDC sensitivity output already exists: {output_dir}")
    starting_git_sha = _git_sha()
    starting_git_dirty = _git_dirty()
    starting_source_hashes = _source_hashes()
    if profile == "full" and starting_git_dirty:
        raise RuntimeError("The full RDC sensitivity profile requires a clean source tree")
    started = time.perf_counter()
    rankings, metrics = run_sensitivity(settings)
    results = build_results(rankings, metrics, settings)
    validate_results(results, settings)
    if profile == "full" and (
        _git_sha() != starting_git_sha or _git_dirty() or _source_hashes() != starting_source_hashes
    ):
        raise RuntimeError("Source tree changed during the full RDC sensitivity run")
    return write_results(
        results,
        output_dir,
        profile=profile,
        base_seed=base_seed,
        elapsed_seconds=time.perf_counter() - started,
        git_sha=starting_git_sha,
        git_dirty=starting_git_dirty,
        source_sha256=starting_source_hashes,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("smoke", "quick", "full"),
        default="quick",
        help="RDC sensitivity workload.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="New directory for result tables and receipt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1718,
        help="Top-level replication seed recorded in the receipt.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = execute(
        args.profile,
        args.output_dir,
        base_seed=args.seed,
    )
    print(f"Wrote verified RDC sensitivity {args.profile} artifacts to {output}.")


if __name__ == "__main__":
    main()
