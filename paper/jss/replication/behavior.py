"""Compare fitted citrees and partykit behavior on shared data splits.

The comparison fits classification and regression trees and forests on the same
training folds with the same integer seeds. It retains roots, native feature
summaries, held-out predictions, and exact same-seed refit checks.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict, cast

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import (
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.utils import Bunch

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
)
from citrees._tree import Node
from paper.benchmark.pipeline.r_methods import (
    RCForestBehavior,
    RCTreeBehavior,
    get_r_runtime_versions,
    r_cforest_behavior,
    r_ctree_behavior,
)

Task = Literal["classification", "regression"]
Profile = Literal["smoke", "quick", "full"]
ModelFamily = Literal["tree", "forest"]
Method = Literal["citrees", "partykit"]
METHODS: tuple[Method, ...] = ("citrees", "partykit")
MODEL_FAMILIES: tuple[ModelFamily, ...] = ("tree", "forest")

BASE_SEED = 1718
ALPHA = 0.05
MAX_DEPTH = 3
MIN_SAMPLES_SPLIT = 20
MIN_SAMPLES_LEAF = 7
TOP_K = 5
PARTYKIT_TESTTYPE = ("Bonferroni", "MonteCarlo")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "behavior"
REPO_ROOT = Path(__file__).resolve().parents[3]


class CitreesControl(TypedDict):
    """Shared typed controls for citrees trees and forests."""

    alpha_selector: float
    adjust_alpha_selector: bool
    n_resamples_selector: str | int | None
    early_stopping_selector: str | None
    n_resamples_splitter: str | int | None
    adjust_alpha_splitter: bool
    early_stopping_splitter: str | None
    feature_muting: bool
    feature_scanning: bool
    threshold_scanning: bool
    threshold_method: str
    max_thresholds: str | float | int | None
    max_depth: int | None
    min_samples_split: int
    min_samples_leaf: int
    random_state: int | None
    verbose: int


class CitreesForestControl(CitreesControl):
    """Additional typed controls for citrees forests."""

    n_estimators: int
    max_features: str | float | int | None
    bootstrap: bool
    max_samples: int | float | None
    n_jobs: int | None


class PartykitControl(TypedDict):
    """Shared typed controls for partykit trees and forests."""

    task: str
    teststat: str
    testtype: str | tuple[str, ...]
    mincriterion: float
    nresample: int
    maxdepth: int | None
    minsplit: int
    minbucket: int
    random_state: int


class PredictionMetrics(TypedDict):
    """Typed paired prediction metrics for one held-out fold."""

    prediction_metric: str
    citrees_prediction_score: float
    partykit_prediction_score: float
    prediction_agreement: float
    prediction_correlation: float
    prediction_mean_absolute_difference: float
    citrees_mean_absolute_error: float
    partykit_mean_absolute_error: float
    probability_mean_absolute_difference: float
    probability_spearman_correlation: float
    citrees_roc_auc: float
    partykit_roc_auc: float
    citrees_log_loss: float
    partykit_log_loss: float


BEHAVIOR_RESULT_SCHEMAS: dict[str, tuple[str, ...]] = {
    "behavior_fit_raw": (
        "task",
        "dataset",
        "dataset_sha256",
        "model_family",
        "method",
        "repeat",
        "fold",
        "split_seed",
        "model_seed",
        "train_ids_sha256",
        "test_ids_sha256",
        "n_train",
        "n_test",
        "n_features",
        "root_feature",
        "native_output_reproducible",
        "predictions_reproducible",
    ),
    "behavior_feature_raw": (
        "task",
        "dataset",
        "dataset_sha256",
        "model_family",
        "method",
        "repeat",
        "fold",
        "model_seed",
        "feature_id",
        "feature_name",
        "feature_definition",
        "feature_value",
        "feature_value_missing",
        "ranking_value",
        "rank",
        "is_root",
    ),
    "behavior_prediction_raw": (
        "task",
        "dataset",
        "dataset_sha256",
        "model_family",
        "method",
        "repeat",
        "fold",
        "model_seed",
        "sample_id",
        "y_true",
        "prediction",
    ),
    "behavior_probability_raw": (
        "task",
        "dataset",
        "dataset_sha256",
        "model_family",
        "method",
        "repeat",
        "fold",
        "model_seed",
        "sample_id",
        "class_label",
        "probability",
    ),
    "behavior_fold_summary": (
        "task",
        "dataset",
        "dataset_sha256",
        "model_family",
        "repeat",
        "fold",
        "model_seed",
        "n_test",
        "split_decision_agreement",
        "root_agreement_given_both_split",
        "root_comparison",
        "citrees_no_split",
        "partykit_no_split",
        "citrees_native_ranking_informative",
        "partykit_native_ranking_informative",
        "both_native_rankings_informative",
        "native_ranking_kendall_tau_b",
        "top_k",
        "native_top_k_with_ties_jaccard",
        "prediction_metric",
        "citrees_prediction_score",
        "partykit_prediction_score",
        "prediction_agreement",
        "prediction_correlation",
        "prediction_mean_absolute_difference",
        "citrees_mean_absolute_error",
        "partykit_mean_absolute_error",
        "probability_mean_absolute_difference",
        "probability_spearman_correlation",
        "citrees_roc_auc",
        "partykit_roc_auc",
        "citrees_log_loss",
        "partykit_log_loss",
        "citrees_performance_advantage",
        "citrees_native_output_reproducible",
        "citrees_predictions_reproducible",
        "partykit_native_output_reproducible",
        "partykit_predictions_reproducible",
    ),
    "behavior_summary": (
        "task",
        "dataset",
        "model_family",
        "metric",
        "estimate",
        "lower_95",
        "upper_95",
        "n_fold_values",
        "n_repeats",
        "bootstrap_resamples",
    ),
}
BEHAVIOR_RAW_TABLES = tuple(name for name in BEHAVIOR_RESULT_SCHEMAS if name != "behavior_summary")
BEHAVIOR_STORAGE_DTYPES: dict[str, dict[str, str | type[object]]] = {
    "behavior_fold_summary": {"root_comparison": object},
    "behavior_probability_raw": {
        "task": "str",
        "dataset": "str",
        "dataset_sha256": "str",
        "model_family": "str",
        "method": "str",
        "repeat": "int64",
        "fold": "int64",
        "model_seed": "int64",
        "sample_id": "int64",
        "class_label": "int64",
        "probability": "float64",
    },
}
_FLOAT_COMPARISON_TOLERANCE = 1e-12

SUMMARY_METRICS = (
    "split_decision_agreement",
    "root_agreement_given_both_split",
    "citrees_native_ranking_informative",
    "partykit_native_ranking_informative",
    "both_native_rankings_informative",
    "native_ranking_kendall_tau_b",
    "native_top_k_with_ties_jaccard",
    "citrees_prediction_score",
    "partykit_prediction_score",
    "citrees_performance_advantage",
    "prediction_agreement",
    "prediction_correlation",
    "prediction_mean_absolute_difference",
    "citrees_mean_absolute_error",
    "partykit_mean_absolute_error",
    "probability_mean_absolute_difference",
    "probability_spearman_correlation",
    "citrees_roc_auc",
    "partykit_roc_auc",
    "citrees_log_loss",
    "partykit_log_loss",
)


def normalize_behavior_table(
    table_name: str,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the frozen in-memory dtypes used for Parquet serialization."""
    normalized = frame.copy()
    for column, dtype in BEHAVIOR_STORAGE_DTYPES.get(table_name, {}).items():
        normalized[column] = normalized[column].astype(dtype)
    return normalized


@dataclass(frozen=True)
class BehaviorSettings:
    """Workload and model controls for one replication profile."""

    n_splits: int
    n_repeats: int
    n_resamples: int
    n_trees: int
    importance_permutations: int
    summary_resamples: int


@dataclass(frozen=True, order=True)
class BehaviorCell:
    """One paired implementation comparison on one held-out partition."""

    task: Task
    dataset: str
    model_family: ModelFamily
    split_index: int
    repeat: int
    fold: int


@dataclass(frozen=True)
class DatasetSpec:
    """One fixed local dataset and its content identity."""

    task: Task
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class ModelBehavior:
    """Common representation for one fitted implementation."""

    root_feature: int | None
    feature_values: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray | None
    classes: np.ndarray | None


def _settings(profile: Profile) -> BehaviorSettings:
    if profile == "smoke":
        return BehaviorSettings(
            n_splits=2,
            n_repeats=1,
            n_resamples=39,
            n_trees=5,
            importance_permutations=1,
            summary_resamples=99,
        )
    if profile == "quick":
        return BehaviorSettings(
            n_splits=3,
            n_repeats=1,
            n_resamples=199,
            n_trees=25,
            importance_permutations=3,
            summary_resamples=999,
        )
    if profile == "full":
        return BehaviorSettings(
            n_splits=5,
            n_repeats=10,
            n_resamples=999,
            n_trees=100,
            importance_permutations=10,
            summary_resamples=9_999,
        )
    raise ValueError(f"unknown behavior profile: {profile}")


def _stream_seed(base_seed: int, *parts: object) -> int:
    """Derive one deterministic uint32 seed from a named stream."""
    key = "__".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("ascii")).digest()
    sequence = np.random.SeedSequence(
        [base_seed, int.from_bytes(digest[:4], byteorder="little", signed=False)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _array_sha256(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...],
) -> str:
    """Hash array shape, dtype, values, and feature names."""
    digest = hashlib.sha256()
    for array in (np.ascontiguousarray(X), np.ascontiguousarray(y)):
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    digest.update("\0".join(feature_names).encode("utf-8"))
    return digest.hexdigest()


def _index_sha256(indices: np.ndarray) -> str:
    """Hash one sample-index set in a platform-independent form."""
    values = np.sort(np.asarray(indices, dtype="<i8"))
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def load_behavior_datasets() -> tuple[DatasetSpec, DatasetSpec]:
    """Load the fixed bundled classification and regression datasets."""
    breast = load_breast_cancer()
    diabetes = load_diabetes()

    def build_dataset(task: Task, name: str, bunch: Bunch) -> DatasetSpec:
        X = np.asarray(bunch.data, dtype=np.float64)
        y = np.asarray(
            bunch.target,
            dtype=np.int64 if task == "classification" else np.float64,
        )
        feature_names = tuple(str(value) for value in bunch.feature_names)
        return DatasetSpec(
            task=task,
            name=name,
            X=X,
            y=y,
            feature_names=feature_names,
            sha256=_array_sha256(X, y, feature_names),
        )

    return (
        build_dataset(
            "classification",
            "breast_cancer_wisconsin_diagnostic",
            breast,
        ),
        build_dataset("regression", "diabetes", diabetes),
    )


def _tree_split_counts(tree: Node, n_features: int) -> np.ndarray:
    """Count split-variable usage in a fitted citrees tree."""
    counts: np.ndarray = np.zeros(n_features, dtype=np.float64)
    stack = [tree]
    while stack:
        node = stack.pop()
        if "feature" not in node:
            continue
        feature = int(node["feature"])
        if feature < 0 or feature >= n_features:
            raise ValueError(f"citrees returned an invalid feature index: {feature}")
        counts[feature] += 1
        stack.append(node["right_child"])
        stack.append(node["left_child"])
    return counts


def _fit_citrees(
    task: Task,
    family: ModelFamily,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    settings: BehaviorSettings,
    seed: int,
) -> ModelBehavior:
    """Fit one citrees model under frozen structural controls."""
    common: CitreesControl = {
        "alpha_selector": ALPHA,
        "adjust_alpha_selector": True,
        "n_resamples_selector": settings.n_resamples,
        "early_stopping_selector": None,
        "n_resamples_splitter": None,
        "adjust_alpha_splitter": False,
        "early_stopping_splitter": None,
        "feature_muting": False,
        "feature_scanning": False,
        "threshold_scanning": False,
        "threshold_method": "exact",
        "max_thresholds": None,
        "max_depth": MAX_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "random_state": seed,
        "verbose": 0,
    }
    if family == "tree":
        model = (
            ConditionalInferenceTreeClassifier(selector="mc", **common)
            if task == "classification"
            else ConditionalInferenceTreeRegressor(selector="pc", **common)
        )
        model.fit(X_train, y_train)
        root = -1 if "feature" not in model.tree_ else int(model.tree_["feature"])
        return ModelBehavior(
            root_feature=root,
            feature_values=_tree_split_counts(model.tree_, X_train.shape[1]),
            predictions=np.asarray(model.predict(X_test)),
            probabilities=(
                np.asarray(model.predict_proba(X_test), dtype=np.float64)
                if task == "classification"
                else None
            ),
            classes=(
                np.asarray(model.classes_, dtype=np.int64) if task == "classification" else None
            ),
        )

    forest_common: CitreesForestControl = {
        **common,
        "n_estimators": settings.n_trees,
        "max_features": None,
        "bootstrap": True,
        "max_samples": None,
        "n_jobs": 1,
    }
    model = (
        ConditionalInferenceForestClassifier(
            selector="mc",
            sampling_method=None,
            **forest_common,
        )
        if task == "classification"
        else ConditionalInferenceForestRegressor(
            selector="pc",
            **forest_common,
        )
    )
    model.fit(X_train, y_train)
    return ModelBehavior(
        root_feature=None,
        feature_values=np.asarray(model.feature_importances_, dtype=np.float64),
        predictions=np.asarray(model.predict(X_test)),
        probabilities=(
            np.asarray(model.predict_proba(X_test), dtype=np.float64)
            if task == "classification"
            else None
        ),
        classes=(np.asarray(model.classes_, dtype=np.int64) if task == "classification" else None),
    )


def _fit_partykit(
    task: Task,
    family: ModelFamily,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    settings: BehaviorSettings,
    seed: int,
) -> ModelBehavior:
    """Fit one partykit model under the matched structural controls."""
    common: PartykitControl = {
        "task": task,
        "teststat": "quadratic",
        "testtype": PARTYKIT_TESTTYPE,
        "mincriterion": 1.0 - ALPHA,
        "nresample": settings.n_resamples,
        "maxdepth": MAX_DEPTH,
        "minsplit": MIN_SAMPLES_SPLIT,
        "minbucket": MIN_SAMPLES_LEAF,
        "random_state": seed,
    }
    if family == "tree":
        result: RCTreeBehavior = r_ctree_behavior(
            X_train,
            y_train,
            X_test,
            **common,
        )
        return ModelBehavior(
            root_feature=result.root_feature,
            feature_values=np.asarray(result.split_counts, dtype=np.float64),
            predictions=np.asarray(result.predictions),
            probabilities=result.probabilities,
            classes=result.classes,
        )

    result_forest: RCForestBehavior = r_cforest_behavior(
        X_train,
        y_train,
        X_test,
        ntree=settings.n_trees,
        mtry="all",
        replace=True,
        fraction=1.0,
        varimp_conditional=False,
        varimp_nperm=settings.importance_permutations,
        cores=1,
        **common,
    )
    return ModelBehavior(
        root_feature=None,
        feature_values=np.asarray(result_forest.importances, dtype=np.float64),
        predictions=np.asarray(result_forest.predictions),
        probabilities=result_forest.probabilities,
        classes=result_forest.classes,
    )


def _fit_method(
    method: Method,
    task: Task,
    family: ModelFamily,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    settings: BehaviorSettings,
    seed: int,
) -> ModelBehavior:
    if method == "citrees":
        return _fit_citrees(
            task,
            family,
            X_train,
            y_train,
            X_test,
            settings=settings,
            seed=seed,
        )
    return _fit_partykit(
        task,
        family,
        X_train,
        y_train,
        X_test,
        settings=settings,
        seed=seed,
    )


def _ranking_values(values: np.ndarray) -> np.ndarray:
    """Map omitted native importance values to explicit structural zeros."""
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or np.isinf(scores).any():
        raise ValueError("feature values must be one-dimensional and non-infinite")
    return np.where(np.isnan(scores), 0.0, scores)


def _tie_ranks(values: np.ndarray) -> np.ndarray:
    """Return zero-based competition ranks while preserving score ties."""
    ranked = np.asarray(
        rankdata(-_ranking_values(values), method="min"),
        dtype=np.int64,
    )
    return ranked - 1


def _top_k_with_ties(values: np.ndarray, k: int) -> set[int]:
    """Return feature IDs at or above the kth score, including ties."""
    scores = _ranking_values(values)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    resolved_k = min(k, len(scores))
    threshold = np.partition(scores, len(scores) - resolved_k)[len(scores) - resolved_k]
    return set(np.flatnonzero(scores >= threshold))


def _native_ranking_informative(values: np.ndarray) -> bool:
    """Return whether a native feature summary distinguishes any features."""
    return bool(np.unique(_ranking_values(values)).size > 1)


def _same_behavior(first: ModelBehavior, second: ModelBehavior) -> tuple[bool, bool]:
    """Compare same-seed reported model outputs and predictions exactly."""
    native_output = first.root_feature == second.root_feature and np.array_equal(
        first.feature_values,
        second.feature_values,
        equal_nan=True,
    )
    hard_predictions = np.array_equal(
        first.predictions,
        second.predictions,
        equal_nan=True,
    )
    classes = (
        first.classes is None
        and second.classes is None
        or first.classes is not None
        and second.classes is not None
        and np.array_equal(first.classes, second.classes)
    )
    probabilities = (
        first.probabilities is None
        and second.probabilities is None
        or first.probabilities is not None
        and second.probabilities is not None
        and np.array_equal(first.probabilities, second.probabilities, equal_nan=True)
    )
    return native_output, hard_predictions and classes and probabilities


def _prediction_metrics(
    task: Task,
    y_true: np.ndarray,
    citrees_behavior: ModelBehavior,
    partykit_behavior: ModelBehavior,
) -> PredictionMetrics:
    """Return task-appropriate paired prediction metrics."""
    citrees_predictions = citrees_behavior.predictions
    partykit_predictions = partykit_behavior.predictions
    if task == "classification":
        if (
            citrees_behavior.probabilities is None
            or partykit_behavior.probabilities is None
            or citrees_behavior.classes is None
            or partykit_behavior.classes is None
            or not np.array_equal(citrees_behavior.classes, partykit_behavior.classes)
        ):
            raise ValueError("Classification behavior requires aligned class probabilities")
        classes = citrees_behavior.classes
        citrees_probabilities = citrees_behavior.probabilities
        partykit_probabilities = partykit_behavior.probabilities
        if len(classes) != 2:
            raise ValueError("The fixed behavior classification dataset must be binary")
        citrees_positive = citrees_probabilities[:, 1]
        partykit_positive = partykit_probabilities[:, 1]
        probability_correlation = (
            float(spearmanr(citrees_positive, partykit_positive).statistic)
            if np.std(citrees_positive) > 0 and np.std(partykit_positive) > 0
            else float("nan")
        )
        return {
            "prediction_metric": "balanced_accuracy",
            "citrees_prediction_score": float(balanced_accuracy_score(y_true, citrees_predictions)),
            "partykit_prediction_score": float(
                balanced_accuracy_score(y_true, partykit_predictions)
            ),
            "prediction_agreement": float(np.mean(citrees_predictions == partykit_predictions)),
            "prediction_correlation": float("nan"),
            "prediction_mean_absolute_difference": float("nan"),
            "citrees_mean_absolute_error": float("nan"),
            "partykit_mean_absolute_error": float("nan"),
            "probability_mean_absolute_difference": float(
                np.mean(np.abs(citrees_probabilities - partykit_probabilities))
            ),
            "probability_spearman_correlation": probability_correlation,
            "citrees_roc_auc": float(roc_auc_score(y_true, citrees_positive)),
            "partykit_roc_auc": float(roc_auc_score(y_true, partykit_positive)),
            "citrees_log_loss": float(log_loss(y_true, citrees_probabilities, labels=classes)),
            "partykit_log_loss": float(log_loss(y_true, partykit_probabilities, labels=classes)),
        }

    correlation = (
        float(np.corrcoef(citrees_predictions, partykit_predictions)[0, 1])
        if np.std(citrees_predictions) > 0 and np.std(partykit_predictions) > 0
        else float("nan")
    )
    return {
        "prediction_metric": "root_mean_squared_error",
        "citrees_prediction_score": float(mean_squared_error(y_true, citrees_predictions) ** 0.5),
        "partykit_prediction_score": float(mean_squared_error(y_true, partykit_predictions) ** 0.5),
        "prediction_agreement": float("nan"),
        "prediction_correlation": correlation,
        "prediction_mean_absolute_difference": float(
            np.mean(np.abs(citrees_predictions - partykit_predictions))
        ),
        "citrees_mean_absolute_error": float(mean_absolute_error(y_true, citrees_predictions)),
        "partykit_mean_absolute_error": float(mean_absolute_error(y_true, partykit_predictions)),
        "probability_mean_absolute_difference": float("nan"),
        "probability_spearman_correlation": float("nan"),
        "citrees_roc_auc": float("nan"),
        "partykit_roc_auc": float("nan"),
        "citrees_log_loss": float("nan"),
        "partykit_log_loss": float("nan"),
    }


def _fold_pair_row(
    dataset: DatasetSpec,
    family: ModelFamily,
    repeat: int,
    fold: int,
    model_seed: int,
    y_test: np.ndarray,
    behaviors: dict[Method, ModelBehavior],
    reproducibility: dict[Method, tuple[bool, bool]],
) -> dict[str, object]:
    """Build one paired fold-level comparison row."""
    citrees_behavior = behaviors["citrees"]
    partykit_behavior = behaviors["partykit"]
    citrees_values = _ranking_values(citrees_behavior.feature_values)
    partykit_values = _ranking_values(partykit_behavior.feature_values)
    citrees_ranking_informative = _native_ranking_informative(citrees_values)
    partykit_ranking_informative = _native_ranking_informative(partykit_values)
    both_rankings_informative = citrees_ranking_informative and partykit_ranking_informative
    top_k = min(TOP_K, len(citrees_values))
    tau = float("nan")
    top_jaccard = float("nan")
    if both_rankings_informative:
        tau = float(
            kendalltau(
                citrees_values,
                partykit_values,
                variant="b",
            ).statistic
        )
        citrees_top = _top_k_with_ties(citrees_values, top_k)
        partykit_top = _top_k_with_ties(partykit_values, top_k)
        top_jaccard = len(citrees_top & partykit_top) / len(citrees_top | partykit_top)
    prediction_metrics = _prediction_metrics(
        dataset.task,
        y_test,
        citrees_behavior,
        partykit_behavior,
    )
    citrees_performance_advantage = (
        prediction_metrics["citrees_prediction_score"]
        - prediction_metrics["partykit_prediction_score"]
        if dataset.task == "classification"
        else prediction_metrics["partykit_prediction_score"]
        - prediction_metrics["citrees_prediction_score"]
    )
    root_comparison: str | object = pd.NA
    split_decision_agreement = float("nan")
    root_agreement_given_both_split = float("nan")
    if family == "tree":
        citrees_root = citrees_behavior.root_feature
        partykit_root = partykit_behavior.root_feature
        citrees_split = citrees_root != -1
        partykit_split = partykit_root != -1
        split_decision_agreement = float(citrees_split == partykit_split)
        if citrees_split and partykit_split:
            root_agreement_given_both_split = float(citrees_root == partykit_root)
        if citrees_root == -1 and partykit_root == -1:
            root_comparison = "both_no_split"
        elif citrees_root == -1:
            root_comparison = "partykit_only_split"
        elif partykit_root == -1:
            root_comparison = "citrees_only_split"
        elif citrees_root == partykit_root:
            root_comparison = "same_root"
        else:
            root_comparison = "different_root"
    return {
        "task": dataset.task,
        "dataset": dataset.name,
        "dataset_sha256": dataset.sha256,
        "model_family": family,
        "repeat": repeat,
        "fold": fold,
        "model_seed": model_seed,
        "n_test": len(y_test),
        "split_decision_agreement": split_decision_agreement,
        "root_agreement_given_both_split": root_agreement_given_both_split,
        "root_comparison": root_comparison,
        "citrees_no_split": (
            bool(citrees_behavior.root_feature == -1) if family == "tree" else pd.NA
        ),
        "partykit_no_split": (
            bool(partykit_behavior.root_feature == -1) if family == "tree" else pd.NA
        ),
        "citrees_native_ranking_informative": citrees_ranking_informative,
        "partykit_native_ranking_informative": partykit_ranking_informative,
        "both_native_rankings_informative": both_rankings_informative,
        "native_ranking_kendall_tau_b": tau,
        "top_k": top_k,
        "native_top_k_with_ties_jaccard": top_jaccard,
        **prediction_metrics,
        "citrees_performance_advantage": float(citrees_performance_advantage),
        "citrees_native_output_reproducible": reproducibility["citrees"][0],
        "citrees_predictions_reproducible": reproducibility["citrees"][1],
        "partykit_native_output_reproducible": reproducibility["partykit"][0],
        "partykit_predictions_reproducible": reproducibility["partykit"][1],
    }


def _splitter(
    dataset: DatasetSpec,
    settings: BehaviorSettings,
    seed: int,
) -> RepeatedStratifiedKFold | RepeatedKFold:
    if dataset.task == "classification":
        return RepeatedStratifiedKFold(
            n_splits=settings.n_splits,
            n_repeats=settings.n_repeats,
            random_state=seed,
        )
    return RepeatedKFold(
        n_splits=settings.n_splits,
        n_repeats=settings.n_repeats,
        random_state=seed,
    )


def _feature_definition(method: Method, family: ModelFamily) -> str:
    if family == "tree":
        return "split_count"
    return (
        "normalized_impurity_decrease" if method == "citrees" else "partykit_permutation_importance"
    )


def _run_dataset(
    dataset: DatasetSpec,
    settings: BehaviorSettings,
    *,
    base_seed: int,
    selected_cells: frozenset[tuple[int, ModelFamily]] | None = None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    fit_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    split_seed = _stream_seed(base_seed, dataset.task, dataset.name, "folds")
    splitter = _splitter(dataset, settings, split_seed)
    for split_index, (train_indices, test_indices) in enumerate(
        splitter.split(dataset.X, dataset.y)
    ):
        families = tuple(
            family
            for family in MODEL_FAMILIES
            if selected_cells is None or (split_index, family) in selected_cells
        )
        if not families:
            continue
        repeat, fold = divmod(split_index, settings.n_splits)
        X_train = dataset.X[train_indices]
        X_test = dataset.X[test_indices]
        y_train = dataset.y[train_indices]
        y_test = dataset.y[test_indices]
        train_ids_sha256 = _index_sha256(train_indices)
        test_ids_sha256 = _index_sha256(test_indices)
        for family in families:
            model_seed = _stream_seed(
                base_seed,
                dataset.task,
                dataset.name,
                family,
                repeat,
                fold,
            )
            behaviors: dict[Method, ModelBehavior] = {}
            reproducibility: dict[Method, tuple[bool, bool]] = {}
            for method in METHODS:
                first = _fit_method(
                    method,
                    dataset.task,
                    family,
                    X_train,
                    y_train,
                    X_test,
                    settings=settings,
                    seed=model_seed,
                )
                second = _fit_method(
                    method,
                    dataset.task,
                    family,
                    X_train,
                    y_train,
                    X_test,
                    settings=settings,
                    seed=model_seed,
                )
                behaviors[method] = first
                reproducibility[method] = _same_behavior(first, second)
                ranking_values = _ranking_values(first.feature_values)
                ranks = _tie_ranks(first.feature_values)
                fit_rows.append(
                    {
                        "task": dataset.task,
                        "dataset": dataset.name,
                        "dataset_sha256": dataset.sha256,
                        "model_family": family,
                        "method": method,
                        "repeat": repeat,
                        "fold": fold,
                        "split_seed": split_seed,
                        "model_seed": model_seed,
                        "train_ids_sha256": train_ids_sha256,
                        "test_ids_sha256": test_ids_sha256,
                        "n_train": len(train_indices),
                        "n_test": len(test_indices),
                        "n_features": dataset.X.shape[1],
                        "root_feature": (first.root_feature if family == "tree" else pd.NA),
                        "native_output_reproducible": reproducibility[method][0],
                        "predictions_reproducible": reproducibility[method][1],
                    }
                )
                for feature_id, feature_name in enumerate(dataset.feature_names):
                    value = float(first.feature_values[feature_id])
                    feature_rows.append(
                        {
                            "task": dataset.task,
                            "dataset": dataset.name,
                            "dataset_sha256": dataset.sha256,
                            "model_family": family,
                            "method": method,
                            "repeat": repeat,
                            "fold": fold,
                            "model_seed": model_seed,
                            "feature_id": feature_id,
                            "feature_name": feature_name,
                            "feature_definition": _feature_definition(method, family),
                            "feature_value": value,
                            "feature_value_missing": bool(np.isnan(value)),
                            "ranking_value": float(ranking_values[feature_id]),
                            "rank": int(ranks[feature_id]),
                            "is_root": (
                                bool(first.root_feature == feature_id)
                                if family == "tree"
                                else pd.NA
                            ),
                        }
                    )
                for position, sample_id in enumerate(test_indices):
                    prediction_rows.append(
                        {
                            "task": dataset.task,
                            "dataset": dataset.name,
                            "dataset_sha256": dataset.sha256,
                            "model_family": family,
                            "method": method,
                            "repeat": repeat,
                            "fold": fold,
                            "model_seed": model_seed,
                            "sample_id": int(sample_id),
                            "y_true": float(y_test[position]),
                            "prediction": float(first.predictions[position]),
                        }
                    )
                if dataset.task == "classification":
                    if first.probabilities is None or first.classes is None:
                        raise ValueError(
                            "Classification fits must return probabilities and class labels"
                        )
                    for position, sample_id in enumerate(test_indices):
                        for class_index, class_label in enumerate(first.classes):
                            probability_rows.append(
                                {
                                    "task": dataset.task,
                                    "dataset": dataset.name,
                                    "dataset_sha256": dataset.sha256,
                                    "model_family": family,
                                    "method": method,
                                    "repeat": repeat,
                                    "fold": fold,
                                    "model_seed": model_seed,
                                    "sample_id": int(sample_id),
                                    "class_label": int(class_label),
                                    "probability": float(
                                        first.probabilities[position, class_index]
                                    ),
                                }
                            )
            pair_rows.append(
                _fold_pair_row(
                    dataset,
                    family,
                    repeat,
                    fold,
                    model_seed,
                    y_test,
                    behaviors,
                    reproducibility,
                )
            )
    return fit_rows, feature_rows, prediction_rows, probability_rows, pair_rows


def _repeat_means(values: np.ndarray, repeats: np.ndarray) -> np.ndarray:
    """Average finite fold values within each observed partition repeat."""
    return np.asarray(
        [values[repeats == repeat].mean() for repeat in np.unique(repeats)],
        dtype=np.float64,
    )


def _repeat_bootstrap_interval(
    repeat_means: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap the mean across partition-repeat means."""
    if repeat_means.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        repeat_means,
        size=(n_resamples, len(repeat_means)),
        replace=True,
    ).mean(axis=1)
    lower, upper = np.quantile(sampled, [0.025, 0.975])
    return float(lower), float(upper)


def summarize_behavior(
    fold_rows: pd.DataFrame,
    settings: BehaviorSettings,
    *,
    base_seed: int,
) -> pd.DataFrame:
    """Summarize fold metrics with repeat-clustered bootstrap intervals."""
    rows: list[dict[str, object]] = []
    for keys, group in fold_rows.groupby(
        ["task", "dataset", "model_family"],
        sort=False,
    ):
        task, dataset, family = keys
        for metric in SUMMARY_METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            finite = np.isfinite(values)
            if not finite.any():
                continue
            observed = values[finite]
            repeats = group.loc[finite, "repeat"].to_numpy(dtype=np.int64)
            repeat_means = _repeat_means(observed, repeats)
            lower, upper = _repeat_bootstrap_interval(
                repeat_means,
                n_resamples=settings.summary_resamples,
                seed=_stream_seed(base_seed, task, dataset, family, metric, "summary"),
            )
            rows.append(
                {
                    "task": task,
                    "dataset": dataset,
                    "model_family": family,
                    "metric": metric,
                    "estimate": float(repeat_means.mean()),
                    "lower_95": lower,
                    "upper_95": upper,
                    "n_fold_values": len(observed),
                    "n_repeats": len(repeat_means),
                    "bootstrap_resamples": (
                        settings.summary_resamples if len(repeat_means) > 1 else 0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _require_integral_values(
    frame: pd.DataFrame,
    table_name: str,
    columns: tuple[str, ...],
    *,
    nullable: frozenset[str] = frozenset(),
) -> None:
    """Require semantic integers before any validation-time casts."""
    numeric_types = (int, float, np.integer, np.floating)
    for column in columns:
        for value in frame[column]:
            if pd.isna(value):
                if column in nullable:
                    continue
                raise ValueError(f"{table_name} {column} values must be integers")
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, numeric_types)
                or not np.isfinite(value)
                or float(value) != np.floor(float(value))
            ):
                raise ValueError(f"{table_name} {column} values must be integers")


def _require_boolean_values(
    frame: pd.DataFrame,
    table_name: str,
    columns: tuple[str, ...],
    *,
    nullable: frozenset[str] = frozenset(),
) -> None:
    """Require actual boolean scalars rather than truthy coercions."""
    for column in columns:
        for value in frame[column]:
            if pd.isna(value):
                if column in nullable:
                    continue
                raise ValueError(f"{table_name} {column} values must be boolean")
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{table_name} {column} values must be boolean")


def validate_behavior_results(
    results: dict[str, pd.DataFrame],
    settings: BehaviorSettings,
    *,
    base_seed: int,
) -> None:
    """Require complete paired folds, tie-aware rankings, and predictions."""
    if set(results) != set(BEHAVIOR_RESULT_SCHEMAS):
        raise ValueError(f"Behavior tables differ from the required set: {sorted(results)}")
    for name, schema in BEHAVIOR_RESULT_SCHEMAS.items():
        frame = results[name]
        if frame.empty or tuple(frame.columns) != schema:
            raise ValueError(f"{name} differs from its required nonempty schema")

    fits = results["behavior_fit_raw"]
    features = results["behavior_feature_raw"]
    predictions = results["behavior_prediction_raw"]
    probabilities = results["behavior_probability_raw"]
    folds = results["behavior_fold_summary"]
    summary = results["behavior_summary"]
    _require_integral_values(
        fits,
        "behavior_fit_raw",
        (
            "repeat",
            "fold",
            "split_seed",
            "model_seed",
            "n_train",
            "n_test",
            "n_features",
            "root_feature",
        ),
        nullable=frozenset({"root_feature"}),
    )
    _require_integral_values(
        features,
        "behavior_feature_raw",
        ("repeat", "fold", "model_seed", "feature_id", "rank"),
    )
    _require_integral_values(
        predictions,
        "behavior_prediction_raw",
        ("repeat", "fold", "model_seed", "sample_id"),
    )
    _require_integral_values(
        probabilities,
        "behavior_probability_raw",
        ("repeat", "fold", "model_seed", "sample_id", "class_label"),
    )
    _require_integral_values(
        folds,
        "behavior_fold_summary",
        ("repeat", "fold", "model_seed", "n_test", "top_k"),
    )
    _require_integral_values(
        summary,
        "behavior_summary",
        ("n_fold_values", "n_repeats", "bootstrap_resamples"),
    )
    _require_boolean_values(
        fits,
        "behavior_fit_raw",
        ("native_output_reproducible", "predictions_reproducible"),
    )
    _require_boolean_values(
        features,
        "behavior_feature_raw",
        ("feature_value_missing", "is_root"),
        nullable=frozenset({"is_root"}),
    )
    _require_boolean_values(
        folds,
        "behavior_fold_summary",
        (
            "citrees_no_split",
            "partykit_no_split",
            "citrees_native_ranking_informative",
            "partykit_native_ranking_informative",
            "both_native_rankings_informative",
            "citrees_native_output_reproducible",
            "citrees_predictions_reproducible",
            "partykit_native_output_reproducible",
            "partykit_predictions_reproducible",
        ),
        nullable=frozenset({"citrees_no_split", "partykit_no_split"}),
    )

    datasets = load_behavior_datasets()
    dataset_by_name = {dataset.name: dataset for dataset in datasets}
    if len(dataset_by_name) != len(datasets):
        raise ValueError("Behavior dataset names must be unique")
    expected_datasets = set(dataset_by_name)
    classification_datasets = {
        dataset.name for dataset in datasets if dataset.task == "classification"
    }
    for table_name, frame in results.items():
        observed_datasets = set(frame["dataset"].astype(str))
        expected_table_datasets = (
            classification_datasets
            if table_name == "behavior_probability_raw"
            else expected_datasets
        )
        if observed_datasets != expected_table_datasets:
            raise ValueError(f"{table_name} differs from the frozen dataset inventory")
        for row in frame.itertuples(index=False):
            dataset = dataset_by_name[str(row.dataset)]
            if str(row.task) != dataset.task:
                raise ValueError(f"{table_name} task labels differ from dataset provenance")
            if hasattr(row, "dataset_sha256") and str(row.dataset_sha256) != dataset.sha256:
                raise ValueError(f"{table_name} hashes differ from dataset provenance")

    expected_splits: dict[tuple[str, str, int, int], dict[str, object]] = {}
    expected_fit_keys: set[tuple[str, str, str, str, int, int]] = set()
    for dataset in datasets:
        split_seed = _stream_seed(base_seed, dataset.task, dataset.name, "folds")
        splitter = _splitter(dataset, settings, split_seed)
        for split_index, (train_indices, test_indices) in enumerate(
            splitter.split(dataset.X, dataset.y)
        ):
            repeat, fold = divmod(split_index, settings.n_splits)
            split_key = (dataset.task, dataset.name, repeat, fold)
            expected_splits[split_key] = {
                "split_seed": split_seed,
                "train_ids_sha256": _index_sha256(train_indices),
                "test_ids_sha256": _index_sha256(test_indices),
                "n_train": len(train_indices),
                "n_test": len(test_indices),
                "n_features": dataset.X.shape[1],
            }
            expected_fit_keys.update(
                (
                    dataset.task,
                    dataset.name,
                    family,
                    method,
                    repeat,
                    fold,
                )
                for family in MODEL_FAMILIES
                for method in METHODS
            )

    fit_key = ["task", "dataset", "model_family", "method", "repeat", "fold"]
    pair_key = ["task", "dataset", "model_family", "repeat", "fold"]
    if fits.duplicated(fit_key).any():
        raise ValueError("Behavior fit keys must be unique")
    observed_fit_keys = {
        (
            str(task),
            str(dataset),
            str(family),
            str(method),
            int(repeat),
            int(fold),
        )
        for task, dataset, family, method, repeat, fold in fits[fit_key].itertuples(
            index=False,
            name=None,
        )
    }
    if observed_fit_keys != expected_fit_keys:
        raise ValueError("Behavior fits differ from the required dataset and fold coverage")
    if len(fits) != len(folds) * 2:
        raise ValueError("Every paired fold must contain both implementations")
    if not fits.groupby(pair_key)["method"].agg(set).eq({"citrees", "partykit"}).all():
        raise ValueError("Every fit pair must contain citrees and partykit")
    paired_controls = (
        "dataset_sha256",
        "split_seed",
        "model_seed",
        "train_ids_sha256",
        "test_ids_sha256",
        "n_train",
        "n_test",
        "n_features",
    )
    if not fits.groupby(pair_key)[list(paired_controls)].nunique().eq(1).all().all():
        raise ValueError("Paired implementations must share folds, seeds, and dimensions")
    for fit in fits.itertuples(index=False):
        split = expected_splits[
            (
                str(fit.task),
                str(fit.dataset),
                int(fit.repeat),
                int(fit.fold),
            )
        ]
        expected_split_seed = _stream_seed(base_seed, fit.task, fit.dataset, "folds")
        expected_model_seed = _stream_seed(
            base_seed,
            fit.task,
            fit.dataset,
            fit.model_family,
            fit.repeat,
            fit.fold,
        )
        if fit.split_seed != expected_split_seed or fit.model_seed != expected_model_seed:
            raise ValueError("Recorded behavior seeds must be derived from the base seed")
        if any(
            getattr(fit, column) != split[column]
            for column in (
                "train_ids_sha256",
                "test_ids_sha256",
                "n_train",
                "n_test",
                "n_features",
            )
        ):
            raise ValueError("Recorded held-out fold differs from the frozen dataset split")
        if fit.model_family == "tree":
            root = int(fit.root_feature)
            if root < -1 or root >= int(fit.n_features):
                raise ValueError("Tree root feature must be -1 or a valid feature index")
        elif pd.notna(fit.root_feature):
            raise ValueError("Forest fits must not contain tree-root values")
    reproducible = fits[["native_output_reproducible", "predictions_reproducible"]].astype(bool)
    if not reproducible.all().all():
        failed = fits.loc[
            ~reproducible.all(axis=1),
            [*fit_key, "native_output_reproducible", "predictions_reproducible"],
        ].to_dict(orient="records")
        raise ValueError(f"Same-seed refits changed reported outputs or predictions: {failed}")

    fit_index = fits.set_index(fit_key)
    fit_seeds = fit_index["model_seed"]
    for table_name, frame in (
        ("feature", features),
        ("prediction", predictions),
        ("probability", probabilities),
    ):
        grouped_seeds = frame.groupby(fit_key, sort=False)["model_seed"]
        observed_seeds = grouped_seeds.first()
        expected_seeds = fit_seeds.reindex(observed_seeds.index)
        if (
            not grouped_seeds.nunique().eq(1).all()
            or expected_seeds.isna().any()
            or not np.array_equal(
                observed_seeds.to_numpy(dtype=np.uint64),
                expected_seeds.to_numpy(dtype=np.uint64),
            )
        ):
            raise ValueError(f"{table_name} raw model seeds must match fitted model seeds")

    feature_key = [*fit_key, "feature_id"]
    if features.duplicated(feature_key).any():
        raise ValueError("Behavior feature keys must be unique")
    if not features.groupby(["dataset", "feature_id"])["feature_name"].nunique().eq(1).all():
        raise ValueError("Feature names must remain fixed within each dataset")
    for keys, group in features.groupby(fit_key, sort=False):
        fit = fit_index.loc[keys]
        n_features = int(fit["n_features"])
        ordered = group.sort_values("feature_id")
        if len(ordered) != n_features or not np.array_equal(
            ordered["feature_id"].to_numpy(dtype=np.int64),
            np.arange(n_features),
        ):
            raise ValueError("Every fit must retain every feature in original order")
        dataset = dataset_by_name[str(keys[1])]
        if tuple(ordered["feature_name"].astype(str)) != dataset.feature_names:
            raise ValueError("Feature names differ from the frozen dataset")
        if set(ordered["feature_definition"]) != {
            _feature_definition(cast(Method, str(keys[3])), cast(ModelFamily, str(keys[2])))
        }:
            raise ValueError("Feature definitions differ from the fitted implementation")
        values = ordered["feature_value"].to_numpy(dtype=np.float64)
        missing = np.isnan(values)
        if np.isinf(values).any() or not np.array_equal(
            ordered["feature_value_missing"].to_numpy(dtype=bool),
            missing,
        ):
            raise ValueError("Feature values and missingness indicators are inconsistent")
        expected_ranking_values = _ranking_values(values)
        if not np.array_equal(
            ordered["ranking_value"].to_numpy(dtype=np.float64),
            expected_ranking_values,
        ) or not np.array_equal(
            ordered["rank"].to_numpy(dtype=np.int64),
            _tie_ranks(values),
        ):
            raise ValueError("Feature rankings must preserve native ties and structural zeros")

        family_name = str(keys[2])
        method_name = str(keys[3])
        if missing.any() and (family_name != "forest" or method_name != "partykit"):
            raise ValueError("Only omitted partykit forest importance values may be missing")
        if family_name == "forest":
            if ordered["is_root"].notna().any() or pd.notna(fit["root_feature"]):
                raise ValueError("Forest rows must not contain tree-root values")
            continue

        if missing.any() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Tree split counts must be complete nonnegative integers")
        root = int(fit["root_feature"])
        expected_root = np.arange(n_features) == root
        if not np.array_equal(ordered["is_root"].to_numpy(dtype=bool), expected_root):
            raise ValueError("Tree root indicators must match the fitted root feature")
        if (root == -1) != (values.sum() == 0):
            raise ValueError("Tree root and split counts are inconsistent")
        if root >= 0 and values[root] < 1:
            raise ValueError("The fitted tree root must occur in its split counts")

    prediction_key = [*fit_key, "sample_id"]
    if predictions.duplicated(prediction_key).any():
        raise ValueError("Behavior prediction keys must be unique")
    if not np.isfinite(predictions[["y_true", "prediction"]].to_numpy(dtype=np.float64)).all():
        raise ValueError("Behavior predictions and outcomes must be finite")
    if not predictions.groupby(["dataset", "sample_id"])["y_true"].nunique().eq(1).all():
        raise ValueError("Sample outcomes must remain fixed across fits")
    if not predictions.groupby([*pair_key, "sample_id"])["method"].nunique().eq(2).all():
        raise ValueError("Every held-out sample must have both implementation predictions")
    for keys, group in predictions.groupby(fit_key, sort=False):
        fit = fit_index.loc[keys]
        sample_ids = group["sample_id"].to_numpy(dtype=np.int64)
        n_test = int(fit["n_test"])
        n_total = int(fit["n_train"]) + n_test
        if (
            len(sample_ids) != n_test
            or len(np.unique(sample_ids)) != n_test
            or (sample_ids < 0).any()
            or (sample_ids >= n_total).any()
            or _index_sha256(sample_ids) != fit["test_ids_sha256"]
        ):
            raise ValueError("Prediction rows differ from the recorded held-out fold")
        dataset_spec = dataset_by_name[str(keys[1])]
        expected_y = np.asarray(dataset_spec.y)[sample_ids].astype(np.float64)
        if not np.array_equal(
            group["y_true"].to_numpy(dtype=np.float64),
            expected_y,
        ):
            raise ValueError("Recorded outcomes differ from the frozen dataset")
        train_ids = np.setdiff1d(np.arange(n_total), sample_ids)
        if _index_sha256(train_ids) != fit["train_ids_sha256"]:
            raise ValueError("Recorded training and held-out folds are not complementary")
    for _keys, group in predictions.groupby(
        ["task", "dataset", "model_family", "method", "repeat"],
        sort=False,
    ):
        if not group["sample_id"].value_counts().eq(1).all():
            raise ValueError("Each repeat must predict every sample exactly once")

    probability_key = [*fit_key, "sample_id", "class_label"]
    if probabilities.duplicated(probability_key).any():
        raise ValueError("Behavior probability keys must be unique")
    if (
        not probabilities["task"].eq("classification").all()
        or not probabilities["probability"].between(0.0, 1.0).all()
    ):
        raise ValueError("Probability rows must contain valid classification probabilities")
    classification_predictions = predictions[predictions["task"] == "classification"]
    class_labels_by_dataset = {
        dataset: np.sort(group["y_true"].unique().astype(np.int64))
        for dataset, group in classification_predictions.groupby("dataset", sort=False)
    }
    for keys, group in probabilities.groupby([*fit_key, "sample_id"], sort=False):
        dataset_name = str(keys[1])
        expected_classes = class_labels_by_dataset[dataset_name]
        ordered = group.sort_values("class_label")
        if not np.array_equal(
            ordered["class_label"].to_numpy(dtype=np.int64),
            expected_classes,
        ) or not np.isclose(ordered["probability"].sum(), 1.0):
            raise ValueError("Probability rows must align to every observed class and sum to one")
    if (
        not classification_predictions["prediction"]
        .isin(probabilities["class_label"].unique())
        .all()
    ):
        raise ValueError("Classification predictions must use recorded class labels")
    if (
        not probabilities.groupby([*pair_key, "sample_id", "class_label"])["method"]
        .nunique()
        .eq(2)
        .all()
    ):
        raise ValueError("Every held-out class probability must include both implementations")

    if folds.duplicated(pair_key).any() or len(folds) != (
        settings.n_splits * settings.n_repeats * 2 * 2
    ):
        raise ValueError("Behavior fold summaries have incomplete task and family coverage")
    feature_groups = {
        keys: group.sort_values("feature_id")
        for keys, group in features.groupby(fit_key, sort=False)
    }
    prediction_groups = {
        keys: group.sort_values("sample_id")
        for keys, group in predictions.groupby(fit_key, sort=False)
    }
    probability_groups = {
        keys: group.pivot(
            index="sample_id",
            columns="class_label",
            values="probability",
        ).sort_index()
        for keys, group in probabilities.groupby(fit_key, sort=False)
    }
    expected_fold_rows: list[dict[str, object]] = []
    for keys, fit_pair in fits.groupby(pair_key, sort=False):
        task = cast(Task, str(keys[0]))
        dataset_name = str(keys[1])
        family = cast(ModelFamily, str(keys[2]))
        repeat = int(keys[3])
        fold = int(keys[4])
        behaviors: dict[Method, ModelBehavior] = {}
        reproducibility: dict[Method, tuple[bool, bool]] = {}
        y_test: np.ndarray | None = None
        model_seed: int | None = None
        for method in METHODS:
            method_key = (task, dataset_name, family, method, repeat, fold)
            fit = fit_index.loc[method_key]
            feature_group = feature_groups[method_key]
            prediction_group = prediction_groups[method_key]
            classes: np.ndarray | None = None
            method_probabilities: np.ndarray | None = None
            if task == "classification":
                probability_group = probability_groups[method_key]
                if not np.array_equal(
                    probability_group.index.to_numpy(dtype=np.int64),
                    prediction_group["sample_id"].to_numpy(dtype=np.int64),
                ):
                    raise ValueError("Probability and hard-prediction sample IDs differ")
                classes = probability_group.columns.to_numpy(dtype=np.int64)
                method_probabilities = probability_group.to_numpy(dtype=np.float64)
            behaviors[method] = ModelBehavior(
                root_feature=int(fit["root_feature"]) if family == "tree" else None,
                feature_values=feature_group["feature_value"].to_numpy(dtype=np.float64),
                predictions=prediction_group["prediction"].to_numpy(),
                probabilities=method_probabilities,
                classes=classes,
            )
            reproducibility[method] = (
                bool(fit["native_output_reproducible"]),
                bool(fit["predictions_reproducible"]),
            )
            current_y = prediction_group["y_true"].to_numpy()
            if y_test is None:
                y_test = current_y
            elif not np.array_equal(y_test, current_y):
                raise ValueError("Paired methods must retain identical held-out outcomes")
            current_seed = int(fit["model_seed"])
            if model_seed is None:
                model_seed = current_seed
            elif model_seed != current_seed:
                raise ValueError("Paired methods must retain one model seed")

        if y_test is None or model_seed is None:
            raise ValueError("Behavior fold reconstruction requires both fitted methods")
        expected_fold_rows.append(
            _fold_pair_row(
                DatasetSpec(
                    task=task,
                    name=dataset_name,
                    X=np.empty((0, 0)),
                    y=np.empty(0),
                    feature_names=(),
                    sha256=str(fit_pair["dataset_sha256"].iloc[0]),
                ),
                family,
                repeat,
                fold,
                model_seed,
                y_test,
                behaviors,
                reproducibility,
            )
        )
    expected_folds = pd.DataFrame(expected_fold_rows).loc[
        :,
        BEHAVIOR_RESULT_SCHEMAS["behavior_fold_summary"],
    ]
    try:
        pd.testing.assert_frame_equal(
            folds.reset_index(drop=True).convert_dtypes(),
            expected_folds.convert_dtypes(),
            check_exact=False,
            check_dtype=False,
            rtol=_FLOAT_COMPARISON_TOLERANCE,
            atol=_FLOAT_COMPARISON_TOLERANCE,
        )
    except AssertionError as error:
        raise ValueError("Behavior fold summaries differ from the validated raw tables") from error

    numeric_fold_columns = [
        column
        for column in BEHAVIOR_RESULT_SCHEMAS["behavior_fold_summary"]
        if column
        not in {
            "task",
            "dataset",
            "dataset_sha256",
            "model_family",
            "root_comparison",
            "citrees_no_split",
            "partykit_no_split",
            "citrees_native_ranking_informative",
            "partykit_native_ranking_informative",
            "both_native_rankings_informative",
            "prediction_metric",
            "citrees_native_output_reproducible",
            "citrees_predictions_reproducible",
            "partykit_native_output_reproducible",
            "partykit_predictions_reproducible",
        }
    ]
    if (
        not folds[numeric_fold_columns]
        .apply(lambda column: np.isfinite(column) | column.isna())
        .all()
        .all()
    ):
        raise ValueError("Fold summaries contain invalid numeric values")
    native_jaccard = folds["native_top_k_with_ties_jaccard"].dropna()
    if not native_jaccard.between(0.0, 1.0).all():
        raise ValueError("Top-k-with-ties Jaccard values must lie in [0, 1]")
    reproducibility_columns = [column for column in folds if column.endswith("_reproducible")]
    if not folds[reproducibility_columns].astype(bool).all().all():
        raise ValueError("Fold summaries require reproducible same-seed refits")

    expected_summary = summarize_behavior(folds, settings, base_seed=base_seed)
    if not summary.convert_dtypes().equals(expected_summary.convert_dtypes()):
        raise ValueError("Behavior aggregate summaries differ from the validated fold estimates")


def behavior_cell_inventory(profile: Profile) -> tuple[BehaviorCell, ...]:
    """Return the complete deterministic paired-cell inventory."""
    settings = _settings(profile)
    return tuple(
        BehaviorCell(
            task=dataset.task,
            dataset=dataset.name,
            model_family=family,
            split_index=split_index,
            repeat=split_index // settings.n_splits,
            fold=split_index % settings.n_splits,
        )
        for dataset in load_behavior_datasets()
        for split_index in range(settings.n_splits * settings.n_repeats)
        for family in MODEL_FAMILIES
    )


def _resolve_behavior_cells(
    profile: Profile,
    selected: Sequence[BehaviorCell] | None,
) -> tuple[BehaviorCell, ...]:
    """Return selected cells in canonical inventory order."""
    inventory = behavior_cell_inventory(profile)
    if selected is None:
        return inventory
    if not selected:
        raise ValueError("selected behavior cells must not be empty")
    if any(not isinstance(cell, BehaviorCell) for cell in selected):
        raise TypeError("selected behavior cells must contain BehaviorCell values")
    if len(set(selected)) != len(selected):
        raise ValueError("selected behavior cells must be unique")
    selected_set = set(selected)
    unknown = selected_set.difference(inventory)
    if unknown:
        raise ValueError(f"selected behavior cells are outside the profile inventory: {unknown}")
    return tuple(cell for cell in inventory if cell in selected_set)


def run_behavior_raw(
    profile: Profile,
    *,
    base_seed: int = BASE_SEED,
    selected_cells: Sequence[BehaviorCell] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run a complete or selected set of paired behavior cells."""
    settings = _settings(profile)
    cells = _resolve_behavior_cells(profile, selected_cells)
    fit_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for dataset in load_behavior_datasets():
        dataset_cells = frozenset(
            (cell.split_index, cell.model_family) for cell in cells if cell.dataset == dataset.name
        )
        if not dataset_cells:
            continue
        (
            dataset_fits,
            dataset_features,
            dataset_predictions,
            dataset_probabilities,
            dataset_pairs,
        ) = _run_dataset(
            dataset,
            settings,
            base_seed=base_seed,
            selected_cells=dataset_cells,
        )
        fit_rows.extend(dataset_fits)
        feature_rows.extend(dataset_features)
        prediction_rows.extend(dataset_predictions)
        probability_rows.extend(dataset_probabilities)
        pair_rows.extend(dataset_pairs)
    rows = {
        "behavior_fit_raw": fit_rows,
        "behavior_feature_raw": feature_rows,
        "behavior_prediction_raw": prediction_rows,
        "behavior_probability_raw": probability_rows,
        "behavior_fold_summary": pair_rows,
    }
    return {
        name: pd.DataFrame(table_rows, columns=BEHAVIOR_RESULT_SCHEMAS[name])
        for name, table_rows in rows.items()
    }


def assemble_behavior_results(
    raw_results: dict[str, pd.DataFrame],
    profile: Profile,
    *,
    base_seed: int = BASE_SEED,
) -> dict[str, pd.DataFrame]:
    """Build and validate final behavior tables from complete raw inputs."""
    if set(raw_results) != set(BEHAVIOR_RAW_TABLES):
        raise ValueError("behavior raw tables differ from the required inventory")
    for name in BEHAVIOR_RAW_TABLES:
        if tuple(raw_results[name].columns) != BEHAVIOR_RESULT_SCHEMAS[name]:
            raise ValueError(f"{name} differs from its required schema")
    normalized_raw = {
        name: normalize_behavior_table(name, raw_results[name]) for name in BEHAVIOR_RAW_TABLES
    }
    folds = normalized_raw["behavior_fold_summary"]
    settings = _settings(profile)
    results = {
        **normalized_raw,
        "behavior_summary": summarize_behavior(
            folds,
            settings,
            base_seed=base_seed,
        ).loc[:, BEHAVIOR_RESULT_SCHEMAS["behavior_summary"]],
    }
    validate_behavior_results(results, settings, base_seed=base_seed)
    return results


def run_behavior(
    profile: Profile,
    *,
    base_seed: int = BASE_SEED,
) -> dict[str, pd.DataFrame]:
    """Run all paired behavior comparisons and return validated tables."""
    return assemble_behavior_results(
        run_behavior_raw(profile, base_seed=base_seed),
        profile,
        base_seed=base_seed,
    )


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout.strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_results(
    results: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    profile: Profile,
    base_seed: int,
    elapsed_seconds: float,
) -> None:
    """Write behavior tables and a machine-readable execution receipt."""
    results = {name: normalize_behavior_table(name, frame) for name, frame in results.items()}
    settings = _settings(profile)
    validate_behavior_results(results, settings, base_seed=base_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: list[Path] = []
    parquet_paths: dict[str, Path] = {}
    csv_paths: dict[str, Path] = {}
    for name, frame in results.items():
        parquet_path = output_dir / f"{name}.parquet"
        frame.to_parquet(parquet_path, index=False)
        parquet_paths[name] = parquet_path
        artifact_paths.append(parquet_path)
        if name.endswith("_summary"):
            csv_path = output_dir / f"{name}.csv"
            frame.to_csv(csv_path, index=False)
            csv_paths[name] = csv_path
            artifact_paths.append(csv_path)

    written_results = {
        name: pd.read_parquet(parquet_path) for name, parquet_path in parquet_paths.items()
    }
    validate_behavior_results(
        written_results,
        settings,
        base_seed=base_seed,
    )
    for name, csv_path in csv_paths.items():
        pd.testing.assert_frame_equal(
            pd.read_csv(csv_path).convert_dtypes(),
            results[name].convert_dtypes(),
            check_exact=False,
            rtol=_FLOAT_COMPARISON_TOLERANCE,
            atol=_FLOAT_COMPARISON_TOLERANCE,
        )

    repo_root = Path(__file__).resolve().parents[3]
    source_files = (
        Path(__file__).resolve(),
        repo_root / "paper" / "benchmark" / "pipeline" / "r_methods.py",
        repo_root / "citrees" / "_forest.py",
        repo_root / "citrees" / "_permutation.py",
        repo_root / "citrees" / "_selector.py",
        repo_root / "citrees" / "_sequential.py",
        repo_root / "citrees" / "_splitter.py",
        repo_root / "citrees" / "_threshold_method.py",
        repo_root / "citrees" / "_tree.py",
        repo_root / "citrees" / "_types.py",
        repo_root / "pyproject.toml",
        repo_root / "uv.lock",
    )
    versions = {
        package: importlib.metadata.version(package)
        for package in ("citrees", "numpy", "pandas", "scikit-learn", "scipy")
    }
    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        versions["rpy2"] = importlib.metadata.version("rpy2")
    datasets = load_behavior_datasets()
    receipt = {
        "analysis": "behavior",
        "schema_version": 4,
        "profile": profile,
        "base_seed": base_seed,
        "settings": asdict(settings),
        "controls": {
            "alpha": ALPHA,
            "citrees_selector": {
                "classification": "multiple_correlation",
                "regression": "pearson_correlation",
                "bonferroni_adjustment": True,
                "sequential_stopping": False,
            },
            "partykit_selector": {
                "test_statistic": "quadratic",
                "test_distribution": list(PARTYKIT_TESTTYPE),
                "split_test": False,
            },
            "split_threshold_search": {
                "citrees": "exact_impurity",
                "partykit": "native_quadratic",
                "citrees_stage_b_significance_test": False,
            },
            "max_depth": MAX_DEPTH,
            "min_samples_split": MIN_SAMPLES_SPLIT,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "top_k": TOP_K,
            "ranking": {
                "missing_partykit_importance": "structural_zero",
                "rank_ties": "minimum",
                "comparison": "implementation_native_feature_summaries",
                "agreement": "conditional_on_both_summaries_distinguishing_features",
                "metrics": "kendall_tau_b_and_top_k_with_ties_jaccard",
            },
            "root_comparison": {
                "split_decision": "all_tree_fits",
                "root_identity": "conditional_on_both_trees_splitting",
            },
            "summary": {
                "estimate": "unweighted_mean_of_partition_repeat_means",
                "interval": "percentile_bootstrap_over_partition_repeat_means",
                "interpretation": "partition_variability_conditional_on_fixed_datasets",
            },
            "citrees_performance_advantage": {
                "classification": "citrees_balanced_accuracy_minus_partykit",
                "regression": "partykit_rmse_minus_citrees",
                "interpretation": "positive_values_favor_citrees",
            },
            "forest": {
                "bootstrap": "n_out_of_n_with_replacement",
                "candidate_features": "all",
                "native_importance_scales_retained": True,
            },
            "randomness": {
                "paired_seed": "same_integer",
                "citrees_generator": "NumPy",
                "partykit_generator": "R_L_Ecuyer_CMRG",
            },
            "r_cores": 1,
            "python_n_jobs": 1,
        },
        "datasets": {
            dataset.name: {
                "task": dataset.task,
                "sha256": dataset.sha256,
                "n_samples": len(dataset.y),
                "n_features": dataset.X.shape[1],
            }
            for dataset in datasets
        },
        "created_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "r_environment": get_r_runtime_versions(),
        "source_sha256": {str(path.relative_to(repo_root)): _sha256(path) for path in source_files},
        "versions": versions,
        "tables": {
            name: {"rows": len(frame), "columns": list(frame.columns)}
            for name, frame in results.items()
        },
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(artifact_paths)
        },
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("smoke", "quick", "full"),
        default="quick",
        help="Replication workload.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Parquet, CSV, and receipt outputs.",
    )
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Base random seed.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.profile == "full" and _git_dirty():
        raise RuntimeError("The full behavior profile requires a clean source tree")
    started = time.perf_counter()
    results = run_behavior(args.profile, base_seed=args.seed)
    elapsed = time.perf_counter() - started
    write_results(
        results,
        args.output_dir,
        profile=args.profile,
        base_seed=args.seed,
        elapsed_seconds=elapsed,
    )
    print(f"Wrote {len(results)} tables to {args.output_dir} in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
