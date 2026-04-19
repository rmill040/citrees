"""Build paper-facing mechanism summary tables.

This script migrates the scratch fixed-design studies into the paper analysis
layer. It separates two diagnostic questions that were previously conflated
under "synthetic performance":

1. Candidate-set coverage:
   On very sparse problems, how much can CIF's default `max_features` regime
   suppress signal recovery before permutation testing even starts?

2. False-feature diffusion:
   On simple fixed designs, when methods recover the signal, how diffuse is the
   residual false-feature tail across repeated fits?

Outputs:
  - paper/results/tables/paper_mechanism_candidate_set_runs.csv
  - paper/results/tables/paper_mechanism_candidate_set_summary.csv
  - paper/results/tables/paper_mechanism_frequency_runs.csv
  - paper/results/tables/paper_mechanism_frequency_summary.csv
  - paper/results/tables/paper_mechanism_feature_counts.csv
  - paper/results/figures/paper_mechanism_feature_frequency.png

Usage:
  UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_mechanism_summary_tables.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
)
from citrees._tree import Node


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TOP_K_FREQUENCY: Final[tuple[int, ...]] = (2, 10)
SPLIT_COUNT_METHODS_SINGLE_TREE: Final[tuple[str, ...]] = ("cit", "dt", "rt")
SPLIT_COUNT_METHODS_ENSEMBLE: Final[tuple[str, ...]] = ("cif", "cif_all", "rf", "et")


@dataclass(frozen=True)
class CandidateSetCase:
    """One fixed-design CIF candidate-set configuration."""

    label: str
    max_features: str | int | None
    bootstrap: bool
    n_estimators: int
    n_seeds: int


@dataclass(frozen=True)
class FixedDesignSpec:
    """One fixed-design frequency study."""

    name: str
    kind: str
    n_features: int = 100
    n_informative: int = 2
    n_samples: int = 1000
    dataset_seed: int = 1718
    class_sep: float = 2.0
    flip_y: float = 0.0
    mean_shift: float = 1.5
    design_family: str = "fixed"
    task: str = "classification"
    noise: float = 0.0


def informative_coverage_probability(p: int, sampled_features: int, n_informative: int) -> float:
    """Probability that at least one informative feature is eligible at the root."""
    if sampled_features >= p:
        return 1.0
    return 1.0 - comb(p - n_informative, sampled_features) / comb(p, sampled_features)


def resolve_max_features_count(p: int, value: str | int | None) -> int:
    """Return the sampled feature count implied by `max_features`."""
    if value is None:
        return p
    if value == "sqrt":
        return int(np.sqrt(p))
    if isinstance(value, int):
        return value
    raise ValueError(f"Unsupported max_features specification: {value!r}")


def make_fixed_dataset(spec: FixedDesignSpec) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build one fixed dataset and return shuffled informative indices."""
    if spec.kind == "easy_shuffled_classification":
        # sklearn requires n_classes * n_clusters_per_class <= 2**n_informative.
        # Our binary mechanism grids include the n_informative=1 case, which
        # must therefore use a single cluster per class.
        n_clusters_per_class = 1 if spec.n_informative == 1 else 2
        X, y = make_classification(
            n_samples=spec.n_samples,
            n_features=spec.n_features,
            n_informative=spec.n_informative,
            n_redundant=0,
            n_clusters_per_class=n_clusters_per_class,
            class_sep=spec.class_sep,
            flip_y=spec.flip_y,
            random_state=spec.dataset_seed,
            shuffle=False,
        )
    elif spec.kind == "symmetric_two_signal_gaussian":
        rng = np.random.RandomState(spec.dataset_seed)
        n_per_class = spec.n_samples // 2
        y = np.concatenate([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
        X = rng.normal(size=(spec.n_samples, spec.n_features))
        informative_count = min(spec.n_informative, spec.n_features)
        for feature_idx in range(informative_count):
            X[y == 1, feature_idx] += spec.mean_shift
            X[y == 0, feature_idx] -= spec.mean_shift
    elif spec.kind == "easy_shuffled_regression":
        X, y, coef = make_regression(
            n_samples=spec.n_samples,
            n_features=spec.n_features,
            n_informative=spec.n_informative,
            noise=spec.noise,
            random_state=spec.dataset_seed,
            shuffle=False,
            coef=True,
        )
    else:
        raise ValueError(f"Unknown fixed-design kind: {spec.kind}")

    rng = np.random.RandomState(spec.dataset_seed)
    perm = rng.permutation(spec.n_features)
    X = X[:, perm]
    if spec.kind == "easy_shuffled_regression":
        informative_original = np.flatnonzero(np.abs(coef) > 0).tolist()
        informative = np.argsort(perm)[informative_original].tolist()
    else:
        informative = np.argsort(perm)[: spec.n_informative].tolist()
    return X, y, [int(i) for i in informative]


def spec_metadata(spec: FixedDesignSpec) -> dict[str, object]:
    """Return shared metadata fields for one fixed-design study."""
    return {
        "dataset": spec.name,
        "task": spec.task,
        "kind": spec.kind,
        "design_family": spec.design_family,
        "n_samples": spec.n_samples,
        "n_features": spec.n_features,
        "n_informative": spec.n_informative,
        "informative_fraction": spec.n_informative / spec.n_features,
        "dataset_seed": spec.dataset_seed,
    }


def build_cif_model(
    *,
    seed: int,
    max_features: str | int | None,
    bootstrap: bool,
    n_estimators: int,
    verbose: int = 0,
) -> ConditionalInferenceForestClassifier:
    """Instantiate a CIF model for the candidate-set study."""
    return ConditionalInferenceForestClassifier(
        n_estimators=n_estimators,
        selector="mc",
        splitter="gini",
        alpha_selector=0.05,
        alpha_splitter=0.05,
        adjust_alpha_selector=True,
        adjust_alpha_splitter=True,
        n_resamples_selector="minimum",
        n_resamples_splitter="minimum",
        early_stopping_selector="adaptive",
        early_stopping_splitter="adaptive",
        feature_muting=True,
        feature_scanning=True,
        max_features=max_features,
        threshold_method="histogram",
        max_thresholds=256,
        bootstrap=bootstrap,
        sampling_method="stratified" if bootstrap else None,
        n_jobs=1,
        random_state=seed,
        verbose=verbose,
    )


def build_frequency_model(method: str, seed: int):
    """Instantiate one fixed-design frequency-study model."""
    if method == "cit":
        return ConditionalInferenceTreeClassifier(
            selector="mc",
            splitter="gini",
            alpha_selector=0.05,
            alpha_splitter=0.05,
            n_resamples_selector="minimum",
            n_resamples_splitter="minimum",
            early_stopping_selector="adaptive",
            early_stopping_splitter="adaptive",
            threshold_method="histogram",
            max_thresholds=256,
            feature_scanning=True,
            feature_muting=True,
            random_state=seed,
            verbose=0,
        )
    if method == "cif_all":
        return ConditionalInferenceForestClassifier(
            n_estimators=1,
            selector="mc",
            splitter="gini",
            alpha_selector=0.05,
            alpha_splitter=0.05,
            n_resamples_selector="minimum",
            n_resamples_splitter="minimum",
            early_stopping_selector="adaptive",
            early_stopping_splitter="adaptive",
            threshold_method="histogram",
            max_thresholds=256,
            feature_scanning=True,
            feature_muting=True,
            max_features=None,
            bootstrap=True,
            n_jobs=1,
            random_state=seed,
            verbose=0,
        )
    if method == "rf":
        return RandomForestClassifier(n_estimators=1, n_jobs=1, random_state=seed)
    if method == "et":
        return ExtraTreesClassifier(n_estimators=1, n_jobs=1, random_state=seed)
    raise ValueError(f"Unknown method: {method}")


def build_single_tree_split_model(task: str, method: str, seed: int, verbose: int = 0):
    """Instantiate one single-tree model for split-count diagnostics."""
    if task == "classification" and method == "cit":
        return ConditionalInferenceTreeClassifier(
            selector="mc",
            splitter="gini",
            alpha_selector=0.05,
            alpha_splitter=0.05,
            n_resamples_selector="minimum",
            n_resamples_splitter="minimum",
            early_stopping_selector="adaptive",
            early_stopping_splitter="adaptive",
            threshold_method="histogram",
            max_thresholds=256,
            feature_scanning=True,
            feature_muting=True,
            random_state=seed,
            verbose=verbose,
        )
    if task == "classification" and method == "dt":
        return DecisionTreeClassifier(random_state=seed)
    if task == "classification" and method == "rt":
        return ExtraTreeClassifier(random_state=seed)
    if task == "regression" and method == "cit":
        return ConditionalInferenceTreeRegressor(
            selector="pc",
            splitter="mse",
            alpha_selector=0.05,
            alpha_splitter=0.05,
            n_resamples_selector="minimum",
            n_resamples_splitter="minimum",
            early_stopping_selector="adaptive",
            early_stopping_splitter="adaptive",
            threshold_method="histogram",
            max_thresholds=256,
            feature_scanning=True,
            feature_muting=True,
            random_state=seed,
            verbose=verbose,
        )
    if task == "regression" and method == "dt":
        return DecisionTreeRegressor(random_state=seed)
    if task == "regression" and method == "rt":
        return ExtraTreeRegressor(random_state=seed)
    raise ValueError(f"Unknown method: {method}")


def build_ensemble_split_model(
    task: str,
    method: str,
    seed: int,
    n_estimators: int,
    *,
    verbose: int = 0,
    n_jobs: int = 1,
):
    """Instantiate one ensemble for split-count diagnostics."""
    if task == "classification" and method == "cif":
        return ConditionalInferenceForestClassifier(
            n_estimators=n_estimators,
            selector="mc",
            splitter="gini",
            alpha_selector=0.05,
            alpha_splitter=0.05,
            adjust_alpha_selector=True,
            adjust_alpha_splitter=True,
            n_resamples_selector="minimum",
            n_resamples_splitter="minimum",
            early_stopping_selector="adaptive",
            early_stopping_splitter="adaptive",
            feature_muting=True,
            feature_scanning=True,
            threshold_method="histogram",
            max_thresholds=256,
            bootstrap=True,
            n_jobs=n_jobs,
            random_state=seed,
            verbose=verbose,
        )
    if task == "classification" and method == "cif_all":
        return ConditionalInferenceForestClassifier(
            n_estimators=n_estimators,
            selector="mc",
            splitter="gini",
            alpha_selector=0.05,
            alpha_splitter=0.05,
            adjust_alpha_selector=True,
            adjust_alpha_splitter=True,
            n_resamples_selector="minimum",
            n_resamples_splitter="minimum",
            early_stopping_selector="adaptive",
            early_stopping_splitter="adaptive",
            feature_muting=True,
            feature_scanning=True,
            threshold_method="histogram",
            max_thresholds=256,
            max_features=None,
            bootstrap=True,
            n_jobs=n_jobs,
            random_state=seed,
            verbose=verbose,
        )
    if task == "classification" and method == "rf":
        return RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, random_state=seed, verbose=verbose)
    if task == "classification" and method == "et":
        return ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=n_jobs, random_state=seed, verbose=verbose)
    if task == "regression" and method == "cif":
        return ConditionalInferenceForestRegressor(
            n_estimators=n_estimators,
            selector="pc",
            splitter="mse",
            alpha_selector=0.05,
            alpha_splitter=0.05,
            adjust_alpha_selector=True,
            adjust_alpha_splitter=True,
            n_resamples_selector="minimum",
            n_resamples_splitter="minimum",
            early_stopping_selector="adaptive",
            early_stopping_splitter="adaptive",
            feature_muting=True,
            feature_scanning=True,
            threshold_method="histogram",
            max_thresholds=256,
            bootstrap=True,
            n_jobs=n_jobs,
            random_state=seed,
            verbose=verbose,
        )
    if task == "regression" and method == "cif_all":
        return ConditionalInferenceForestRegressor(
            n_estimators=n_estimators,
            selector="pc",
            splitter="mse",
            alpha_selector=0.05,
            alpha_splitter=0.05,
            adjust_alpha_selector=True,
            adjust_alpha_splitter=True,
            n_resamples_selector="minimum",
            n_resamples_splitter="minimum",
            early_stopping_selector="adaptive",
            early_stopping_splitter="adaptive",
            feature_muting=True,
            feature_scanning=True,
            threshold_method="histogram",
            max_thresholds=256,
            max_features=None,
            bootstrap=True,
            n_jobs=n_jobs,
            random_state=seed,
            verbose=verbose,
        )
    if task == "regression" and method == "rf":
        return RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, random_state=seed, verbose=verbose)
    if task == "regression" and method == "et":
        return ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=n_jobs, random_state=seed, verbose=verbose)
    raise ValueError(f"Unknown method: {method}")


def collect_cit_split_features(tree: Node) -> list[int]:
    """Return split features from a fitted CIT tree."""
    features: list[int] = []
    stack: list[Node] = [tree]
    while stack:
        node = stack.pop()
        feature = node.get("feature")
        if feature is None:
            continue
        features.append(int(feature))
        left_child = node.get("left_child")
        right_child = node.get("right_child")
        if left_child is not None:
            stack.append(left_child)
        if right_child is not None:
            stack.append(right_child)
    return features


def count_split_features_from_tree(model, p: int) -> np.ndarray:
    """Count how many internal-node splits use each feature in one fitted tree."""
    counts = np.zeros(p, dtype=int)
    if isinstance(getattr(model, "tree_", None), dict):
        for feature in collect_cit_split_features(model.tree_):
            counts[feature] += 1
        return counts

    features = np.asarray(model.tree_.feature, dtype=int)
    features = features[features >= 0]
    if features.size:
        counts += np.bincount(features, minlength=p)
    return counts


def count_split_features_from_ensemble(model, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Count split usage across a fitted forest and tree-level usage frequency."""
    split_counts = np.zeros(p, dtype=int)
    tree_use_counts = np.zeros(p, dtype=int)
    for estimator in model.estimators_:
        estimator_counts = count_split_features_from_tree(estimator, p)
        split_counts += estimator_counts
        tree_use_counts += (estimator_counts > 0).astype(int)
    return split_counts, tree_use_counts


def _ranking_summary(ranking: np.ndarray, informative: list[int], k: int) -> int:
    """Count informative features appearing in the first k ranks."""
    return sum(int(i in ranking[:k]) for i in informative)


def build_candidate_set_study(spec: FixedDesignSpec, *, verbose: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the CIF candidate-set sweep on one fixed design."""
    X, y, informative = make_fixed_dataset(spec)
    p = X.shape[1]
    cases = [
        CandidateSetCase("sqrt_bootstrap_1tree", "sqrt", True, 1, 2),
        CandidateSetCase("20_bootstrap_1tree", 20, True, 1, 2),
        CandidateSetCase("50_bootstrap_1tree", 50, True, 1, 2),
        CandidateSetCase("all_bootstrap_1tree", None, True, 1, 2),
        CandidateSetCase("all_no_bootstrap_1tree", None, False, 1, 2),
        CandidateSetCase("sqrt_bootstrap_5tree", "sqrt", True, 5, 2),
        CandidateSetCase("all_bootstrap_5tree", None, True, 5, 2),
    ]

    rows: list[dict[str, object]] = []
    for case in cases:
        sampled_features = resolve_max_features_count(p, case.max_features)
        coverage = informative_coverage_probability(p, sampled_features, len(informative))
        for seed in range(case.n_seeds):
            if verbose:
                print(
                    f"[candidate_set] dataset={spec.name} case={case.label} "
                    f"seed={seed + 1}/{case.n_seeds}",
                    flush=True,
                )
            model = build_cif_model(
                seed=seed,
                max_features=case.max_features,
                bootstrap=case.bootstrap,
                n_estimators=case.n_estimators,
                verbose=0,
            )
            model.fit(X, y)
            ranking = np.argsort(model.feature_importances_)[::-1]
            nonzero = np.flatnonzero(model.feature_importances_ > 0.0)
            rows.append(
                {
                    "study": "candidate_set",
                    **spec_metadata(spec),
                    "seed": seed,
                    "case": case.label,
                    "max_features": case.max_features,
                    "sampled_features_per_node": sampled_features,
                    "bootstrap": case.bootstrap,
                    "n_estimators": case.n_estimators,
                    "root_informative_coverage_probability": coverage,
                    "top1_feature": int(ranking[0]),
                    "top1_hit": int(ranking[0] in informative),
                    "n_true_top2": _ranking_summary(ranking, informative, 2),
                    "n_true_top10": _ranking_summary(ranking, informative, 10),
                    "nonzero_features": int(len(nonzero)),
                    "nonzero_contains_true": int(any(i in informative for i in nonzero)),
                    "informative_indices": ",".join(str(i) for i in informative),
                }
            )

    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(
            [
                "study",
                "dataset",
                "case",
                "max_features",
                "sampled_features_per_node",
                "bootstrap",
                "n_estimators",
                "root_informative_coverage_probability",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            top1_hit_rate=("top1_hit", "mean"),
            mean_true_top2=("n_true_top2", "mean"),
            mean_true_top10=("n_true_top10", "mean"),
            mean_nonzero_features=("nonzero_features", "mean"),
            nonzero_contains_true_rate=("nonzero_contains_true", "mean"),
        )
        .sort_values(["n_estimators", "sampled_features_per_node"])
        .reset_index(drop=True)
    )
    return runs, summary


def build_frequency_study(
    specs: list[FixedDesignSpec],
    n_seeds: int,
    *,
    verbose: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the fixed-design false-feature diffusion study."""
    methods = ["cit", "cif_all", "rf", "et"]
    run_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []

    for spec in specs:
        X, y, informative = make_fixed_dataset(spec)
        for method in methods:
            counts_by_k = {k: np.zeros(X.shape[1], dtype=int) for k in TOP_K_FREQUENCY}
            for seed in range(n_seeds):
                if verbose:
                    print(
                        f"[frequency] dataset={spec.name} method={method} seed={seed + 1}/{n_seeds}",
                        flush=True,
                    )
                model = build_frequency_model(method, seed)
                model.fit(X, y)
                ranking = np.argsort(model.feature_importances_)[::-1]
                for k in TOP_K_FREQUENCY:
                    counts_by_k[k][np.array(ranking[:k], dtype=int)] += 1

                run_rows.append(
                    {
                "study": "false_feature_diffusion",
                **spec_metadata(spec),
                "method": method,
                "seed": seed,
                "informative_indices": ",".join(str(i) for i in informative),
                        "top1_hit": int(ranking[0] in informative),
                        "n_true_top2": _ranking_summary(ranking, informative, 2),
                        "n_true_top10": _ranking_summary(ranking, informative, 10),
                    }
                )

            for k, counts in counts_by_k.items():
                for feature_idx, count in enumerate(counts.tolist()):
                    count_rows.append(
                        {
                            "study": "false_feature_diffusion",
                            **spec_metadata(spec),
                            "method": method,
                            "k": k,
                            "feature_idx": feature_idx,
                            "selection_count": int(count),
                            "selection_rate": count / n_seeds,
                            "is_informative": int(feature_idx in informative),
                        }
                    )

    runs = pd.DataFrame(run_rows)
    counts = pd.DataFrame(count_rows)

    summary_rows: list[dict[str, object]] = []
    group_cols = ["dataset", "kind", "design_family", "n_samples", "n_features", "n_informative", "informative_fraction", "method"]
    for group_key, sub in runs.groupby(group_cols):
        dataset, kind, design_family, n_samples, n_features, n_informative, informative_fraction, method = group_key
        for k in TOP_K_FREQUENCY:
            count_sub = counts[
                (counts["dataset"] == dataset)
                & (counts["method"] == method)
                & (counts["k"] == k)
            ]
            false = count_sub[count_sub["is_informative"] == 0]
            true = count_sub[count_sub["is_informative"] == 1]
            summary_rows.append(
                {
                    "study": "false_feature_diffusion",
                    "dataset": dataset,
                    "kind": kind,
                    "design_family": design_family,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "n_informative": n_informative,
                    "informative_fraction": informative_fraction,
                    "method": method,
                    "k": k,
                    "n_seeds": int(sub["seed"].nunique()),
                    "top1_hit_rate": float(sub["top1_hit"].mean()),
                    "mean_true_top2": float(sub["n_true_top2"].mean()),
                    "mean_true_top10": float(sub["n_true_top10"].mean()),
                    "distinct_false_features": int((false["selection_count"] > 0).sum()),
                    "max_false_selection_count": int(false["selection_count"].max()),
                    "mean_false_selection_count": float(false["selection_count"].mean()),
                    "true_selection_counts": ",".join(str(int(v)) for v in true["selection_count"].tolist()),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(["dataset", "k", "method"]).reset_index(drop=True)
    return runs, summary, counts


def build_single_tree_split_study(
    specs: list[FixedDesignSpec],
    n_seeds: int,
    *,
    verbose: int = 0,
    estimator_verbose: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run repeated single-tree fits and count split usage by feature."""
    run_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []

    for spec in specs:
        X, y, informative = make_fixed_dataset(spec)
        p = X.shape[1]
        for method in SPLIT_COUNT_METHODS_SINGLE_TREE:
            split_counts_total = np.zeros(p, dtype=int)
            tree_use_counts = np.zeros(p, dtype=int)
            for seed in range(n_seeds):
                if verbose:
                    print(
                        f"[single_tree] task={spec.task} dataset={spec.name} method={method} "
                        f"seed={seed + 1}/{n_seeds}",
                        flush=True,
                    )
                model = build_single_tree_split_model(spec.task, method, seed, verbose=estimator_verbose)
                model.fit(X, y)
                split_counts = count_split_features_from_tree(model, p)
                split_counts_total += split_counts
                tree_use_counts += (split_counts > 0).astype(int)

                run_rows.append(
                    {
                        "study": "single_tree_split_counts",
                        **spec_metadata(spec),
                        "method": method,
                        "seed": seed,
                        "n_splits": int(split_counts.sum()),
                        "n_true_split_events": int(split_counts[informative].sum()),
                        "n_noise_split_events": int(split_counts.sum() - split_counts[informative].sum()),
                        "distinct_noise_features_used": int(((split_counts > 0) & ~np.isin(np.arange(p), informative)).sum()),
                        "informative_indices": ",".join(str(i) for i in informative),
                    }
                )

            for feature_idx, split_count in enumerate(split_counts_total.tolist()):
                count_rows.append(
                    {
                        "study": "single_tree_split_counts",
                        **spec_metadata(spec),
                        "method": method,
                        "feature_idx": feature_idx,
                        "split_count": int(split_count),
                        "split_rate_per_fit": split_count / n_seeds,
                        "tree_use_count": int(tree_use_counts[feature_idx]),
                        "tree_use_rate": tree_use_counts[feature_idx] / n_seeds,
                        "is_informative": int(feature_idx in informative),
                    }
                )

    runs = pd.DataFrame(run_rows)
    counts = pd.DataFrame(count_rows)

    summary_rows: list[dict[str, object]] = []
    group_cols = ["dataset", "task", "kind", "design_family", "n_samples", "n_features", "n_informative", "informative_fraction", "method"]
    for group_key, sub in runs.groupby(group_cols):
        dataset, task, kind, design_family, n_samples, n_features, n_informative, informative_fraction, method = group_key
        count_sub = counts[(counts["dataset"] == dataset) & (counts["method"] == method)]
        false = count_sub[count_sub["is_informative"] == 0]
        true = count_sub[count_sub["is_informative"] == 1]
        summary_rows.append(
            {
                "study": "single_tree_split_counts",
                "dataset": dataset,
                "task": task,
                "kind": kind,
                "design_family": design_family,
                "n_samples": n_samples,
                "n_features": n_features,
                "n_informative": n_informative,
                "informative_fraction": informative_fraction,
                "method": method,
                "n_seeds": int(sub["seed"].nunique()),
                "mean_total_splits_per_fit": float(sub["n_splits"].mean()),
                "mean_true_split_events_per_fit": float(sub["n_true_split_events"].mean()),
                "mean_noise_split_events_per_fit": float(sub["n_noise_split_events"].mean()),
                "informative_split_share": float(
                    true["split_count"].sum() / count_sub["split_count"].sum()
                )
                if count_sub["split_count"].sum()
                else 0.0,
                "distinct_false_features_used": int((false["tree_use_count"] > 0).sum()),
                "max_false_tree_use_count": int(false["tree_use_count"].max()),
                "mean_false_tree_use_count": float(false["tree_use_count"].mean()),
                "true_split_counts": ",".join(str(int(v)) for v in true["split_count"].tolist()),
                "true_tree_use_counts": ",".join(str(int(v)) for v in true["tree_use_count"].tolist()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["dataset", "method"]).reset_index(drop=True)
    return runs, summary, counts


def build_ensemble_split_study(
    specs: list[FixedDesignSpec],
    n_estimators: int,
    n_seeds: int,
    *,
    verbose: int = 0,
    estimator_verbose: int = 0,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run large-ensemble split-count diagnostics on fixed sparse-signal designs."""
    run_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []

    for spec in specs:
        X, y, informative = make_fixed_dataset(spec)
        p = X.shape[1]
        for method in SPLIT_COUNT_METHODS_ENSEMBLE:
            split_counts_total = np.zeros(p, dtype=int)
            tree_use_counts_total = np.zeros(p, dtype=int)
            total_trees = 0
            for seed in range(n_seeds):
                if verbose:
                    print(
                        f"[ensemble] task={spec.task} dataset={spec.name} method={method} "
                        f"seed={seed + 1}/{n_seeds} trees={n_estimators}",
                        flush=True,
                    )
                model = build_ensemble_split_model(
                    spec.task,
                    method,
                    seed,
                    n_estimators,
                    verbose=estimator_verbose,
                    n_jobs=n_jobs,
                )
                model.fit(X, y)
                split_counts, tree_use_counts = count_split_features_from_ensemble(model, p)
                split_counts_total += split_counts
                tree_use_counts_total += tree_use_counts
                total_trees += len(model.estimators_)

                run_rows.append(
                    {
                        "study": "ensemble_split_counts",
                        **spec_metadata(spec),
                        "method": method,
                        "seed": seed,
                        "n_estimators": len(model.estimators_),
                        "n_split_events": int(split_counts.sum()),
                        "n_true_split_events": int(split_counts[informative].sum()),
                        "n_noise_split_events": int(split_counts.sum() - split_counts[informative].sum()),
                        "distinct_noise_features_used": int(((tree_use_counts > 0) & ~np.isin(np.arange(p), informative)).sum()),
                        "informative_indices": ",".join(str(i) for i in informative),
                    }
                )

            for feature_idx, split_count in enumerate(split_counts_total.tolist()):
                count_rows.append(
                    {
                        "study": "ensemble_split_counts",
                        **spec_metadata(spec),
                        "method": method,
                        "n_fit_seeds": n_seeds,
                        "n_estimators_per_fit": n_estimators,
                        "total_trees": total_trees,
                        "feature_idx": feature_idx,
                        "split_count": int(split_count),
                        "split_rate_per_tree": split_count / total_trees,
                        "tree_use_count": int(tree_use_counts_total[feature_idx]),
                        "tree_use_rate": tree_use_counts_total[feature_idx] / total_trees,
                        "is_informative": int(feature_idx in informative),
                    }
                )

    runs = pd.DataFrame(run_rows)
    counts = pd.DataFrame(count_rows)

    summary_rows: list[dict[str, object]] = []
    group_cols = ["dataset", "task", "kind", "design_family", "n_samples", "n_features", "n_informative", "informative_fraction", "method"]
    for group_key, sub in runs.groupby(group_cols):
        dataset, task, kind, design_family, n_samples, n_features, n_informative, informative_fraction, method = group_key
        count_sub = counts[(counts["dataset"] == dataset) & (counts["method"] == method)]
        false = count_sub[count_sub["is_informative"] == 0]
        true = count_sub[count_sub["is_informative"] == 1]
        total_split_count = count_sub["split_count"].sum()
        summary_rows.append(
            {
                "study": "ensemble_split_counts",
                "dataset": dataset,
                "task": task,
                "kind": kind,
                "design_family": design_family,
                "n_samples": n_samples,
                "n_features": n_features,
                "n_informative": n_informative,
                "informative_fraction": informative_fraction,
                "method": method,
                "n_fit_seeds": int(sub["seed"].nunique()),
                "n_estimators_per_fit": int(sub["n_estimators"].iloc[0]),
                "total_trees": int(count_sub["total_trees"].iloc[0]),
                "mean_total_splits_per_fit": float(sub["n_split_events"].mean()),
                "mean_true_split_events_per_fit": float(sub["n_true_split_events"].mean()),
                "mean_noise_split_events_per_fit": float(sub["n_noise_split_events"].mean()),
                "informative_split_share": float(true["split_count"].sum() / total_split_count) if total_split_count else 0.0,
                "distinct_false_features_used": int((false["tree_use_count"] > 0).sum()),
                "max_false_tree_use_count": int(false["tree_use_count"].max()),
                "mean_false_tree_use_count": float(false["tree_use_count"].mean()),
                "true_split_counts": ",".join(str(int(v)) for v in true["split_count"].tolist()),
                "true_tree_use_counts": ",".join(str(int(v)) for v in true["tree_use_count"].tolist()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["dataset", "method"]).reset_index(drop=True)
    return runs, summary, counts


def plot_frequency_counts(counts: pd.DataFrame, output_path: Path) -> None:
    """Plot top-10 feature-frequency counts for each fixed design and method."""
    datasets = list(dict.fromkeys(counts["dataset"].tolist()))
    methods = ["cit", "cif_all", "rf", "et"]
    display_names = {"cit": "CIT", "cif_all": "CIF (all features)", "rf": "RF", "et": "ExtraTrees"}

    fig, axes = plt.subplots(len(datasets), len(methods), figsize=(16, 3.6 * len(datasets)), sharex=False, sharey=True)
    axes = np.atleast_2d(axes)

    for row_idx, dataset in enumerate(datasets):
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx, col_idx]
            sub = counts[(counts["dataset"] == dataset) & (counts["method"] == method) & (counts["k"] == 10)].copy()
            sub = sub.sort_values(["selection_count", "feature_idx"], ascending=[False, True]).reset_index(drop=True)
            colors = ["#2563EB" if flag else "#D1D5DB" for flag in sub["is_informative"]]
            ax.bar(range(len(sub)), sub["selection_count"], color=colors, width=0.9)
            if row_idx == 0:
                ax.set_title(display_names[method], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{dataset}\nTop-10 count", fontsize=9)
            if row_idx == len(datasets) - 1:
                ax.set_xlabel("Features sorted by top-10 count", fontsize=9)
            ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Fixed-design feature-frequency diagnostics", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_split_counts_by_feature_index(
    counts: pd.DataFrame,
    *,
    methods: tuple[str, ...],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot raw feature-index split/use counts with informative features highlighted."""
    datasets = list(dict.fromkeys(counts["dataset"].tolist()))
    fig, axes = plt.subplots(len(datasets), len(methods), figsize=(16, 3.8 * len(datasets)), sharex=False, sharey=False)
    axes = np.atleast_2d(axes)
    display_names = {
        "cit": "CIT",
        "dt": "Decision Tree",
        "rt": "Random Tree",
        "cif": "CIF",
        "cif_all": "CIF (all features)",
        "rf": "RF",
        "et": "ExtraTrees",
    }

    for row_idx, dataset in enumerate(datasets):
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx, col_idx]
            sub = counts[(counts["dataset"] == dataset) & (counts["method"] == method)].sort_values("feature_idx")
            if sub.empty:
                ax.axis("off")
                continue
            y_col = "tree_use_count" if "tree_use_count" in sub.columns else "split_count"
            colors = ["#2563EB" if flag else "#D1D5DB" for flag in sub["is_informative"]]
            ax.bar(sub["feature_idx"], sub[y_col], color=colors, width=0.85)
            if row_idx == 0:
                ax.set_title(display_names[method], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{dataset}\n{ylabel}", fontsize=9)
            if row_idx == len(datasets) - 1:
                ax.set_xlabel("Feature index", fontsize=9)
            ax.grid(axis="y", alpha=0.25)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_summary_curves(
    summary: pd.DataFrame,
    *,
    methods: tuple[str, ...],
    x_col: str,
    metrics: tuple[tuple[str, str], ...],
    title: str,
    output_path: Path,
) -> None:
    """Plot summary metrics over a design grid for each method."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 4.0), sharex=False)
    axes = np.atleast_1d(axes)
    display_names = {
        "cit": "CIT",
        "dt": "Decision Tree",
        "rt": "Random Tree",
        "cif": "CIF",
        "cif_all": "CIF (all features)",
        "rf": "RF",
        "et": "ExtraTrees",
    }
    palette = {
        "cit": "#2563EB",
        "dt": "#DC2626",
        "rt": "#F59E0B",
        "cif": "#1D4ED8",
        "cif_all": "#0F766E",
        "rf": "#B91C1C",
        "et": "#A16207",
    }

    for ax, (metric_col, ylabel) in zip(axes, metrics, strict=False):
        for method in methods:
            sub = summary[summary["method"] == method].sort_values(x_col)
            if sub.empty:
                continue
            ax.plot(
                sub[x_col],
                sub[metric_col],
                marker="o",
                linewidth=2,
                markersize=5,
                label=display_names[method],
                color=palette[method],
            )
        ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.25)

    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    """Build and save paper-facing screening mechanism diagnostics."""
    parser = argparse.ArgumentParser(description="Build paper-facing screening mechanism diagnostics")
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR, help="Directory for CSV outputs")
    parser.add_argument("--figure-dir", type=Path, default=FIGURES_DIR, help="Directory for figure outputs")
    parser.add_argument(
        "--frequency-seeds",
        type=int,
        default=4,
        help="Number of model seeds for the fixed-design frequency study",
    )
    parser.add_argument(
        "--single-tree-seeds",
        type=int,
        default=5,
        help="Number of repeated fits for the single-tree split-count study",
    )
    parser.add_argument(
        "--ensemble-estimators",
        type=int,
        default=1000,
        help="Number of trees per forest for the ensemble split-count study",
    )
    parser.add_argument(
        "--ensemble-seeds",
        type=int,
        default=5,
        help="Number of repeated forest fits for the ensemble split-count study",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Progress verbosity for mechanism studies and compatible estimators",
    )
    parser.add_argument(
        "--estimator-verbose",
        type=int,
        default=0,
        help="Estimator-internal verbosity; keep 0 unless debugging estimator internals",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for forest-based mechanism studies",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    candidate_spec = FixedDesignSpec(name="easy_shuffled_classification", kind="easy_shuffled_classification")
    frequency_specs = [
        candidate_spec,
        FixedDesignSpec(
            name="symmetric_two_signal_gaussian",
            kind="symmetric_two_signal_gaussian",
            dataset_seed=2718,
            mean_shift=1.5,
        ),
    ]
    density_specs = [
        FixedDesignSpec(
            name=f"symmetric_two_signal_gaussian_n250_p100_i{n_informative}",
            kind="symmetric_two_signal_gaussian",
            dataset_seed=2718 + n_informative,
            n_samples=250,
            n_features=100,
            n_informative=n_informative,
            mean_shift=1.5,
            design_family="vary_informative_fraction",
        )
        for n_informative in (2, 5, 10, 20, 50)
    ]
    make_classification_density_specs = [
        FixedDesignSpec(
            name=f"make_classification_n250_p100_i{n_informative}",
            kind="easy_shuffled_classification",
            dataset_seed=3718 + n_informative,
            n_samples=250,
            n_features=100,
            n_informative=n_informative,
            class_sep=2.0,
            flip_y=0.0,
            design_family="vary_informative_fraction",
        )
        for n_informative in (2, 5, 10, 20, 50)
    ]
    make_classification_dimension_specs = [
        FixedDesignSpec(
            name=f"make_classification_n250_p{n_features}_i2",
            kind="easy_shuffled_classification",
            dataset_seed=4718 + n_features,
            n_samples=250,
            n_features=n_features,
            n_informative=2,
            class_sep=2.0,
            flip_y=0.0,
            design_family="vary_dimension_fixed_signal",
        )
        for n_features in (100, 250, 500, 1000)
    ]
    dimension_specs = [
        FixedDesignSpec(
            name=f"symmetric_two_signal_gaussian_n250_p{n_features}_i2",
            kind="symmetric_two_signal_gaussian",
            dataset_seed=3141 + n_features,
            n_samples=250,
            n_features=n_features,
            n_informative=2,
            mean_shift=1.5,
            design_family="vary_dimension_fixed_signal",
        )
        for n_features in (100, 250, 500, 1000)
    ]
    regression_density_specs = [
        FixedDesignSpec(
            name=f"make_regression_n250_p100_i{n_informative}",
            kind="easy_shuffled_regression",
            dataset_seed=5718 + n_informative,
            n_samples=250,
            n_features=100,
            n_informative=n_informative,
            noise=5.0,
            design_family="vary_informative_fraction",
            task="regression",
        )
        for n_informative in (2, 5, 10, 20, 50)
    ]
    regression_dimension_specs = [
        FixedDesignSpec(
            name=f"make_regression_n250_p{n_features}_i2",
            kind="easy_shuffled_regression",
            dataset_seed=6718 + n_features,
            n_samples=250,
            n_features=n_features,
            n_informative=2,
            noise=5.0,
            design_family="vary_dimension_fixed_signal",
            task="regression",
        )
        for n_features in (100, 250, 500, 1000)
    ]
    split_count_specs = (
        density_specs
        + make_classification_density_specs
        + dimension_specs
        + make_classification_dimension_specs
        + regression_density_specs
        + regression_dimension_specs
    )

    candidate_runs, candidate_summary = build_candidate_set_study(candidate_spec, verbose=args.verbose)
    frequency_runs, frequency_summary, feature_counts = build_frequency_study(
        frequency_specs,
        n_seeds=args.frequency_seeds,
        verbose=args.verbose,
    )
    single_tree_runs, single_tree_summary, single_tree_feature_counts = build_single_tree_split_study(
        split_count_specs,
        n_seeds=args.single_tree_seeds,
        verbose=args.verbose,
        estimator_verbose=args.estimator_verbose,
    )
    ensemble_runs, ensemble_summary, ensemble_feature_counts = build_ensemble_split_study(
        split_count_specs,
        n_estimators=args.ensemble_estimators,
        n_seeds=args.ensemble_seeds,
        verbose=args.verbose,
        estimator_verbose=args.estimator_verbose,
        n_jobs=args.n_jobs,
    )

    outputs = {
        "paper_mechanism_candidate_set_runs.csv": candidate_runs,
        "paper_mechanism_candidate_set_summary.csv": candidate_summary,
        "paper_mechanism_frequency_runs.csv": frequency_runs,
        "paper_mechanism_frequency_summary.csv": frequency_summary,
        "paper_mechanism_feature_counts.csv": feature_counts,
        "paper_mechanism_single_tree_split_runs.csv": single_tree_runs,
        "paper_mechanism_single_tree_split_summary.csv": single_tree_summary,
        "paper_mechanism_single_tree_split_feature_counts.csv": single_tree_feature_counts,
        "paper_mechanism_ensemble_split_runs.csv": ensemble_runs,
        "paper_mechanism_ensemble_split_summary.csv": ensemble_summary,
        "paper_mechanism_ensemble_split_feature_counts.csv": ensemble_feature_counts,
    }

    for filename, frame in outputs.items():
        out_path = output_dir / filename
        frame.to_csv(out_path, index=False)
        print(f"Saved {out_path}")

    fig_path = figure_dir / "paper_mechanism_feature_frequency.png"
    plot_frequency_counts(feature_counts, fig_path)
    print(f"Saved {fig_path}")

    single_tree_fig = figure_dir / "paper_mechanism_single_tree_split_counts.png"
    plot_split_counts_by_feature_index(
        single_tree_feature_counts[single_tree_feature_counts["dataset"] == "symmetric_two_signal_gaussian_n250_p100_i2"],
        methods=SPLIT_COUNT_METHODS_SINGLE_TREE,
        ylabel="Tree-use count",
        title="Repeated single-tree split usage by feature",
        output_path=single_tree_fig,
    )
    print(f"Saved {single_tree_fig}")

    ensemble_fig = figure_dir / "paper_mechanism_ensemble_split_counts.png"
    plot_split_counts_by_feature_index(
        ensemble_feature_counts[ensemble_feature_counts["dataset"] == "symmetric_two_signal_gaussian_n250_p100_i2"],
        methods=SPLIT_COUNT_METHODS_ENSEMBLE,
        ylabel="Tree-use count",
        title="Large-ensemble feature usage by feature",
        output_path=ensemble_fig,
    )
    print(f"Saved {ensemble_fig}")

    single_tree_density_fig = figure_dir / "paper_mechanism_single_tree_density_curves.png"
    plot_summary_curves(
        single_tree_summary[
            (single_tree_summary["design_family"] == "vary_informative_fraction")
            & (single_tree_summary["task"] == "classification")
            & (single_tree_summary["kind"] == "symmetric_two_signal_gaussian")
        ],
        methods=SPLIT_COUNT_METHODS_SINGLE_TREE,
        x_col="n_informative",
        metrics=(
            ("informative_split_share", "Informative split share"),
            ("distinct_false_features_used", "Distinct noise features used"),
        ),
        title="Single-tree split concentration as informative fraction increases",
        output_path=single_tree_density_fig,
    )
    print(f"Saved {single_tree_density_fig}")

    ensemble_density_fig = figure_dir / "paper_mechanism_ensemble_density_curves.png"
    plot_summary_curves(
        ensemble_summary[
            (ensemble_summary["design_family"] == "vary_informative_fraction")
            & (ensemble_summary["task"] == "classification")
            & (ensemble_summary["kind"] == "symmetric_two_signal_gaussian")
        ],
        methods=SPLIT_COUNT_METHODS_ENSEMBLE,
        x_col="n_informative",
        metrics=(
            ("informative_split_share", "Informative split share"),
            ("distinct_false_features_used", "Distinct noise features used"),
        ),
        title="Ensemble split concentration as informative fraction increases",
        output_path=ensemble_density_fig,
    )
    print(f"Saved {ensemble_density_fig}")

    single_tree_dimension_fig = figure_dir / "paper_mechanism_single_tree_dimension_curves.png"
    plot_summary_curves(
        single_tree_summary[
            (single_tree_summary["design_family"] == "vary_dimension_fixed_signal")
            & (single_tree_summary["task"] == "classification")
            & (single_tree_summary["kind"] == "symmetric_two_signal_gaussian")
        ],
        methods=SPLIT_COUNT_METHODS_SINGLE_TREE,
        x_col="n_features",
        metrics=(
            ("informative_split_share", "Informative split share"),
            ("distinct_false_features_used", "Distinct noise features used"),
        ),
        title="Single-tree split concentration as dimension increases",
        output_path=single_tree_dimension_fig,
    )
    print(f"Saved {single_tree_dimension_fig}")

    ensemble_dimension_fig = figure_dir / "paper_mechanism_ensemble_dimension_curves.png"
    plot_summary_curves(
        ensemble_summary[
            (ensemble_summary["design_family"] == "vary_dimension_fixed_signal")
            & (ensemble_summary["task"] == "classification")
            & (ensemble_summary["kind"] == "symmetric_two_signal_gaussian")
        ],
        methods=SPLIT_COUNT_METHODS_ENSEMBLE,
        x_col="n_features",
        metrics=(
            ("informative_split_share", "Informative split share"),
            ("distinct_false_features_used", "Distinct noise features used"),
        ),
        title="Ensemble split concentration as dimension increases",
        output_path=ensemble_dimension_fig,
    )
    print(f"Saved {ensemble_dimension_fig}")

    single_tree_regression_density_fig = figure_dir / "paper_mechanism_single_tree_regression_density_curves.png"
    plot_summary_curves(
        single_tree_summary[
            (single_tree_summary["design_family"] == "vary_informative_fraction")
            & (single_tree_summary["task"] == "regression")
        ],
        methods=SPLIT_COUNT_METHODS_SINGLE_TREE,
        x_col="n_informative",
        metrics=(
            ("informative_split_share", "Informative split share"),
            ("distinct_false_features_used", "Distinct noise features used"),
        ),
        title="Regression single-tree split concentration as informative fraction increases",
        output_path=single_tree_regression_density_fig,
    )
    print(f"Saved {single_tree_regression_density_fig}")

    ensemble_regression_density_fig = figure_dir / "paper_mechanism_ensemble_regression_density_curves.png"
    plot_summary_curves(
        ensemble_summary[
            (ensemble_summary["design_family"] == "vary_informative_fraction")
            & (ensemble_summary["task"] == "regression")
        ],
        methods=SPLIT_COUNT_METHODS_ENSEMBLE,
        x_col="n_informative",
        metrics=(
            ("informative_split_share", "Informative split share"),
            ("distinct_false_features_used", "Distinct noise features used"),
        ),
        title="Regression ensemble split concentration as informative fraction increases",
        output_path=ensemble_regression_density_fig,
    )
    print(f"Saved {ensemble_regression_density_fig}")


if __name__ == "__main__":
    main()
