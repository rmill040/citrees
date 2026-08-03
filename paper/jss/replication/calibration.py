"""Run statistical calibration and split-variable bias experiments.

The analysis separates three questions:

1. Are fixed-budget selector permutation p-values calibrated under independence?
2. How often does a fitted tree split under the global null when the feature
   gate, threshold gate, or both gates are active?
3. Does variable cardinality affect which null feature is selected at the root?

Use the quick profile for reviewer-facing replication and the full profile for
the manuscript estimates.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import chisquare, kendalltau, norm, rankdata
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from citrees import (
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
)
from citrees._permutation import collect_permutation_counts
from citrees._selector import (
    _ptest_multi,
    dc,
    mc,
    pc,
    ptest_dc,
    ptest_mc,
    ptest_mi,
    ptest_pc,
    ptest_rdc_classifier,
    ptest_rdc_regressor,
    rdc_classifier,
    rdc_regressor,
)

Task = Literal["classification", "regression"]
Stopping = Literal["exhaustive", "adaptive", "simple"]
FeatureDistribution = Literal["normal", "binary", "ordinal4"]
Gate = Literal["selector", "splitter", "combined"]
Profile = Literal["smoke", "quick", "full"]
PValueComparison = Literal["<", "<="]
ReplicateStatistic = Callable[[np.ndarray, np.ndarray], float]
SupportIndices = Sequence[int] | np.ndarray

BASE_SEED = 1718
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "calibration"
CARDINALITY_N_SAMPLES = 200
CARDINALITY_SUPPORTS = (2, 4, 10, 20, 200)
CARDINALITY_LABELS = ("binary", "4 levels", "10 levels", "20 levels", "continuous")
CARDINALITY_B = 999
CARDINALITY_ALPHA = 0.05


@dataclass(frozen=True)
class ProfileSettings:
    """Simulation counts and permutation budgets for one replication profile."""

    selector_replicates: int
    root_replicates: int
    cardinality_replicates: int
    selector_resamples: int
    root_resamples: int
    cardinality_resamples: int = CARDINALITY_B


@dataclass(frozen=True)
class CardinalityFeatureDesign:
    """Metadata for one exact-support null feature."""

    feature_id: int
    label: str
    nominal_support: int
    realized_support: int
    maximum_multiplicity: int
    tied_pair_fraction: float
    position: int


@dataclass(frozen=True)
class ControlledSplitResult:
    """Bonferroni-controlled winner weights with an explicit no-split outcome."""

    feature_weights: tuple[float, ...]
    no_split_weight: float
    minimum_p_value: float
    minimum_p_tie_size: int

    @property
    def split(self) -> bool:
        """Return whether the adjusted candidate test selected a feature."""
        return self.no_split_weight == 0.0


@dataclass(frozen=True)
class RandomizationTestResult:
    """Observed statistic and upper-tail label-randomization p-value."""

    statistic: float
    p_value: float
    extreme_count: int
    n_resamples: int


@dataclass(frozen=True)
class SelectorNullScenario:
    """Configuration for one fixed-budget selector-level null experiment."""

    task: Task
    selector: str
    feature_distribution: FeatureDistribution
    n_samples: int
    n_resamples: int
    alpha: float = 0.05

    @property
    def scenario(self) -> str:
        return (
            f"{self.task}__{self.selector}__fixed_budget__"
            f"{self.feature_distribution}__n{self.n_samples}__b{self.n_resamples}"
        )

    @property
    def data_design(self) -> str:
        return f"selector_null__{self.task}__{self.feature_distribution}__n{self.n_samples}"


@dataclass(frozen=True)
class SelectorTestResult:
    """Fixed-budget p-value and actual permutations from one selector test."""

    p_value: float
    permutations: int


@dataclass(frozen=True)
class RootNullScenario:
    """Configuration for one fitted-tree null experiment."""

    task: Task
    gate: Gate
    stopping: Stopping
    feature_scanning: bool
    threshold_scanning: bool
    feature_distribution: FeatureDistribution
    n_samples: int
    n_features: int
    base_resamples: int
    threshold_method: str = "exact"
    max_thresholds: int | None = None
    alpha: float = 0.05

    def __post_init__(self) -> None:
        """Disable scan flags that are inert under exhaustive permutation tests."""
        if self.stopping == "exhaustive":
            object.__setattr__(self, "feature_scanning", False)
            object.__setattr__(self, "threshold_scanning", False)

    @property
    def scenario(self) -> str:
        threshold = (
            self.threshold_method
            if self.max_thresholds is None
            else f"{self.threshold_method}{self.max_thresholds}"
        )
        return (
            f"{self.task}__{self.gate}__{self.stopping}__"
            f"feature_scan{int(self.feature_scanning)}__"
            f"threshold_scan{int(self.threshold_scanning)}__"
            f"{self.feature_distribution}__n{self.n_samples}__p{self.n_features}__"
            f"base_b{self.base_resamples}__{threshold}"
        )

    @property
    def data_design(self) -> str:
        return (
            f"root_null__{self.task}__{self.feature_distribution}__"
            f"n{self.n_samples}__p{self.n_features}"
        )

    @property
    def model_design(self) -> str:
        """Return the seed-pairing key with stopping and scanning factors removed."""
        threshold = (
            self.threshold_method
            if self.max_thresholds is None
            else f"{self.threshold_method}{self.max_thresholds}"
        )
        return (
            f"root_null__{self.task}__{self.gate}__{self.feature_distribution}__"
            f"n{self.n_samples}__p{self.n_features}__base_b{self.base_resamples}__"
            f"{threshold}__alpha{self.alpha:.17g}"
        )


@dataclass(frozen=True)
class RootFitResult:
    """Root split decision and actual permutation counts from one fitted tree."""

    split: bool
    selector_permutations: int
    splitter_permutations: int


def _settings(profile: Profile) -> ProfileSettings:
    if profile == "smoke":
        return ProfileSettings(
            selector_replicates=4,
            root_replicates=3,
            cardinality_replicates=8,
            selector_resamples=39,
            root_resamples=39,
        )
    if profile == "quick":
        return ProfileSettings(
            selector_replicates=200,
            root_replicates=100,
            cardinality_replicates=500,
            selector_resamples=199,
            root_resamples=199,
        )
    return ProfileSettings(
        selector_replicates=5_000,
        root_replicates=5_000,
        cardinality_replicates=10_000,
        selector_resamples=999,
        root_resamples=999,
    )


def _stream_seed(base_seed: int, design: str, replicate: int, stream: str) -> int:
    """Derive a deterministic seed for one independent simulation stream."""
    digest = hashlib.sha256(f"{stream}__{design}".encode("ascii")).digest()
    scenario_key = int.from_bytes(digest[:4], byteorder="little", signed=False)
    sequence = np.random.SeedSequence([base_seed, scenario_key, replicate])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _balanced_classes(rng: np.random.Generator, n_samples: int, n_classes: int = 2) -> np.ndarray:
    y = np.arange(n_samples, dtype=np.int64) % n_classes
    rng.shuffle(y)
    return y


def _response(rng: np.random.Generator, task: Task, n_samples: int) -> np.ndarray:
    if task == "classification":
        return _balanced_classes(rng, n_samples)
    return rng.standard_normal(n_samples)


def _feature(
    rng: np.random.Generator,
    distribution: FeatureDistribution,
    n_samples: int,
) -> np.ndarray:
    if distribution == "normal":
        return rng.standard_normal(n_samples)
    if distribution == "binary":
        return rng.integers(0, 2, size=n_samples).astype(np.float64)
    return rng.integers(0, 4, size=n_samples).astype(np.float64)


def _matrix(
    rng: np.random.Generator,
    distribution: FeatureDistribution,
    n_samples: int,
    n_features: int,
) -> np.ndarray:
    return np.column_stack([_feature(rng, distribution, n_samples) for _ in range(n_features)])


def _early_stopping(stopping: Stopping) -> str | None:
    """Map the exhaustive reference level to a full permutation test."""
    return None if stopping == "exhaustive" else stopping


def _fixed_budget_selector_p_value(
    scenario: SelectorNullScenario,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> float:
    common = {
        "x": x,
        "y": y,
        "n_resamples": scenario.n_resamples,
        "early_stopping": None,
        "alpha": scenario.alpha,
        "random_state": seed,
    }
    if scenario.task == "classification":
        if scenario.selector == "mc+rdc":
            n_classes = int(np.unique(y).size)
            return float(
                _ptest_multi(
                    funcs=[mc, rdc_classifier],
                    func_args=[n_classes, n_classes],
                    take_abs=[True, True],
                    **common,
                )
            )
        kwargs = {**common, "n_classes": int(np.unique(y).size)}
        if scenario.selector == "mc":
            return float(ptest_mc(**kwargs))
        if scenario.selector == "mi":
            return float(ptest_mi(**kwargs))
        if scenario.selector == "rdc":
            return float(ptest_rdc_classifier(**kwargs))
    else:
        if scenario.selector == "pc+dc+rdc":
            return float(
                _ptest_multi(
                    funcs=[pc, dc, rdc_regressor],
                    func_args=[True, True, True],
                    take_abs=[True, True, True],
                    **common,
                )
            )
        kwargs = {**common, "standardize": True}
        if scenario.selector == "pc":
            return float(ptest_pc(**kwargs))
        if scenario.selector == "dc":
            return float(ptest_dc(**kwargs))
        if scenario.selector == "rdc":
            return float(ptest_rdc_regressor(**kwargs))
    raise ValueError(f"Unsupported selector scenario: {scenario}")


def _fixed_budget_selector_test(
    scenario: SelectorNullScenario,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> SelectorTestResult:
    """Run one fixed-budget selector test and collect actual permutations."""
    with collect_permutation_counts() as counts:
        p_value = _fixed_budget_selector_p_value(scenario, x, y, seed)
    return SelectorTestResult(
        p_value=p_value,
        permutations=counts["selector"],
    )


def _selector_scenarios(profile: Profile, settings: ProfileSettings) -> list[SelectorNullScenario]:
    if profile == "smoke":
        return [
            SelectorNullScenario("classification", "mc", "normal", 80, settings.selector_resamples),
            SelectorNullScenario("regression", "pc", "binary", 80, settings.selector_resamples),
        ]

    scenarios = [
        SelectorNullScenario(
            task,
            selector,
            "normal",
            200,
            settings.selector_resamples,
        )
        for task, selectors in (
            ("classification", ("mc", "mi", "rdc")),
            ("regression", ("pc", "dc", "rdc")),
        )
        for selector in selectors
    ]
    scenarios.extend(
        SelectorNullScenario(
            task,
            selector,
            distribution,
            200,
            settings.selector_resamples,
        )
        for task, selector in (("classification", "mc"), ("regression", "pc"))
        for distribution in ("binary", "ordinal4")
    )
    if profile == "full":
        scenarios.extend(
            SelectorNullScenario(
                task,
                selector,
                distribution,
                n_samples,
                settings.selector_resamples,
            )
            for task, selector in (("classification", "mc"), ("regression", "pc"))
            for distribution in ("normal", "binary", "ordinal4")
            for n_samples in (100, 500)
            if n_samples != 200
        )
        scenarios.extend(
            SelectorNullScenario(
                task,
                selector,
                "normal",
                200,
                settings.selector_resamples,
            )
            for task, selector in (
                ("classification", "mc+rdc"),
                ("regression", "pc+dc+rdc"),
            )
        )
    return scenarios


def run_selector_null(
    profile: Profile,
    *,
    base_seed: int = BASE_SEED,
) -> pd.DataFrame:
    """Run fixed-budget selector permutation tests under independence."""
    settings = _settings(profile)
    rows: list[dict[str, object]] = []
    for scenario in _selector_scenarios(profile, settings):
        for replicate in range(settings.selector_replicates):
            data_seed = _stream_seed(base_seed, scenario.data_design, replicate, "data")
            model_seed = _stream_seed(base_seed, scenario.scenario, replicate, "model")
            rng = np.random.default_rng(data_seed)
            x = _feature(rng, scenario.feature_distribution, scenario.n_samples)
            y = _response(rng, scenario.task, scenario.n_samples)
            test_result = _fixed_budget_selector_test(scenario, x, y, model_seed)
            rows.append(
                {
                    **asdict(scenario),
                    "experiment": "selector_null",
                    "estimand": "fixed_budget_permutation_p_value",
                    "scenario": scenario.scenario,
                    "replicate": replicate,
                    "data_seed": data_seed,
                    "model_seed": model_seed,
                    "realized_permutations": test_result.permutations,
                    "p_value": test_result.p_value,
                    "rejected": test_result.p_value < scenario.alpha,
                }
            )
    return pd.DataFrame(rows)


def _root_scenarios(profile: Profile, settings: ProfileSettings) -> list[RootNullScenario]:
    if profile == "smoke":
        scenarios = [
            RootNullScenario(
                "classification",
                "combined",
                "adaptive",
                feature_scanning,
                threshold_scanning,
                "normal",
                80,
                3,
                settings.root_resamples,
                threshold_method="histogram",
                max_thresholds=8,
            )
            for feature_scanning in (False, True)
            for threshold_scanning in (False, True)
        ]
        scenarios.extend(
            [
                RootNullScenario(
                    "regression",
                    "selector",
                    "exhaustive",
                    False,
                    False,
                    "normal",
                    80,
                    3,
                    settings.root_resamples,
                ),
                RootNullScenario(
                    "regression",
                    "splitter",
                    "simple",
                    False,
                    True,
                    "normal",
                    80,
                    1,
                    settings.root_resamples,
                    threshold_method="histogram",
                    max_thresholds=8,
                ),
            ]
        )
        return scenarios

    stopping_modes: tuple[Stopping, ...] = (
        ("exhaustive", "adaptive", "simple") if profile == "full" else ("exhaustive", "adaptive")
    )
    scenarios = [
        RootNullScenario(
            task,
            "selector",
            stopping,
            feature_scanning,
            False,
            "normal",
            200,
            n_features,
            settings.root_resamples,
        )
        for task in ("classification", "regression")
        for stopping in stopping_modes
        for n_features in (5, 20)
        for feature_scanning in ((False,) if stopping == "exhaustive" else (False, True))
    ]
    scenarios.extend(
        RootNullScenario(
            task,
            "splitter",
            stopping,
            False,
            threshold_scanning,
            distribution,
            200,
            1,
            settings.root_resamples,
            threshold_method="histogram" if distribution == "normal" else "exact",
            max_thresholds=16 if distribution == "normal" else None,
        )
        for task in ("classification", "regression")
        for stopping in stopping_modes
        for distribution in ("binary", "ordinal4", "normal")
        for threshold_scanning in ((False,) if stopping == "exhaustive" else (False, True))
    )
    scenarios.extend(
        RootNullScenario(
            task,
            "combined",
            stopping,
            feature_scanning,
            threshold_scanning,
            "normal",
            200,
            5,
            settings.root_resamples,
            threshold_method="histogram",
            max_thresholds=16,
        )
        for task in ("classification", "regression")
        for stopping in stopping_modes
        for feature_scanning in ((False,) if stopping == "exhaustive" else (False, True))
        for threshold_scanning in ((False,) if stopping == "exhaustive" else (False, True))
    )
    if profile == "full":
        scenarios.extend(
            RootNullScenario(
                task,
                "splitter",
                stopping,
                False,
                threshold_scanning,
                "normal",
                50,
                1,
                settings.root_resamples,
            )
            for task in ("classification", "regression")
            for stopping in stopping_modes
            for threshold_scanning in ((False,) if stopping == "exhaustive" else (False, True))
        )
    return scenarios


def _fit_null_tree(
    scenario: RootNullScenario,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> RootFitResult:
    selector_enabled = scenario.gate in {"selector", "combined"}
    splitter_enabled = scenario.gate in {"splitter", "combined"}
    common: dict[str, object] = {
        "alpha_selector": scenario.alpha,
        "alpha_splitter": scenario.alpha,
        "adjust_alpha_selector": selector_enabled,
        "adjust_alpha_splitter": splitter_enabled,
        "n_resamples_selector": scenario.base_resamples if selector_enabled else None,
        "n_resamples_splitter": scenario.base_resamples if splitter_enabled else None,
        "early_stopping_selector": (
            _early_stopping(scenario.stopping) if selector_enabled else None
        ),
        "early_stopping_splitter": (
            _early_stopping(scenario.stopping) if splitter_enabled else None
        ),
        "feature_muting": False,
        "feature_scanning": scenario.feature_scanning,
        "threshold_method": scenario.threshold_method,
        "threshold_scanning": scenario.threshold_scanning,
        "max_thresholds": scenario.max_thresholds,
        "max_depth": 1,
        "random_state": seed,
        "verbose": 0,
    }
    if scenario.task == "classification":
        model = ConditionalInferenceTreeClassifier(**common)
    else:
        model = ConditionalInferenceTreeRegressor(**common)
    model.fit(X, y)
    counts = model.realized_permutation_counts_
    return RootFitResult(
        split=bool(model._n_nodes > 1),
        selector_permutations=counts["selector"],
        splitter_permutations=counts["splitter"],
    )


def run_root_null(
    profile: Profile,
    *,
    base_seed: int = BASE_SEED,
) -> pd.DataFrame:
    """Run root-level tree tests under the complete global null."""
    settings = _settings(profile)
    rows: list[dict[str, object]] = []
    for scenario in _root_scenarios(profile, settings):
        for replicate in range(settings.root_replicates):
            data_seed = _stream_seed(base_seed, scenario.data_design, replicate, "data")
            model_seed = _stream_seed(base_seed, scenario.model_design, replicate, "model")
            rng = np.random.default_rng(data_seed)
            X = _matrix(
                rng,
                scenario.feature_distribution,
                scenario.n_samples,
                scenario.n_features,
            )
            y = _response(rng, scenario.task, scenario.n_samples)
            fit_result = _fit_null_tree(scenario, X, y, model_seed)
            rows.append(
                {
                    **asdict(scenario),
                    "experiment": "root_null",
                    "estimand": "fitted_tree_split_decision",
                    "scenario": scenario.scenario,
                    "replicate": replicate,
                    "data_seed": data_seed,
                    "model_seed": model_seed,
                    "realized_selector_permutations": fit_result.selector_permutations,
                    "realized_splitter_permutations": fit_result.splitter_permutations,
                    "realized_permutations": (
                        fit_result.selector_permutations + fit_result.splitter_permutations
                    ),
                    "split": fit_result.split,
                }
            )
    return pd.DataFrame(rows)


def _cardinality_matrix(
    rng: np.random.Generator,
    n_samples: int,
) -> tuple[np.ndarray, tuple[CardinalityFeatureDesign, ...]]:
    """Construct shuffled columns with exact balanced empirical supports."""
    if n_samples != CARDINALITY_N_SAMPLES:
        raise ValueError(
            f"The cardinality design requires n_samples={CARDINALITY_N_SAMPLES}, "
            f"received {n_samples}"
        )

    columns: list[np.ndarray] = []
    for support in CARDINALITY_SUPPORTS:
        if n_samples % support != 0:
            raise ValueError(f"Support {support} does not divide n_samples={n_samples}")
        column = np.repeat(
            np.arange(support, dtype=np.float64),
            n_samples // support,
        )
        rng.shuffle(column)
        columns.append(column)

    column_order = rng.permutation(len(columns))
    X = np.column_stack([columns[feature_id] for feature_id in column_order])
    positions = {int(feature_id): int(position) for position, feature_id in enumerate(column_order)}
    total_pairs = n_samples * (n_samples - 1) // 2
    features: list[CardinalityFeatureDesign] = []
    for feature_id, (label, support, column) in enumerate(
        zip(CARDINALITY_LABELS, CARDINALITY_SUPPORTS, columns, strict=True)
    ):
        _, counts = np.unique(column, return_counts=True)
        tied_pairs = int(np.sum(counts * (counts - 1) // 2))
        features.append(
            CardinalityFeatureDesign(
                feature_id=feature_id,
                label=label,
                nominal_support=support,
                realized_support=int(counts.size),
                maximum_multiplicity=int(counts.max()),
                tied_pair_fraction=tied_pairs / total_pairs,
                position=positions[feature_id],
            )
        )
    return X, tuple(features)


def _validate_support_indices(
    support_indices: SupportIndices,
    n_resamples: int,
) -> np.ndarray:
    """Return a validated vector of zero-based permutation support indices."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    values = np.asarray(support_indices)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("support_indices must be a non-empty one-dimensional sequence")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("support_indices must contain integers")
    indices = values.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices > n_resamples):
        raise ValueError(f"support_indices must be between 0 and {n_resamples}")
    return indices


def _permutation_p_value(support_index: int, n_resamples: int) -> float:
    """Map a zero-based exceedance index to its corrected permutation p-value."""
    indices = _validate_support_indices([support_index], n_resamples)
    return float((indices[0] + 1) / (n_resamples + 1))


def _permutation_support_index(p_value: float, n_resamples: int) -> int:
    """Recover the exact zero-based support index for a permutation p-value."""
    if not np.isfinite(p_value):
        raise ValueError("p_value must be finite")
    scaled_index = p_value * (n_resamples + 1) - 1
    support_index = int(round(scaled_index))
    if not np.isclose(scaled_index, support_index, rtol=0.0, atol=1e-10):
        raise ValueError(f"p_value={p_value} is not on the B={n_resamples} permutation support")
    _validate_support_indices([support_index], n_resamples)
    return support_index


def _attainable_cutoff(
    alpha: float,
    n_resamples: int,
    comparison: PValueComparison,
) -> float | None:
    """Return the largest attainable p-value satisfying the rejection rule."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    support = np.arange(1, n_resamples + 2, dtype=np.float64) / (n_resamples + 1)
    if comparison == "<":
        eligible = support[support < alpha]
    elif comparison == "<=":
        eligible = support[support <= alpha]
    else:
        raise ValueError(f"Unsupported p-value comparison: {comparison}")
    return None if eligible.size == 0 else float(eligible[-1])


def fractional_minimum_p_weights(
    support_indices: SupportIndices,
    *,
    n_resamples: int = CARDINALITY_B,
) -> np.ndarray:
    """Assign equal winner weight to candidates tied at the minimum p-value."""
    indices = _validate_support_indices(support_indices, n_resamples)
    tied = indices == indices.min()
    weights = np.zeros(indices.size, dtype=np.float64)
    weights[tied] = 1.0 / int(tied.sum())
    return weights


def controlled_split_weights(
    support_indices: SupportIndices,
    *,
    n_resamples: int = CARDINALITY_B,
    alpha: float = CARDINALITY_ALPHA,
    comparison: PValueComparison = "<",
) -> ControlledSplitResult:
    """Apply a Bonferroni gate and retain failure to split as an outcome."""
    indices = _validate_support_indices(support_indices, n_resamples)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    forced_weights = fractional_minimum_p_weights(indices, n_resamples=n_resamples)
    minimum_p_value = _permutation_p_value(int(indices.min()), n_resamples)
    adjusted_alpha = alpha / indices.size
    if comparison == "<":
        split = minimum_p_value < adjusted_alpha
    elif comparison == "<=":
        split = minimum_p_value <= adjusted_alpha
    else:
        raise ValueError(f"Unsupported p-value comparison: {comparison}")
    feature_weights = forced_weights if split else np.zeros_like(forced_weights)
    return ControlledSplitResult(
        feature_weights=tuple(float(value) for value in feature_weights),
        no_split_weight=0.0 if split else 1.0,
        minimum_p_value=minimum_p_value,
        minimum_p_tie_size=int(np.count_nonzero(indices == indices.min())),
    )


def build_cardinality_tree_raw(
    features: Sequence[CardinalityFeatureDesign],
    support_indices: SupportIndices,
    *,
    task: Task,
    method: str,
    replicate: int,
    data_seed: int,
    model_seed: int,
    native_selected_position: int | None,
    alpha: float = CARDINALITY_ALPHA,
    comparison: PValueComparison = "<",
) -> pd.DataFrame:
    """Build one candidate-level tree record from aligned permutation results."""
    n_resamples = CARDINALITY_B
    ordered_features = tuple(sorted(features, key=lambda feature: feature.feature_id))
    expected_feature_ids = set(range(len(CARDINALITY_SUPPORTS)))
    if {feature.feature_id for feature in ordered_features} != expected_feature_ids:
        raise ValueError("features must contain each cardinality feature exactly once")
    if {feature.position for feature in ordered_features} != expected_feature_ids:
        raise ValueError("features must occupy each matrix position exactly once")

    indices = _validate_support_indices(support_indices, n_resamples)
    if indices.size != len(ordered_features):
        raise ValueError("support_indices must align with the matrix columns")
    if (
        native_selected_position is not None
        and native_selected_position not in expected_feature_ids
    ):
        raise ValueError("native_selected_position is outside the matrix")

    forced_weights = fractional_minimum_p_weights(indices, n_resamples=n_resamples)
    controlled = controlled_split_weights(
        indices,
        n_resamples=n_resamples,
        alpha=alpha,
        comparison=comparison,
    )
    adjusted_weights = np.asarray(controlled.feature_weights)
    denominator = n_resamples + 1
    rows: list[dict[str, object]] = []
    for feature in ordered_features:
        support_index = int(indices[feature.position])
        rows.append(
            {
                "task": task,
                "method": method,
                "replicate": replicate,
                "feature_id": feature.feature_id,
                "cardinality": feature.label,
                "cardinality_rank": feature.feature_id,
                "nominal_support": feature.nominal_support,
                "realized_support": feature.realized_support,
                "maximum_multiplicity": feature.maximum_multiplicity,
                "tied_pair_fraction": feature.tied_pair_fraction,
                "feature_position": feature.position,
                "data_seed": data_seed,
                "model_seed": model_seed,
                "B": n_resamples,
                "selection_alpha": alpha,
                "adjusted_alpha": alpha / len(ordered_features),
                "p_value_comparison": comparison,
                "p_value_support_index": support_index,
                "p_value_numerator": support_index + 1,
                "p_value_denominator": denominator,
                "p_value": _permutation_p_value(support_index, n_resamples),
                "minimum_p_tie_size": controlled.minimum_p_tie_size,
                "forced_winner_weight": float(forced_weights[feature.position]),
                "adjusted_winner_weight": float(adjusted_weights[feature.position]),
                "adjusted_split": controlled.split,
                "adjusted_no_split": not controlled.split,
                "no_split_weight": controlled.no_split_weight,
                "native_winner": feature.position == native_selected_position,
                "native_split": native_selected_position is not None,
                "native_no_split": native_selected_position is None,
            }
        )
    frame = pd.DataFrame(rows)
    validate_cardinality_tree_raw(frame)
    return frame


def validate_cardinality_tree_raw(raw: pd.DataFrame) -> None:
    """Validate candidate keys, seed pairing, and explicit split outcomes."""
    required = {
        "task",
        "method",
        "replicate",
        "feature_id",
        "data_seed",
        "model_seed",
        "B",
        "nominal_support",
        "realized_support",
        "feature_position",
        "p_value_support_index",
        "p_value_numerator",
        "p_value_denominator",
        "p_value",
        "forced_winner_weight",
        "adjusted_winner_weight",
        "adjusted_split",
        "adjusted_no_split",
        "no_split_weight",
        "native_winner",
        "native_split",
        "native_no_split",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing cardinality tree columns: {sorted(missing)}")
    if not raw.columns.is_unique:
        raise ValueError("Cardinality tree columns must be unique")

    keys = ["task", "replicate", "method", "feature_id"]
    if raw.duplicated(keys).any():
        raise ValueError("Cardinality tree candidate keys must be unique")
    if not raw["B"].eq(CARDINALITY_B).all():
        raise ValueError(f"Cardinality tree records require B={CARDINALITY_B}")
    paired_data_seeds = raw.groupby(["task", "replicate"], sort=False)["data_seed"].nunique()
    if not paired_data_seeds.eq(1).all():
        raise ValueError("Methods within a replicate must share one data seed")

    expected_feature_ids = set(range(len(CARDINALITY_SUPPORTS)))
    for _, group in raw.groupby(["task", "replicate", "method"], sort=False):
        if set(group["feature_id"]) != expected_feature_ids:
            raise ValueError("Every method replicate must retain all cardinality features")
        if set(group["nominal_support"]) != set(CARDINALITY_SUPPORTS):
            raise ValueError("Every method replicate must retain the planned supports")
        if not group["nominal_support"].equals(group["realized_support"]):
            raise ValueError("Realized supports must equal the planned supports")
        if set(group["feature_position"]) != expected_feature_ids:
            raise ValueError("Every method replicate must retain each feature position")
        if not (
            group["p_value_numerator"].eq(group["p_value_support_index"] + 1).all()
            and group["p_value_denominator"].eq(group["B"] + 1).all()
            and np.allclose(
                group["p_value"].to_numpy(dtype=np.float64),
                (group["p_value_numerator"] / group["p_value_denominator"]).to_numpy(
                    dtype=np.float64
                ),
            )
        ):
            raise ValueError("Permutation p-values must match their recorded support")
        if not np.isclose(float(group["forced_winner_weight"].sum()), 1.0):
            raise ValueError("Forced winner weights must sum to one")
        adjusted_no_split = group["adjusted_no_split"].astype(bool)
        if adjusted_no_split.nunique() != 1:
            raise ValueError("Adjusted split status must be constant within a replicate")
        expected_adjusted_total = 0.0 if bool(adjusted_no_split.iloc[0]) else 1.0
        if not np.isclose(
            float(group["adjusted_winner_weight"].sum()),
            expected_adjusted_total,
        ):
            raise ValueError("Adjusted winner weights disagree with split status")
        if not group["no_split_weight"].eq(float(adjusted_no_split.iloc[0])).all():
            raise ValueError("No-split weight disagrees with adjusted split status")
        if not group["adjusted_split"].eq(not bool(adjusted_no_split.iloc[0])).all():
            raise ValueError("Adjusted split indicators disagree")

        native_no_split = group["native_no_split"].astype(bool)
        if native_no_split.nunique() != 1:
            raise ValueError("Native split status must be constant within a replicate")
        expected_native_winners = 0 if bool(native_no_split.iloc[0]) else 1
        if int(group["native_winner"].sum()) != expected_native_winners:
            raise ValueError("Native winner indicators disagree with split status")
        if not group["native_split"].eq(not bool(native_no_split.iloc[0])).all():
            raise ValueError("Native split indicators disagree")


def _selected_root_feature(model: object) -> int:
    if isinstance(
        model,
        (ConditionalInferenceTreeClassifier, ConditionalInferenceTreeRegressor),
    ):
        return -1 if "value" in model.tree_ else int(model.tree_["feature"])
    return int(model.tree_.feature[0])


def _fit_cardinality_model(
    task: Task,
    method: Literal["citrees", "cart"],
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_resamples: int,
) -> object:
    if method == "citrees":
        common: dict[str, object] = {
            "alpha_selector": 1.0,
            "adjust_alpha_selector": False,
            "n_resamples_selector": n_resamples,
            "early_stopping_selector": None,
            "n_resamples_splitter": None,
            "feature_muting": False,
            "feature_scanning": False,
            "max_depth": 1,
            "random_state": seed,
            "verbose": 0,
        }
        model = (
            ConditionalInferenceTreeClassifier(**common)
            if task == "classification"
            else ConditionalInferenceTreeRegressor(**common)
        )
    else:
        model = (
            DecisionTreeClassifier(max_depth=1, random_state=seed)
            if task == "classification"
            else DecisionTreeRegressor(max_depth=1, random_state=seed)
        )
    model.fit(X, y)
    return model


def _select_cardinality_feature(
    task: Task,
    method: Literal["citrees", "partykit", "cart"],
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_resamples: int,
) -> int:
    if method == "partykit":
        from paper.benchmark.pipeline.r_methods import r_ctree_root_feature

        return r_ctree_root_feature(
            X,
            y,
            task=task,
            teststat="quadratic",
            testtype="Univariate",
            mincriterion=0.0,
            minsplit=2,
            minbucket=1,
            random_state=seed,
        )
    model = _fit_cardinality_model(task, method, X, y, seed, n_resamples)
    return _selected_root_feature(model)


def run_cardinality_bias(
    profile: Profile,
    *,
    base_seed: int = BASE_SEED,
) -> pd.DataFrame:
    """Measure root-variable selection when null features differ in cardinality."""
    settings = _settings(profile)
    rows: list[dict[str, object]] = []
    methods: tuple[Literal["citrees", "partykit", "cart"], ...] = (
        ("citrees", "cart") if profile == "smoke" else ("citrees", "partykit", "cart")
    )
    for task in ("classification", "regression"):
        data_design = f"cardinality_bias__{task}__n{CARDINALITY_N_SAMPLES}"
        for replicate in range(settings.cardinality_replicates):
            data_seed = _stream_seed(base_seed, data_design, replicate, "data")
            rng = np.random.default_rng(data_seed)
            X, features = _cardinality_matrix(rng, CARDINALITY_N_SAMPLES)
            features_by_position = {feature.position: feature for feature in features}
            y = _response(rng, task, CARDINALITY_N_SAMPLES)
            for method in methods:
                scenario = f"{task}__{method}__n{CARDINALITY_N_SAMPLES}"
                model_seed = _stream_seed(base_seed, scenario, replicate, "model")
                selected_column = _select_cardinality_feature(
                    task,
                    method,
                    X,
                    y,
                    model_seed,
                    CARDINALITY_B,
                )
                selected_design = (
                    None if selected_column < 0 else features_by_position[selected_column]
                )
                selected_feature = -1 if selected_design is None else selected_design.feature_id
                rows.append(
                    {
                        "experiment": "cardinality_bias",
                        "scenario": scenario,
                        "task": task,
                        "method": method,
                        "replicate": replicate,
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "n_samples": CARDINALITY_N_SAMPLES,
                        "n_features": len(CARDINALITY_LABELS),
                        "selection_test": (
                            "fixed_monte_carlo"
                            if method == "citrees"
                            else "asymptotic"
                            if method == "partykit"
                            else "none"
                        ),
                        "n_resamples": CARDINALITY_B if method == "citrees" else None,
                        "selected_feature": selected_feature,
                        "selected_cardinality": (
                            "no split" if selected_design is None else selected_design.label
                        ),
                    }
                )
    return pd.DataFrame(rows)


def directional_effect(
    cardinality_ranks: np.ndarray,
    winner_weights: np.ndarray,
) -> float:
    """Return the weighted cardinality rank relative to the design center."""
    ranks = np.asarray(cardinality_ranks, dtype=np.float64)
    weights = np.asarray(winner_weights, dtype=np.float64)
    if ranks.ndim != 1 or weights.ndim != 1 or ranks.shape != weights.shape:
        raise ValueError("cardinality_ranks and winner_weights must be aligned vectors")
    if ranks.size == 0 or not np.isfinite(ranks).all() or not np.isfinite(weights).all():
        raise ValueError("cardinality_ranks and winner_weights must be finite and non-empty")
    if np.any(weights < 0.0):
        raise ValueError("winner_weights must be non-negative")
    weight_total = float(weights.sum())
    if weight_total > 1.0 and not np.isclose(weight_total, 1.0):
        raise ValueError("winner_weights must sum to at most one")
    return float(np.dot(weights, ranks - ranks.mean()))


def _replicate_statistic_mean(
    labels: np.ndarray,
    values: np.ndarray,
    statistic: ReplicateStatistic,
) -> float:
    """Average a statistic over replicates while retaining defined values."""
    replicate_values = np.asarray(
        [
            statistic(label_row, value_row)
            for label_row, value_row in zip(labels, values, strict=True)
        ],
        dtype=np.float64,
    )
    defined = np.isfinite(replicate_values)
    if not defined.any():
        raise ValueError("The replicate statistic is undefined for every replicate")
    return float(replicate_values[defined].mean())


def one_sided_label_randomization(
    labels: np.ndarray,
    values: np.ndarray,
    *,
    statistic: ReplicateStatistic,
    n_resamples: int,
    random_state: int,
) -> RandomizationTestResult:
    """Test for a positive trend by permuting labels within each replicate."""
    label_matrix = np.asarray(labels, dtype=np.float64)
    value_matrix = np.asarray(values, dtype=np.float64)
    if label_matrix.ndim == 1:
        label_matrix = label_matrix[np.newaxis, :]
    if value_matrix.ndim == 1:
        value_matrix = value_matrix[np.newaxis, :]
    if label_matrix.ndim != 2 or value_matrix.ndim != 2 or label_matrix.shape != value_matrix.shape:
        raise ValueError("labels and values must be aligned replicate-by-feature matrices")
    if label_matrix.shape[1] < 2:
        raise ValueError("Each replicate must contain at least two features")
    if not np.isfinite(label_matrix).all() or not np.isfinite(value_matrix).all():
        raise ValueError("labels and values must be finite")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    observed = _replicate_statistic_mean(label_matrix, value_matrix, statistic)
    rng = np.random.default_rng(random_state)
    extreme_count = 0
    permuted_labels = np.empty_like(label_matrix)
    for _ in range(n_resamples):
        for replicate in range(label_matrix.shape[0]):
            permuted_labels[replicate] = rng.permutation(label_matrix[replicate])
        null_statistic = _replicate_statistic_mean(
            permuted_labels,
            value_matrix,
            statistic,
        )
        extreme_count += int(null_statistic >= observed)
    return RandomizationTestResult(
        statistic=observed,
        p_value=(extreme_count + 1) / (n_resamples + 1),
        extreme_count=extreme_count,
        n_resamples=n_resamples,
    )


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return Holm familywise-error adjusted p-values in the original order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("p_values must be finite and between zero and one")

    order = np.argsort(values, kind="stable")
    ordered = values[order]
    multipliers = np.arange(values.size, 0, -1, dtype=np.float64)
    adjusted_ordered = np.minimum(
        np.maximum.accumulate(ordered * multipliers),
        1.0,
    )
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = adjusted_ordered
    return adjusted


def average_ranks(values: Sequence[float]) -> np.ndarray:
    """Return ascending average ranks without resolving equal values by position."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a finite, non-empty vector")
    return np.asarray(rankdata(array, method="average"), dtype=np.float64)


def kendall_tau_b(
    cardinality_ranks: np.ndarray,
    raw_importances: np.ndarray,
) -> float:
    """Return Kendall tau-b between cardinality rank and raw importance."""
    ranks = np.asarray(cardinality_ranks, dtype=np.float64)
    importances = np.asarray(raw_importances, dtype=np.float64)
    if ranks.ndim != 1 or importances.ndim != 1 or ranks.shape != importances.shape:
        raise ValueError("cardinality_ranks and raw_importances must be aligned vectors")
    if ranks.size < 2 or not np.isfinite(ranks).all() or not np.isfinite(importances).all():
        raise ValueError("cardinality_ranks and raw_importances must be finite")
    result = kendalltau(ranks, importances, variant="b")
    return float(result.statistic)


def summarize_cardinality_forest_trends(
    raw: pd.DataFrame,
    *,
    n_resamples: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize replicate and method-level tau-b trends from raw importances."""
    required = {
        "task",
        "method",
        "replicate",
        "feature_id",
        "cardinality_rank",
        "raw_importance",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing cardinality forest columns: {sorted(missing)}")
    keys = ["task", "replicate", "method", "feature_id"]
    if raw.duplicated(keys).any():
        raise ValueError("Cardinality forest candidate keys must be unique")

    replicate_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    grouped = raw.groupby(["task", "method"], sort=True)
    for (task, method), method_group in grouped:
        label_rows: list[np.ndarray] = []
        importance_rows: list[np.ndarray] = []
        for replicate, replicate_group in method_group.groupby("replicate", sort=True):
            ordered = replicate_group.sort_values("feature_id")
            if set(ordered["feature_id"]) != set(range(len(CARDINALITY_SUPPORTS))):
                raise ValueError("Every forest replicate must contain all cardinality features")
            labels = ordered["cardinality_rank"].to_numpy(dtype=np.float64)
            importances = ordered["raw_importance"].to_numpy(dtype=np.float64)
            importance_ranks = average_ranks(importances)
            tau = kendall_tau_b(labels, importances)
            _, tie_counts = np.unique(importances, return_counts=True)
            replicate_rows.append(
                {
                    "task": task,
                    "method": method,
                    "replicate": int(replicate),
                    "kendall_tau_b": tau,
                    "maximum_importance_tie_size": int(tie_counts.max()),
                    "zero_importance_count": int(np.count_nonzero(importances == 0.0)),
                    "average_importance_ranks": tuple(float(value) for value in importance_ranks),
                }
            )
            label_rows.append(labels)
            importance_rows.append(importances)

        label_matrix = np.vstack(label_rows)
        importance_matrix = np.vstack(importance_rows)
        defined = np.asarray(
            [
                kendall_tau_b(label_row, importance_row)
                for label_row, importance_row in zip(
                    label_matrix,
                    importance_matrix,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        defined_count = int(np.isfinite(defined).sum())
        if defined_count:
            seed = _stream_seed(
                random_state,
                f"cardinality_forest__{task}__{method}",
                0,
                "randomization",
            )
            test = one_sided_label_randomization(
                label_matrix,
                importance_matrix,
                statistic=kendall_tau_b,
                n_resamples=n_resamples,
                random_state=seed,
            )
            mean_tau = test.statistic
            p_value = test.p_value
            extreme_count: int | None = test.extreme_count
        else:
            mean_tau = float("nan")
            p_value = float("nan")
            extreme_count = None
        summary_rows.append(
            {
                "task": task,
                "method": method,
                "n_replicates": label_matrix.shape[0],
                "n_defined_trends": defined_count,
                "mean_kendall_tau_b": mean_tau,
                "randomization_p_value": p_value,
                "randomization_extreme_count": extreme_count,
                "randomization_resamples": n_resamples,
            }
        )
    return pd.DataFrame(replicate_rows), pd.DataFrame(summary_rows)


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if total <= 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(0.5 + confidence / 2))
    estimate = successes / total
    denominator = 1 + z**2 / total
    center = (estimate + z**2 / (2 * total)) / denominator
    half_width = (
        z * np.sqrt(estimate * (1 - estimate) / total + z**2 / (4 * total**2)) / denominator
    )
    return max(float(center - half_width), 0.0), min(float(center + half_width), 1.0)


def _attainable_alpha(alpha: float, n_resamples: int) -> float:
    cutoff = _attainable_cutoff(alpha, n_resamples, "<")
    return 0.0 if cutoff is None else cutoff


def summarize_selector_null(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize selector-level rejection rates and Wilson intervals."""
    group_columns = [
        "estimand",
        "scenario",
        "task",
        "selector",
        "feature_distribution",
        "n_samples",
        "n_resamples",
        "alpha",
    ]
    rows = []
    for keys, group in raw.groupby(group_columns, sort=True, dropna=False):
        values = dict(zip(group_columns, keys, strict=True))
        successes = int(group["rejected"].sum())
        total = int(len(group))
        lower, upper = wilson_interval(successes, total)
        rows.append(
            {
                **values,
                "n_replicates": total,
                "n_rejections": successes,
                "rejection_rate": successes / total,
                "confidence_lower": lower,
                "confidence_upper": upper,
                "attainable_alpha": _attainable_alpha(
                    float(values["alpha"]),
                    int(values["n_resamples"]),
                ),
                "realized_permutations_total": int(group["realized_permutations"].sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_root_null(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize root split rates and Wilson intervals."""
    group_columns = [
        "estimand",
        "scenario",
        "task",
        "gate",
        "stopping",
        "feature_scanning",
        "threshold_scanning",
        "feature_distribution",
        "n_samples",
        "n_features",
        "base_resamples",
        "threshold_method",
        "max_thresholds",
        "alpha",
    ]
    rows = []
    for keys, group in raw.groupby(group_columns, sort=True, dropna=False):
        values = dict(zip(group_columns, keys, strict=True))
        successes = int(group["split"].sum())
        total = int(len(group))
        lower, upper = wilson_interval(successes, total)
        rows.append(
            {
                **values,
                "n_replicates": total,
                "n_splits": successes,
                "split_rate": successes / total,
                "confidence_lower": lower,
                "confidence_upper": upper,
                "realized_selector_permutations_total": int(
                    group["realized_selector_permutations"].sum()
                ),
                "realized_splitter_permutations_total": int(
                    group["realized_splitter_permutations"].sum()
                ),
                "realized_permutations_total": int(group["realized_permutations"].sum()),
                "realized_permutations_mean": float(group["realized_permutations"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_cardinality_bias(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize conditional root selection frequencies and global tests."""
    feature_rows: list[dict[str, object]] = []
    global_rows: list[dict[str, object]] = []
    for (task, method), group in raw.groupby(["task", "method"], sort=True):
        selected = group[group["selected_feature"] >= 0]
        total = int(len(group))
        n_splits = int(len(selected))
        counts = (
            selected["selected_feature"]
            .value_counts()
            .reindex(range(len(CARDINALITY_LABELS)), fill_value=0)
            .astype(int)
        )
        for feature_index, label in enumerate(CARDINALITY_LABELS):
            count = int(counts.loc[feature_index])
            lower, upper = wilson_interval(count, n_splits)
            feature_rows.append(
                {
                    "task": task,
                    "method": method,
                    "feature_index": feature_index,
                    "cardinality": label,
                    "n_replicates": total,
                    "n_splits": n_splits,
                    "selection_count": count,
                    "selection_rate_given_split": (count / n_splits if n_splits else float("nan")),
                    "confidence_lower": lower,
                    "confidence_upper": upper,
                }
            )
        if n_splits:
            statistic, p_value = chisquare(counts.to_numpy())
        else:
            statistic, p_value = float("nan"), float("nan")
        split_lower, split_upper = wilson_interval(n_splits, total)
        global_rows.append(
            {
                "task": task,
                "method": method,
                "n_replicates": total,
                "n_splits": n_splits,
                "split_rate": n_splits / total,
                "split_confidence_lower": split_lower,
                "split_confidence_upper": split_upper,
                "uniform_selection_chisquare": float(statistic),
                "uniform_selection_p_value": float(p_value),
            }
        )
    return pd.DataFrame(feature_rows), pd.DataFrame(global_rows)


def run_calibration(
    profile: Profile,
    *,
    base_seed: int = BASE_SEED,
) -> dict[str, pd.DataFrame]:
    """Run all calibration analyses and return raw and summary tables."""
    selector_raw = run_selector_null(profile, base_seed=base_seed)
    root_raw = run_root_null(profile, base_seed=base_seed)
    cardinality_raw = run_cardinality_bias(profile, base_seed=base_seed)
    cardinality_summary, cardinality_global = summarize_cardinality_bias(cardinality_raw)
    return {
        "selector_null_raw": selector_raw,
        "selector_null_summary": summarize_selector_null(selector_raw),
        "root_null_raw": root_raw,
        "root_null_summary": summarize_root_null(root_raw),
        "cardinality_bias_raw": cardinality_raw,
        "cardinality_bias_summary": cardinality_summary,
        "cardinality_bias_global": cardinality_global,
    }


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _r_environment() -> dict[str, str] | None:
    if shutil.which("Rscript") is None:
        return None
    result = subprocess.run(
        [
            "Rscript",
            "-e",
            'cat(R.version.string, as.character(packageVersion("partykit")), sep="\\t")',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    r_version, partykit_version = result.stdout.strip().split("\t", maxsplit=1)
    return {"r": r_version, "partykit": partykit_version}


def write_results(
    results: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    profile: Profile,
    base_seed: int,
    elapsed_seconds: float,
) -> None:
    """Write analysis tables and a machine-readable execution receipt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: list[Path] = []
    for name, frame in results.items():
        parquet_name = f"{name}.parquet"
        parquet_path = output_dir / parquet_name
        frame.to_parquet(parquet_path, index=False)
        artifact_paths.append(parquet_path)
        if name.endswith("_summary") or name.endswith("_global"):
            csv_name = f"{name}.csv"
            csv_path = output_dir / csv_name
            frame.to_csv(csv_path, index=False)
            artifact_paths.append(csv_path)

    repo_root = Path(__file__).resolve().parents[3]
    source_files = [
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
    ]
    versions = {
        package: importlib.metadata.version(package)
        for package in ("citrees", "numpy", "pandas", "scikit-learn", "scipy")
    }
    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        versions["rpy2"] = importlib.metadata.version("rpy2")

    receipt = {
        "analysis": "calibration",
        "schema_version": 3,
        "profile": profile,
        "base_seed": base_seed,
        "settings": asdict(_settings(profile)),
        "created_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "r_environment": _r_environment(),
        "source_sha256": {str(path.relative_to(repo_root)): _sha256(path) for path in source_files},
        "versions": versions,
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
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
        raise RuntimeError("The full calibration profile requires a clean source tree")
    started = time.perf_counter()
    results = run_calibration(args.profile, base_seed=args.seed)
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
