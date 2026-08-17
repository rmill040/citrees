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
import subprocess
import time
from collections.abc import Callable, Sequence, Set
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
from itertools import combinations, permutations
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
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
from paper.benchmark.pipeline.r_methods import (
    RCTreeRootDiagnostics,
    get_r_runtime_versions,
    r_cforest_importance,
    r_ctree_root_diagnostics,
)

Task = Literal["classification", "regression"]
Stopping = Literal["exhaustive", "adaptive", "simple"]
FeatureDistribution = Literal["normal", "binary", "ordinal4"]
Gate = Literal["selector", "splitter", "combined"]
Profile = Literal["smoke", "quick", "full"]
PValueComparison = Literal["<", "<="]
PValueRule = Literal["plus_one", "r_monte_carlo"]
ConditionalTreeMethod = Literal["citrees", "partykit"]
TreeMethod = Literal["citrees", "partykit", "cart"]
ForestMethod = Literal["citrees_cif", "partykit_cforest", "sklearn_rf"]
ReplicateStatistic = Callable[[np.ndarray, np.ndarray], float]
SupportIndices = Sequence[int] | np.ndarray

BASE_SEED = 1718
SEED_MODULUS = 2**32 - 1
SEED_MULTIPLIER = 2_654_435_761
SEED_BASE_MULTIPLIER = 2_246_822_519
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "calibration"
REPO_ROOT = Path(__file__).resolve().parents[3]
CARDINALITY_N_SAMPLES = 200
CARDINALITY_SUPPORTS = (2, 4, 10, 20, 200)
CARDINALITY_LABELS = ("2 levels", "4 levels", "10 levels", "20 levels", "200 levels")
CARDINALITY_DGP = "exact_balanced_ordered_discrete"
CARDINALITY_MAX_SUPPORT_FEATURE_TYPE = "balanced_200_level_ordered_discrete"
CARDINALITY_B = 999
CARDINALITY_ALPHA = 0.05
CARDINALITY_FOREST_TREES = 100
CARDINALITY_FOREST_IMPORTANCE_PERMUTATIONS = 10
CARDINALITY_EXACT_LABEL_PERMUTATIONS = 120
CARDINALITY_TREND_RESAMPLES = 9_999
CARDINALITY_TREND_RANDOMIZATION = "monte_carlo_independent_within_replicate"
CARDINALITY_TREND_P_VALUE_RULE = "plus_one"
CARDINALITY_HOLM_FAMILY = "cardinality_tree_and_forest_positive_trends"
CARDINALITY_HOLM_FAMILY_SIZE = 14


@dataclass(frozen=True)
class ProfileSettings:
    """Simulation counts and permutation budgets for one replication profile."""

    selector_replicates: int
    root_replicates: int
    cardinality_replicates: int
    cardinality_forest_replicates: int
    selector_resamples: int
    root_resamples: int


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
class PermutationSupport:
    """Exact rational support representation for one permutation p-value."""

    index: int
    numerator: int
    denominator: int


@dataclass(frozen=True)
class CardinalityCandidateDiagnostics:
    """Candidate-level diagnostics and native root outcome for one method fit."""

    statistics: np.ndarray
    p_values: np.ndarray
    native_p_values: np.ndarray
    realized_permutations: np.ndarray
    native_selected_position: int | None
    native_p_value_rule: PValueRule


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
            cardinality_replicates=1,
            cardinality_forest_replicates=1,
            selector_resamples=39,
            root_resamples=39,
        )
    if profile == "quick":
        return ProfileSettings(
            selector_replicates=200,
            root_replicates=100,
            cardinality_replicates=500,
            cardinality_forest_replicates=10,
            selector_resamples=199,
            root_resamples=199,
        )
    return ProfileSettings(
        selector_replicates=5_000,
        root_replicates=5_000,
        cardinality_replicates=10_000,
        cardinality_forest_replicates=2_500,
        selector_resamples=999,
        root_resamples=999,
    )


def _resolve_replicates(
    total: int,
    selected: Sequence[int] | None,
) -> tuple[int, ...]:
    """Return one sorted, unique, in-range replicate subset."""
    if selected is None:
        return tuple(range(total))
    if not selected:
        raise ValueError("selected replicates must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in selected
    ):
        raise TypeError("selected replicates must contain integers")
    replicates = tuple(sorted(int(value) for value in selected))
    if len(set(replicates)) != len(replicates):
        raise ValueError("selected replicates must be unique")
    if replicates[0] < 0 or replicates[-1] >= total:
        raise ValueError(f"selected replicates must lie in [0, {total})")
    return replicates


def _stream_seed(base_seed: int, design: str, replicate: int, stream: str) -> int:
    """Return an injective 32-bit seed for one frozen simulation stream."""
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    if base_seed < 0:
        raise ValueError("base_seed must be nonnegative")
    if isinstance(replicate, bool) or not isinstance(replicate, int):
        raise TypeError("replicate must be an integer")
    namespace = (stream, design)
    try:
        offset, replicate_limit = _seed_namespace_offsets()[namespace]
    except KeyError as error:
        raise ValueError(f"unregistered calibration seed namespace: {namespace!r}") from error
    if replicate < 0 or replicate >= replicate_limit:
        raise ValueError(
            f"replicate must lie in [0, {replicate_limit}) for seed namespace {namespace!r}"
        )
    ordinal = offset + replicate
    return int((base_seed * SEED_BASE_MULTIPLIER + ordinal * SEED_MULTIPLIER) % SEED_MODULUS)


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
                "rdc",
                distribution,
                200,
                settings.selector_resamples,
            )
            for task in ("classification", "regression")
            for distribution in ("binary", "ordinal4")
        )
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
    replicate_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Run fixed-budget selector permutation tests under independence."""
    settings = _settings(profile)
    replicates = _resolve_replicates(settings.selector_replicates, replicate_indices)
    rows: list[dict[str, object]] = []
    for scenario in _selector_scenarios(profile, settings):
        for replicate in replicates:
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


@lru_cache(maxsize=1)
def _seed_namespace_offsets() -> dict[tuple[str, str], tuple[int, int]]:
    """Assign disjoint ordinal ranges to every frozen calibration RNG stream."""
    limits: dict[tuple[str, str], int] = {}

    def register(stream: str, design: str, replicate_limit: int) -> None:
        key = (stream, design)
        limits[key] = max(limits.get(key, 0), replicate_limit)

    for profile in ("smoke", "quick", "full"):
        typed_profile = cast(Profile, profile)
        settings = _settings(typed_profile)
        for selector_scenario in _selector_scenarios(typed_profile, settings):
            register(
                "data",
                selector_scenario.data_design,
                settings.selector_replicates,
            )
            register(
                "model",
                selector_scenario.scenario,
                settings.selector_replicates,
            )
        for root_scenario in _root_scenarios(typed_profile, settings):
            register("data", root_scenario.data_design, settings.root_replicates)
            register("model", root_scenario.model_design, settings.root_replicates)

    full_settings = _settings("full")
    for task in ("classification", "regression"):
        data_design = f"cardinality_bias__{task}__n{CARDINALITY_N_SAMPLES}"
        register("data", data_design, full_settings.cardinality_replicates)
        for method in ("citrees", "partykit", "cart"):
            register(
                "model",
                f"{task}__{method}__n{CARDINALITY_N_SAMPLES}",
                full_settings.cardinality_replicates,
            )
        for method in ("citrees_cif", "partykit_cforest", "sklearn_rf"):
            register(
                "model",
                f"{task}__{method}__n{CARDINALITY_N_SAMPLES}",
                full_settings.cardinality_forest_replicates,
            )
        for method in ("citrees", "partykit"):
            for estimand in ("forced_winner", "adjusted_winner_given_split"):
                register(
                    "trend_randomization",
                    f"tree__{task}__{method}__{estimand}",
                    1,
                )
        for method in ("citrees_cif", "partykit_cforest", "sklearn_rf"):
            register(
                "trend_randomization",
                f"forest__{task}__{method}__importance_trend",
                1,
            )

    offsets: dict[tuple[str, str], tuple[int, int]] = {}
    offset = 0
    for namespace in sorted(limits):
        replicate_limit = limits[namespace]
        offsets[namespace] = (offset, replicate_limit)
        offset += replicate_limit
    if offset >= SEED_MODULUS:
        raise RuntimeError("calibration seed inventory exceeds the supported 32-bit space")
    return offsets


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
    replicate_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Run root-level tree tests under the complete global null."""
    settings = _settings(profile)
    replicates = _resolve_replicates(settings.root_replicates, replicate_indices)
    rows: list[dict[str, object]] = []
    for scenario in _root_scenarios(profile, settings):
        for replicate in replicates:
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
    """Construct shuffled exact-support ordered discrete columns.

    All five predictors are discrete. In particular, the maximum-support
    predictor contains each integer level from 0 through 199 exactly once; it
    is not sampled from a continuous distribution.
    """
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


def cardinality_design() -> pd.DataFrame:
    """Return the frozen cardinality data-generating process as a table."""
    total_pairs = CARDINALITY_N_SAMPLES * (CARDINALITY_N_SAMPLES - 1) // 2
    rows: list[dict[str, object]] = []
    for feature_id, (label, support) in enumerate(
        zip(CARDINALITY_LABELS, CARDINALITY_SUPPORTS, strict=True)
    ):
        multiplicity = CARDINALITY_N_SAMPLES // support
        tied_pairs = support * multiplicity * (multiplicity - 1) // 2
        rows.append(
            {
                "experiment": "cardinality_design",
                "dgp": CARDINALITY_DGP,
                "n_samples": CARDINALITY_N_SAMPLES,
                "n_features": len(CARDINALITY_SUPPORTS),
                "feature_id": feature_id,
                "cardinality": label,
                "feature_type": "ordered_discrete",
                "nominal_support": support,
                "realized_support": support,
                "maximum_multiplicity": multiplicity,
                "tied_pair_fraction": tied_pairs / total_pairs,
                "balance": "exact",
                "column_order": "randomized_per_replicate",
                "maximum_support_feature_type": CARDINALITY_MAX_SUPPORT_FEATURE_TYPE,
            }
        )
    return pd.DataFrame(rows)


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


def _permutation_support(
    p_value: float,
    n_resamples: int,
    rule: PValueRule,
) -> PermutationSupport:
    """Return the exact support code for a method-specific permutation p-value."""
    if not np.isfinite(p_value):
        raise ValueError("p_value must be finite")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if rule == "plus_one":
        denominator = n_resamples + 1
        numerator = int(round(p_value * denominator))
        minimum_numerator = 1
        support_index = numerator - 1
    elif rule == "r_monte_carlo":
        denominator = n_resamples
        numerator = int(round(p_value * denominator))
        minimum_numerator = 0
        support_index = numerator
    else:
        raise ValueError(f"Unsupported p-value rule: {rule}")
    if (
        numerator < minimum_numerator
        or numerator > denominator
        or not np.isclose(
            p_value,
            numerator / denominator,
            rtol=0.0,
            atol=1e-10,
        )
    ):
        raise ValueError(f"p_value={p_value} is not on the B={n_resamples} support for rule={rule}")
    return PermutationSupport(
        index=support_index,
        numerator=numerator,
        denominator=denominator,
    )


def _permutation_p_value(support_index: int, n_resamples: int) -> float:
    """Map a zero-based exceedance index to its corrected permutation p-value."""
    indices = _validate_support_indices([support_index], n_resamples)
    return float((indices[0] + 1) / (n_resamples + 1))


def _permutation_support_index(p_value: float, n_resamples: int) -> int:
    """Recover the exact zero-based support index for a permutation p-value."""
    return _permutation_support(p_value, n_resamples, "plus_one").index


def _attainable_cutoff(
    alpha: float,
    n_resamples: int,
    comparison: PValueComparison,
    *,
    rule: PValueRule = "plus_one",
) -> float | None:
    """Return the largest attainable p-value satisfying the rejection rule."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if rule == "plus_one":
        support = np.arange(1, n_resamples + 2, dtype=np.float64) / (n_resamples + 1)
    elif rule == "r_monte_carlo":
        support = np.arange(0, n_resamples + 1, dtype=np.float64) / n_resamples
    else:
        raise ValueError(f"Unsupported p-value rule: {rule}")
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
    rule: PValueRule = "plus_one",
    alpha: float = CARDINALITY_ALPHA,
    comparison: PValueComparison = "<",
) -> ControlledSplitResult:
    """Apply a Bonferroni gate and retain failure to split as an outcome."""
    indices = _validate_support_indices(support_indices, n_resamples)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    forced_weights = fractional_minimum_p_weights(
        indices,
        n_resamples=n_resamples,
    )
    if rule == "plus_one":
        minimum_p_value = _permutation_p_value(int(indices.min()), n_resamples)
    elif rule == "r_monte_carlo":
        minimum_p_value = float(indices.min() / n_resamples)
    else:
        raise ValueError(f"Unsupported p-value rule: {rule}")
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
    diagnostics: CardinalityCandidateDiagnostics,
    *,
    task: Task,
    method: ConditionalTreeMethod,
    selector: str,
    replicate: int,
    data_seed: int,
    model_seed: int,
    alpha: float = CARDINALITY_ALPHA,
    comparison: PValueComparison = "<",
) -> pd.DataFrame:
    """Build one candidate-level tree record from aligned permutation results."""
    ordered_features = tuple(sorted(features, key=lambda feature: feature.feature_id))
    expected_feature_ids = set(range(len(CARDINALITY_SUPPORTS)))
    if {feature.feature_id for feature in ordered_features} != expected_feature_ids:
        raise ValueError("features must contain each cardinality feature exactly once")
    if {feature.position for feature in ordered_features} != expected_feature_ids:
        raise ValueError("features must occupy each matrix position exactly once")
    expected_native_rule: PValueRule = "plus_one" if method == "citrees" else "r_monte_carlo"
    if diagnostics.native_p_value_rule != expected_native_rule:
        raise ValueError(
            f"{method} cardinality diagnostics require native rule={expected_native_rule}"
        )
    statistics = np.asarray(diagnostics.statistics, dtype=np.float64)
    p_values = np.asarray(diagnostics.p_values, dtype=np.float64)
    native_p_values = np.asarray(diagnostics.native_p_values, dtype=np.float64)
    realized_permutations = np.asarray(diagnostics.realized_permutations)
    for name, values in (
        ("statistics", statistics),
        ("p_values", p_values),
        ("native_p_values", native_p_values),
        ("realized_permutations", realized_permutations),
    ):
        if values.ndim != 1 or values.size != len(ordered_features):
            raise ValueError(f"{name} must align with the matrix columns")
    if (
        not np.isfinite(statistics).all()
        or not np.isfinite(p_values).all()
        or not np.isfinite(native_p_values).all()
    ):
        raise ValueError("Candidate statistics and p-values must be finite")
    expected_resamples = (
        CARDINALITY_B * len(ordered_features) if method == "citrees" else CARDINALITY_B
    )
    if (
        not np.issubdtype(realized_permutations.dtype, np.integer)
        or not np.asarray(realized_permutations == expected_resamples).all()
    ):
        raise ValueError(f"{method} candidate tests must realize {expected_resamples} permutations")
    supports = [
        _permutation_support(float(p_value), expected_resamples, "plus_one") for p_value in p_values
    ]
    native_supports = [
        _permutation_support(
            float(p_value),
            expected_resamples,
            diagnostics.native_p_value_rule,
        )
        for p_value in native_p_values
    ]
    indices = np.asarray([support.index for support in supports], dtype=np.int64)
    native_selected_position = diagnostics.native_selected_position
    if (
        native_selected_position is not None
        and native_selected_position not in expected_feature_ids
    ):
        raise ValueError("native_selected_position is outside the matrix")

    forced_weights = fractional_minimum_p_weights(
        indices,
        n_resamples=expected_resamples,
    )
    controlled = controlled_split_weights(
        indices,
        n_resamples=expected_resamples,
        rule="plus_one",
        alpha=alpha,
        comparison=comparison,
    )
    adjusted_weights = np.asarray(controlled.feature_weights)
    minimum_index = int(indices.min())
    all_p_equal = bool(np.all(indices == minimum_index))
    native_selected_feature = (
        -1
        if native_selected_position is None
        else next(
            feature.feature_id
            for feature in ordered_features
            if feature.position == native_selected_position
        )
    )
    rows: list[dict[str, object]] = []
    for feature in ordered_features:
        support = supports[feature.position]
        rows.append(
            {
                "experiment": "cardinality_tree",
                "dgp": CARDINALITY_DGP,
                "maximum_support_feature_type": CARDINALITY_MAX_SUPPORT_FEATURE_TYPE,
                "task": task,
                "method": method,
                "selection_test": "fixed_monte_carlo",
                "selector": selector,
                "replicate": replicate,
                "feature_id": feature.feature_id,
                "cardinality": feature.label,
                "cardinality_rank": feature.feature_id,
                "feature_type": "ordered_discrete",
                "nominal_support": feature.nominal_support,
                "realized_support": feature.realized_support,
                "maximum_multiplicity": feature.maximum_multiplicity,
                "tied_pair_fraction": feature.tied_pair_fraction,
                "feature_position": feature.position,
                "n_samples": CARDINALITY_N_SAMPLES,
                "n_features": len(CARDINALITY_SUPPORTS),
                "data_seed": data_seed,
                "model_seed": model_seed,
                "B": CARDINALITY_B,
                "realized_permutations": int(realized_permutations[feature.position]),
                "selection_alpha": alpha,
                "adjusted_alpha": alpha / len(ordered_features),
                "p_value_comparison": comparison,
                "p_value_rule": "plus_one",
                "native_p_value_rule": diagnostics.native_p_value_rule,
                "attainable_adjusted_cutoff": _attainable_cutoff(
                    alpha / len(ordered_features),
                    expected_resamples,
                    comparison,
                    rule="plus_one",
                ),
                "candidate_statistic": float(statistics[feature.position]),
                "p_value_support_index": support.index,
                "p_value_numerator": support.numerator,
                "p_value_denominator": support.denominator,
                "p_value": float(p_values[feature.position]),
                "native_p_value_support_index": native_supports[feature.position].index,
                "native_p_value_numerator": native_supports[feature.position].numerator,
                "native_p_value_denominator": native_supports[feature.position].denominator,
                "native_p_value": float(native_p_values[feature.position]),
                "minimum_p_tie_size": controlled.minimum_p_tie_size,
                "all_p_equal": all_p_equal,
                "is_minimum_p": support.index == minimum_index,
                "forced_winner_weight": float(forced_weights[feature.position]),
                "adjusted_winner_weight": float(adjusted_weights[feature.position]),
                "adjusted_split": controlled.split,
                "adjusted_no_split": not controlled.split,
                "no_split_weight": controlled.no_split_weight,
                "native_winner": feature.position == native_selected_position,
                "native_split": native_selected_position is not None,
                "native_no_split": native_selected_position is None,
                "native_selected_position": (
                    -1 if native_selected_position is None else native_selected_position
                ),
                "native_selected_feature": native_selected_feature,
                "native_score_tie_size": pd.NA,
            }
        )
    frame = pd.DataFrame(rows)
    validate_cardinality_tree_raw(frame)
    return frame


def build_cardinality_cart_raw(
    features: Sequence[CardinalityFeatureDesign],
    candidate_scores: np.ndarray,
    *,
    task: Task,
    replicate: int,
    data_seed: int,
    model_seed: int,
    native_selected_position: int | None,
) -> pd.DataFrame:
    """Build candidate-level native CART evidence without permutation fields."""
    ordered_features = tuple(sorted(features, key=lambda feature: feature.feature_id))
    scores = np.asarray(candidate_scores, dtype=np.float64)
    if scores.shape != (len(ordered_features),) or not np.isfinite(scores).all():
        raise ValueError("candidate_scores must be one finite value per matrix column")
    expected_positions = set(range(len(ordered_features)))
    if {feature.position for feature in ordered_features} != expected_positions:
        raise ValueError("features must occupy each matrix position exactly once")
    if native_selected_position is not None and native_selected_position not in expected_positions:
        raise ValueError("native_selected_position is outside the matrix")
    maximum_score = float(scores.max())
    score_tie_size = int(np.count_nonzero(scores == maximum_score))
    native_selected_feature = (
        -1
        if native_selected_position is None
        else next(
            feature.feature_id
            for feature in ordered_features
            if feature.position == native_selected_position
        )
    )
    rows: list[dict[str, object]] = []
    for feature in ordered_features:
        rows.append(
            {
                "experiment": "cardinality_tree",
                "dgp": CARDINALITY_DGP,
                "maximum_support_feature_type": CARDINALITY_MAX_SUPPORT_FEATURE_TYPE,
                "task": task,
                "method": "cart",
                "selection_test": "impurity",
                "selector": "gini" if task == "classification" else "squared_error",
                "replicate": replicate,
                "feature_id": feature.feature_id,
                "cardinality": feature.label,
                "cardinality_rank": feature.feature_id,
                "feature_type": "ordered_discrete",
                "nominal_support": feature.nominal_support,
                "realized_support": feature.realized_support,
                "maximum_multiplicity": feature.maximum_multiplicity,
                "tied_pair_fraction": feature.tied_pair_fraction,
                "feature_position": feature.position,
                "n_samples": CARDINALITY_N_SAMPLES,
                "n_features": len(CARDINALITY_SUPPORTS),
                "data_seed": data_seed,
                "model_seed": model_seed,
                "B": pd.NA,
                "realized_permutations": 0,
                "selection_alpha": pd.NA,
                "adjusted_alpha": pd.NA,
                "p_value_comparison": "none",
                "p_value_rule": "none",
                "native_p_value_rule": "none",
                "attainable_adjusted_cutoff": pd.NA,
                "candidate_statistic": float(scores[feature.position]),
                "p_value_support_index": pd.NA,
                "p_value_numerator": pd.NA,
                "p_value_denominator": pd.NA,
                "p_value": pd.NA,
                "native_p_value_support_index": pd.NA,
                "native_p_value_numerator": pd.NA,
                "native_p_value_denominator": pd.NA,
                "native_p_value": pd.NA,
                "minimum_p_tie_size": pd.NA,
                "all_p_equal": pd.NA,
                "is_minimum_p": pd.NA,
                "forced_winner_weight": pd.NA,
                "adjusted_winner_weight": pd.NA,
                "adjusted_split": pd.NA,
                "adjusted_no_split": pd.NA,
                "no_split_weight": pd.NA,
                "native_winner": feature.position == native_selected_position,
                "native_split": native_selected_position is not None,
                "native_no_split": native_selected_position is None,
                "native_selected_position": (
                    -1 if native_selected_position is None else native_selected_position
                ),
                "native_selected_feature": native_selected_feature,
                "native_score_tie_size": score_tie_size,
            }
        )
    frame = pd.DataFrame(rows)
    validate_cardinality_tree_raw(frame)
    return frame


def validate_cardinality_tree_raw(raw: pd.DataFrame) -> None:
    """Validate candidate support, pairing, weights, and split outcomes."""
    required = {
        "experiment",
        "dgp",
        "maximum_support_feature_type",
        "task",
        "method",
        "selection_test",
        "selector",
        "replicate",
        "feature_id",
        "cardinality",
        "cardinality_rank",
        "feature_type",
        "data_seed",
        "model_seed",
        "B",
        "nominal_support",
        "realized_support",
        "maximum_multiplicity",
        "tied_pair_fraction",
        "feature_position",
        "n_samples",
        "n_features",
        "realized_permutations",
        "selection_alpha",
        "adjusted_alpha",
        "p_value_comparison",
        "p_value_rule",
        "native_p_value_rule",
        "attainable_adjusted_cutoff",
        "candidate_statistic",
        "p_value_support_index",
        "p_value_numerator",
        "p_value_denominator",
        "p_value",
        "native_p_value_support_index",
        "native_p_value_numerator",
        "native_p_value_denominator",
        "native_p_value",
        "minimum_p_tie_size",
        "all_p_equal",
        "is_minimum_p",
        "forced_winner_weight",
        "adjusted_winner_weight",
        "adjusted_split",
        "adjusted_no_split",
        "no_split_weight",
        "native_winner",
        "native_split",
        "native_no_split",
        "native_selected_position",
        "native_selected_feature",
        "native_score_tie_size",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing cardinality tree columns: {sorted(missing)}")
    if not raw.columns.is_unique:
        raise ValueError("Cardinality tree columns must be unique")

    keys = ["task", "replicate", "method", "feature_id"]
    if raw.duplicated(keys).any():
        raise ValueError("Cardinality tree candidate keys must be unique")
    if not raw["experiment"].eq("cardinality_tree").all():
        raise ValueError("Tree rows require experiment='cardinality_tree'")
    if not raw["dgp"].eq(CARDINALITY_DGP).all():
        raise ValueError(f"Tree rows require dgp={CARDINALITY_DGP}")
    if not raw["maximum_support_feature_type"].eq(CARDINALITY_MAX_SUPPORT_FEATURE_TYPE).all():
        raise ValueError("Tree rows do not identify the 200-level discrete predictor")
    if not raw["feature_type"].eq("ordered_discrete").all():
        raise ValueError("Every cardinality predictor must be labeled ordered discrete")
    if not raw["n_samples"].eq(CARDINALITY_N_SAMPLES).all():
        raise ValueError("Tree rows have an inconsistent sample size")
    if not raw["n_features"].eq(len(CARDINALITY_SUPPORTS)).all():
        raise ValueError("Tree rows have an inconsistent feature count")
    paired_data_seeds = raw.groupby(["task", "replicate"], sort=False)["data_seed"].nunique()
    if not paired_data_seeds.eq(1).all():
        raise ValueError("Methods within a replicate must share one data seed")

    expected_feature_ids = set(range(len(CARDINALITY_SUPPORTS)))
    expected_multiplicities = np.asarray(
        [CARDINALITY_N_SAMPLES // support for support in CARDINALITY_SUPPORTS],
        dtype=np.int64,
    )
    total_pairs = CARDINALITY_N_SAMPLES * (CARDINALITY_N_SAMPLES - 1) // 2
    expected_tied_fractions = np.asarray(
        [
            support * (multiplicity * (multiplicity - 1) // 2) / total_pairs
            for support, multiplicity in zip(
                CARDINALITY_SUPPORTS,
                expected_multiplicities,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    position_mappings: dict[tuple[object, object], tuple[int, ...]] = {}
    for _, group in raw.groupby(["task", "replicate", "method"], sort=False):
        group = group.sort_values("feature_id")
        if set(group["feature_id"]) != expected_feature_ids:
            raise ValueError("Every method replicate must retain all cardinality features")
        if not group["cardinality_rank"].eq(group["feature_id"]).all():
            raise ValueError("Cardinality rank must equal feature_id")
        if tuple(group["cardinality"]) != CARDINALITY_LABELS:
            raise ValueError("Cardinality labels must follow the frozen DGP")
        if tuple(int(value) for value in group["nominal_support"]) != CARDINALITY_SUPPORTS:
            raise ValueError("Every method replicate must retain the planned supports")
        if not np.array_equal(
            group["nominal_support"].to_numpy(dtype=np.int64),
            group["realized_support"].to_numpy(dtype=np.int64),
        ):
            raise ValueError("Realized supports must equal the planned supports")
        if not np.array_equal(
            group["maximum_multiplicity"].to_numpy(dtype=np.int64),
            expected_multiplicities,
        ):
            raise ValueError("Maximum multiplicities disagree with the frozen DGP")
        if not np.allclose(
            group["tied_pair_fraction"].to_numpy(dtype=np.float64),
            expected_tied_fractions,
        ):
            raise ValueError("Tied-pair fractions disagree with the frozen DGP")
        if set(group["feature_position"]) != expected_feature_ids:
            raise ValueError("Every method replicate must retain each feature position")
        pair_key = (group["task"].iloc[0], group["replicate"].iloc[0])
        positions = tuple(int(value) for value in group["feature_position"])
        reference_positions = position_mappings.setdefault(pair_key, positions)
        if positions != reference_positions:
            raise ValueError("Methods within a replicate must share the position mapping")

        native_no_split = group["native_no_split"].astype(bool)
        if native_no_split.nunique() != 1:
            raise ValueError("Native split status must be constant within a replicate")
        expected_native_winners = 0 if bool(native_no_split.iloc[0]) else 1
        if int(group["native_winner"].sum()) != expected_native_winners:
            raise ValueError("Native winner indicators disagree with split status")
        if not group["native_split"].eq(not bool(native_no_split.iloc[0])).all():
            raise ValueError("Native split indicators disagree")
        selected_positions = group["native_selected_position"].astype(int)
        selected_features = group["native_selected_feature"].astype(int)
        if selected_positions.nunique() != 1 or selected_features.nunique() != 1:
            raise ValueError("Native selected-feature metadata must be constant")
        selected_position = int(selected_positions.iloc[0])
        selected_feature = int(selected_features.iloc[0])
        if bool(native_no_split.iloc[0]):
            if selected_position != -1 or selected_feature != -1:
                raise ValueError("Native no-split rows require selected-feature sentinel -1")
        else:
            winner = group[group["native_winner"]]
            if (
                selected_position not in expected_feature_ids
                or selected_feature not in expected_feature_ids
                or int(winner["feature_position"].iloc[0]) != selected_position
                or int(winner["feature_id"].iloc[0]) != selected_feature
            ):
                raise ValueError("Native winner disagrees with selected-feature metadata")

        method = str(group["method"].iloc[0])
        if method == "cart":
            nullable = [
                "B",
                "selection_alpha",
                "adjusted_alpha",
                "attainable_adjusted_cutoff",
                "p_value_support_index",
                "p_value_numerator",
                "p_value_denominator",
                "p_value",
                "native_p_value_support_index",
                "native_p_value_numerator",
                "native_p_value_denominator",
                "native_p_value",
                "minimum_p_tie_size",
                "all_p_equal",
                "is_minimum_p",
                "forced_winner_weight",
                "adjusted_winner_weight",
                "adjusted_split",
                "adjusted_no_split",
                "no_split_weight",
            ]
            if not group[nullable].isna().all().all():
                raise ValueError("CART rows must not contain permutation outcomes")
            if not group["p_value_comparison"].eq("none").all():
                raise ValueError("CART rows require p_value_comparison='none'")
            if not group["p_value_rule"].eq("none").all():
                raise ValueError("CART rows require p_value_rule='none'")
            if not group["native_p_value_rule"].eq("none").all():
                raise ValueError("CART rows require native_p_value_rule='none'")
            scores = group["candidate_statistic"].to_numpy(dtype=np.float64)
            if not np.isfinite(scores).all():
                raise ValueError("CART candidate statistics must be finite")
            tie_size = int(np.count_nonzero(scores == scores.max()))
            if not group["native_score_tie_size"].eq(tie_size).all():
                raise ValueError("CART score-tie size is inconsistent")
            if not bool(native_no_split.iloc[0]):
                winner_score = float(
                    group.loc[group["native_winner"], "candidate_statistic"].iloc[0]
                )
                if not np.isclose(winner_score, float(scores.max())):
                    raise ValueError("CART native winner must maximize impurity decrease")
            continue

        if method not in {"citrees", "partykit"}:
            raise ValueError(f"Unsupported cardinality tree method: {method}")
        if not group["B"].eq(CARDINALITY_B).all():
            raise ValueError(f"Conditional tree records require base B={CARDINALITY_B}")
        n_tests = len(CARDINALITY_SUPPORTS)
        realized = CARDINALITY_B * n_tests if method == "citrees" else CARDINALITY_B
        native_rule: PValueRule = "plus_one" if method == "citrees" else "r_monte_carlo"
        if not group["realized_permutations"].eq(realized).all():
            raise ValueError(f"{method} rows require {realized} realized permutations")
        if not group["p_value_rule"].eq("plus_one").all():
            raise ValueError("Common p-values require the plus-one rule")
        if not group["native_p_value_rule"].eq(native_rule).all():
            raise ValueError(f"{method} rows require native rule={native_rule}")
        if group["p_value_comparison"].nunique() != 1 or group["p_value_comparison"].iloc[
            0
        ] not in {"<", "<="}:
            raise ValueError("Conditional rows require one explicit comparison rule")
        comparison = group["p_value_comparison"].iloc[0]

        supports = [
            _permutation_support(float(value), realized, "plus_one") for value in group["p_value"]
        ]
        native_supports = [
            _permutation_support(float(value), realized, native_rule)
            for value in group["native_p_value"]
        ]
        for prefix, expected_supports in (
            ("p_value", supports),
            ("native_p_value", native_supports),
        ):
            if not np.array_equal(
                group[f"{prefix}_support_index"].to_numpy(dtype=np.int64),
                np.asarray([support.index for support in expected_supports]),
            ):
                raise ValueError(f"{prefix} support indices are invalid")
            if not np.array_equal(
                group[f"{prefix}_numerator"].to_numpy(dtype=np.int64),
                np.asarray([support.numerator for support in expected_supports]),
            ):
                raise ValueError(f"{prefix} numerators are invalid")
            if not np.array_equal(
                group[f"{prefix}_denominator"].to_numpy(dtype=np.int64),
                np.asarray([support.denominator for support in expected_supports]),
            ):
                raise ValueError(f"{prefix} denominators are invalid")
        if method == "partykit":
            expected_corrected = (
                group["native_p_value_numerator"].to_numpy(dtype=np.int64) + 1
            ) / (CARDINALITY_B + 1)
            if not np.allclose(
                group["p_value"].to_numpy(dtype=np.float64),
                expected_corrected,
            ):
                raise ValueError("partykit p-values must use the plus-one conversion")
        elif not np.allclose(
            group["p_value"].to_numpy(dtype=np.float64),
            group["native_p_value"].to_numpy(dtype=np.float64),
        ):
            raise ValueError("citrees native and common p-values must match")

        indices = np.asarray([support.index for support in supports], dtype=np.int64)
        minimum = int(indices.min())
        is_minimum = indices == minimum
        tie_size = int(is_minimum.sum())
        if not group["is_minimum_p"].astype(bool).eq(is_minimum).all():
            raise ValueError("Minimum-p indicators are inconsistent")
        if not group["minimum_p_tie_size"].eq(tie_size).all():
            raise ValueError("Minimum-p tie sizes are inconsistent")
        if not group["all_p_equal"].eq(tie_size == n_tests).all():
            raise ValueError("All-p-equal indicators are inconsistent")
        expected_forced = np.zeros(n_tests, dtype=np.float64)
        expected_forced[is_minimum] = 1.0 / tie_size
        forced = group["forced_winner_weight"].to_numpy(dtype=np.float64)
        if not np.allclose(forced, expected_forced):
            raise ValueError("Forced winner weights must cover only minimum-p ties")

        controlled = controlled_split_weights(
            indices,
            n_resamples=realized,
            rule="plus_one",
            alpha=CARDINALITY_ALPHA,
            comparison=comparison,
        )
        expected_adjusted = np.asarray(controlled.feature_weights)
        adjusted = group["adjusted_winner_weight"].to_numpy(dtype=np.float64)
        if not np.allclose(adjusted, expected_adjusted):
            raise ValueError("Adjusted winner weights are inconsistent")
        if not group["adjusted_split"].eq(controlled.split).all():
            raise ValueError("Adjusted split indicators are inconsistent")
        if not group["adjusted_no_split"].eq(not controlled.split).all():
            raise ValueError("Adjusted no-split indicators are inconsistent")
        if not group["no_split_weight"].eq(controlled.no_split_weight).all():
            raise ValueError("No-split weights are inconsistent")
        expected_cutoff = _attainable_cutoff(
            CARDINALITY_ALPHA / n_tests,
            realized,
            comparison,
            rule="plus_one",
        )
        if not np.allclose(
            group["attainable_adjusted_cutoff"].to_numpy(dtype=np.float64),
            expected_cutoff,
        ):
            raise ValueError("Attainable adjusted cutoff is inconsistent")
        if not group["selection_alpha"].eq(CARDINALITY_ALPHA).all():
            raise ValueError("Selection alpha is inconsistent")
        if not group["adjusted_alpha"].eq(CARDINALITY_ALPHA / n_tests).all():
            raise ValueError("Adjusted alpha is inconsistent")
        if not np.isfinite(group["candidate_statistic"].to_numpy(dtype=np.float64)).all():
            raise ValueError("Candidate statistics must be finite")
        if not bool(native_no_split.iloc[0]):
            native_indices = np.asarray(
                [support.index for support in native_supports],
                dtype=np.int64,
            )
            winner_index = int(
                group.loc[group["native_winner"], "native_p_value_support_index"].iloc[0]
            )
            if winner_index != int(native_indices.min()):
                raise ValueError("Native winner must have a minimum candidate p-value")
        if method == "citrees" and bool(native_no_split.iloc[0]) == controlled.split:
            raise ValueError("citrees native and reconstructed adjusted split statuses disagree")
        if method == "partykit" and controlled.split and bool(native_no_split.iloc[0]):
            raise ValueError("partykit cannot reject after correction but retain no split")


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
) -> object:
    if method == "citrees":
        common: dict[str, object] = {
            "alpha_selector": CARDINALITY_ALPHA,
            "adjust_alpha_selector": True,
            "n_resamples_selector": CARDINALITY_B,
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


def _citrees_cardinality_diagnostics(
    task: Task,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> CardinalityCandidateDiagnostics:
    """Compute candidate tests and the native citrees Bonferroni decision."""
    n_tests = X.shape[1]
    adjusted_alpha = CARDINALITY_ALPHA / n_tests
    realized_budget = CARDINALITY_B * n_tests
    statistics = np.empty(n_tests, dtype=np.float64)
    p_values = np.empty(n_tests, dtype=np.float64)
    realized = np.empty(n_tests, dtype=np.int64)
    for position in range(n_tests):
        x = X[:, position]
        with collect_permutation_counts() as counts:
            if task == "classification":
                statistics[position] = mc(
                    x,
                    y,
                    int(np.unique(y).size),
                    random_state=seed,
                )
                p_values[position] = ptest_mc(
                    x=x,
                    y=y,
                    n_classes=int(np.unique(y).size),
                    n_resamples=realized_budget,
                    early_stopping=None,
                    alpha=adjusted_alpha,
                    random_state=seed,
                )
            else:
                statistics[position] = abs(pc(x, y, True, random_state=seed))
                p_values[position] = ptest_pc(
                    x=x,
                    y=y,
                    standardize=True,
                    n_resamples=realized_budget,
                    early_stopping=None,
                    alpha=adjusted_alpha,
                    random_state=seed,
                )
        realized[position] = counts["selector"]
    model = _fit_cardinality_model(task, "citrees", X, y, seed)
    selected = _selected_root_feature(model)
    return CardinalityCandidateDiagnostics(
        statistics=statistics,
        p_values=p_values,
        native_p_values=p_values.copy(),
        realized_permutations=realized,
        native_selected_position=None if selected < 0 else selected,
        native_p_value_rule="plus_one",
    )


def _partykit_cardinality_diagnostics(
    task: Task,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> CardinalityCandidateDiagnostics:
    """Retain partykit's native lattice alongside corrected p-values."""
    diagnostics: RCTreeRootDiagnostics = r_ctree_root_diagnostics(
        X,
        y,
        task=task,
        teststat="quadratic",
        testtype="MonteCarlo",
        nresample=CARDINALITY_B,
        mincriterion=1.0 - CARDINALITY_ALPHA / X.shape[1],
        minsplit=2,
        minbucket=1,
        random_state=seed,
    )
    corrected = np.asarray(diagnostics.p_values, dtype=np.float64)
    corrected_supports = [
        _permutation_support(float(value), CARDINALITY_B, "plus_one") for value in corrected
    ]
    native_p_values = np.asarray(
        [(support.numerator - 1) / CARDINALITY_B for support in corrected_supports],
        dtype=np.float64,
    )
    return CardinalityCandidateDiagnostics(
        statistics=np.asarray(diagnostics.statistics, dtype=np.float64),
        p_values=corrected,
        native_p_values=native_p_values,
        realized_permutations=np.full(X.shape[1], CARDINALITY_B, dtype=np.int64),
        native_selected_position=(
            None if diagnostics.root_feature < 0 else diagnostics.root_feature
        ),
        native_p_value_rule="r_monte_carlo",
    )


def _cart_impurity_decrease(model: object) -> float:
    """Return the weighted root impurity decrease for a fitted CART stump."""
    tree = model.tree_
    if tree.node_count == 1:
        return 0.0
    left = int(tree.children_left[0])
    right = int(tree.children_right[0])
    root_weight = float(tree.weighted_n_node_samples[0])
    decrease = (
        root_weight * float(tree.impurity[0])
        - float(tree.weighted_n_node_samples[left]) * float(tree.impurity[left])
        - float(tree.weighted_n_node_samples[right]) * float(tree.impurity[right])
    ) / root_weight
    return max(decrease, 0.0)


def _cart_cardinality_evidence(
    task: Task,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int | None]:
    """Compute per-feature CART stump scores and the native full-matrix winner."""
    scores = np.empty(X.shape[1], dtype=np.float64)
    for position in range(X.shape[1]):
        model = _fit_cardinality_model(
            task,
            "cart",
            X[:, [position]],
            y,
            seed,
        )
        scores[position] = _cart_impurity_decrease(model)
    full_model = _fit_cardinality_model(task, "cart", X, y, seed)
    selected = _selected_root_feature(full_model)
    return scores, None if selected < 0 else selected


def run_cardinality_trees(
    profile: Profile,
    *,
    base_seed: int = BASE_SEED,
    replicate_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Run candidate-level conditional tests and native CART root selection."""
    settings = _settings(profile)
    replicates = _resolve_replicates(settings.cardinality_replicates, replicate_indices)
    frames: list[pd.DataFrame] = []
    for task in ("classification", "regression"):
        data_design = f"cardinality_bias__{task}__n{CARDINALITY_N_SAMPLES}"
        for replicate in replicates:
            data_seed = _stream_seed(base_seed, data_design, replicate, "data")
            rng = np.random.default_rng(data_seed)
            X, features = _cardinality_matrix(rng, CARDINALITY_N_SAMPLES)
            y = _response(rng, task, CARDINALITY_N_SAMPLES)
            for method in ("citrees", "partykit"):
                scenario = f"{task}__{method}__n{CARDINALITY_N_SAMPLES}"
                model_seed = _stream_seed(base_seed, scenario, replicate, "model")
                diagnostics = (
                    _citrees_cardinality_diagnostics(task, X, y, model_seed)
                    if method == "citrees"
                    else _partykit_cardinality_diagnostics(task, X, y, model_seed)
                )
                frames.append(
                    build_cardinality_tree_raw(
                        features,
                        diagnostics,
                        task=task,
                        method=method,
                        selector=(
                            "mc"
                            if task == "classification" and method == "citrees"
                            else "pc"
                            if task == "regression" and method == "citrees"
                            else "quadratic"
                        ),
                        replicate=replicate,
                        data_seed=data_seed,
                        model_seed=model_seed,
                    )
                )
            cart_seed = _stream_seed(
                base_seed,
                f"{task}__cart__n{CARDINALITY_N_SAMPLES}",
                replicate,
                "model",
            )
            cart_scores, cart_selected = _cart_cardinality_evidence(
                task,
                X,
                y,
                cart_seed,
            )
            frames.append(
                build_cardinality_cart_raw(
                    features,
                    cart_scores,
                    task=task,
                    replicate=replicate,
                    data_seed=data_seed,
                    model_seed=cart_seed,
                    native_selected_position=cart_selected,
                )
            )
    raw = pd.concat(frames, ignore_index=True)
    validate_cardinality_tree_raw(raw)
    return raw


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
    if np.isclose(weight_total, 0.0):
        return float("nan")
    return float(np.dot(weights, ranks - ranks.mean()))


def _validate_label_value_matrices(
    labels: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated replicate-by-feature matrices for exact tests."""
    label_matrix = np.asarray(labels, dtype=np.float64)
    value_matrix = np.asarray(values, dtype=np.float64)
    if label_matrix.ndim == 1:
        label_matrix = label_matrix[np.newaxis, :]
    if value_matrix.ndim == 1:
        value_matrix = value_matrix[np.newaxis, :]
    if label_matrix.ndim != 2 or value_matrix.ndim != 2 or label_matrix.shape != value_matrix.shape:
        raise ValueError("labels and values must be aligned replicate-by-feature matrices")
    if label_matrix.shape[1] != len(CARDINALITY_SUPPORTS):
        raise ValueError("Exact cardinality randomization requires five features")
    if not np.isfinite(label_matrix).all() or not np.isfinite(value_matrix).all():
        raise ValueError("labels and values must be finite")
    if not np.all(label_matrix == label_matrix[0]):
        raise ValueError("Cardinality labels must be shared across replicates")
    if np.unique(label_matrix[0]).size != label_matrix.shape[1]:
        raise ValueError("Cardinality labels must be unique")
    return label_matrix, value_matrix


def _label_permutations(labels: np.ndarray) -> np.ndarray:
    """Enumerate all 5! permutations of one cardinality label vector."""
    label_permutations = np.asarray(list(permutations(labels)), dtype=np.float64)
    if label_permutations.shape[0] != CARDINALITY_EXACT_LABEL_PERMUTATIONS:
        raise RuntimeError("The frozen five-feature design must have exactly 120 permutations")
    return label_permutations


def _joint_randomization_test(
    per_replicate_null: np.ndarray,
    observed: float,
    *,
    n_resamples: int,
    random_state: int,
) -> RandomizationTestResult:
    """Sample independent within-replicate assignments from exact null grids."""
    null_grid = np.asarray(per_replicate_null, dtype=np.float64)
    if (
        null_grid.ndim != 2
        or null_grid.shape[0] == 0
        or null_grid.shape[1] != CARDINALITY_EXACT_LABEL_PERMUTATIONS
        or not np.isfinite(null_grid).all()
    ):
        raise ValueError("per_replicate_null must be a finite R by 120 matrix")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int):
        raise TypeError("n_resamples must be an integer")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not np.isfinite(observed):
        raise ValueError("observed must be finite")

    rng = np.random.default_rng(random_state)
    replicate_indices = np.arange(null_grid.shape[0], dtype=np.int64)
    extreme_count = 0
    batch_size = 256
    for start in range(0, n_resamples, batch_size):
        current_size = min(batch_size, n_resamples - start)
        sampled_permutations = rng.integers(
            0,
            CARDINALITY_EXACT_LABEL_PERMUTATIONS,
            size=(current_size, null_grid.shape[0]),
        )
        sampled_statistics = null_grid[
            replicate_indices[np.newaxis, :],
            sampled_permutations,
        ].mean(axis=1)
        extreme_count += int(np.count_nonzero(sampled_statistics >= observed - 1e-12))
    return RandomizationTestResult(
        statistic=observed,
        p_value=(extreme_count + 1) / (n_resamples + 1),
        extreme_count=extreme_count,
        n_resamples=n_resamples,
    )


def one_sided_label_randomization(
    labels: np.ndarray,
    values: np.ndarray,
    *,
    statistic: ReplicateStatistic,
    n_resamples: int = CARDINALITY_TREND_RESAMPLES,
    random_state: int = BASE_SEED,
) -> RandomizationTestResult:
    """Test a mean effect using independent within-replicate label assignments."""
    label_matrix, value_matrix = _validate_label_value_matrices(labels, values)
    observed_values = np.asarray(
        [
            statistic(label_row, value_row)
            for label_row, value_row in zip(label_matrix, value_matrix, strict=True)
        ],
        dtype=np.float64,
    )
    defined = np.isfinite(observed_values)
    if not defined.any():
        raise ValueError("The statistic is undefined for every replicate")
    observed = float(observed_values[defined].mean())
    per_replicate_null = np.empty(
        (int(defined.sum()), CARDINALITY_EXACT_LABEL_PERMUTATIONS),
        dtype=np.float64,
    )
    for permutation_index, permuted_labels in enumerate(_label_permutations(label_matrix[0])):
        per_replicate_null[:, permutation_index] = np.asarray(
            [statistic(permuted_labels, value_row) for value_row in value_matrix[defined]],
            dtype=np.float64,
        )
    return _joint_randomization_test(
        per_replicate_null,
        observed,
        n_resamples=n_resamples,
        random_state=random_state,
    )


def _directional_randomization(
    labels: np.ndarray,
    winner_weights: np.ndarray,
    *,
    n_resamples: int,
    random_state: int,
) -> RandomizationTestResult:
    """Enumerate each replicate exactly, then sample their joint null."""
    label_matrix, weight_matrix = _validate_label_value_matrices(
        labels,
        winner_weights,
    )
    active = ~np.isclose(weight_matrix.sum(axis=1), 0.0)
    if not active.any():
        raise ValueError("Directional effect is undefined when every replicate has no split")
    label_permutations = _label_permutations(label_matrix[0])
    centered = label_permutations - label_permutations.mean(axis=1, keepdims=True)
    per_replicate_null = weight_matrix[active] @ centered.T
    observed = float(
        np.mean(
            np.sum(
                weight_matrix[active]
                * (label_matrix[active] - label_matrix[active].mean(axis=1, keepdims=True)),
                axis=1,
            )
        )
    )
    return _joint_randomization_test(
        per_replicate_null,
        observed,
        n_resamples=n_resamples,
        random_state=random_state,
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
    """Return ascending average ranks without resolving ties by position."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a finite, non-empty vector")
    return np.asarray(rankdata(array, method="average"), dtype=np.float64)


def _pairwise_signs(values: np.ndarray) -> np.ndarray:
    """Return pairwise comparison signs for each row of a matrix."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("values must be a finite two-dimensional matrix")
    pairs = tuple(combinations(range(matrix.shape[1]), 2))
    return np.column_stack([np.sign(matrix[:, left] - matrix[:, right]) for left, right in pairs])


def _kendall_tau_b_grid(
    label_rows: np.ndarray,
    value_rows: np.ndarray,
) -> np.ndarray:
    """Compute all row-pair tau-b values by vectorized pairwise signs."""
    label_signs = _pairwise_signs(label_rows)
    value_signs = _pairwise_signs(value_rows)
    numerator = label_signs @ value_signs.T
    denominator = np.sqrt(
        np.count_nonzero(label_signs, axis=1)[:, np.newaxis]
        * np.count_nonzero(value_signs, axis=1)[np.newaxis, :]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(
            numerator,
            denominator,
            out=np.full(numerator.shape, np.nan, dtype=np.float64),
            where=denominator > 0.0,
        )


def kendall_tau_b(
    cardinality_ranks: np.ndarray,
    raw_importances: np.ndarray,
) -> float:
    """Return Kendall tau-b between cardinality rank and importance."""
    ranks = np.asarray(cardinality_ranks, dtype=np.float64)
    importances = np.asarray(raw_importances, dtype=np.float64)
    if (
        ranks.ndim != 1
        or importances.ndim != 1
        or ranks.shape != importances.shape
        or ranks.size < 2
        or not np.isfinite(ranks).all()
        or not np.isfinite(importances).all()
    ):
        raise ValueError("cardinality_ranks and raw_importances must be finite aligned vectors")
    return float(_kendall_tau_b_grid(ranks[np.newaxis, :], importances[np.newaxis, :])[0, 0])


def _kendall_randomization(
    labels: np.ndarray,
    importances: np.ndarray,
    *,
    n_resamples: int,
    random_state: int,
) -> RandomizationTestResult:
    """Enumerate replicate tau-b grids, then sample their joint null."""
    label_matrix, importance_matrix = _validate_label_value_matrices(
        labels,
        importances,
    )
    tau_grid = _kendall_tau_b_grid(
        _label_permutations(label_matrix[0]),
        importance_matrix,
    )
    defined = np.isfinite(tau_grid[0])
    if not defined.any():
        raise ValueError("Kendall tau-b is undefined for every replicate")
    observed = float(_kendall_tau_b_grid(label_matrix[:1], importance_matrix)[0, defined].mean())
    return _joint_randomization_test(
        tau_grid[:, defined].T,
        observed,
        n_resamples=n_resamples,
        random_state=random_state,
    )


def _candidate_matrices(
    raw: pd.DataFrame,
    value_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return feature-ordered label and value matrices across replicates."""
    label_rows: list[np.ndarray] = []
    value_rows: list[np.ndarray] = []
    for _, replicate_group in raw.groupby("replicate", sort=True):
        ordered = replicate_group.sort_values("feature_id")
        label_rows.append(ordered["cardinality_rank"].to_numpy(dtype=np.float64))
        value_rows.append(ordered[value_column].to_numpy(dtype=np.float64))
    return np.vstack(label_rows), np.vstack(value_rows)


def summarize_cardinality_tree(
    raw: pd.DataFrame,
    *,
    random_state: int = BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return replicate, planned-trend, and method summaries for tree rows."""
    validate_cardinality_tree_raw(raw)
    replicate_rows: list[dict[str, object]] = []
    trend_rows: list[dict[str, object]] = []
    for (task, method), method_group in raw.groupby(["task", "method"], sort=True):
        for replicate, replicate_group in method_group.groupby("replicate", sort=True):
            ordered = replicate_group.sort_values("feature_id")
            labels = ordered["cardinality_rank"].to_numpy(dtype=np.float64)
            native_effect = directional_effect(
                labels,
                ordered["native_winner"].to_numpy(dtype=np.float64),
            )
            conditional = method in {"citrees", "partykit"}
            replicate_rows.append(
                {
                    "experiment": "cardinality_tree_replicate",
                    "dgp": CARDINALITY_DGP,
                    "task": task,
                    "method": method,
                    "replicate": int(replicate),
                    "data_seed": int(ordered["data_seed"].iloc[0]),
                    "model_seed": int(ordered["model_seed"].iloc[0]),
                    "n_features": len(CARDINALITY_SUPPORTS),
                    "forced_directional_effect": (
                        directional_effect(
                            labels,
                            ordered["forced_winner_weight"].to_numpy(dtype=np.float64),
                        )
                        if conditional
                        else float("nan")
                    ),
                    "adjusted_directional_effect": (
                        directional_effect(
                            labels,
                            ordered["adjusted_winner_weight"].to_numpy(dtype=np.float64),
                        )
                        if conditional
                        else float("nan")
                    ),
                    "adjusted_split": (
                        bool(ordered["adjusted_split"].iloc[0]) if conditional else None
                    ),
                    "adjusted_no_split": (
                        bool(ordered["adjusted_no_split"].iloc[0]) if conditional else None
                    ),
                    "native_directional_effect": native_effect,
                    "native_split": bool(ordered["native_split"].iloc[0]),
                    "native_no_split": bool(ordered["native_no_split"].iloc[0]),
                    "minimum_p_tie_size": (
                        int(ordered["minimum_p_tie_size"].iloc[0]) if conditional else float("nan")
                    ),
                    "all_p_equal": (bool(ordered["all_p_equal"].iloc[0]) if conditional else None),
                    "native_score_tie_size": (
                        int(ordered["native_score_tie_size"].iloc[0])
                        if method == "cart"
                        else float("nan")
                    ),
                }
            )

        if method not in {"citrees", "partykit"}:
            continue
        labels, forced_weights = _candidate_matrices(
            method_group,
            "forced_winner_weight",
        )
        _, adjusted_weights = _candidate_matrices(
            method_group,
            "adjusted_winner_weight",
        )
        adjusted_splits = ~np.isclose(adjusted_weights.sum(axis=1), 0.0)
        for estimand, weights in (
            ("forced_winner", forced_weights),
            ("adjusted_winner_given_split", adjusted_weights),
        ):
            hypothesis_id = f"tree__{task}__{method}__{estimand}"
            defined = ~np.isclose(weights.sum(axis=1), 0.0)
            if defined.any():
                test = _directional_randomization(
                    labels,
                    weights,
                    n_resamples=CARDINALITY_TREND_RESAMPLES,
                    random_state=_stream_seed(
                        random_state,
                        hypothesis_id,
                        0,
                        "trend_randomization",
                    ),
                )
                statistic = test.statistic
                p_value = test.p_value
                extreme_count: int | float = test.extreme_count
            else:
                statistic = float("nan")
                p_value = float("nan")
                extreme_count = float("nan")
            trend_rows.append(
                {
                    "experiment": "cardinality_tree_trend",
                    "dgp": CARDINALITY_DGP,
                    "hypothesis_id": hypothesis_id,
                    "task": task,
                    "method": method,
                    "estimand": estimand,
                    "alternative": "positive",
                    "n_replicates": weights.shape[0],
                    "n_defined_effects": int(defined.sum()),
                    "n_adjusted_splits": int(adjusted_splits.sum()),
                    "n_adjusted_no_splits": int((~adjusted_splits).sum()),
                    "adjusted_no_split_rate": float((~adjusted_splits).mean()),
                    "mean_directional_effect": statistic,
                    "randomization_p_value": p_value,
                    "randomization_extreme_count": extreme_count,
                    "per_replicate_label_permutations": (CARDINALITY_EXACT_LABEL_PERMUTATIONS),
                    "joint_randomization_resamples": CARDINALITY_TREND_RESAMPLES,
                    "randomization_scheme": CARDINALITY_TREND_RANDOMIZATION,
                    "randomization_p_value_rule": CARDINALITY_TREND_P_VALUE_RULE,
                }
            )

    replicate_summary = pd.DataFrame(replicate_rows)
    method_rows: list[dict[str, object]] = []
    for (task, method), group in replicate_summary.groupby(
        ["task", "method"],
        sort=True,
    ):
        conditional = method in {"citrees", "partykit"}
        native_splits = group["native_split"].astype(bool)
        native_effects = group.loc[
            native_splits,
            "native_directional_effect",
        ].to_numpy(dtype=np.float64)
        if conditional:
            adjusted_splits = group["adjusted_split"].astype(bool)
            tie_sizes = group["minimum_p_tie_size"].to_numpy(dtype=np.float64)
            score_tie_count: int | float = float("nan")
            mean_score_tie: float = float("nan")
        else:
            adjusted_splits = pd.Series(dtype=bool)
            tie_sizes = np.asarray([], dtype=np.float64)
            score_ties = group["native_score_tie_size"].to_numpy(dtype=np.float64)
            score_tie_count = int(np.count_nonzero(score_ties > 1))
            mean_score_tie = float(score_ties.mean())
        method_rows.append(
            {
                "experiment": "cardinality_tree_method",
                "dgp": CARDINALITY_DGP,
                "task": task,
                "method": method,
                "n_replicates": len(group),
                "n_adjusted_splits": (int(adjusted_splits.sum()) if conditional else float("nan")),
                "n_adjusted_no_splits": (
                    int((~adjusted_splits).sum()) if conditional else float("nan")
                ),
                "adjusted_split_rate": (
                    float(adjusted_splits.mean()) if conditional else float("nan")
                ),
                "n_native_splits": int(native_splits.sum()),
                "n_native_no_splits": int((~native_splits).sum()),
                "native_split_rate": float(native_splits.mean()),
                "mean_native_directional_effect_given_split": (
                    float(native_effects.mean()) if native_effects.size else float("nan")
                ),
                "n_minimum_p_ties": (
                    int(np.count_nonzero(tie_sizes > 1)) if conditional else float("nan")
                ),
                "minimum_p_tie_rate": (
                    float(np.mean(tie_sizes > 1)) if conditional else float("nan")
                ),
                "mean_minimum_p_tie_size": (
                    float(tie_sizes.mean()) if conditional else float("nan")
                ),
                "n_all_p_equal": (
                    int(group["all_p_equal"].astype(bool).sum()) if conditional else float("nan")
                ),
                "n_native_score_ties": score_tie_count,
                "mean_native_score_tie_size": mean_score_tie,
            }
        )
    return replicate_summary, pd.DataFrame(trend_rows), pd.DataFrame(method_rows)


def _fit_cardinality_forest_importance(
    task: Task,
    method: ForestMethod,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Fit one matched depth-one forest and return native importances."""
    if method == "partykit_cforest":
        return np.asarray(
            r_cforest_importance(
                X,
                y,
                task=task,
                teststat="quadratic",
                testtype="MonteCarlo",
                mincriterion=1.0 - CARDINALITY_ALPHA / X.shape[1],
                nresample=CARDINALITY_B,
                ntree=CARDINALITY_FOREST_TREES,
                mtry="all",
                maxdepth=1,
                minsplit=2,
                minbucket=1,
                replace=True,
                fraction=1.0,
                varimp_conditional=False,
                varimp_nperm=CARDINALITY_FOREST_IMPORTANCE_PERMUTATIONS,
                cores=1,
                random_state=seed,
            ),
            dtype=np.float64,
        )

    if method == "sklearn_rf":
        model = (
            RandomForestClassifier(
                n_estimators=CARDINALITY_FOREST_TREES,
                max_depth=1,
                max_features=None,
                min_samples_split=2,
                min_samples_leaf=1,
                bootstrap=True,
                max_samples=None,
                n_jobs=1,
                random_state=seed,
            )
            if task == "classification"
            else RandomForestRegressor(
                n_estimators=CARDINALITY_FOREST_TREES,
                max_depth=1,
                max_features=None,
                min_samples_split=2,
                min_samples_leaf=1,
                bootstrap=True,
                max_samples=None,
                n_jobs=1,
                random_state=seed,
            )
        )
    else:
        common: dict[str, object] = {
            "n_estimators": CARDINALITY_FOREST_TREES,
            "alpha_selector": CARDINALITY_ALPHA,
            "adjust_alpha_selector": True,
            "adjust_alpha_splitter": False,
            "n_resamples_selector": CARDINALITY_B,
            "n_resamples_splitter": None,
            "early_stopping_selector": None,
            "early_stopping_splitter": None,
            "feature_muting": False,
            "feature_scanning": False,
            "max_features": None,
            "threshold_scanning": False,
            "max_depth": 1,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "bootstrap": True,
            "max_samples": None,
            "n_jobs": 1,
            "random_state": seed,
            "verbose": 0,
        }
        model = (
            ConditionalInferenceForestClassifier(
                sampling_method=None,
                **common,
            )
            if task == "classification"
            else ConditionalInferenceForestRegressor(**common)
        )
    model.fit(X, y)
    return np.asarray(model.feature_importances_, dtype=np.float64)


def build_cardinality_forest_raw(
    features: Sequence[CardinalityFeatureDesign],
    raw_importances: np.ndarray,
    *,
    task: Task,
    method: ForestMethod,
    replicate: int,
    data_seed: int,
    model_seed: int,
) -> pd.DataFrame:
    """Build feature-level forest records with explicit structural zeros."""
    ordered_features = tuple(sorted(features, key=lambda feature: feature.feature_id))
    values = np.asarray(raw_importances, dtype=np.float64)
    if values.shape != (len(ordered_features),) or np.isinf(values).any():
        raise ValueError("raw_importances must be one non-infinite value per matrix column")
    if method != "partykit_cforest" and np.isnan(values).any():
        raise ValueError(f"{method} importances must not be missing")
    trend_values = np.where(np.isnan(values), 0.0, values)
    if method == "citrees_cif":
        forest_family = "conditional_inference"
        selector = "mc" if task == "classification" else "pc"
        base_budget: int | float = CARDINALITY_B
        candidate_budget = CARDINALITY_B * len(CARDINALITY_SUPPORTS)
        importance_type = "normalized_impurity_decrease"
        importance_permutations = 0
    elif method == "partykit_cforest":
        forest_family = "conditional_inference"
        selector = "quadratic"
        base_budget = CARDINALITY_B
        candidate_budget = CARDINALITY_B
        importance_type = "partykit_permutation"
        importance_permutations = CARDINALITY_FOREST_IMPORTANCE_PERMUTATIONS
    else:
        forest_family = "cart"
        selector = "gini" if task == "classification" else "squared_error"
        base_budget = float("nan")
        candidate_budget = 0
        importance_type = "normalized_impurity_decrease"
        importance_permutations = 0

    rows: list[dict[str, object]] = []
    for feature in ordered_features:
        raw_value = float(values[feature.position])
        missing = bool(np.isnan(raw_value))
        rows.append(
            {
                "experiment": "cardinality_forest",
                "dgp": CARDINALITY_DGP,
                "maximum_support_feature_type": CARDINALITY_MAX_SUPPORT_FEATURE_TYPE,
                "task": task,
                "method": method,
                "forest_family": forest_family,
                "selector": selector,
                "replicate": replicate,
                "feature_id": feature.feature_id,
                "cardinality": feature.label,
                "cardinality_rank": feature.feature_id,
                "feature_type": "ordered_discrete",
                "nominal_support": feature.nominal_support,
                "realized_support": feature.realized_support,
                "maximum_multiplicity": feature.maximum_multiplicity,
                "tied_pair_fraction": feature.tied_pair_fraction,
                "feature_position": feature.position,
                "n_samples": CARDINALITY_N_SAMPLES,
                "n_features": len(CARDINALITY_SUPPORTS),
                "data_seed": data_seed,
                "model_seed": model_seed,
                "B": base_budget,
                "realized_candidate_resamples": candidate_budget,
                "n_trees": CARDINALITY_FOREST_TREES,
                "max_depth": 1,
                "max_features": len(CARDINALITY_SUPPORTS),
                "sampling_with_replacement": True,
                "sample_size": CARDINALITY_N_SAMPLES,
                "importance_type": importance_type,
                "importance_permutations": importance_permutations,
                "raw_importance": raw_value,
                "raw_importance_missing": missing,
                "trend_importance": float(trend_values[feature.position]),
                "structural_zero_imputed": missing,
            }
        )
    frame = pd.DataFrame(rows)
    validate_cardinality_forest_raw(frame)
    return frame


def validate_cardinality_forest_raw(raw: pd.DataFrame) -> None:
    """Validate paired DGP metadata, controls, and forest importances."""
    required = {
        "experiment",
        "dgp",
        "maximum_support_feature_type",
        "task",
        "method",
        "forest_family",
        "selector",
        "replicate",
        "feature_id",
        "cardinality",
        "cardinality_rank",
        "feature_type",
        "nominal_support",
        "realized_support",
        "maximum_multiplicity",
        "tied_pair_fraction",
        "feature_position",
        "n_samples",
        "n_features",
        "data_seed",
        "model_seed",
        "B",
        "realized_candidate_resamples",
        "n_trees",
        "max_depth",
        "max_features",
        "sampling_with_replacement",
        "sample_size",
        "importance_type",
        "importance_permutations",
        "raw_importance",
        "raw_importance_missing",
        "trend_importance",
        "structural_zero_imputed",
    }
    missing_columns = required.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing cardinality forest columns: {sorted(missing_columns)}")
    if not raw.columns.is_unique:
        raise ValueError("Cardinality forest columns must be unique")
    if raw.duplicated(["task", "replicate", "method", "feature_id"]).any():
        raise ValueError("Cardinality forest candidate keys must be unique")
    if not raw["experiment"].eq("cardinality_forest").all():
        raise ValueError("Forest rows require experiment='cardinality_forest'")
    if not raw["dgp"].eq(CARDINALITY_DGP).all():
        raise ValueError(f"Forest rows require dgp={CARDINALITY_DGP}")
    if not raw["maximum_support_feature_type"].eq(CARDINALITY_MAX_SUPPORT_FEATURE_TYPE).all():
        raise ValueError("Forest rows do not identify the 200-level discrete predictor")
    if not raw["feature_type"].eq("ordered_discrete").all():
        raise ValueError("Every cardinality predictor must be labeled ordered discrete")
    if not raw.groupby(["task", "replicate"])["data_seed"].nunique().eq(1).all():
        raise ValueError("Forest methods within a replicate must share one data seed")

    expected_ids = set(range(len(CARDINALITY_SUPPORTS)))
    expected_multiplicities = np.asarray(
        [CARDINALITY_N_SAMPLES // support for support in CARDINALITY_SUPPORTS],
        dtype=np.int64,
    )
    total_pairs = CARDINALITY_N_SAMPLES * (CARDINALITY_N_SAMPLES - 1) // 2
    expected_tied_fractions = np.asarray(
        [
            support * (multiplicity * (multiplicity - 1) // 2) / total_pairs
            for support, multiplicity in zip(
                CARDINALITY_SUPPORTS,
                expected_multiplicities,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    position_mappings: dict[tuple[object, object], tuple[int, ...]] = {}
    for _, group in raw.groupby(["task", "replicate", "method"], sort=False):
        ordered = group.sort_values("feature_id")
        if set(ordered["feature_id"]) != expected_ids:
            raise ValueError("Every forest replicate must retain all cardinality features")
        if tuple(ordered["cardinality"]) != CARDINALITY_LABELS:
            raise ValueError("Forest cardinality labels must follow the frozen DGP")
        if not ordered["cardinality_rank"].eq(ordered["feature_id"]).all():
            raise ValueError("Forest cardinality rank must equal feature_id")
        if tuple(int(value) for value in ordered["nominal_support"]) != CARDINALITY_SUPPORTS:
            raise ValueError("Forest supports disagree with the frozen DGP")
        if not np.array_equal(
            ordered["nominal_support"].to_numpy(dtype=np.int64),
            ordered["realized_support"].to_numpy(dtype=np.int64),
        ):
            raise ValueError("Forest realized supports must equal planned supports")
        if not np.array_equal(
            ordered["maximum_multiplicity"].to_numpy(dtype=np.int64),
            expected_multiplicities,
        ):
            raise ValueError("Forest multiplicities disagree with the frozen DGP")
        if not np.allclose(
            ordered["tied_pair_fraction"].to_numpy(dtype=np.float64),
            expected_tied_fractions,
        ):
            raise ValueError("Forest tied-pair fractions disagree with the frozen DGP")
        if set(ordered["feature_position"]) != expected_ids:
            raise ValueError("Every forest replicate must retain each feature position")
        pair_key = (ordered["task"].iloc[0], ordered["replicate"].iloc[0])
        positions = tuple(int(value) for value in ordered["feature_position"])
        reference = position_mappings.setdefault(pair_key, positions)
        if positions != reference:
            raise ValueError("Forest methods within a replicate must share position mappings")
        if (
            not ordered["n_samples"].eq(CARDINALITY_N_SAMPLES).all()
            or not ordered["n_features"].eq(len(CARDINALITY_SUPPORTS)).all()
            or not ordered["n_trees"].eq(CARDINALITY_FOREST_TREES).all()
            or not ordered["max_depth"].eq(1).all()
            or not ordered["max_features"].eq(len(CARDINALITY_SUPPORTS)).all()
            or not ordered["sampling_with_replacement"].astype(bool).all()
            or not ordered["sample_size"].eq(CARDINALITY_N_SAMPLES).all()
        ):
            raise ValueError("Forest controls disagree with the frozen design")

        method = str(ordered["method"].iloc[0])
        raw_values = ordered["raw_importance"].to_numpy(dtype=np.float64)
        trend_values = ordered["trend_importance"].to_numpy(dtype=np.float64)
        missing = np.isnan(raw_values)
        if not np.isfinite(trend_values).all():
            raise ValueError("Trend importances must be finite")
        if not np.array_equal(
            ordered["raw_importance_missing"].astype(bool).to_numpy(),
            missing,
        ):
            raise ValueError("Raw-importance missing indicators are inconsistent")
        if not np.array_equal(
            ordered["structural_zero_imputed"].astype(bool).to_numpy(),
            missing,
        ):
            raise ValueError("Structural-zero indicators are inconsistent")
        if not np.allclose(trend_values, np.where(missing, 0.0, raw_values)):
            raise ValueError("Trend importance must only replace omitted values with zero")

        if method == "partykit_cforest":
            if (
                not ordered["B"].eq(CARDINALITY_B).all()
                or not ordered["realized_candidate_resamples"].eq(CARDINALITY_B).all()
                or not ordered["importance_type"].eq("partykit_permutation").all()
                or not ordered["importance_permutations"]
                .eq(CARDINALITY_FOREST_IMPORTANCE_PERMUTATIONS)
                .all()
            ):
                raise ValueError("partykit cforest controls are inconsistent")
        elif method == "citrees_cif":
            if missing.any():
                raise ValueError("citrees CIF importances must not be missing")
            if (
                not ordered["B"].eq(CARDINALITY_B).all()
                or not ordered["realized_candidate_resamples"]
                .eq(CARDINALITY_B * len(CARDINALITY_SUPPORTS))
                .all()
                or not ordered["importance_type"].eq("normalized_impurity_decrease").all()
            ):
                raise ValueError("citrees CIF controls are inconsistent")
        elif method == "sklearn_rf":
            if missing.any() or not ordered["B"].isna().all():
                raise ValueError("sklearn RF rows have invalid permutation metadata")
            if (
                not ordered["realized_candidate_resamples"].eq(0).all()
                or not ordered["importance_type"].eq("normalized_impurity_decrease").all()
            ):
                raise ValueError("sklearn RF controls are inconsistent")
        else:
            raise ValueError(f"Unsupported cardinality forest method: {method}")


def run_cardinality_forests(
    profile: Profile,
    *,
    base_seed: int = BASE_SEED,
    replicate_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Run matched CIF, cforest, and CART random-forest comparisons."""
    settings = _settings(profile)
    replicates = _resolve_replicates(
        settings.cardinality_forest_replicates,
        replicate_indices,
    )
    frames: list[pd.DataFrame] = []
    methods: tuple[ForestMethod, ...] = (
        "citrees_cif",
        "partykit_cforest",
        "sklearn_rf",
    )
    for task in ("classification", "regression"):
        data_design = f"cardinality_bias__{task}__n{CARDINALITY_N_SAMPLES}"
        for replicate in replicates:
            data_seed = _stream_seed(base_seed, data_design, replicate, "data")
            rng = np.random.default_rng(data_seed)
            X, features = _cardinality_matrix(rng, CARDINALITY_N_SAMPLES)
            y = _response(rng, task, CARDINALITY_N_SAMPLES)
            for method in methods:
                model_seed = _stream_seed(
                    base_seed,
                    f"{task}__{method}__n{CARDINALITY_N_SAMPLES}",
                    replicate,
                    "model",
                )
                importances = _fit_cardinality_forest_importance(
                    task,
                    method,
                    X,
                    y,
                    model_seed,
                )
                frames.append(
                    build_cardinality_forest_raw(
                        features,
                        importances,
                        task=task,
                        method=method,
                        replicate=replicate,
                        data_seed=data_seed,
                        model_seed=model_seed,
                    )
                )
    raw = pd.concat(frames, ignore_index=True)
    validate_cardinality_forest_raw(raw)
    return raw


def summarize_cardinality_forest_trends(
    raw: pd.DataFrame,
    *,
    random_state: int = BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize replicate and Monte Carlo method-level tau-b forest trends."""
    validate_cardinality_forest_raw(raw)
    replicate_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for (task, method), method_group in raw.groupby(["task", "method"], sort=True):
        labels, importances = _candidate_matrices(
            method_group,
            "trend_importance",
        )
        tau_values = _kendall_tau_b_grid(labels[:1], importances)[0]
        for row_index, (replicate, replicate_group) in enumerate(
            method_group.groupby("replicate", sort=True)
        ):
            ordered = replicate_group.sort_values("feature_id")
            values = ordered["trend_importance"].to_numpy(dtype=np.float64)
            importance_ranks = average_ranks(values)
            _, tie_counts = np.unique(values, return_counts=True)
            replicate_rows.append(
                {
                    "experiment": "cardinality_forest_replicate",
                    "dgp": CARDINALITY_DGP,
                    "task": task,
                    "method": method,
                    "replicate": int(replicate),
                    "data_seed": int(ordered["data_seed"].iloc[0]),
                    "model_seed": int(ordered["model_seed"].iloc[0]),
                    "kendall_tau_b": float(tau_values[row_index]),
                    "maximum_importance_tie_size": int(tie_counts.max()),
                    "zero_importance_count": int(np.count_nonzero(values == 0.0)),
                    "raw_importance_missing_count": int(
                        ordered["raw_importance_missing"].astype(bool).sum()
                    ),
                    "all_importances_equal": bool(tie_counts.max() == len(values)),
                    **{
                        f"average_rank_feature_{feature_id}": float(importance_ranks[feature_id])
                        for feature_id in range(len(CARDINALITY_SUPPORTS))
                    },
                }
            )
        defined_count = int(np.isfinite(tau_values).sum())
        hypothesis_id = f"forest__{task}__{method}__importance_trend"
        if defined_count:
            test = _kendall_randomization(
                labels,
                importances,
                n_resamples=CARDINALITY_TREND_RESAMPLES,
                random_state=_stream_seed(
                    random_state,
                    hypothesis_id,
                    0,
                    "trend_randomization",
                ),
            )
            mean_tau = test.statistic
            p_value = test.p_value
            extreme_count: int | float = test.extreme_count
        else:
            mean_tau = float("nan")
            p_value = float("nan")
            extreme_count = float("nan")
        summary_rows.append(
            {
                "experiment": "cardinality_forest_trend",
                "dgp": CARDINALITY_DGP,
                "hypothesis_id": hypothesis_id,
                "task": task,
                "method": method,
                "estimand": "importance_kendall_tau_b",
                "alternative": "positive",
                "n_replicates": importances.shape[0],
                "n_defined_trends": defined_count,
                "n_undefined_trends": int(importances.shape[0] - defined_count),
                "mean_kendall_tau_b": mean_tau,
                "randomization_p_value": p_value,
                "randomization_extreme_count": extreme_count,
                "per_replicate_label_permutations": (CARDINALITY_EXACT_LABEL_PERMUTATIONS),
                "joint_randomization_resamples": CARDINALITY_TREND_RESAMPLES,
                "randomization_scheme": CARDINALITY_TREND_RANDOMIZATION,
                "randomization_p_value_rule": CARDINALITY_TREND_P_VALUE_RULE,
                "mean_zero_importance_count": float(
                    np.mean(np.count_nonzero(importances == 0.0, axis=1))
                ),
                "mean_raw_importance_missing_count": float(
                    method_group.groupby("replicate")["raw_importance_missing"].sum().mean()
                ),
            }
        )
    return pd.DataFrame(replicate_rows), pd.DataFrame(summary_rows)


def apply_cardinality_holm(
    tree_summary: pd.DataFrame,
    forest_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply one fixed Holm family to the 14 planned positive-trend tests."""
    planned = pd.concat(
        [
            tree_summary[
                [
                    "hypothesis_id",
                    "experiment",
                    "task",
                    "method",
                    "estimand",
                    "alternative",
                    "randomization_p_value",
                ]
            ],
            forest_summary[
                [
                    "hypothesis_id",
                    "experiment",
                    "task",
                    "method",
                    "estimand",
                    "alternative",
                    "randomization_p_value",
                ]
            ],
        ],
        ignore_index=True,
    )
    if len(planned) != CARDINALITY_HOLM_FAMILY_SIZE or planned["hypothesis_id"].duplicated().any():
        raise ValueError(
            f"The cardinality Holm family requires {CARDINALITY_HOLM_FAMILY_SIZE} "
            "unique planned hypotheses"
        )
    raw_p_values = planned["randomization_p_value"].to_numpy(dtype=np.float64)
    for p_value in raw_p_values[np.isfinite(raw_p_values)]:
        _permutation_support(
            float(p_value),
            CARDINALITY_TREND_RESAMPLES,
            "plus_one",
        )
    holm_inputs = np.where(np.isfinite(raw_p_values), raw_p_values, 1.0)
    adjusted = holm_adjust(holm_inputs)
    minimum_raw_p_value = 1.0 / (CARDINALITY_TREND_RESAMPLES + 1)
    minimum_holm_p_value = float(
        holm_adjust(
            [minimum_raw_p_value] * CARDINALITY_HOLM_FAMILY_SIZE,
        )[0]
    )
    if minimum_holm_p_value >= CARDINALITY_ALPHA:
        raise ValueError("The planned Holm family does not have attainable 0.05 resolution")
    planned["holm_family"] = CARDINALITY_HOLM_FAMILY
    planned["holm_family_size"] = CARDINALITY_HOLM_FAMILY_SIZE
    planned["test_defined"] = np.isfinite(raw_p_values)
    planned["holm_input_p_value"] = holm_inputs
    planned["holm_adjusted_p_value"] = adjusted
    planned["holm_reject_0_05"] = adjusted < CARDINALITY_ALPHA
    planned["minimum_attainable_raw_p_value"] = minimum_raw_p_value
    planned["minimum_attainable_holm_p_value"] = minimum_holm_p_value
    planned["undefined_test_treatment"] = np.where(
        planned["test_defined"],
        "monte_carlo_p_value",
        "set_to_one",
    )

    holm_columns = [
        "holm_family",
        "holm_family_size",
        "holm_input_p_value",
        "holm_adjusted_p_value",
        "holm_reject_0_05",
    ]
    lookup = planned.set_index("hypothesis_id")
    adjusted_summaries: list[pd.DataFrame] = []
    for summary in (tree_summary, forest_summary):
        adjusted_summary = summary.copy()
        for column in holm_columns:
            adjusted_summary[column] = adjusted_summary["hypothesis_id"].map(lookup[column])
        adjusted_summaries.append(adjusted_summary)
    family = planned[
        [
            "holm_family",
            "holm_family_size",
            "hypothesis_id",
            "experiment",
            "task",
            "method",
            "estimand",
            "alternative",
            "test_defined",
            "randomization_p_value",
            "holm_input_p_value",
            "holm_adjusted_p_value",
            "holm_reject_0_05",
            "minimum_attainable_raw_p_value",
            "minimum_attainable_holm_p_value",
            "undefined_test_treatment",
        ]
    ].rename(
        columns={
            "holm_family": "family",
            "holm_family_size": "family_size",
        }
    )
    return adjusted_summaries[0], adjusted_summaries[1], family


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


CALIBRATION_RESULT_SCHEMAS: dict[str, tuple[str, ...]] = {
    "selector_null_raw": (
        "task",
        "selector",
        "feature_distribution",
        "n_samples",
        "n_resamples",
        "alpha",
        "experiment",
        "estimand",
        "scenario",
        "replicate",
        "data_seed",
        "model_seed",
        "realized_permutations",
        "p_value",
        "rejected",
    ),
    "selector_null_summary": (
        "estimand",
        "scenario",
        "task",
        "selector",
        "feature_distribution",
        "n_samples",
        "n_resamples",
        "alpha",
        "n_replicates",
        "n_rejections",
        "rejection_rate",
        "confidence_lower",
        "confidence_upper",
        "attainable_alpha",
        "realized_permutations_total",
    ),
    "root_null_raw": (
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
        "experiment",
        "estimand",
        "scenario",
        "replicate",
        "data_seed",
        "model_seed",
        "realized_selector_permutations",
        "realized_splitter_permutations",
        "realized_permutations",
        "split",
    ),
    "root_null_summary": (
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
        "n_replicates",
        "n_splits",
        "split_rate",
        "confidence_lower",
        "confidence_upper",
        "realized_selector_permutations_total",
        "realized_splitter_permutations_total",
        "realized_permutations_total",
        "realized_permutations_mean",
    ),
    "cardinality_design": (
        "experiment",
        "dgp",
        "n_samples",
        "n_features",
        "feature_id",
        "cardinality",
        "feature_type",
        "nominal_support",
        "realized_support",
        "maximum_multiplicity",
        "tied_pair_fraction",
        "balance",
        "column_order",
        "maximum_support_feature_type",
    ),
    "cardinality_tree_raw": (
        "experiment",
        "dgp",
        "maximum_support_feature_type",
        "task",
        "method",
        "selection_test",
        "selector",
        "replicate",
        "feature_id",
        "cardinality",
        "cardinality_rank",
        "feature_type",
        "nominal_support",
        "realized_support",
        "maximum_multiplicity",
        "tied_pair_fraction",
        "feature_position",
        "n_samples",
        "n_features",
        "data_seed",
        "model_seed",
        "B",
        "realized_permutations",
        "selection_alpha",
        "adjusted_alpha",
        "p_value_comparison",
        "p_value_rule",
        "native_p_value_rule",
        "attainable_adjusted_cutoff",
        "candidate_statistic",
        "p_value_support_index",
        "p_value_numerator",
        "p_value_denominator",
        "p_value",
        "native_p_value_support_index",
        "native_p_value_numerator",
        "native_p_value_denominator",
        "native_p_value",
        "minimum_p_tie_size",
        "all_p_equal",
        "is_minimum_p",
        "forced_winner_weight",
        "adjusted_winner_weight",
        "adjusted_split",
        "adjusted_no_split",
        "no_split_weight",
        "native_winner",
        "native_split",
        "native_no_split",
        "native_selected_position",
        "native_selected_feature",
        "native_score_tie_size",
    ),
    "cardinality_tree_replicate": (
        "experiment",
        "dgp",
        "task",
        "method",
        "replicate",
        "data_seed",
        "model_seed",
        "n_features",
        "forced_directional_effect",
        "adjusted_directional_effect",
        "adjusted_split",
        "adjusted_no_split",
        "native_directional_effect",
        "native_split",
        "native_no_split",
        "minimum_p_tie_size",
        "all_p_equal",
        "native_score_tie_size",
    ),
    "cardinality_tree_trend_summary": (
        "experiment",
        "dgp",
        "hypothesis_id",
        "task",
        "method",
        "estimand",
        "alternative",
        "n_replicates",
        "n_defined_effects",
        "n_adjusted_splits",
        "n_adjusted_no_splits",
        "adjusted_no_split_rate",
        "mean_directional_effect",
        "randomization_p_value",
        "randomization_extreme_count",
        "per_replicate_label_permutations",
        "joint_randomization_resamples",
        "randomization_scheme",
        "randomization_p_value_rule",
        "holm_family",
        "holm_family_size",
        "holm_input_p_value",
        "holm_adjusted_p_value",
        "holm_reject_0_05",
    ),
    "cardinality_tree_method_summary": (
        "experiment",
        "dgp",
        "task",
        "method",
        "n_replicates",
        "n_adjusted_splits",
        "n_adjusted_no_splits",
        "adjusted_split_rate",
        "n_native_splits",
        "n_native_no_splits",
        "native_split_rate",
        "mean_native_directional_effect_given_split",
        "n_minimum_p_ties",
        "minimum_p_tie_rate",
        "mean_minimum_p_tie_size",
        "n_all_p_equal",
        "n_native_score_ties",
        "mean_native_score_tie_size",
    ),
    "cardinality_forest_raw": (
        "experiment",
        "dgp",
        "maximum_support_feature_type",
        "task",
        "method",
        "forest_family",
        "selector",
        "replicate",
        "feature_id",
        "cardinality",
        "cardinality_rank",
        "feature_type",
        "nominal_support",
        "realized_support",
        "maximum_multiplicity",
        "tied_pair_fraction",
        "feature_position",
        "n_samples",
        "n_features",
        "data_seed",
        "model_seed",
        "B",
        "realized_candidate_resamples",
        "n_trees",
        "max_depth",
        "max_features",
        "sampling_with_replacement",
        "sample_size",
        "importance_type",
        "importance_permutations",
        "raw_importance",
        "raw_importance_missing",
        "trend_importance",
        "structural_zero_imputed",
    ),
    "cardinality_forest_replicate": (
        "experiment",
        "dgp",
        "task",
        "method",
        "replicate",
        "data_seed",
        "model_seed",
        "kendall_tau_b",
        "maximum_importance_tie_size",
        "zero_importance_count",
        "raw_importance_missing_count",
        "all_importances_equal",
        "average_rank_feature_0",
        "average_rank_feature_1",
        "average_rank_feature_2",
        "average_rank_feature_3",
        "average_rank_feature_4",
    ),
    "cardinality_forest_summary": (
        "experiment",
        "dgp",
        "hypothesis_id",
        "task",
        "method",
        "estimand",
        "alternative",
        "n_replicates",
        "n_defined_trends",
        "n_undefined_trends",
        "mean_kendall_tau_b",
        "randomization_p_value",
        "randomization_extreme_count",
        "per_replicate_label_permutations",
        "joint_randomization_resamples",
        "randomization_scheme",
        "randomization_p_value_rule",
        "mean_zero_importance_count",
        "mean_raw_importance_missing_count",
        "holm_family",
        "holm_family_size",
        "holm_input_p_value",
        "holm_adjusted_p_value",
        "holm_reject_0_05",
    ),
    "cardinality_holm_family": (
        "family",
        "family_size",
        "hypothesis_id",
        "experiment",
        "task",
        "method",
        "estimand",
        "alternative",
        "test_defined",
        "randomization_p_value",
        "holm_input_p_value",
        "holm_adjusted_p_value",
        "holm_reject_0_05",
        "minimum_attainable_raw_p_value",
        "minimum_attainable_holm_p_value",
        "undefined_test_treatment",
    ),
}

CALIBRATION_STORAGE_DTYPES: dict[str, dict[str, str]] = {
    "cardinality_tree_raw": {
        "B": "Int64",
        "selection_alpha": "Float64",
        "adjusted_alpha": "Float64",
        "attainable_adjusted_cutoff": "Float64",
        "p_value_support_index": "Int64",
        "p_value_numerator": "Int64",
        "p_value_denominator": "Int64",
        "p_value": "Float64",
        "native_p_value_support_index": "Int64",
        "native_p_value_numerator": "Int64",
        "native_p_value_denominator": "Int64",
        "native_p_value": "Float64",
        "minimum_p_tie_size": "Int64",
        "forced_winner_weight": "Float64",
        "adjusted_winner_weight": "Float64",
        "no_split_weight": "Float64",
        "native_score_tie_size": "Int64",
    },
}


def normalize_calibration_table(
    table_name: str,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the frozen in-memory dtypes used for Parquet serialization."""
    normalized = frame.copy()
    for column, dtype in CALIBRATION_STORAGE_DTYPES.get(table_name, {}).items():
        normalized[column] = normalized[column].astype(dtype)
    return normalized


def _validate_serializable_frame(name: str, frame: pd.DataFrame) -> None:
    """Reject nested cells and infinite values before Parquet/CSV writing."""
    if not frame.columns.is_unique:
        raise ValueError(f"{name} columns must be unique")
    numeric = frame.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy(dtype=np.float64)).any():
        raise ValueError(f"{name} contains infinite numeric values")
    nested_types = (dict, list, set, tuple, np.ndarray)
    for column in frame.columns:
        if frame[column].map(lambda value: isinstance(value, nested_types)).any():
            raise TypeError(f"{name}.{column} contains a non-serializable nested value")


def _validate_exact_keys(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    expected: Set[tuple[object, ...]],
    label: str,
) -> None:
    if frame.duplicated(list(columns)).any():
        raise ValueError(f"{label} contains duplicate keys")
    expected_set = set(expected)
    observed = set(frame.loc[:, columns].itertuples(index=False, name=None))
    if observed != expected_set or len(frame) != len(expected_set):
        missing = sorted(expected_set.difference(observed), key=repr)
        extra = sorted(observed.difference(expected_set), key=repr)
        raise ValueError(f"{label} inventory differs: missing={missing}, extra={extra}")


def _matches_expected_value(observed: object, expected: object) -> bool:
    if expected is None:
        return bool(pd.isna(observed))
    return bool(observed == expected)


def _require_integral(value: object, label: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or float(value) != np.floor(float(value))
    ):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def validate_selector_null_raw(
    raw: pd.DataFrame,
    profile: Profile,
    *,
    base_seed: int,
    replicate_indices: Sequence[int] | None = None,
) -> None:
    """Require the exact selector-null inventory, metadata, seeds, and decisions."""
    table_name = "selector_null_raw"
    if tuple(raw.columns) != CALIBRATION_RESULT_SCHEMAS[table_name]:
        raise ValueError(f"{table_name} differs from its required schema")
    settings = _settings(profile)
    replicates = _resolve_replicates(settings.selector_replicates, replicate_indices)
    scenarios = _selector_scenarios(profile, settings)
    scenario_by_name = {scenario.scenario: scenario for scenario in scenarios}
    expected = {
        (scenario.scenario, replicate) for scenario in scenarios for replicate in replicates
    }
    _validate_exact_keys(
        raw,
        columns=("scenario", "replicate"),
        expected=expected,
        label=table_name,
    )
    for row in raw.itertuples(index=False):
        scenario = scenario_by_name[str(row.scenario)]
        replicate = _require_integral(row.replicate, f"{table_name}.replicate")
        for field, expected_value in asdict(scenario).items():
            if not _matches_expected_value(getattr(row, field), expected_value):
                raise ValueError(f"{table_name} scenario metadata differs for {row.scenario}")
        if row.experiment != "selector_null" or row.estimand != (
            "fixed_budget_permutation_p_value"
        ):
            raise ValueError(f"{table_name} experiment metadata differs")
        if row.data_seed != _stream_seed(
            base_seed,
            scenario.data_design,
            replicate,
            "data",
        ) or row.model_seed != _stream_seed(
            base_seed,
            scenario.scenario,
            replicate,
            "model",
        ):
            raise ValueError(f"{table_name} seeds differ from the frozen derivation")
        realized = _require_integral(
            row.realized_permutations,
            f"{table_name}.realized_permutations",
        )
        if realized != scenario.n_resamples:
            raise ValueError(f"{table_name} must exhaust its fixed permutation budget")
        p_value = float(row.p_value)
        numerator = p_value * (scenario.n_resamples + 1)
        if (
            not np.isfinite(p_value)
            or not 0.0 < p_value <= 1.0
            or not np.isclose(numerator, round(numerator), rtol=0.0, atol=1e-10)
        ):
            raise ValueError(f"{table_name} p-values differ from the permutation support")
        if not isinstance(row.rejected, (bool, np.bool_)) or bool(row.rejected) != (
            p_value < scenario.alpha
        ):
            raise ValueError(f"{table_name} rejection flags differ from p < alpha")


def validate_root_null_raw(
    raw: pd.DataFrame,
    profile: Profile,
    *,
    base_seed: int,
    replicate_indices: Sequence[int] | None = None,
) -> None:
    """Require the exact fitted-root inventory, metadata, seeds, and counts."""
    table_name = "root_null_raw"
    if tuple(raw.columns) != CALIBRATION_RESULT_SCHEMAS[table_name]:
        raise ValueError(f"{table_name} differs from its required schema")
    settings = _settings(profile)
    replicates = _resolve_replicates(settings.root_replicates, replicate_indices)
    scenarios = _root_scenarios(profile, settings)
    scenario_by_name = {scenario.scenario: scenario for scenario in scenarios}
    expected = {
        (scenario.scenario, replicate) for scenario in scenarios for replicate in replicates
    }
    _validate_exact_keys(
        raw,
        columns=("scenario", "replicate"),
        expected=expected,
        label=table_name,
    )
    for row in raw.itertuples(index=False):
        scenario = scenario_by_name[str(row.scenario)]
        replicate = _require_integral(row.replicate, f"{table_name}.replicate")
        for field, expected_value in asdict(scenario).items():
            if not _matches_expected_value(getattr(row, field), expected_value):
                raise ValueError(f"{table_name} scenario metadata differs for {row.scenario}")
        if row.experiment != "root_null" or row.estimand != "fitted_tree_split_decision":
            raise ValueError(f"{table_name} experiment metadata differs")
        if row.data_seed != _stream_seed(
            base_seed,
            scenario.data_design,
            replicate,
            "data",
        ) or row.model_seed != _stream_seed(
            base_seed,
            scenario.model_design,
            replicate,
            "model",
        ):
            raise ValueError(f"{table_name} seeds differ from the frozen derivation")
        selector_count = _require_integral(
            row.realized_selector_permutations,
            f"{table_name}.realized_selector_permutations",
        )
        splitter_count = _require_integral(
            row.realized_splitter_permutations,
            f"{table_name}.realized_splitter_permutations",
        )
        total_count = _require_integral(
            row.realized_permutations,
            f"{table_name}.realized_permutations",
        )
        if min(selector_count, splitter_count) < 0 or total_count != (
            selector_count + splitter_count
        ):
            raise ValueError(f"{table_name} permutation counts are inconsistent")
        if scenario.gate == "selector" and (selector_count < 1 or splitter_count != 0):
            raise ValueError(f"{table_name} selector-gate counts are inconsistent")
        if scenario.gate == "splitter" and (selector_count != 0 or splitter_count < 1):
            raise ValueError(f"{table_name} splitter-gate counts are inconsistent")
        if scenario.gate == "combined" and selector_count < 1:
            raise ValueError(f"{table_name} combined-gate counts are inconsistent")
        if not isinstance(row.split, (bool, np.bool_)):
            raise ValueError(f"{table_name}.split must be boolean")


def validate_cardinality_tree_experiment(
    raw: pd.DataFrame,
    profile: Profile,
    *,
    base_seed: int,
    replicate_indices: Sequence[int] | None = None,
) -> None:
    """Require complete tree-method coverage and frozen per-fit seeds."""
    table_name = "cardinality_tree_raw"
    if tuple(raw.columns) != CALIBRATION_RESULT_SCHEMAS[table_name]:
        raise ValueError(f"{table_name} differs from its required schema")
    validate_cardinality_tree_raw(raw)
    replicates = _resolve_replicates(
        _settings(profile).cardinality_replicates,
        replicate_indices,
    )
    methods: tuple[TreeMethod, ...] = ("citrees", "partykit", "cart")
    expected = {
        (task, method, replicate, feature_id)
        for task in ("classification", "regression")
        for replicate in replicates
        for method in methods
        for feature_id in range(len(CARDINALITY_SUPPORTS))
    }
    _validate_exact_keys(
        raw,
        columns=("task", "method", "replicate", "feature_id"),
        expected=expected,
        label=table_name,
    )
    for (task, replicate, method), group in raw.groupby(
        ["task", "replicate", "method"],
        sort=False,
    ):
        replicate_value = _require_integral(replicate, f"{table_name}.replicate")
        data_design = f"cardinality_bias__{task}__n{CARDINALITY_N_SAMPLES}"
        model_design = f"{task}__{method}__n{CARDINALITY_N_SAMPLES}"
        if set(group["data_seed"]) != {
            _stream_seed(base_seed, data_design, replicate_value, "data")
        } or set(group["model_seed"]) != {
            _stream_seed(base_seed, model_design, replicate_value, "model")
        }:
            raise ValueError(f"{table_name} seeds differ from the frozen derivation")


def validate_cardinality_forest_experiment(
    raw: pd.DataFrame,
    profile: Profile,
    *,
    base_seed: int,
    replicate_indices: Sequence[int] | None = None,
) -> None:
    """Require complete forest-method coverage and frozen per-fit seeds."""
    table_name = "cardinality_forest_raw"
    if tuple(raw.columns) != CALIBRATION_RESULT_SCHEMAS[table_name]:
        raise ValueError(f"{table_name} differs from its required schema")
    validate_cardinality_forest_raw(raw)
    replicates = _resolve_replicates(
        _settings(profile).cardinality_forest_replicates,
        replicate_indices,
    )
    methods: tuple[ForestMethod, ...] = (
        "citrees_cif",
        "partykit_cforest",
        "sklearn_rf",
    )
    expected = {
        (task, method, replicate, feature_id)
        for task in ("classification", "regression")
        for replicate in replicates
        for method in methods
        for feature_id in range(len(CARDINALITY_SUPPORTS))
    }
    _validate_exact_keys(
        raw,
        columns=("task", "method", "replicate", "feature_id"),
        expected=expected,
        label=table_name,
    )
    for (task, replicate, method), group in raw.groupby(
        ["task", "replicate", "method"],
        sort=False,
    ):
        replicate_value = _require_integral(replicate, f"{table_name}.replicate")
        data_design = f"cardinality_bias__{task}__n{CARDINALITY_N_SAMPLES}"
        model_design = f"{task}__{method}__n{CARDINALITY_N_SAMPLES}"
        if set(group["data_seed"]) != {
            _stream_seed(base_seed, data_design, replicate_value, "data")
        } or set(group["model_seed"]) != {
            _stream_seed(base_seed, model_design, replicate_value, "model")
        }:
            raise ValueError(f"{table_name} seeds differ from the frozen derivation")


def validate_calibration_results(
    results: dict[str, pd.DataFrame],
    *,
    profile: Profile,
    base_seed: int,
) -> None:
    """Validate the complete production result and writer contract."""
    if set(results) != set(CALIBRATION_RESULT_SCHEMAS):
        missing = sorted(set(CALIBRATION_RESULT_SCHEMAS).difference(results))
        extra = sorted(set(results).difference(CALIBRATION_RESULT_SCHEMAS))
        raise ValueError(f"Calibration result tables differ: missing={missing}, extra={extra}")
    for name, expected_columns in CALIBRATION_RESULT_SCHEMAS.items():
        frame = results[name]
        if tuple(frame.columns) != expected_columns:
            raise ValueError(
                f"{name} schema differs: expected={expected_columns}, "
                f"observed={tuple(frame.columns)}"
            )
        _validate_serializable_frame(name, frame)

    validate_selector_null_raw(
        results["selector_null_raw"],
        profile,
        base_seed=base_seed,
    )
    validate_root_null_raw(
        results["root_null_raw"],
        profile,
        base_seed=base_seed,
    )
    validate_cardinality_tree_experiment(
        results["cardinality_tree_raw"],
        profile,
        base_seed=base_seed,
    )
    validate_cardinality_forest_experiment(
        results["cardinality_forest_raw"],
        profile,
        base_seed=base_seed,
    )
    expected_selector_summary = summarize_selector_null(results["selector_null_raw"])
    if (
        not results["selector_null_summary"]
        .convert_dtypes()
        .equals(expected_selector_summary.convert_dtypes())
    ):
        raise ValueError("selector_null_summary differs from the validated raw rows")
    expected_root_summary = summarize_root_null(results["root_null_raw"])
    if (
        not results["root_null_summary"]
        .convert_dtypes()
        .equals(expected_root_summary.convert_dtypes())
    ):
        raise ValueError("root_null_summary differs from the validated raw rows")
    design = results["cardinality_design"]
    if (
        len(design) != len(CARDINALITY_SUPPORTS)
        or tuple(design["nominal_support"]) != CARDINALITY_SUPPORTS
        or not design["maximum_support_feature_type"].eq(CARDINALITY_MAX_SUPPORT_FEATURE_TYPE).all()
    ):
        raise ValueError("The serialized cardinality design differs from the frozen DGP")
    family = results["cardinality_holm_family"]
    if (
        len(family) != CARDINALITY_HOLM_FAMILY_SIZE
        or family["hypothesis_id"].duplicated().any()
        or not family["family"].eq(CARDINALITY_HOLM_FAMILY).all()
        or not family["family_size"].eq(CARDINALITY_HOLM_FAMILY_SIZE).all()
    ):
        raise ValueError("The serialized cardinality Holm family is incomplete")
    for name in (
        "cardinality_tree_trend_summary",
        "cardinality_forest_summary",
    ):
        summary = results[name]
        if (
            not summary["per_replicate_label_permutations"]
            .eq(CARDINALITY_EXACT_LABEL_PERMUTATIONS)
            .all()
            or not summary["joint_randomization_resamples"].eq(CARDINALITY_TREND_RESAMPLES).all()
            or not summary["randomization_scheme"].eq(CARDINALITY_TREND_RANDOMIZATION).all()
            or not summary["randomization_p_value_rule"].eq(CARDINALITY_TREND_P_VALUE_RULE).all()
        ):
            raise ValueError(f"{name} has inconsistent randomization metadata")
    if (
        not family["minimum_attainable_raw_p_value"]
        .eq(1.0 / (CARDINALITY_TREND_RESAMPLES + 1))
        .all()
        or not family["minimum_attainable_holm_p_value"].lt(CARDINALITY_ALPHA).all()
    ):
        raise ValueError("The serialized Holm family has insufficient p-value resolution")


def assemble_calibration_results(
    selector_raw: pd.DataFrame,
    root_raw: pd.DataFrame,
    tree_raw: pd.DataFrame,
    forest_raw: pd.DataFrame,
    *,
    profile: Profile,
    base_seed: int = BASE_SEED,
) -> dict[str, pd.DataFrame]:
    """Build and validate final calibration tables from complete raw inputs."""
    selector_raw = normalize_calibration_table("selector_null_raw", selector_raw)
    root_raw = normalize_calibration_table("root_null_raw", root_raw)
    tree_raw = normalize_calibration_table("cardinality_tree_raw", tree_raw)
    forest_raw = normalize_calibration_table("cardinality_forest_raw", forest_raw)
    validate_selector_null_raw(
        selector_raw,
        profile,
        base_seed=base_seed,
    )
    validate_root_null_raw(
        root_raw,
        profile,
        base_seed=base_seed,
    )
    validate_cardinality_tree_experiment(
        tree_raw,
        profile,
        base_seed=base_seed,
    )
    validate_cardinality_forest_experiment(
        forest_raw,
        profile,
        base_seed=base_seed,
    )
    tree_replicate, tree_trend, tree_method = summarize_cardinality_tree(
        tree_raw,
        random_state=base_seed,
    )
    forest_replicate, forest_summary = summarize_cardinality_forest_trends(
        forest_raw,
        random_state=base_seed,
    )
    tree_trend, forest_summary, holm_family = apply_cardinality_holm(
        tree_trend,
        forest_summary,
    )
    results = {
        "selector_null_raw": selector_raw,
        "selector_null_summary": summarize_selector_null(selector_raw),
        "root_null_raw": root_raw,
        "root_null_summary": summarize_root_null(root_raw),
        "cardinality_design": cardinality_design(),
        "cardinality_tree_raw": tree_raw,
        "cardinality_tree_replicate": tree_replicate,
        "cardinality_tree_trend_summary": tree_trend,
        "cardinality_tree_method_summary": tree_method,
        "cardinality_forest_raw": forest_raw,
        "cardinality_forest_replicate": forest_replicate,
        "cardinality_forest_summary": forest_summary,
        "cardinality_holm_family": holm_family,
    }
    validate_calibration_results(
        results,
        profile=profile,
        base_seed=base_seed,
    )
    return results


def run_calibration(
    profile: Profile,
    *,
    base_seed: int = BASE_SEED,
) -> dict[str, pd.DataFrame]:
    """Run all calibration analyses and return validated production tables."""
    return assemble_calibration_results(
        run_selector_null(profile, base_seed=base_seed),
        run_root_null(profile, base_seed=base_seed),
        run_cardinality_trees(profile, base_seed=base_seed),
        run_cardinality_forests(profile, base_seed=base_seed),
        profile=profile,
        base_seed=base_seed,
    )


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return result.stdout.strip()


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return bool(result.stdout.strip())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _r_environment() -> dict[str, str]:
    return get_r_runtime_versions()


def write_results(
    results: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    profile: Profile,
    base_seed: int,
    elapsed_seconds: float,
) -> None:
    """Write analysis tables and a machine-readable execution receipt."""
    results = {name: normalize_calibration_table(name, frame) for name, frame in results.items()}
    validate_calibration_results(
        results,
        profile=profile,
        base_seed=base_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: list[Path] = []
    for name, frame in results.items():
        parquet_name = f"{name}.parquet"
        parquet_path = output_dir / parquet_name
        frame.to_parquet(parquet_path, index=False)
        artifact_paths.append(parquet_path)
        if name.endswith("_summary") or name.endswith("_family") or name == "cardinality_design":
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
        "profile": profile,
        "base_seed": base_seed,
        "settings": asdict(_settings(profile)),
        "cardinality_design": {
            "dgp": CARDINALITY_DGP,
            "n_samples": CARDINALITY_N_SAMPLES,
            "supports": list(CARDINALITY_SUPPORTS),
            "maximum_support_feature_type": CARDINALITY_MAX_SUPPORT_FEATURE_TYPE,
            "base_permutations": CARDINALITY_B,
            "forest_trees": CARDINALITY_FOREST_TREES,
            "forest_importance_permutations": (CARDINALITY_FOREST_IMPORTANCE_PERMUTATIONS),
            "exact_label_permutations": CARDINALITY_EXACT_LABEL_PERMUTATIONS,
            "trend_randomization_resamples": CARDINALITY_TREND_RESAMPLES,
            "trend_randomization_scheme": CARDINALITY_TREND_RANDOMIZATION,
            "trend_randomization_p_value_rule": CARDINALITY_TREND_P_VALUE_RULE,
            "holm_family": CARDINALITY_HOLM_FAMILY,
            "holm_family_size": CARDINALITY_HOLM_FAMILY_SIZE,
            "minimum_attainable_raw_p_value": 1.0 / (CARDINALITY_TREND_RESAMPLES + 1),
            "minimum_attainable_holm_p_value": CARDINALITY_HOLM_FAMILY_SIZE
            / (CARDINALITY_TREND_RESAMPLES + 1),
        },
        "created_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "r_environment": _r_environment(),
        "source_sha256": {str(path.relative_to(repo_root)): _sha256(path) for path in source_files},
        "versions": versions,
        "tables": {
            name: {
                "rows": len(frame),
                "columns": list(frame.columns),
            }
            for name, frame in results.items()
        },
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
