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
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import chisquare, norm
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

BASE_SEED = 1718
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "calibration"


@dataclass(frozen=True)
class ProfileSettings:
    """Simulation counts and permutation budgets for one replication profile."""

    selector_replicates: int
    root_replicates: int
    cardinality_replicates: int
    selector_resamples: int
    root_resamples: int
    cardinality_resamples: int


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
            cardinality_resamples=39,
        )
    if profile == "quick":
        return ProfileSettings(
            selector_replicates=200,
            root_replicates=100,
            cardinality_replicates=500,
            selector_resamples=199,
            root_resamples=199,
            cardinality_resamples=199,
        )
    return ProfileSettings(
        selector_replicates=5_000,
        root_replicates=5_000,
        cardinality_replicates=10_000,
        selector_resamples=999,
        root_resamples=999,
        cardinality_resamples=999,
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


CARDINALITY_LABELS = ("binary", "4 levels", "10 levels", "20 levels", "continuous")


def _cardinality_matrix(
    rng: np.random.Generator,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    columns = [
        rng.integers(0, 2, n_samples),
        rng.integers(0, 4, n_samples),
        rng.integers(0, 10, n_samples),
        rng.integers(0, 20, n_samples),
        rng.standard_normal(n_samples),
    ]
    permutation = rng.permutation(len(columns))
    X = np.column_stack([columns[index] for index in permutation]).astype(np.float64)
    return X, permutation


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
    methods = ("citrees", "cart") if profile == "smoke" else ("citrees", "partykit", "cart")
    for task in ("classification", "regression"):
        data_design = f"cardinality_bias__{task}__n200"
        for replicate in range(settings.cardinality_replicates):
            data_seed = _stream_seed(base_seed, data_design, replicate, "data")
            rng = np.random.default_rng(data_seed)
            X, permutation = _cardinality_matrix(rng, 200)
            y = _response(rng, task, 200)
            for method in methods:
                scenario = f"{task}__{method}__n200"
                model_seed = _stream_seed(base_seed, scenario, replicate, "model")
                selected_column = _select_cardinality_feature(
                    task,
                    method,
                    X,
                    y,
                    model_seed,
                    settings.cardinality_resamples,
                )
                selected_feature = -1 if selected_column < 0 else int(permutation[selected_column])
                rows.append(
                    {
                        "experiment": "cardinality_bias",
                        "scenario": scenario,
                        "task": task,
                        "method": method,
                        "replicate": replicate,
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "n_samples": 200,
                        "n_features": len(CARDINALITY_LABELS),
                        "selection_test": (
                            "fixed_monte_carlo"
                            if method == "citrees"
                            else "asymptotic"
                            if method == "partykit"
                            else "none"
                        ),
                        "n_resamples": (
                            settings.cardinality_resamples if method == "citrees" else None
                        ),
                        "selected_feature": selected_feature,
                        "selected_cardinality": (
                            "no split"
                            if selected_feature < 0
                            else CARDINALITY_LABELS[selected_feature]
                        ),
                    }
                )
    return pd.DataFrame(rows)


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
    support = n_resamples + 1
    rejected_values = max(int(np.ceil(alpha * support) - 1), 0)
    return rejected_values / support


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
