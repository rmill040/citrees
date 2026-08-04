"""Tests for the JSS calibration replication analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paper.jss.replication import calibration
from paper.jss.replication.calibration import (
    ProfileSettings,
    RootNullScenario,
    SelectorNullScenario,
    _fit_null_tree,
    _fixed_budget_selector_p_value,
    _fixed_budget_selector_test,
    _root_scenarios,
    _selector_scenarios,
    _settings,
    _stream_seed,
    run_root_null,
    run_selector_null,
    wilson_interval,
)

pytestmark = pytest.mark.paper


def test_selector_null_is_deterministic() -> None:
    first = run_selector_null("smoke", base_seed=42)
    second = run_selector_null("smoke", base_seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_root_null_is_deterministic() -> None:
    first = run_root_null("smoke", base_seed=42)
    second = run_root_null("smoke", base_seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_smoke_selector_and_root_schema_and_counts() -> None:
    selector = run_selector_null("smoke", base_seed=42)
    root = run_root_null("smoke", base_seed=42)

    assert len(selector) == 8
    assert len(root) == 18
    assert selector["p_value"].between(0.0, 1.0).all()
    assert selector["estimand"].eq("fixed_budget_permutation_p_value").all()
    assert selector["realized_permutations"].between(0, selector["n_resamples"]).all()
    assert "stopping" not in selector
    assert root["estimand"].eq("fitted_tree_split_decision").all()
    assert "p_value" not in root
    assert (
        root["realized_permutations"]
        == root["realized_selector_permutations"] + root["realized_splitter_permutations"]
    ).all()
    selector_only = root[root["gate"] == "selector"]
    splitter_only = root[root["gate"] == "splitter"]
    combined = root[root["gate"] == "combined"]
    assert selector_only["realized_selector_permutations"].gt(0).all()
    assert selector_only["realized_splitter_permutations"].eq(0).all()
    assert splitter_only["realized_selector_permutations"].eq(0).all()
    assert splitter_only["realized_splitter_permutations"].gt(0).all()
    assert combined["realized_selector_permutations"].gt(0).all()
    assert combined["realized_splitter_permutations"].ge(0).all()
    assert {"data_seed", "model_seed"} <= set(selector)
    assert {"data_seed", "model_seed"} <= set(root)
    paired_root = combined.groupby("replicate")[["data_seed", "model_seed"]].nunique()
    assert paired_root["data_seed"].eq(1).all()
    assert paired_root["model_seed"].eq(1).all()


def test_full_profile_separates_fixed_budget_and_fitted_tree_estimands() -> None:
    settings = _settings("full")
    assert settings == ProfileSettings(
        selector_replicates=5_000,
        root_replicates=5_000,
        cardinality_replicates=10_000,
        cardinality_forest_replicates=2_500,
        selector_resamples=999,
        root_resamples=999,
    )

    selector_scenarios = _selector_scenarios("full", settings)
    assert {scenario.selector for scenario in selector_scenarios} >= {
        "mc+rdc",
        "pc+dc+rdc",
    }
    assert all("fixed_budget" in scenario.scenario for scenario in selector_scenarios)

    root_scenarios = _root_scenarios("full", settings)
    assert {scenario.stopping for scenario in root_scenarios} == {
        "exhaustive",
        "adaptive",
        "simple",
    }
    assert {scenario.gate for scenario in root_scenarios} == {
        "selector",
        "splitter",
        "combined",
    }
    quick_scenarios = _root_scenarios("quick", _settings("quick"))
    assert {scenario.stopping for scenario in quick_scenarios} == {
        "exhaustive",
        "adaptive",
    }


def test_smoke_root_scenarios_cover_the_representative_design() -> None:
    scenarios = _root_scenarios("smoke", _settings("smoke"))

    assert len(scenarios) == 6
    assert {scenario.task for scenario in scenarios} == {"classification", "regression"}
    assert {scenario.gate for scenario in scenarios} == {"selector", "splitter", "combined"}
    assert {scenario.stopping for scenario in scenarios} == {
        "exhaustive",
        "adaptive",
        "simple",
    }
    combined = [
        scenario
        for scenario in scenarios
        if scenario.task == "classification"
        and scenario.gate == "combined"
        and scenario.stopping == "adaptive"
    ]
    assert {(scenario.feature_scanning, scenario.threshold_scanning) for scenario in combined} == {
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    }


def test_root_scanning_factors_are_independent_and_paired() -> None:
    settings = _settings("quick")
    scenarios = [
        scenario
        for scenario in _root_scenarios("quick", settings)
        if scenario.task == "classification"
        and scenario.gate == "combined"
        and scenario.feature_distribution == "normal"
    ]
    assert {scenario.stopping for scenario in scenarios} == {"exhaustive", "adaptive"}
    assert {(scenario.feature_scanning, scenario.threshold_scanning) for scenario in scenarios} == {
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    }

    data_seeds = {
        _stream_seed(42, scenario.data_design, replicate=3, stream="data") for scenario in scenarios
    }
    model_seeds = {
        _stream_seed(42, scenario.model_design, replicate=3, stream="model")
        for scenario in scenarios
    }
    assert len(data_seeds) == 1
    assert len(model_seeds) == 1
    assert data_seeds.isdisjoint(model_seeds)


def test_inactive_scanning_factor_is_always_disabled() -> None:
    scenarios = _root_scenarios("full", _settings("full"))

    assert all(
        not scenario.threshold_scanning for scenario in scenarios if scenario.gate == "selector"
    )
    assert all(
        not scenario.feature_scanning for scenario in scenarios if scenario.gate == "splitter"
    )
    assert all(
        not scenario.feature_scanning and not scenario.threshold_scanning
        for scenario in scenarios
        if scenario.stopping == "exhaustive"
    )


def test_exhaustive_fit_disables_stopping_and_scanning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class FakeTree:
        def __init__(self, **kwargs: object) -> None:
            observed.update(kwargs)
            self._n_nodes = 1
            self.realized_permutation_counts_ = {"selector": 39, "splitter": 0}

        def fit(self, X: np.ndarray, y: np.ndarray) -> None:
            assert X.shape == (40, 2)
            assert y.shape == (40,)

    monkeypatch.setattr(calibration, "ConditionalInferenceTreeClassifier", FakeTree)
    scenario = RootNullScenario(
        task="classification",
        gate="combined",
        stopping="exhaustive",
        feature_scanning=True,
        threshold_scanning=True,
        feature_distribution="normal",
        n_samples=40,
        n_features=2,
        base_resamples=39,
    )

    result = _fit_null_tree(
        scenario,
        np.zeros((40, 2), dtype=float),
        np.tile(np.array([0, 1], dtype=np.int64), 20),
        seed=7,
    )

    assert not scenario.feature_scanning
    assert not scenario.threshold_scanning
    assert observed["early_stopping_selector"] is None
    assert observed["early_stopping_splitter"] is None
    assert observed["feature_scanning"] is False
    assert observed["threshold_scanning"] is False
    assert result.selector_permutations == 39
    assert result.splitter_permutations == 0


def test_permutation_count_property_requires_fit() -> None:
    model = calibration.ConditionalInferenceTreeClassifier()

    with pytest.raises(
        AttributeError,
        match="realized_permutation_counts_ is available only after fitting",
    ):
        _ = model.realized_permutation_counts_


def test_selector_test_forces_fixed_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_ptest_mc(**kwargs: object) -> float:
        observed.update(kwargs)
        return 0.25

    monkeypatch.setattr(calibration, "ptest_mc", fake_ptest_mc)
    scenario = SelectorNullScenario(
        task="classification",
        selector="mc",
        feature_distribution="normal",
        n_samples=40,
        n_resamples=39,
    )
    x = np.linspace(-1.0, 1.0, 40)
    y = np.tile(np.array([0, 1], dtype=np.int64), 20)

    p_value = _fixed_budget_selector_p_value(scenario, x, y, seed=7)

    assert p_value == 0.25
    assert observed["early_stopping"] is None
    assert observed["n_resamples"] == 39


def test_degenerate_parallel_selector_records_zero_permutations() -> None:
    scenario = SelectorNullScenario(
        task="classification",
        selector="mc",
        feature_distribution="normal",
        n_samples=40,
        n_resamples=200,
    )
    x = np.ones(40)
    y = np.tile(np.array([0, 1], dtype=np.int64), 20)

    result = _fixed_budget_selector_test(scenario, x, y, seed=7)

    assert result.p_value == 1.0
    assert result.permutations == 0


@pytest.mark.parametrize(
    ("task", "selector"),
    [
        ("classification", "mc+rdc"),
        ("regression", "pc+dc+rdc"),
    ],
)
def test_multi_selector_scenario_executes(task: str, selector: str) -> None:
    scenario = SelectorNullScenario(
        task=task,
        selector=selector,
        feature_distribution="normal",
        n_samples=40,
        n_resamples=39,
    )
    rng = np.random.default_rng(42)
    x = rng.standard_normal(40)
    y = (
        np.tile(np.array([0, 1], dtype=np.int64), 20)
        if task == "classification"
        else rng.standard_normal(40)
    )

    p_value = _fixed_budget_selector_p_value(scenario, x, y, seed=7)

    assert 0.0 < p_value <= 1.0


def test_wilson_interval_contains_observed_proportion() -> None:
    lower, upper = wilson_interval(5, 20)
    assert lower < 0.25 < upper
    assert np.isnan(wilson_interval(0, 0)[0])
