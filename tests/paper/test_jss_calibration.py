"""Tests for the JSS calibration replication analysis."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from paper.jss.replication.calibration import (
    CARDINALITY_LABELS,
    ProfileSettings,
    SelectorNullScenario,
    _root_scenarios,
    _selector_p_value,
    _selector_scenarios,
    _settings,
    _stream_seed,
    run_calibration,
    run_selector_null,
    wilson_interval,
    write_results,
)

pytestmark = pytest.mark.paper


def test_selector_null_is_deterministic() -> None:
    first = run_selector_null("smoke", base_seed=42)
    second = run_selector_null("smoke", base_seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_smoke_profile_schema_and_counts() -> None:
    results = run_calibration("smoke", base_seed=42)

    assert set(results) == {
        "selector_null_raw",
        "selector_null_summary",
        "root_null_raw",
        "root_null_summary",
        "cardinality_bias_raw",
        "cardinality_bias_summary",
        "cardinality_bias_global",
    }
    assert len(results["selector_null_raw"]) == 8
    assert len(results["root_null_raw"]) == 6
    assert len(results["cardinality_bias_raw"]) == 32
    assert set(results["cardinality_bias_summary"]["cardinality"]) == set(CARDINALITY_LABELS)
    assert results["selector_null_raw"]["p_value"].between(0.0, 1.0).all()
    assert {"data_seed", "model_seed"} <= set(results["selector_null_raw"])
    assert {"data_seed", "model_seed"} <= set(results["root_null_raw"])
    assert results["selector_null_summary"].columns.is_unique

    cardinality = results["cardinality_bias_raw"]
    paired_data_counts = cardinality.groupby(["task", "replicate"])["data_seed"].nunique()
    assert paired_data_counts.eq(1).all()
    assert cardinality.groupby(["task", "replicate"])["model_seed"].nunique().eq(2).all()
    assert cardinality.loc[cardinality["method"] == "citrees", "n_resamples"].notna().all()
    assert cardinality.loc[cardinality["method"] == "cart", "n_resamples"].isna().all()


def test_full_profile_covers_all_stopping_modes_and_multi_selectors() -> None:
    settings = _settings("full")
    assert settings == ProfileSettings(
        selector_replicates=5_000,
        root_replicates=5_000,
        cardinality_replicates=10_000,
        selector_resamples=999,
        root_resamples=999,
        cardinality_resamples=999,
    )

    selector_scenarios = _selector_scenarios("full", settings)
    assert {scenario.stopping for scenario in selector_scenarios} == {
        "fixed",
        "adaptive",
        "simple",
    }
    assert {scenario.selector for scenario in selector_scenarios} >= {
        "mc+rdc",
        "pc+dc+rdc",
    }

    root_scenarios = _root_scenarios("full", settings)
    assert {scenario.stopping for scenario in root_scenarios} == {
        "fixed",
        "adaptive",
        "simple",
    }
    assert {scenario.gate for scenario in root_scenarios} == {
        "selector",
        "splitter",
        "combined",
    }


def test_selector_design_pairs_data_but_separates_model_randomness() -> None:
    settings = _settings("quick")
    scenarios = [
        scenario
        for scenario in _selector_scenarios("quick", settings)
        if scenario.task == "classification"
        and scenario.selector == "mc"
        and scenario.feature_distribution == "normal"
    ]
    assert {scenario.stopping for scenario in scenarios} == {"fixed", "adaptive"}

    data_seeds = {
        _stream_seed(42, scenario.data_design, replicate=3, stream="data") for scenario in scenarios
    }
    model_seeds = {
        _stream_seed(42, scenario.scenario, replicate=3, stream="model") for scenario in scenarios
    }
    assert len(data_seeds) == 1
    assert len(model_seeds) == 2
    assert data_seeds.isdisjoint(model_seeds)


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
        stopping="fixed",
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

    p_value = _selector_p_value(scenario, x, y, seed=7)

    assert 0.0 < p_value <= 1.0


def test_wilson_interval_contains_observed_proportion() -> None:
    lower, upper = wilson_interval(5, 20)
    assert lower < 0.25 < upper
    assert np.isnan(wilson_interval(0, 0)[0])


def test_write_results_records_artifacts(tmp_path) -> None:
    results = run_calibration("smoke", base_seed=7)
    write_results(
        results,
        tmp_path,
        profile="smoke",
        base_seed=7,
        elapsed_seconds=1.25,
    )

    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="ascii"))
    assert receipt["analysis"] == "calibration"
    assert receipt["schema_version"] == 2
    assert receipt["profile"] == "smoke"
    assert receipt["base_seed"] == 7
    assert isinstance(receipt["git_dirty"], bool)
    assert "paper/jss/replication/calibration.py" in receipt["source_sha256"]
    assert "citrees/_selector.py" in receipt["source_sha256"]
    assert "citrees/_tree.py" in receipt["source_sha256"]
    assert "uv.lock" in receipt["source_sha256"]
    assert len(receipt["artifacts"]) > len(results)
    for artifact, metadata in receipt["artifacts"].items():
        artifact_path = tmp_path / artifact
        assert artifact_path.exists()
        assert metadata["bytes"] == artifact_path.stat().st_size
        assert metadata["sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
