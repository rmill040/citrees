"""Focused tests for the JSS cardinality production analysis."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from paper.jss.replication import calibration
from paper.jss.replication.calibration import (
    CALIBRATION_RESULT_SCHEMAS,
    CARDINALITY_ALPHA,
    CARDINALITY_B,
    CARDINALITY_DGP,
    CARDINALITY_EXACT_LABEL_PERMUTATIONS,
    CARDINALITY_HOLM_FAMILY_SIZE,
    CARDINALITY_MAX_SUPPORT_FEATURE_TYPE,
    CARDINALITY_N_SAMPLES,
    CARDINALITY_SUPPORTS,
    CARDINALITY_TREND_P_VALUE_RULE,
    CARDINALITY_TREND_RANDOMIZATION,
    CARDINALITY_TREND_RESAMPLES,
    CardinalityCandidateDiagnostics,
    _attainable_alpha,
    _attainable_cutoff,
    _cardinality_matrix,
    _citrees_cardinality_diagnostics,
    _fit_cardinality_model,
    _partykit_cardinality_diagnostics,
    _permutation_p_value,
    _permutation_support_index,
    _settings,
    apply_cardinality_holm,
    average_ranks,
    build_cardinality_cart_raw,
    build_cardinality_forest_raw,
    build_cardinality_tree_raw,
    cardinality_design,
    controlled_split_weights,
    directional_effect,
    fractional_minimum_p_weights,
    holm_adjust,
    kendall_tau_b,
    one_sided_label_randomization,
    run_calibration,
    summarize_cardinality_forest_trends,
    summarize_cardinality_tree,
    validate_calibration_results,
    validate_cardinality_forest_raw,
    validate_cardinality_tree_raw,
    write_results,
)

pytestmark = pytest.mark.paper


def _plus_one_values(indices: np.ndarray, n_resamples: int) -> np.ndarray:
    return (indices + 1) / (n_resamples + 1)


def _synthetic_tree_raw(base_seed: int | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for task_index, task in enumerate(("classification", "regression")):
        _, features = _cardinality_matrix(
            np.random.default_rng(100 + task_index),
            CARDINALITY_N_SAMPLES,
        )
        data_seed = (
            500 + task_index
            if base_seed is None
            else calibration._stream_seed(
                base_seed,
                f"cardinality_bias__{task}__n{CARDINALITY_N_SAMPLES}",
                0,
                "data",
            )
        )
        citrees_indices = np.asarray([100, 200, 300, 400, 500], dtype=np.int64)
        citrees_budget = CARDINALITY_B * len(CARDINALITY_SUPPORTS)
        citrees_p_values = _plus_one_values(citrees_indices, citrees_budget)
        frames.append(
            build_cardinality_tree_raw(
                features,
                CardinalityCandidateDiagnostics(
                    statistics=np.linspace(0.1, 0.5, len(CARDINALITY_SUPPORTS)),
                    p_values=citrees_p_values,
                    native_p_values=citrees_p_values.copy(),
                    realized_permutations=np.full(
                        len(CARDINALITY_SUPPORTS),
                        citrees_budget,
                        dtype=np.int64,
                    ),
                    native_selected_position=None,
                    native_p_value_rule="plus_one",
                ),
                task=task,
                method="citrees",
                selector="mc" if task == "classification" else "pc",
                replicate=0,
                data_seed=data_seed,
                model_seed=(
                    700 + task_index
                    if base_seed is None
                    else calibration._stream_seed(
                        base_seed,
                        f"{task}__citrees__n{CARDINALITY_N_SAMPLES}",
                        0,
                        "model",
                    )
                ),
            )
        )

        partykit_counts = np.asarray([100, 200, 300, 400, 500], dtype=np.int64)
        frames.append(
            build_cardinality_tree_raw(
                features,
                CardinalityCandidateDiagnostics(
                    statistics=np.linspace(1.0, 5.0, len(CARDINALITY_SUPPORTS)),
                    p_values=_plus_one_values(partykit_counts, CARDINALITY_B),
                    native_p_values=partykit_counts / CARDINALITY_B,
                    realized_permutations=np.full(
                        len(CARDINALITY_SUPPORTS),
                        CARDINALITY_B,
                        dtype=np.int64,
                    ),
                    native_selected_position=None,
                    native_p_value_rule="r_monte_carlo",
                ),
                task=task,
                method="partykit",
                selector="quadratic",
                replicate=0,
                data_seed=data_seed,
                model_seed=(
                    800 + task_index
                    if base_seed is None
                    else calibration._stream_seed(
                        base_seed,
                        f"{task}__partykit__n{CARDINALITY_N_SAMPLES}",
                        0,
                        "model",
                    )
                ),
            )
        )

        scores = np.linspace(0.1, 0.5, len(CARDINALITY_SUPPORTS))
        frames.append(
            build_cardinality_cart_raw(
                features,
                scores,
                task=task,
                replicate=0,
                data_seed=data_seed,
                model_seed=(
                    900 + task_index
                    if base_seed is None
                    else calibration._stream_seed(
                        base_seed,
                        f"{task}__cart__n{CARDINALITY_N_SAMPLES}",
                        0,
                        "model",
                    )
                ),
                native_selected_position=int(np.argmax(scores)),
            )
        )
    raw = pd.concat(frames, ignore_index=True)
    validate_cardinality_tree_raw(raw)
    return raw


def _position_aligned_values(
    features: tuple[calibration.CardinalityFeatureDesign, ...],
    feature_values: list[float],
) -> np.ndarray:
    values = np.empty(len(features), dtype=np.float64)
    for feature in features:
        values[feature.position] = feature_values[feature.feature_id]
    return values


def _synthetic_forest_raw(base_seed: int | None = None) -> pd.DataFrame:
    patterns = {
        "citrees_cif": [1.0, 2.0, 3.0, 4.0, 5.0],
        "partykit_cforest": [3.0, 2.0, 5.0, 1.0, 4.0],
        "sklearn_rf": [5.0, 4.0, 3.0, 2.0, 1.0],
    }
    frames: list[pd.DataFrame] = []
    for task_index, task in enumerate(("classification", "regression")):
        for replicate in range(2):
            _, features = _cardinality_matrix(
                np.random.default_rng(200 + task_index * 10 + replicate),
                CARDINALITY_N_SAMPLES,
            )
            data_seed = (
                1_000 + task_index * 10 + replicate
                if base_seed is None
                else calibration._stream_seed(
                    base_seed,
                    f"cardinality_bias__{task}__n{CARDINALITY_N_SAMPLES}",
                    replicate,
                    "data",
                )
            )
            for method_index, (method, pattern) in enumerate(patterns.items()):
                frames.append(
                    build_cardinality_forest_raw(
                        features,
                        _position_aligned_values(features, pattern),
                        task=task,
                        method=method,  # type: ignore[arg-type]
                        replicate=replicate,
                        data_seed=data_seed,
                        model_seed=(
                            1_100 + task_index * 100 + replicate * 10 + method_index
                            if base_seed is None
                            else calibration._stream_seed(
                                base_seed,
                                f"{task}__{method}__n{CARDINALITY_N_SAMPLES}",
                                replicate,
                                "model",
                            )
                        ),
                    )
                )
    raw = pd.concat(frames, ignore_index=True)
    validate_cardinality_forest_raw(raw)
    return raw


def test_cardinality_budget_is_fixed_and_profiles_only_change_replicates() -> None:
    settings = {profile: _settings(profile) for profile in ("smoke", "quick", "full")}

    assert CARDINALITY_B == 999
    assert all(not hasattr(value, "cardinality_resamples") for value in settings.values())
    assert {profile: value.cardinality_replicates for profile, value in settings.items()} == {
        "smoke": 1,
        "quick": 500,
        "full": 10_000,
    }
    assert {
        profile: value.cardinality_forest_replicates for profile, value in settings.items()
    } == {"smoke": 1, "quick": 10, "full": 2_500}


def test_cardinality_dgp_is_exact_balanced_and_explicitly_discrete() -> None:
    matrix, features = _cardinality_matrix(
        np.random.default_rng(42),
        CARDINALITY_N_SAMPLES,
    )
    repeated_matrix, repeated_features = _cardinality_matrix(
        np.random.default_rng(42),
        CARDINALITY_N_SAMPLES,
    )

    np.testing.assert_array_equal(matrix, repeated_matrix)
    assert features == repeated_features
    assert matrix.shape == (200, 5)
    assert {feature.position for feature in features} == set(range(5))
    total_pairs = CARDINALITY_N_SAMPLES * (CARDINALITY_N_SAMPLES - 1) // 2
    for feature, support in zip(features, CARDINALITY_SUPPORTS, strict=True):
        values, counts = np.unique(matrix[:, feature.position], return_counts=True)
        multiplicity = CARDINALITY_N_SAMPLES // support
        tied_pairs = support * multiplicity * (multiplicity - 1) // 2
        assert values.tolist() == list(np.arange(support, dtype=np.float64))
        assert counts.tolist() == [multiplicity] * support
        assert feature.nominal_support == feature.realized_support == support
        assert feature.maximum_multiplicity == multiplicity
        assert feature.tied_pair_fraction == pytest.approx(tied_pairs / total_pairs)

    maximum_support = features[-1]
    assert maximum_support.nominal_support == 200
    assert maximum_support.maximum_multiplicity == 1
    design = cardinality_design()
    assert design["dgp"].eq(CARDINALITY_DGP).all()
    assert design["feature_type"].eq("ordered_discrete").all()
    assert design["maximum_support_feature_type"].eq(CARDINALITY_MAX_SUPPORT_FEATURE_TYPE).all()
    assert design.iloc[-1]["cardinality"] == "200 levels"

    with pytest.raises(ValueError, match="requires n_samples=200"):
        _cardinality_matrix(np.random.default_rng(42), 100)


def test_fractional_minimum_p_weights_are_permutation_invariant() -> None:
    support_indices = np.array([7, 2, 2, 9, 4], dtype=np.int64)
    expected = np.array([0.0, 0.5, 0.5, 0.0, 0.0])
    order = np.array([4, 2, 0, 3, 1])

    weights = fractional_minimum_p_weights(support_indices)
    permuted_weights = fractional_minimum_p_weights(support_indices[order])
    restored_weights = np.empty_like(permuted_weights)
    restored_weights[order] = permuted_weights

    np.testing.assert_array_equal(weights, expected)
    np.testing.assert_array_equal(restored_weights, expected)


def test_permutation_support_and_strict_cutoffs_are_explicit() -> None:
    for support_index in (0, 8, 9, 48, 49, CARDINALITY_B):
        p_value = _permutation_p_value(support_index, CARDINALITY_B)
        assert _permutation_support_index(p_value, CARDINALITY_B) == support_index

    assert _attainable_cutoff(0.05, CARDINALITY_B, "<") == pytest.approx(0.049)
    assert _attainable_cutoff(0.05, CARDINALITY_B, "<=") == pytest.approx(0.05)
    assert _attainable_alpha(0.05, CARDINALITY_B) == pytest.approx(0.049)
    assert _attainable_cutoff(0.001, CARDINALITY_B, "<") is None
    assert _attainable_cutoff(0.001, CARDINALITY_B, "<=") == pytest.approx(0.001)

    strict = controlled_split_weights([9, 20, 30, 40, 50], comparison="<")
    inclusive = controlled_split_weights([9, 20, 30, 40, 50], comparison="<=")
    assert not strict.split
    assert strict.no_split_weight == 1.0
    assert inclusive.split
    assert inclusive.no_split_weight == 0.0

    with pytest.raises(ValueError, match="not on"):
        _permutation_support_index(0.0105, CARDINALITY_B)


def test_directional_effect_retains_no_split_as_nan_and_randomized_trends() -> None:
    ranks = np.arange(5, dtype=np.float64)
    positive_weights = np.tile([0.0, 0.1, 0.2, 0.3, 0.4], (4, 1))
    zero_weights = np.full((4, 5), 0.2)
    negative_weights = np.tile([0.4, 0.3, 0.2, 0.1, 0.0], (4, 1))
    no_split_weights = np.zeros(5, dtype=np.float64)
    rank_matrix = np.tile(ranks, (4, 1))

    assert directional_effect(ranks, positive_weights[0]) > 0.0
    assert directional_effect(ranks, zero_weights[0]) == pytest.approx(0.0)
    assert directional_effect(ranks, negative_weights[0]) < 0.0
    assert np.isnan(directional_effect(ranks, no_split_weights))

    positive = one_sided_label_randomization(
        rank_matrix,
        positive_weights,
        statistic=directional_effect,
        random_state=42,
    )
    zero = one_sided_label_randomization(
        rank_matrix,
        zero_weights,
        statistic=directional_effect,
        random_state=42,
    )
    negative = one_sided_label_randomization(
        rank_matrix,
        negative_weights,
        statistic=directional_effect,
        random_state=42,
    )

    repeated = one_sided_label_randomization(
        rank_matrix,
        positive_weights,
        statistic=directional_effect,
        random_state=42,
    )
    assert positive == repeated
    assert CARDINALITY_EXACT_LABEL_PERMUTATIONS == 120
    assert positive.n_resamples == CARDINALITY_TREND_RESAMPLES == 9_999
    assert positive.p_value == pytest.approx(1 / (CARDINALITY_TREND_RESAMPLES + 1))
    assert zero.p_value == pytest.approx(1.0)
    assert negative.p_value == pytest.approx(1.0)


def test_holm_adjustment_matches_reference_values() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.002])

    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.008])


def test_independent_joint_randomization_has_attainable_holm_resolution() -> None:
    ranks = np.arange(len(CARDINALITY_SUPPORTS), dtype=np.float64)
    rank_matrix = np.tile(ranks, (2, 1))
    positive_weights = np.tile([0.0, 0.1, 0.2, 0.3, 0.4], (2, 1))
    result = one_sided_label_randomization(
        rank_matrix,
        positive_weights,
        statistic=directional_effect,
        random_state=42,
    )

    exact_joint_assignments = CARDINALITY_EXACT_LABEL_PERMUTATIONS**2
    minimum_exact_raw = 1 / exact_joint_assignments
    minimum_raw = 1 / (CARDINALITY_TREND_RESAMPLES + 1)
    adjusted = holm_adjust([minimum_raw] * CARDINALITY_HOLM_FAMILY_SIZE)

    assert exact_joint_assignments == 14_400
    assert minimum_exact_raw * CARDINALITY_HOLM_FAMILY_SIZE < CARDINALITY_ALPHA
    assert result.p_value * CARDINALITY_HOLM_FAMILY_SIZE < CARDINALITY_ALPHA
    np.testing.assert_allclose(
        adjusted,
        np.full(
            CARDINALITY_HOLM_FAMILY_SIZE,
            CARDINALITY_HOLM_FAMILY_SIZE * minimum_raw,
        ),
    )
    assert adjusted.max() < CARDINALITY_ALPHA


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_citrees_candidate_tests_match_native_bonferroni_controls(
    task: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict[str, object]] = []
    realized_budget = CARDINALITY_B * len(CARDINALITY_SUPPORTS)

    @contextmanager
    def fake_counts() -> Any:
        yield {"selector": realized_budget, "splitter": 0}

    def fake_test(**kwargs: object) -> float:
        observed.append(kwargs)
        return 1 / (realized_budget + 1)

    monkeypatch.setattr(calibration, "collect_permutation_counts", fake_counts)
    monkeypatch.setattr(calibration, "ptest_mc", fake_test)
    monkeypatch.setattr(calibration, "ptest_pc", fake_test)
    monkeypatch.setattr(calibration, "mc", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(calibration, "pc", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(calibration, "_fit_cardinality_model", lambda *_args: object())
    monkeypatch.setattr(calibration, "_selected_root_feature", lambda _model: 0)

    X = np.arange(1_000, dtype=np.float64).reshape(200, 5)
    y = (
        np.tile(np.array([0, 1], dtype=np.int64), 100)
        if task == "classification"
        else np.linspace(-1.0, 1.0, 200)
    )
    diagnostics = _citrees_cardinality_diagnostics(task, X, y, seed=7)  # type: ignore[arg-type]

    assert len(observed) == len(CARDINALITY_SUPPORTS)
    assert {call["n_resamples"] for call in observed} == {realized_budget}
    assert {call["alpha"] for call in observed} == {CARDINALITY_ALPHA / len(CARDINALITY_SUPPORTS)}
    assert {call["early_stopping"] for call in observed} == {None}
    np.testing.assert_array_equal(
        diagnostics.realized_permutations,
        np.full(5, realized_budget),
    )


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_native_citrees_model_uses_base_budget_for_internal_multiplication(
    task: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class FakeTree:
        def __init__(self, **kwargs: object) -> None:
            observed.update(kwargs)

        def fit(self, X: np.ndarray, y: np.ndarray) -> None:
            assert X.shape == (200, 5)
            assert y.shape == (200,)

    monkeypatch.setattr(calibration, "ConditionalInferenceTreeClassifier", FakeTree)
    monkeypatch.setattr(calibration, "ConditionalInferenceTreeRegressor", FakeTree)
    X = np.zeros((200, 5), dtype=np.float64)
    y = np.zeros(200, dtype=np.int64)

    _fit_cardinality_model(task, "citrees", X, y, seed=7)  # type: ignore[arg-type]

    assert observed["alpha_selector"] == CARDINALITY_ALPHA
    assert observed["adjust_alpha_selector"] is True
    assert observed["n_resamples_selector"] == CARDINALITY_B
    assert observed["early_stopping_selector"] is None


def test_partykit_diagnostics_retain_native_and_corrected_lattices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    exceedances = np.asarray([0, 9, 99, 499, 999], dtype=np.int64)
    corrected = _plus_one_values(exceedances, CARDINALITY_B)

    def fake_diagnostics(*_args: object, **kwargs: object) -> calibration.RCTreeRootDiagnostics:
        observed.update(kwargs)
        return calibration.RCTreeRootDiagnostics(
            root_feature=-1,
            statistics=np.arange(5, dtype=np.float64),
            p_values=corrected,
        )

    monkeypatch.setattr(calibration, "r_ctree_root_diagnostics", fake_diagnostics)
    diagnostics = _partykit_cardinality_diagnostics(
        "classification",
        np.zeros((200, 5), dtype=np.float64),
        np.tile(np.array([0, 1], dtype=np.int64), 100),
        seed=7,
    )

    assert observed["testtype"] == "MonteCarlo"
    assert observed["nresample"] == CARDINALITY_B
    assert observed["mincriterion"] == pytest.approx(0.99)
    np.testing.assert_allclose(diagnostics.p_values, corrected)
    np.testing.assert_allclose(
        diagnostics.native_p_values,
        exceedances / CARDINALITY_B,
    )
    assert diagnostics.native_p_value_rule == "r_monte_carlo"


def test_tree_raw_schema_is_unique_paired_and_retains_no_split() -> None:
    raw = _synthetic_tree_raw()

    assert raw.columns.is_unique
    assert not raw.duplicated(["task", "replicate", "method", "feature_id"]).any()
    assert raw.groupby(["task", "replicate"])["data_seed"].nunique().eq(1).all()
    assert raw.groupby(["task", "replicate", "method"]).size().eq(5).all()
    conditional = raw[raw["method"].isin(["citrees", "partykit"])]
    assert conditional["B"].eq(CARDINALITY_B).all()
    assert conditional["adjusted_no_split"].astype(bool).all()
    assert conditional["adjusted_winner_weight"].eq(0.0).all()
    assert conditional.groupby(["task", "method"])["forced_winner_weight"].sum().eq(1.0).all()
    assert set(raw["nominal_support"]) == set(CARDINALITY_SUPPORTS)
    assert raw["maximum_support_feature_type"].eq(CARDINALITY_MAX_SUPPORT_FEATURE_TYPE).all()


def test_tree_validator_rejects_corrupt_candidate_contracts() -> None:
    raw = _synthetic_tree_raw()
    conditional_index = raw.index[raw["method"] == "citrees"][0]

    out_of_support = raw.copy()
    out_of_support.loc[conditional_index, "p_value_support_index"] = 9_999
    with pytest.raises(ValueError, match="support indices"):
        validate_cardinality_tree_raw(out_of_support)

    position_mismatch = raw.copy()
    position_mismatch.loc[conditional_index, "feature_position"] = 99
    with pytest.raises(ValueError, match="feature position"):
        validate_cardinality_tree_raw(position_mismatch)

    nonminimum_weight = raw.copy()
    group = nonminimum_weight[
        (nonminimum_weight["task"] == "classification") & (nonminimum_weight["method"] == "citrees")
    ].sort_values("feature_id")
    nonminimum_weight.loc[group.index, "forced_winner_weight"] = 0.0
    nonminimum_index = group.index[~group["is_minimum_p"].astype(bool)][0]
    nonminimum_weight.loc[nonminimum_index, "forced_winner_weight"] = 1.0
    with pytest.raises(ValueError, match="minimum-p ties"):
        validate_cardinality_tree_raw(nonminimum_weight)

    missing_comparison = raw.drop(columns=["p_value_comparison"])
    with pytest.raises(ValueError, match="Missing cardinality tree columns"):
        validate_cardinality_tree_raw(missing_comparison)

    contradictory_native = raw.copy()
    contradictory_native.loc[conditional_index, "native_split"] = True
    with pytest.raises(ValueError, match="Native split indicators"):
        validate_cardinality_tree_raw(contradictory_native)


def test_tree_validator_matches_native_winner_by_feature_position() -> None:
    _, features = _cardinality_matrix(
        np.random.default_rng(100),
        CARDINALITY_N_SAMPLES,
    )
    budget = CARDINALITY_B * len(CARDINALITY_SUPPORTS)
    support_indices = np.asarray([100, 200, 300, 400, 0], dtype=np.int64)
    p_values = _plus_one_values(support_indices, budget)
    raw = build_cardinality_tree_raw(
        features,
        CardinalityCandidateDiagnostics(
            statistics=np.linspace(0.1, 0.5, len(CARDINALITY_SUPPORTS)),
            p_values=p_values,
            native_p_values=p_values.copy(),
            realized_permutations=np.full(
                len(CARDINALITY_SUPPORTS),
                budget,
                dtype=np.int64,
            ),
            native_selected_position=4,
            native_p_value_rule="plus_one",
        ),
        task="classification",
        method="citrees",
        selector="mc",
        replicate=0,
        data_seed=500,
        model_seed=700,
    )
    winner = raw.loc[raw["native_winner"]]

    assert int(winner["feature_position"].iloc[0]) == 4
    assert int(winner["feature_id"].iloc[0]) == 0
    validate_cardinality_tree_raw(raw)

    nonminimum = raw.copy()
    nonminimum["native_winner"] = False
    nonminimum.loc[nonminimum["feature_position"] == 0, "native_winner"] = True
    nonminimum["native_selected_position"] = 0
    nonminimum["native_selected_feature"] = int(
        nonminimum.loc[nonminimum["feature_position"] == 0, "feature_id"].iloc[0]
    )
    with pytest.raises(ValueError, match="minimum candidate p-value"):
        validate_cardinality_tree_raw(nonminimum)


def test_tree_summaries_report_no_split_effects_as_missing() -> None:
    replicate, trend, method = summarize_cardinality_tree(_synthetic_tree_raw())
    conditional = replicate[replicate["method"].isin(["citrees", "partykit"])]

    assert conditional["adjusted_no_split"].astype(bool).all()
    assert conditional["adjusted_directional_effect"].isna().all()
    adjusted = trend[trend["estimand"] == "adjusted_winner_given_split"]
    assert adjusted["n_defined_effects"].eq(0).all()
    assert adjusted["randomization_p_value"].isna().all()
    forced = trend[trend["estimand"] == "forced_winner"]
    assert forced["n_defined_effects"].eq(1).all()
    conditional_methods = method[method["method"].isin(["citrees", "partykit"])]
    assert conditional_methods["adjusted_split_rate"].eq(0.0).all()


def test_average_ranks_and_kendall_tau_b_preserve_ties_and_direction() -> None:
    np.testing.assert_array_equal(
        average_ranks([0.0, 0.0, 2.0, 2.0, 5.0]),
        [1.5, 1.5, 3.5, 3.5, 5.0],
    )
    cardinality_ranks = np.arange(5, dtype=np.float64)
    assert kendall_tau_b(cardinality_ranks, np.arange(1.0, 6.0)) == pytest.approx(1.0)
    assert kendall_tau_b(cardinality_ranks, np.arange(5.0, 0.0, -1.0)) == pytest.approx(-1.0)
    assert kendall_tau_b(
        cardinality_ranks,
        np.array([3.0, 2.0, 5.0, 1.0, 4.0]),
    ) == pytest.approx(0.0)
    assert np.isnan(kendall_tau_b(cardinality_ranks, np.ones(5)))


def test_forest_raw_retains_partykit_missing_values_and_structural_zeros() -> None:
    _, features = _cardinality_matrix(
        np.random.default_rng(7),
        CARDINALITY_N_SAMPLES,
    )
    values = _position_aligned_values(features, [np.nan, -1.0, 0.0, 2.0, np.nan])
    raw = build_cardinality_forest_raw(
        features,
        values,
        task="classification",
        method="partykit_cforest",
        replicate=0,
        data_seed=1,
        model_seed=2,
    )

    validate_cardinality_forest_raw(raw)
    ordered = raw.sort_values("feature_id")
    np.testing.assert_allclose(
        ordered["raw_importance"].to_numpy(dtype=np.float64),
        [np.nan, -1.0, 0.0, 2.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        ordered["trend_importance"].to_numpy(dtype=np.float64),
        [0.0, -1.0, 0.0, 2.0, 0.0],
    )
    assert ordered["structural_zero_imputed"].sum() == 2


def test_forest_trends_use_exact_vectorized_tau_and_serializable_ranks() -> None:
    replicate, method = summarize_cardinality_forest_trends(_synthetic_forest_raw())

    observed = method.set_index(["task", "method"])["mean_kendall_tau_b"]
    assert observed[("classification", "citrees_cif")] == pytest.approx(1.0)
    assert observed[("classification", "partykit_cforest")] == pytest.approx(0.0)
    assert observed[("classification", "sklearn_rf")] == pytest.approx(-1.0)
    assert method["n_defined_trends"].eq(2).all()
    assert method["per_replicate_label_permutations"].eq(120).all()
    assert method["joint_randomization_resamples"].eq(CARDINALITY_TREND_RESAMPLES).all()
    assert method["randomization_scheme"].eq(CARDINALITY_TREND_RANDOMIZATION).all()
    assert method["randomization_p_value_rule"].eq(CARDINALITY_TREND_P_VALUE_RULE).all()
    positive_p_value = method.set_index(["task", "method"]).loc[
        ("classification", "citrees_cif"),
        "randomization_p_value",
    ]
    assert positive_p_value * CARDINALITY_HOLM_FAMILY_SIZE < CARDINALITY_ALPHA
    assert all(
        f"average_rank_feature_{feature_id}" in replicate
        for feature_id in range(len(CARDINALITY_SUPPORTS))
    )
    assert not any(
        isinstance(value, (tuple, list, np.ndarray))
        for value in replicate.to_numpy(dtype=object).ravel()
    )


def test_fixed_holm_family_contains_all_tree_and_forest_hypotheses() -> None:
    _, tree_trend, _ = summarize_cardinality_tree(_synthetic_tree_raw())
    _, forest_trend = summarize_cardinality_forest_trends(_synthetic_forest_raw())

    adjusted_tree, adjusted_forest, family = apply_cardinality_holm(
        tree_trend,
        forest_trend,
    )

    assert len(adjusted_tree) == 8
    assert len(adjusted_forest) == 6
    assert len(family) == CARDINALITY_HOLM_FAMILY_SIZE == 14
    assert not family["hypothesis_id"].duplicated().any()
    undefined = family[~family["test_defined"]]
    assert undefined["holm_input_p_value"].eq(1.0).all()
    assert undefined["undefined_test_treatment"].eq("set_to_one").all()
    assert adjusted_tree["holm_family_size"].eq(14).all()
    assert adjusted_forest["holm_family_size"].eq(14).all()
    assert family["minimum_attainable_raw_p_value"].eq(1 / (CARDINALITY_TREND_RESAMPLES + 1)).all()
    assert family["minimum_attainable_holm_p_value"].lt(CARDINALITY_ALPHA).all()


def test_run_calibration_and_writer_emit_every_validated_table(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tree_raw = _synthetic_tree_raw(base_seed=7)
    forest_raw = (
        _synthetic_forest_raw(base_seed=7)
        .loc[lambda frame: frame["replicate"].eq(0)]
        .reset_index(drop=True)
    )
    monkeypatch.setattr(
        calibration,
        "run_cardinality_trees",
        lambda _profile, *, base_seed: tree_raw.copy(),
    )
    monkeypatch.setattr(
        calibration,
        "run_cardinality_forests",
        lambda _profile, *, base_seed: forest_raw.copy(),
    )

    results = run_calibration("smoke", base_seed=7)
    validate_calibration_results(results, profile="smoke", base_seed=7)

    assert set(results) == set(CALIBRATION_RESULT_SCHEMAS)
    assert all(
        tuple(results[name].columns) == schema
        for name, schema in CALIBRATION_RESULT_SCHEMAS.items()
    )
    assert len(results["cardinality_tree_raw"]) == 30
    assert len(results["cardinality_forest_raw"]) == 30
    assert len(results["cardinality_holm_family"]) == 14

    write_results(
        results,
        tmp_path,
        profile="smoke",
        base_seed=7,
        elapsed_seconds=1.25,
    )
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="ascii"))
    assert receipt["analysis"] == "calibration"
    assert receipt["schema_version"] == 4
    assert receipt["profile"] == "smoke"
    assert receipt["base_seed"] == 7
    assert isinstance(receipt["git_dirty"], bool)
    assert receipt["cardinality_design"]["supports"] == list(CARDINALITY_SUPPORTS)
    assert receipt["cardinality_design"]["maximum_support_feature_type"] == (
        CARDINALITY_MAX_SUPPORT_FEATURE_TYPE
    )
    assert receipt["cardinality_design"]["trend_randomization_resamples"] == (
        CARDINALITY_TREND_RESAMPLES
    )
    assert receipt["cardinality_design"]["minimum_attainable_holm_p_value"] < CARDINALITY_ALPHA
    assert set(receipt["tables"]) == set(CALIBRATION_RESULT_SCHEMAS)
    assert "paper/jss/replication/calibration.py" in receipt["source_sha256"]
    assert "paper/benchmark/pipeline/r_methods.py" in receipt["source_sha256"]
    assert "citrees/_forest.py" in receipt["source_sha256"]
    assert "citrees/_permutation.py" in receipt["source_sha256"]
    assert "citrees/_selector.py" in receipt["source_sha256"]
    assert "citrees/_tree.py" in receipt["source_sha256"]
    assert "uv.lock" in receipt["source_sha256"]
    for table_name in CALIBRATION_RESULT_SCHEMAS:
        assert (tmp_path / f"{table_name}.parquet").exists()
    for artifact, metadata in receipt["artifacts"].items():
        artifact_path = tmp_path / artifact
        assert metadata["bytes"] == artifact_path.stat().st_size
        assert metadata["sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
