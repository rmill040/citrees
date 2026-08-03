"""Focused tests for the JSS cardinality experiment data and statistics layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paper.jss.replication.calibration import (
    CARDINALITY_B,
    CARDINALITY_N_SAMPLES,
    CARDINALITY_SUPPORTS,
    _attainable_alpha,
    _attainable_cutoff,
    _cardinality_matrix,
    _permutation_p_value,
    _permutation_support_index,
    _settings,
    average_ranks,
    build_cardinality_tree_raw,
    controlled_split_weights,
    directional_effect,
    fractional_minimum_p_weights,
    holm_adjust,
    kendall_tau_b,
    one_sided_label_randomization,
    summarize_cardinality_forest_trends,
    validate_cardinality_tree_raw,
)

pytestmark = pytest.mark.paper


def test_cardinality_budget_is_fixed_across_profiles() -> None:
    settings = {profile: _settings(profile) for profile in ("smoke", "quick", "full")}

    assert CARDINALITY_B == 999
    assert {
        profile: profile_settings.cardinality_resamples
        for profile, profile_settings in settings.items()
    } == {"smoke": 999, "quick": 999, "full": 999}
    assert {
        profile: profile_settings.cardinality_replicates
        for profile, profile_settings in settings.items()
    } == {"smoke": 8, "quick": 500, "full": 10_000}


def test_cardinality_matrix_has_exact_balanced_supports_and_metadata() -> None:
    first_matrix, first_features = _cardinality_matrix(
        np.random.default_rng(42),
        CARDINALITY_N_SAMPLES,
    )
    second_matrix, second_features = _cardinality_matrix(
        np.random.default_rng(42),
        CARDINALITY_N_SAMPLES,
    )

    np.testing.assert_array_equal(first_matrix, second_matrix)
    assert first_features == second_features
    assert first_matrix.shape == (200, 5)
    assert {feature.position for feature in first_features} == set(range(5))

    total_pairs = CARDINALITY_N_SAMPLES * (CARDINALITY_N_SAMPLES - 1) // 2
    for feature, expected_support in zip(
        first_features,
        CARDINALITY_SUPPORTS,
        strict=True,
    ):
        _, counts = np.unique(
            first_matrix[:, feature.position],
            return_counts=True,
        )
        expected_multiplicity = CARDINALITY_N_SAMPLES // expected_support
        expected_tied_pairs = expected_support * (
            expected_multiplicity * (expected_multiplicity - 1) // 2
        )
        assert feature.nominal_support == expected_support
        assert feature.realized_support == expected_support
        assert feature.maximum_multiplicity == expected_multiplicity
        assert counts.tolist() == [expected_multiplicity] * expected_support
        assert feature.tied_pair_fraction == pytest.approx(expected_tied_pairs / total_pairs)

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


def test_permutation_support_round_trips_and_cutoffs_are_explicit() -> None:
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


def test_directional_effect_and_randomization_cover_known_trends() -> None:
    ranks = np.arange(5, dtype=np.float64)
    positive_weights = np.tile([0.0, 0.0, 0.0, 0.0, 1.0], (6, 1))
    zero_weights = np.zeros((6, 5), dtype=np.float64)
    negative_weights = np.tile([1.0, 0.0, 0.0, 0.0, 0.0], (6, 1))
    rank_matrix = np.tile(ranks, (6, 1))

    assert directional_effect(ranks, positive_weights[0]) == pytest.approx(2.0)
    assert directional_effect(ranks, zero_weights[0]) == pytest.approx(0.0)
    assert directional_effect(ranks, negative_weights[0]) == pytest.approx(-2.0)

    positive = one_sided_label_randomization(
        rank_matrix,
        positive_weights,
        statistic=directional_effect,
        n_resamples=199,
        random_state=42,
    )
    repeated = one_sided_label_randomization(
        rank_matrix,
        positive_weights,
        statistic=directional_effect,
        n_resamples=199,
        random_state=42,
    )
    zero = one_sided_label_randomization(
        rank_matrix,
        zero_weights,
        statistic=directional_effect,
        n_resamples=39,
        random_state=42,
    )
    negative = one_sided_label_randomization(
        rank_matrix,
        negative_weights,
        statistic=directional_effect,
        n_resamples=39,
        random_state=42,
    )

    assert positive == repeated
    assert positive.statistic == pytest.approx(2.0)
    assert positive.p_value < 0.05
    assert zero.statistic == pytest.approx(0.0)
    assert zero.p_value == pytest.approx(1.0)
    assert negative.statistic == pytest.approx(-2.0)
    assert negative.p_value == pytest.approx(1.0)


def test_holm_adjustment_matches_reference_values() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.002])

    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.008])


def test_tree_raw_schema_is_unique_paired_and_retains_no_split() -> None:
    _, features = _cardinality_matrix(
        np.random.default_rng(7),
        CARDINALITY_N_SAMPLES,
    )
    support_indices = [100, 200, 300, 400, 500]
    first = build_cardinality_tree_raw(
        features,
        support_indices,
        task="classification",
        method="citrees",
        replicate=3,
        data_seed=101,
        model_seed=201,
        native_selected_position=None,
    )
    second = build_cardinality_tree_raw(
        features,
        support_indices,
        task="classification",
        method="partykit",
        replicate=3,
        data_seed=101,
        model_seed=202,
        native_selected_position=None,
    )
    combined = pd.concat([first, second], ignore_index=True)

    validate_cardinality_tree_raw(combined)
    assert combined.columns.is_unique
    assert not combined.duplicated(["task", "replicate", "method", "feature_id"]).any()
    assert combined.groupby(["task", "replicate"])["data_seed"].nunique().eq(1).all()
    assert combined.groupby(["task", "replicate"])["model_seed"].nunique().eq(2).all()
    assert combined.groupby(["task", "replicate", "method"]).size().eq(5).all()
    assert combined["B"].eq(CARDINALITY_B).all()
    assert combined["adjusted_no_split"].all()
    assert combined["native_no_split"].all()
    assert combined["adjusted_winner_weight"].eq(0.0).all()
    assert combined.groupby("method")["forced_winner_weight"].sum().eq(1.0).all()
    assert set(combined["nominal_support"]) == set(CARDINALITY_SUPPORTS)
    assert combined["nominal_support"].equals(combined["realized_support"])

    duplicated = pd.concat([combined, combined.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="candidate keys"):
        validate_cardinality_tree_raw(duplicated)


def test_average_ranks_and_kendall_tau_b_preserve_ties_and_direction() -> None:
    np.testing.assert_array_equal(
        average_ranks([0.0, 0.0, 2.0, 2.0, 5.0]),
        [1.5, 1.5, 3.5, 3.5, 5.0],
    )

    cardinality_ranks = np.arange(5, dtype=np.float64)
    assert kendall_tau_b(
        cardinality_ranks,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    ) == pytest.approx(1.0)
    assert kendall_tau_b(
        cardinality_ranks,
        np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
    ) == pytest.approx(-1.0)
    assert kendall_tau_b(
        cardinality_ranks,
        np.array([3.0, 2.0, 5.0, 1.0, 4.0]),
    ) == pytest.approx(0.0)


def test_forest_trend_summaries_recover_positive_zero_and_negative_effects() -> None:
    patterns = {
        "positive": [1.0, 2.0, 3.0, 4.0, 5.0],
        "zero": [3.0, 2.0, 5.0, 1.0, 4.0],
        "negative": [5.0, 4.0, 3.0, 2.0, 1.0],
    }
    rows = [
        {
            "task": "classification",
            "method": method,
            "replicate": replicate,
            "feature_id": feature_id,
            "cardinality_rank": feature_id,
            "raw_importance": importance,
        }
        for method, importances in patterns.items()
        for replicate in range(2)
        for feature_id, importance in enumerate(importances)
    ]
    raw = pd.DataFrame(rows)

    replicate_summary, method_summary = summarize_cardinality_forest_trends(
        raw,
        n_resamples=39,
        random_state=42,
    )
    repeated_replicates, repeated_methods = summarize_cardinality_forest_trends(
        raw,
        n_resamples=39,
        random_state=42,
    )

    pd.testing.assert_frame_equal(replicate_summary, repeated_replicates)
    pd.testing.assert_frame_equal(method_summary, repeated_methods)
    observed = method_summary.set_index("method")["mean_kendall_tau_b"]
    assert observed["positive"] == pytest.approx(1.0)
    assert observed["zero"] == pytest.approx(0.0)
    assert observed["negative"] == pytest.approx(-1.0)
    assert method_summary["n_defined_trends"].eq(2).all()
