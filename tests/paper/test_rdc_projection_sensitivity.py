"""Tests for the small local RDC projection-count sensitivity study."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from paper.benchmark.adapters.data import get_dataset_identity
from paper.benchmark.experiments import rdc_projection_sensitivity as sensitivity

pytestmark = pytest.mark.paper


def test_study_uses_only_two_small_real_datasets() -> None:
    assert sensitivity.DATASETS == ("wine", "glass")
    for dataset in sensitivity.DATASETS:
        identity = get_dataset_identity(dataset, sensitivity.TASK, source="real")
        assert identity.n_samples <= 250
        assert identity.n_features <= 15


def test_projection_counts_and_projected_columns_are_explicit() -> None:
    assert sensitivity.PROJECTION_COUNTS == (5, 10, 20, 40)
    assert [2 * count for count in sensitivity.PROJECTION_COUNTS] == [10, 20, 40, 80]


def test_seed_parser_rejects_empty_negative_and_duplicate_values() -> None:
    assert sensitivity._parse_int_csv("0,2,4") == (0, 2, 4)
    for value in ("", "-1", "1,1"):
        with pytest.raises(ValueError):
            sensitivity._parse_int_csv(value)


def test_completed_keys_require_every_downstream_model_and_cutoff() -> None:
    rankings = pd.DataFrame(
        [
            {
                "dataset": "glass",
                "seed": 0,
                "fold": 0,
                "projection_count": 10,
                "n_features": 10,
            }
        ]
    )
    metrics = pd.DataFrame(
        [
            {
                "dataset": "glass",
                "seed": 0,
                "fold": 0,
                "projection_count": 10,
                "k": selected,
                "downstream_model": model,
            }
            for selected in (5, 10)
            for model in ("lr", "svm", "knn")
        ]
    )

    assert sensitivity._completed_keys(rankings, metrics) == {("glass", 0, 0, 10)}
    assert sensitivity._completed_keys(rankings, metrics.iloc[:-1]) == set()


def test_stability_uses_ranking_positions_and_stage2_cutoffs() -> None:
    rows = []
    rankings = {
        5: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        10: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        20: [1, 0, 2, 3, 4, 5, 6, 7, 8, 9],
        40: [1, 0, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    for projection_count, ranking in rankings.items():
        rows.append(
            {
                "dataset": "glass",
                "seed": 0,
                "fold": 0,
                "projection_count": projection_count,
                "feature_ranking": json.dumps(ranking),
            }
        )

    result = sensitivity.build_stability(pd.DataFrame(rows))
    comparison = result[
        (result["projection_count_left"] == 10)
        & (result["projection_count_right"] == 20)
    ]

    assert set(comparison["selected_features"]) == {5, 10}
    assert np.allclose(comparison["overlap_fraction"], 1.0)
    assert comparison["spearman_complete_ranking"].iloc[0] < 1.0
