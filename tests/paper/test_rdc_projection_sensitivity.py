"""Tests for the small local RDC projection-count sensitivity study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper.benchmark.experiments import rdc_projection_sensitivity as sensitivity

pytestmark = pytest.mark.paper


def test_study_uses_only_three_small_real_datasets() -> None:
    assert sensitivity.DATASETS == ("wine", "glass", "breast-cancer")
    for dataset_name in sensitivity.DATASETS:
        dataset = sensitivity._load_study_dataset(dataset_name)
        assert dataset.X.shape == (dataset.n_samples, dataset.n_features)
        assert dataset.y.shape == (dataset.n_samples,)
        assert dataset.n_samples <= 600
        assert dataset.n_features <= 30
        assert len(dataset.sha256) == 64


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


def test_run_writes_dataset_names_to_rankings_and_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = np.arange(120, dtype=np.float64).reshape(20, 6)
    y = np.repeat(np.arange(2, dtype=np.int64), 10)
    dataset = sensitivity.StudyDataset(
        name="tiny",
        sha256="a" * 64,
        source="test",
        X=X,
        y=y,
    )

    monkeypatch.setattr(sensitivity, "DATASETS", ("tiny",))
    monkeypatch.setattr(sensitivity, "_warm_rdc", lambda: None)
    monkeypatch.setattr(sensitivity, "_load_study_dataset", lambda name: dataset)
    monkeypatch.setattr(
        sensitivity,
        "permutation_selector",
        lambda X_train, *_args, **_kwargs: np.arange(X_train.shape[1]),
    )

    def fake_evaluate_fold(
        *_args: object,
        k_values: list[int],
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        return [
            {
                "k": selected,
                "downstream_model": model,
                "balanced_accuracy": 0.5,
            }
            for selected in k_values
            for model in ("lr", "svm", "knn")
        ]

    monkeypatch.setattr(sensitivity, "evaluate_fold", fake_evaluate_fold)

    rankings, metrics = sensitivity.execute(seeds=(0,), output_dir=tmp_path)
    result_bytes = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    resumed_rankings, resumed_metrics = sensitivity.execute(
        seeds=(0,),
        output_dir=tmp_path,
    )

    assert set(rankings["dataset"]) == {"tiny"}
    assert set(metrics["dataset"]) == {"tiny"}
    assert all(isinstance(value, str) for value in metrics["dataset"])
    assert len(resumed_rankings) == len(rankings)
    assert len(resumed_metrics) == len(metrics)
    assert {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    } == result_bytes


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
        (result["projection_count_left"] == 10) & (result["projection_count_right"] == 20)
    ]

    assert set(comparison["selected_features"]) == {5, 10}
    assert np.allclose(comparison["overlap_fraction"], 1.0)
    assert comparison["spearman_complete_ranking"].iloc[0] < 1.0
