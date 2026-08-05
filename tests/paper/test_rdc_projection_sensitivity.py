"""Tests for the small local RDC projection-count sensitivity study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper.benchmark.experiments import rdc_projection_sensitivity as sensitivity

pytestmark = pytest.mark.paper


def test_study_uses_only_five_small_real_datasets() -> None:
    expected = {
        "wine": (
            (178, 13),
            3,
            "f559edb359a04d51531da5e80a499c8d7fde49233ffb9be8949f00ee36336b9e",
        ),
        "glass": (
            (214, 10),
            6,
            "8f6eea990c4fdc7bcc55a2bd26743435287d88d3f49b60dd4a8dd68dc931d9de",
        ),
        "heart-statlog": (
            (270, 13),
            2,
            "8c624c057ab20b49e8ce2b4b222ae5655a7652e3991aa997d100570ef9c3f438",
        ),
        "ionosphere": (
            (351, 34),
            2,
            "6693050c2f2a6ece1e4abccd19658743bc17d2a132dcd981c4b6ca1c2d24bffd",
        ),
        "breast-cancer": (
            (569, 30),
            2,
            "1720d92c704161b13f5e67929cbb6912f2fb2348d0add12846f881f65aaf0604",
        ),
    }
    assert tuple(expected) == sensitivity.DATASETS
    for dataset_name, (expected_shape, expected_classes, expected_sha256) in expected.items():
        dataset = sensitivity._load_study_dataset(dataset_name)
        assert dataset.X.shape == expected_shape
        assert dataset.X.shape == (dataset.n_samples, dataset.n_features)
        assert dataset.y.shape == (dataset.n_samples,)
        assert dataset.n_samples <= 600
        assert dataset.n_features <= 34
        assert dataset.sha256 == expected_sha256
        assert np.isfinite(dataset.X).all()
        assert len(np.unique(dataset.y)) == expected_classes


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


def test_comparison_summary_excludes_all_feature_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sensitivity, "DATASETS", ("tiny",))
    rankings = pd.DataFrame([{"dataset": "tiny", "n_features": 10}])
    metrics = pd.DataFrame(
        [
            {
                "dataset": "tiny",
                "seed": 0,
                "fold": 0,
                "k": selected,
                "downstream_model": "lr",
                "projection_count": projection_count,
                "balanced_accuracy": accuracy,
            }
            for selected, projection_count, accuracy in (
                (5, 10, 0.5),
                (5, 20, 0.7),
                (10, 10, 0.0),
                (10, 20, 1.0),
            )
        ]
    )
    timing = pd.DataFrame(
        [
            {"dataset": "tiny", "projection_count": 10, "median_seconds": 1.0},
            {"dataset": "tiny", "projection_count": 20, "median_seconds": 3.0},
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "dataset": "tiny",
                "projection_count_left": 10,
                "projection_count_right": 20,
                "selected_features": selected,
                "overlap_fraction": overlap,
                "spearman_complete_ranking": 0.9,
            }
            for selected, overlap in ((5, 0.8), (10, 1.0))
        ]
    )

    result = sensitivity.build_comparison_summary(rankings, metrics, timing, stability).iloc[0]

    assert result["median_runtime_ratio"] == pytest.approx(3.0)
    assert result["balanced_accuracy_mean_difference_reduced_k"] == pytest.approx(0.2)
    assert result["selected_set_overlap_mean_reduced_k"] == pytest.approx(0.8)
