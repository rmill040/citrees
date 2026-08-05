"""Tests for the small local RDC projection-count sensitivity study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper.benchmark.experiments import rdc_projection_sensitivity as sensitivity

pytestmark = pytest.mark.paper


def _configure_tiny_study(monkeypatch: pytest.MonkeyPatch) -> sensitivity.StudyDataset:
    X = np.arange(120, dtype=np.float64).reshape(20, 6)
    y = np.repeat(np.arange(2, dtype=np.int64), 10)
    dataset = sensitivity.StudyDataset(
        name="tiny",
        sha256="a" * 64,
        source="test",
        X=X,
        y=y,
    )
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
    return dataset


def test_study_uses_only_four_small_real_datasets() -> None:
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
        "parkinsons": (
            (195, 22),
            2,
            "f7d1985c10604b4f65c5a0444767a4b1433395016e155009892abe1cf854f91f",
        ),
    }
    assert tuple(expected) == sensitivity.DATASETS
    for dataset_name, (expected_shape, expected_classes, expected_sha256) in expected.items():
        dataset = sensitivity._load_study_dataset(dataset_name)
        assert dataset.X.shape == expected_shape
        assert dataset.X.shape == (dataset.n_samples, dataset.n_features)
        assert dataset.y.shape == (dataset.n_samples,)
        assert dataset.n_samples <= 270
        assert dataset.n_features <= 22
        assert dataset.sha256 == expected_sha256
        assert np.isfinite(dataset.X).all()
        assert len(np.unique(dataset.y)) == expected_classes
    assert tuple(np.bincount(sensitivity._load_study_dataset("parkinsons").y)) == (48, 147)


def test_projection_counts_and_projected_columns_are_explicit() -> None:
    assert sensitivity.PROJECTION_COUNTS == (5, 10, 20, 40)
    assert [2 * count for count in sensitivity.PROJECTION_COUNTS] == [10, 20, 40, 80]


def test_seed_parser_rejects_empty_negative_and_duplicate_values() -> None:
    assert sensitivity._parse_int_csv("0,2,4") == (0, 2, 4)
    for value in ("", "-1", "1,1"):
        with pytest.raises(ValueError):
            sensitivity._parse_int_csv(value)


def test_dataset_parser_accepts_small_defaults_and_rejects_invalid_values() -> None:
    assert sensitivity._parse_dataset_csv("wine,glass,heart-statlog,parkinsons") == (
        "wine",
        "glass",
        "heart-statlog",
        "parkinsons",
    )
    for value in ("", "wine,wine", "wine,unknown"):
        with pytest.raises(ValueError):
            sensitivity._parse_dataset_csv(value)


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
    assert sensitivity._completed_keys(pd.concat([rankings, rankings]), metrics) == set()
    assert sensitivity._completed_keys(rankings, pd.concat([metrics, metrics.iloc[:1]])) == set()


def test_existing_outputs_cannot_cross_requested_scope() -> None:
    rankings = pd.DataFrame(
        [{"dataset": "glass", "seed": 0, "fold": 0, "projection_count": 5}],
    )
    metrics = pd.DataFrame(
        [{"dataset": "wine", "seed": 1, "fold": 0, "projection_count": 5}],
    )

    with pytest.raises(RuntimeError, match="datasets outside"):
        sensitivity._validate_existing_scope(
            rankings,
            metrics,
            datasets=("wine",),
            seeds=(0, 1),
        )
    with pytest.raises(RuntimeError, match="seeds outside"):
        sensitivity._validate_existing_scope(
            rankings,
            metrics,
            datasets=("wine", "glass"),
            seeds=(0,),
        )


def test_run_writes_dataset_names_to_rankings_and_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_tiny_study(monkeypatch)

    rankings, metrics = sensitivity.execute(
        datasets=("tiny",),
        seeds=(0,),
        output_dir=tmp_path,
    )
    result_bytes = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    resumed_rankings, resumed_metrics = sensitivity.execute(
        datasets=("tiny",),
        seeds=(0,),
        output_dir=tmp_path,
    )
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="ascii"))
    selected_stability = pd.read_csv(tmp_path / "stability.csv")
    ranking_stability = pd.read_csv(tmp_path / "ranking-stability.csv")

    assert set(rankings["dataset"]) == {"tiny"}
    assert set(metrics["dataset"]) == {"tiny"}
    assert all(isinstance(value, str) for value in metrics["dataset"])
    assert len(resumed_rankings) == len(rankings)
    assert len(resumed_metrics) == len(metrics)
    assert "spearman_complete_ranking" not in selected_stability
    assert len(ranking_stability) == 5 * 6
    assert receipt["selector"] == {
        "alpha": 0.05,
        "early_stopping": "adaptive",
        "method": "ptest_rdc",
        "n_resamples": "auto",
        "projection_counts": [5, 10, 20, 40],
    }
    assert receipt["timing"]["scope"] == "permutation_selector call"
    assert {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    } == result_bytes


def test_torn_checkpoint_replaces_incomplete_ranking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _configure_tiny_study(monkeypatch)
    partial_ranking = pd.DataFrame(
        [
            {
                "task": sensitivity.TASK,
                "dataset": "tiny",
                "dataset_sha256": dataset.sha256,
                "n_samples": dataset.n_samples,
                "n_features": dataset.n_features,
                "seed": 0,
                "fold": 0,
                "projection_count": 5,
                "projected_columns": 10,
                "selection_seconds": 0.0,
                "feature_ranking": json.dumps(list(range(dataset.n_features))),
            }
        ]
    )
    partial_ranking.to_csv(tmp_path / "rankings.csv", index=False)

    rankings, metrics = sensitivity.execute(
        datasets=("tiny",),
        seeds=(0,),
        output_dir=tmp_path,
    )

    assert len(rankings) == 20
    assert len(metrics) == 120
    assert not rankings.duplicated(["dataset", "seed", "fold", "projection_count"]).any()
    assert len(sensitivity._completed_keys(rankings, metrics)) == 20


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

    ranking_frame = pd.DataFrame(rows)
    selected_sets = sensitivity.build_stability(ranking_frame)
    complete_rankings = sensitivity.build_ranking_stability(ranking_frame)
    selected_comparison = selected_sets[
        (selected_sets["projection_count_left"] == 10)
        & (selected_sets["projection_count_right"] == 20)
    ]
    ranking_comparison = complete_rankings[
        (complete_rankings["projection_count_left"] == 10)
        & (complete_rankings["projection_count_right"] == 20)
    ]

    assert set(selected_comparison["selected_features"]) == {5, 10}
    assert np.allclose(selected_comparison["overlap_fraction"], 1.0)
    assert "spearman_complete_ranking" not in selected_sets
    assert len(ranking_comparison) == 1
    assert ranking_comparison["spearman_complete_ranking"].iloc[0] < 1.0


def test_comparison_summary_excludes_all_feature_endpoint() -> None:
    rankings = pd.DataFrame(
        [
            {
                "dataset": "tiny",
                "seed": 0,
                "fold": 0,
                "projection_count": projection_count,
                "n_features": 10,
                "selection_seconds": seconds,
            }
            for projection_count, seconds in ((10, 1.0), (20, 3.0))
        ]
    )
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
    stability = pd.DataFrame(
        [
            {
                "dataset": "tiny",
                "seed": 0,
                "fold": 0,
                "projection_count_left": 10,
                "projection_count_right": 20,
                "selected_features": selected,
                "overlap_fraction": overlap,
            }
            for selected, overlap in ((5, 0.8), (10, 1.0))
        ]
    )
    ranking_stability = pd.DataFrame(
        [
            {
                "dataset": "tiny",
                "seed": 0,
                "fold": 0,
                "projection_count_left": 10,
                "projection_count_right": 20,
                "spearman_complete_ranking": 0.9,
            }
        ]
    )

    result = sensitivity.build_comparison_summary(
        rankings,
        metrics,
        stability,
        ranking_stability,
        datasets=("tiny",),
    ).iloc[0]

    assert result["runtime_ratio_median"] == pytest.approx(3.0)
    assert result["reduced_feature_counts"] == "5"
    assert result["balanced_accuracy_mean_difference_reduced_k"] == pytest.approx(0.2)
    assert result["selected_set_overlap_mean_reduced_k"] == pytest.approx(0.8)
