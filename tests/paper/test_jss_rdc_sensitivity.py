"""Tests for the JSS RDC projection-count sensitivity study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper.jss.replication import rdc_sensitivity as sensitivity

pytestmark = pytest.mark.paper


def _tiny_dataset(name: str = "tiny") -> sensitivity.StudyDataset:
    X = np.arange(120, dtype=np.float64).reshape(20, 6)
    y = np.repeat(np.arange(2, dtype=np.int64), 10)
    return sensitivity.StudyDataset(
        name=name,
        sha256="a" * 64,
        source="test",
        X=X,
        y=y,
    )


def _configure_tiny_study(
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str = "tiny",
) -> sensitivity.StudyDataset:
    dataset = _tiny_dataset(name)
    monkeypatch.setattr(sensitivity, "_warm_rdc", lambda settings: None)
    monkeypatch.setattr(sensitivity, "_load_study_dataset", lambda dataset_name: dataset)
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
                "n_features_selected": selected,
                "downstream_model": model,
                "accuracy": 0.5,
                "f1": 0.5,
                "f1_macro": 0.5,
                "balanced_accuracy": 0.5,
                "roc_auc": 0.5,
                "auc": 0.5,
            }
            for selected in k_values
            for model in ("lr", "svm", "knn")
        ]

    monkeypatch.setattr(sensitivity, "evaluate_fold", fake_evaluate_fold)
    return dataset


def _tiny_settings(name: str = "tiny") -> sensitivity.RdcSensitivitySettings:
    return sensitivity.RdcSensitivitySettings(
        datasets=(name,),
        seeds=(0,),
        folds=5,
        projection_counts=(5, 10, 20, 40),
    )


def _tiny_results(
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str = "tiny",
) -> tuple[sensitivity.RdcSensitivitySettings, dict[str, pd.DataFrame]]:
    _configure_tiny_study(monkeypatch, name=name)
    settings = _tiny_settings(name)
    rankings, metrics = sensitivity.run_sensitivity(settings)
    return settings, sensitivity.build_results(rankings, metrics, settings)


def test_profiles_preserve_the_full_design_and_bound_smaller_runs() -> None:
    assert sensitivity._settings("smoke") == sensitivity.RdcSensitivitySettings(
        datasets=("wine",),
        seeds=(0,),
        folds=5,
        projection_counts=(5, 10, 20, 40),
    )
    assert sensitivity._settings("quick") == sensitivity.RdcSensitivitySettings(
        datasets=("wine", "glass", "heart-statlog", "parkinsons"),
        seeds=(0,),
        folds=5,
        projection_counts=(5, 10, 20, 40),
    )
    assert sensitivity._settings("full") == sensitivity.RdcSensitivitySettings(
        datasets=("wine", "glass", "heart-statlog", "parkinsons"),
        seeds=(0, 1, 2, 3, 4),
        folds=5,
        projection_counts=(5, 10, 20, 40),
    )
    with pytest.raises(ValueError, match="unknown RDC sensitivity profile"):
        sensitivity._settings("unknown")  # type: ignore[arg-type]


def test_study_uses_only_four_small_pinned_datasets() -> None:
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
        assert dataset.y.shape == (dataset.n_samples,)
        assert dataset.n_samples <= 270
        assert dataset.n_features <= 22
        assert dataset.sha256 == expected_sha256
        assert np.isfinite(dataset.X).all()
        assert len(np.unique(dataset.y)) == expected_classes
    assert tuple(np.bincount(sensitivity._load_study_dataset("parkinsons").y)) == (48, 147)


def test_tiny_run_has_exact_cells_schemas_and_one_feature_ranking_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, results = _tiny_results(monkeypatch)

    sensitivity.validate_results(results, settings)
    assert len(results["rankings"]) == 20
    assert len(results["metrics"]) == 120
    assert len(results["stability"]) == 60
    assert len(results["ranking_stability"]) == 30
    assert sensitivity.RANKING_COLUMNS.count("feature_ranking") == 1
    assert tuple(results["rankings"].columns) == sensitivity.RANKING_COLUMNS
    assert tuple(results["metrics"].columns) == sensitivity.METRIC_COLUMNS
    assert set(results) == set(sensitivity.RESULT_SCHEMAS)
    assert not results["rankings"].duplicated(["dataset", "seed", "fold", "projection_count"]).any()


@pytest.mark.parametrize(
    ("table", "column", "value", "message"),
    [
        ("rankings", "feature_ranking", "[0, 0, 1, 2, 3, 4]", "complete permutation"),
        ("metrics", "balanced_accuracy", 2.0, "finite probabilities"),
        ("comparison_summary", "runtime_ratio_median", 999.0, "raw-cell aggregates"),
    ],
)
def test_semantic_validation_rejects_corruption(
    monkeypatch: pytest.MonkeyPatch,
    table: str,
    column: str,
    value: object,
    message: str,
) -> None:
    settings, results = _tiny_results(monkeypatch)
    corrupted = {name: frame.copy(deep=True) for name, frame in results.items()}
    corrupted[table].loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        sensitivity.validate_results(corrupted, settings)


def test_semantic_validation_rejects_missing_and_duplicate_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, results = _tiny_results(monkeypatch)
    missing = {name: frame.copy(deep=True) for name, frame in results.items()}
    missing["rankings"] = missing["rankings"].iloc[1:].reset_index(drop=True)
    with pytest.raises(ValueError, match="ranking inventory differs"):
        sensitivity.validate_results(missing, settings)

    duplicated = {name: frame.copy(deep=True) for name, frame in results.items()}
    duplicated["metrics"] = pd.concat(
        [duplicated["metrics"], duplicated["metrics"].iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="metric inventory differs"):
        sensitivity.validate_results(duplicated, settings)


def test_stability_uses_ranking_positions_and_stage2_cutoffs() -> None:
    rankings = {
        5: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        10: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        20: [1, 0, 2, 3, 4, 5, 6, 7, 8, 9],
        40: [1, 0, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    frame = pd.DataFrame(
        [
            {
                "dataset": "glass",
                "seed": 0,
                "fold": 0,
                "projection_count": projection_count,
                "feature_ranking": json.dumps(ranking),
            }
            for projection_count, ranking in rankings.items()
        ]
    )

    selected_sets = sensitivity.build_stability(frame)
    complete_rankings = sensitivity.build_ranking_stability(frame)
    selected = selected_sets[
        (selected_sets["projection_count_left"] == 10)
        & (selected_sets["projection_count_right"] == 20)
    ]
    complete = complete_rankings[
        (complete_rankings["projection_count_left"] == 10)
        & (complete_rankings["projection_count_right"] == 20)
    ]

    assert set(selected["selected_features"]) == {5, 10}
    assert np.allclose(selected["overlap_fraction"], 1.0)
    assert len(complete) == 1
    assert complete["spearman_complete_ranking"].iloc[0] < 1.0


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


def test_writer_atomically_records_sources_tables_and_artifact_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_tiny_study(monkeypatch, name="wine")
    settings = sensitivity._settings("smoke")
    rankings, metrics = sensitivity.run_sensitivity(settings)
    results = sensitivity.build_results(rankings, metrics, settings)
    output_dir = tmp_path / "rdc-output"
    source_hashes = sensitivity._source_hashes()

    observed = sensitivity.write_results(
        results,
        output_dir,
        profile="smoke",
        base_seed=1718,
        elapsed_seconds=1.25,
        git_sha="a" * 40,
        git_dirty=True,
        source_sha256=source_hashes,
    )

    assert observed == output_dir.resolve()
    assert output_dir.is_dir()
    assert not list(tmp_path.glob(".rdc-output-*"))
    receipt = json.loads((output_dir / "receipt.json").read_text(encoding="ascii"))
    assert receipt["analysis"] == "rdc_sensitivity"
    assert receipt["schema_version"] == 1
    assert receipt["semantic_validation"] == "citrees-jss-rdc-sensitivity-v1"
    assert receipt["profile"] == "smoke"
    assert receipt["base_seed"] == 1718
    assert receipt["settings"]["datasets"] == ["wine"]
    assert receipt["settings"]["seeds"] == [0]
    assert "paper/jss/replication/rdc_sensitivity.py" in receipt["source_sha256"]
    assert "pyproject.toml" in receipt["source_sha256"]
    assert "uv.lock" in receipt["source_sha256"]
    assert set(receipt["tables"]) == set(sensitivity.RESULT_FILENAMES)
    assert set(receipt["artifacts"]) == set(sensitivity.RESULT_FILENAMES.values())
    for artifact, metadata in receipt["artifacts"].items():
        path = output_dir / artifact
        assert metadata["bytes"] == path.stat().st_size
        assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_existing_output_is_rejected_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    called = False

    def fail_if_called(
        settings: sensitivity.RdcSensitivitySettings,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        nonlocal called
        called = True
        raise AssertionError(settings)

    monkeypatch.setattr(sensitivity, "run_sensitivity", fail_if_called)
    with pytest.raises(FileExistsError, match="already exists"):
        sensitivity.execute("smoke", output_dir, base_seed=1718)
    assert not called


def test_failed_run_is_not_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "failed"
    monkeypatch.setattr(
        sensitivity,
        "run_sensitivity",
        lambda settings: (_ for _ in ()).throw(RuntimeError("intentional failure")),
    )

    with pytest.raises(RuntimeError, match="intentional failure"):
        sensitivity.execute("smoke", output_dir, base_seed=1718)
    assert not output_dir.exists()


def test_full_profile_requires_clean_source_before_computation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fail_if_called(
        settings: sensitivity.RdcSensitivitySettings,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        nonlocal called
        called = True
        raise AssertionError(settings)

    monkeypatch.setattr(sensitivity, "_git_dirty", lambda: True)
    monkeypatch.setattr(sensitivity, "run_sensitivity", fail_if_called)
    with pytest.raises(RuntimeError, match="requires a clean source tree"):
        sensitivity.execute("full", tmp_path / "full", base_seed=1718)
    assert not called


def test_writer_rejects_dirty_full_receipt_before_publication(tmp_path: Path) -> None:
    output_dir = tmp_path / "full"
    with pytest.raises(RuntimeError, match="requires a clean source tree"):
        sensitivity.write_results(
            {},
            output_dir,
            profile="full",
            base_seed=1718,
            elapsed_seconds=1.0,
            git_sha="a" * 40,
            git_dirty=True,
            source_sha256={},
        )
    assert not output_dir.exists()
