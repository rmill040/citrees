"""Tests for the executable JSS estimator tutorial."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from paper.jss.replication.tutorial import (
    TutorialRun,
    TutorialSettings,
    _settings,
    run_tutorial,
    write_results,
)

pytestmark = [
    pytest.mark.paper,
    pytest.mark.filterwarnings("ignore:Some inputs do not have OOB scores:UserWarning"),
]

EXPECTED_INPUT_SHA256 = "9cbd74e7e166c6dd5c7b78e564a91d9c2ae09d73f99731b6489d0daee489e0a1"


@pytest.fixture(scope="module")
def smoke_run() -> TutorialRun:
    return run_tutorial("smoke")


def test_profiles_define_explicit_workloads() -> None:
    assert _settings("smoke") == TutorialSettings(
        forest_estimators=3,
        search_folds=2,
        search_alphas=(0.05,),
        search_depths=(2, 3),
    )
    assert _settings("quick") == TutorialSettings(
        forest_estimators=25,
        search_folds=3,
        search_alphas=(0.01, 0.05),
        search_depths=(3, 5),
    )
    assert _settings("full") == TutorialSettings(
        forest_estimators=100,
        search_folds=5,
        search_alphas=(0.01, 0.05),
        search_depths=(3, 5),
    )
    with pytest.raises(ValueError, match="unknown tutorial profile"):
        _settings("unknown")  # type: ignore[arg-type]


def test_smoke_tutorial_covers_estimators_pipeline_search_and_inspection(
    smoke_run: TutorialRun,
) -> None:
    assert smoke_run.input_sha256 == EXPECTED_INPUT_SHA256
    assert smoke_run.n_samples == 569
    assert smoke_run.n_features == 30
    assert smoke_run.target_names == ("malignant", "benign")

    metrics = smoke_run.holdout_metrics.set_index("model")
    assert set(metrics.index) == {"tree", "forest", "tuned_pipeline"}
    assert (
        metrics[["balanced_accuracy", "roc_auc"]]
        .map(lambda value: 0.0 <= value <= 1.0)
        .all(axis=None)
    )

    importances = smoke_run.feature_importances
    assert importances.groupby("model").size().to_dict() == {"forest": 30, "tree": 30}
    assert importances.groupby("model")["importance"].sum().to_dict() == pytest.approx(
        {"forest": 1.0, "tree": 1.0}
    )
    assert importances.groupby("model")["rank"].apply(list).to_dict() == {
        "forest": list(range(1, 31)),
        "tree": list(range(1, 31)),
    }

    inspection = smoke_run.inspection.set_index(["object", "measure"])["value"]
    assert inspection[("tree", "depth")] > 0
    assert inspection[("tree", "nodes")] > 1
    assert inspection[("tree", "selector_permutations")] > 0
    assert inspection[("tree", "splitter_permutations")] > 0
    assert inspection[("forest", "estimators")] == 3
    assert inspection[("forest", "leaf_matrix_columns")] == 3
    assert inspection[("forest", "node_offset_entries")] == 4

    search = smoke_run.search_results
    assert len(search) == 2
    assert search["alpha_selector"].eq(0.05).all()
    assert set(search["max_depth"]) == {2, 3}
    assert search["mean_balanced_accuracy"].between(0.0, 1.0).all()
    assert "pipeline feature names:" in smoke_run.transcript
    assert "search best parameters:" in smoke_run.transcript


def test_smoke_tutorial_is_deterministic(smoke_run: TutorialRun) -> None:
    repeated = run_tutorial("smoke")

    assert repeated.input_sha256 == smoke_run.input_sha256
    assert repeated.transcript == smoke_run.transcript
    for field in (
        "inspection",
        "feature_importances",
        "holdout_metrics",
        "search_results",
    ):
        pd.testing.assert_frame_equal(getattr(repeated, field), getattr(smoke_run, field))


def test_write_results_records_inputs_sources_and_artifact_hashes(
    smoke_run: TutorialRun,
    tmp_path,
) -> None:
    write_results(
        smoke_run,
        tmp_path,
        profile="smoke",
        base_seed=1718,
        elapsed_seconds=1.25,
    )

    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="ascii"))
    assert receipt["analysis"] == "tutorial"
    assert receipt["schema_version"] == 1
    assert receipt["profile"] == "smoke"
    assert receipt["base_seed"] == 1718
    assert receipt["inputs"]["sklearn_breast_cancer"] == {
        "sha256": EXPECTED_INPUT_SHA256,
        "n_samples": 569,
        "n_features": 30,
        "target_names": ["malignant", "benign"],
    }
    assert "paper/jss/replication/tutorial.py" in receipt["source_sha256"]
    assert "citrees/_forest.py" in receipt["source_sha256"]
    assert "citrees/_tree.py" in receipt["source_sha256"]
    assert "uv.lock" in receipt["source_sha256"]
    assert len(receipt["artifacts"]) == 9

    for artifact, metadata in receipt["artifacts"].items():
        artifact_path = tmp_path / artifact
        assert artifact_path.exists()
        assert metadata["bytes"] == artifact_path.stat().st_size
        assert metadata["sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    for name, expected in (
        ("inspection", smoke_run.inspection),
        ("feature_importances", smoke_run.feature_importances),
        ("holdout_metrics", smoke_run.holdout_metrics),
        ("search_results", smoke_run.search_results),
    ):
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / f"{name}.parquet"), expected)
