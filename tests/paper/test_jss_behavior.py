"""Tests for the JSS matched-behavior replication analysis."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper.jss.replication import behavior
from paper.jss.replication.behavior import (
    BEHAVIOR_RESULT_SCHEMAS,
    PARTYKIT_TESTTYPE,
    SUMMARY_METRICS,
    BehaviorSettings,
    DatasetSpec,
    ModelBehavior,
    _array_sha256,
    _fold_pair_row,
    _prediction_metrics,
    _settings,
    _tie_ranks,
    _top_k_with_ties,
    run_behavior,
    summarize_behavior,
    validate_behavior_results,
    write_results,
)

pytestmark = pytest.mark.paper

R_AVAILABLE = (
    shutil.which("Rscript") is not None and importlib.util.find_spec("rpy2") is not None
)


def test_profiles_define_explicit_workloads() -> None:
    assert _settings("smoke") == BehaviorSettings(
        n_splits=2,
        n_repeats=1,
        n_resamples=39,
        n_trees=5,
        importance_permutations=1,
        summary_resamples=99,
    )
    assert _settings("quick") == BehaviorSettings(
        n_splits=3,
        n_repeats=1,
        n_resamples=199,
        n_trees=25,
        importance_permutations=3,
        summary_resamples=999,
    )
    assert _settings("full") == BehaviorSettings(
        n_splits=5,
        n_repeats=10,
        n_resamples=999,
        n_trees=100,
        importance_permutations=10,
        summary_resamples=9_999,
    )
    with pytest.raises(ValueError, match="unknown behavior profile"):
        _settings("unknown")  # type: ignore[arg-type]


def test_rankings_preserve_ties_and_structural_zeros() -> None:
    values = np.array([3.0, 3.0, 1.0, np.nan, -1.0])

    np.testing.assert_array_equal(_tie_ranks(values), [0, 0, 2, 3, 4])
    assert _top_k_with_ties(values, 1) == {0, 1}
    assert _top_k_with_ties(values, 3) == {0, 1, 2}

    permutation = np.array([3, 1, 4, 0, 2])
    permuted_top = _top_k_with_ties(values[permutation], 3)
    assert {int(permutation[index]) for index in permuted_top} == {0, 1, 2}


def test_prediction_metrics_cover_probability_and_regression_agreement() -> None:
    classes = np.array([0, 1])
    classification_true = np.array([0, 1, 0, 1])
    citrees_classification = ModelBehavior(
        root_feature=0,
        feature_values=np.array([1.0, 0.0]),
        predictions=classification_true,
        probabilities=np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.1, 0.9],
            ]
        ),
        classes=classes,
    )
    partykit_classification = ModelBehavior(
        root_feature=0,
        feature_values=np.array([1.0, 0.0]),
        predictions=classification_true,
        probabilities=np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
                [0.7, 0.3],
                [0.2, 0.8],
            ]
        ),
        classes=classes,
    )
    classification = _prediction_metrics(
        "classification",
        classification_true,
        citrees_classification,
        partykit_classification,
    )
    assert classification["prediction_agreement"] == 1.0
    assert classification["probability_mean_absolute_difference"] == pytest.approx(0.1)
    assert classification["probability_spearman_correlation"] == pytest.approx(1.0)
    assert classification["citrees_roc_auc"] == 1.0
    assert classification["partykit_roc_auc"] == 1.0

    regression_true = np.array([0.0, 1.0, 2.0])
    citrees_regression = ModelBehavior(
        root_feature=0,
        feature_values=np.array([1.0, 0.0]),
        predictions=np.array([0.0, 1.0, 2.0]),
        probabilities=None,
        classes=None,
    )
    partykit_regression = ModelBehavior(
        root_feature=0,
        feature_values=np.array([1.0, 0.0]),
        predictions=np.array([0.5, 1.5, 2.5]),
        probabilities=None,
        classes=None,
    )
    regression = _prediction_metrics(
        "regression",
        regression_true,
        citrees_regression,
        partykit_regression,
    )
    assert regression["prediction_correlation"] == pytest.approx(1.0)
    assert regression["prediction_mean_absolute_difference"] == pytest.approx(0.5)
    assert regression["citrees_mean_absolute_error"] == 0.0
    assert regression["partykit_mean_absolute_error"] == pytest.approx(0.5)


def _mock_datasets() -> tuple[DatasetSpec, DatasetSpec]:
    sample_ids = np.arange(8, dtype=np.float64)
    X_classification = np.column_stack(
        [sample_ids, sample_ids / 8.0, np.sin(sample_ids)]
    )
    y_classification = (sample_ids.astype(np.int64) % 2).astype(np.int64)
    X_regression = np.column_stack([sample_ids, sample_ids**2, np.cos(sample_ids)])
    y_regression = sample_ids * 1.5
    feature_names = ("sample", "trend", "wave")
    return (
        DatasetSpec(
            task="classification",
            name="mock_classification",
            X=X_classification,
            y=y_classification,
            feature_names=feature_names,
            sha256=_array_sha256(
                X_classification,
                y_classification,
                feature_names,
            ),
        ),
        DatasetSpec(
            task="regression",
            name="mock_regression",
            X=X_regression,
            y=y_regression,
            feature_names=feature_names,
            sha256=_array_sha256(
                X_regression,
                y_regression,
                feature_names,
            ),
        ),
    )


def _mock_fit_method(
    method: behavior.Method,
    task: behavior.Task,
    family: behavior.ModelFamily,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    settings: BehaviorSettings,
    seed: int,
) -> ModelBehavior:
    del X_train, y_train, settings, seed
    if family == "tree":
        root_feature = 0 if method == "citrees" else 1
        feature_values = (
            np.array([2.0, 1.0, 0.0])
            if method == "citrees"
            else np.array([1.0, 2.0, 0.0])
        )
    else:
        root_feature = None
        feature_values = (
            np.array([0.6, 0.3, 0.1])
            if method == "citrees"
            else np.array([0.5, np.nan, 0.2])
        )

    sample_ids = X_test[:, 0].astype(np.int64)
    if task == "classification":
        predictions = sample_ids % 2
        probabilities = np.full((len(sample_ids), 2), 0.2, dtype=np.float64)
        probabilities[np.arange(len(sample_ids)), predictions] = 0.8
        if method == "partykit":
            probabilities = probabilities * 0.875 + 0.0625
        return ModelBehavior(
            root_feature=root_feature,
            feature_values=feature_values,
            predictions=predictions,
            probabilities=probabilities,
            classes=np.array([0, 1]),
        )

    predictions = sample_ids * 1.5 + (0.0 if method == "citrees" else 0.25)
    return ModelBehavior(
        root_feature=root_feature,
        feature_values=feature_values,
        predictions=predictions,
        probabilities=None,
        classes=None,
    )


@pytest.fixture
def mocked_results(monkeypatch: pytest.MonkeyPatch) -> dict[str, pd.DataFrame]:
    monkeypatch.setattr(behavior, "load_behavior_datasets", _mock_datasets)
    monkeypatch.setattr(behavior, "_fit_method", _mock_fit_method)
    return run_behavior("smoke", base_seed=7)


def test_mocked_run_is_complete_and_deterministic(
    mocked_results: dict[str, pd.DataFrame],
) -> None:
    repeated = run_behavior("smoke", base_seed=7)

    assert set(mocked_results) == set(BEHAVIOR_RESULT_SCHEMAS)
    assert all(
        tuple(mocked_results[name].columns) == schema
        for name, schema in BEHAVIOR_RESULT_SCHEMAS.items()
    )
    assert len(mocked_results["behavior_fit_raw"]) == 16
    assert len(mocked_results["behavior_feature_raw"]) == 48
    assert len(mocked_results["behavior_prediction_raw"]) == 64
    assert len(mocked_results["behavior_probability_raw"]) == 64
    assert len(mocked_results["behavior_fold_summary"]) == 8
    assert mocked_results["behavior_fit_raw"][
        ["native_output_reproducible", "predictions_reproducible"]
    ].all(axis=None)
    assert mocked_results["behavior_summary"]["bootstrap_resamples"].eq(0).all()
    assert (
        mocked_results["behavior_summary"][["lower_95", "upper_95"]]
        .isna()
        .all(axis=None)
    )
    fold_summary = mocked_results["behavior_fold_summary"]
    assert (
        fold_summary.loc[
            fold_summary["task"].eq("regression"),
            "citrees_performance_advantage",
        ]
        .gt(0.0)
        .all()
    )
    for name in BEHAVIOR_RESULT_SCHEMAS:
        pd.testing.assert_frame_equal(mocked_results[name], repeated[name])


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("fold_hash", "held-out fold"),
        ("probability", "sum to one"),
        ("tree_missing", "Only omitted partykit forest"),
        ("rank", "preserve native ties"),
        ("seed", "derived from the base seed"),
        ("feature_seed", "feature raw model seeds"),
        ("prediction_seed", "prediction raw model seeds"),
        ("probability_seed", "probability raw model seeds"),
        ("fold_metric", "validated raw tables"),
        ("string_boolean", "must be boolean"),
        ("fractional_root", "must be integers"),
        ("dataset_hash", "dataset provenance"),
        ("feature_name", "frozen dataset"),
    ],
)
def test_validation_rejects_corrupted_behavior_tables(
    mocked_results: dict[str, pd.DataFrame],
    mutation: str,
    match: str,
) -> None:
    corrupted = {name: frame.copy(deep=True) for name, frame in mocked_results.items()}
    if mutation == "fold_hash":
        fits = corrupted["behavior_fit_raw"]
        first = fits.iloc[0]
        pair = (
            fits["task"].eq(first["task"])
            & fits["dataset"].eq(first["dataset"])
            & fits["model_family"].eq(first["model_family"])
            & fits["repeat"].eq(first["repeat"])
            & fits["fold"].eq(first["fold"])
        )
        fits.loc[pair, "test_ids_sha256"] = "0" * 64
    elif mutation == "probability":
        corrupted["behavior_probability_raw"].loc[0, "probability"] = 0.9
    elif mutation == "tree_missing":
        feature = corrupted["behavior_feature_raw"]
        candidate = feature[
            feature["model_family"].eq("tree") & feature["method"].eq("citrees")
        ].iloc[0]
        mask = (
            feature["task"].eq(candidate["task"])
            & feature["dataset"].eq(candidate["dataset"])
            & feature["model_family"].eq(candidate["model_family"])
            & feature["method"].eq(candidate["method"])
            & feature["repeat"].eq(candidate["repeat"])
            & feature["fold"].eq(candidate["fold"])
        )
        feature.loc[mask, "feature_value"] = np.nan
        feature.loc[mask, "feature_value_missing"] = True
        feature.loc[mask, "ranking_value"] = 0.0
        feature.loc[mask, "rank"] = 0
    elif mutation == "rank":
        corrupted["behavior_feature_raw"].loc[0, "rank"] = 2
    elif mutation == "seed":
        fits = corrupted["behavior_fit_raw"]
        first = fits.iloc[0]
        pair = (
            fits["task"].eq(first["task"])
            & fits["dataset"].eq(first["dataset"])
            & fits["model_family"].eq(first["model_family"])
            & fits["repeat"].eq(first["repeat"])
            & fits["fold"].eq(first["fold"])
        )
        fits.loc[pair, "model_seed"] = fits.loc[pair, "model_seed"] + 1
    elif mutation.endswith("_seed"):
        table_name = f"behavior_{mutation.removesuffix('_seed')}_raw"
        corrupted[table_name].loc[0, "model_seed"] += 1
    elif mutation == "string_boolean":
        fits = corrupted["behavior_fit_raw"]
        fits["native_output_reproducible"] = fits["native_output_reproducible"].astype(
            object
        )
        fits.loc[0, "native_output_reproducible"] = "False"
    elif mutation == "fractional_root":
        fits = corrupted["behavior_fit_raw"]
        fits["root_feature"] = fits["root_feature"].astype(object)
        tree_index = fits[fits["model_family"].eq("tree")].index[0]
        fits.loc[tree_index, "root_feature"] = (
            float(fits.loc[tree_index, "root_feature"]) + 0.5
        )
    elif mutation == "dataset_hash":
        for frame in corrupted.values():
            if "dataset_sha256" in frame:
                frame["dataset_sha256"] = "f" * 64
    elif mutation == "feature_name":
        features = corrupted["behavior_feature_raw"]
        first = features.iloc[0]
        feature = features["dataset"].eq(first["dataset"]) & features["feature_id"].eq(
            first["feature_id"]
        )
        features.loc[feature, "feature_name"] = "altered feature"
    else:
        corrupted["behavior_fold_summary"].loc[0, "prediction_agreement"] = 0.0

    with pytest.raises(ValueError, match=match):
        validate_behavior_results(
            corrupted,
            _settings("smoke"),
            base_seed=7,
        )


def test_terminal_trees_separate_split_and_root_agreement() -> None:
    terminal = ModelBehavior(
        root_feature=-1,
        feature_values=np.zeros(3, dtype=np.float64),
        predictions=np.array([0, 1], dtype=np.int64),
        probabilities=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        classes=np.array([0, 1], dtype=np.int64),
    )

    row = _fold_pair_row(
        DatasetSpec(
            task="classification",
            name="terminal",
            X=np.empty((0, 0)),
            y=np.empty(0),
            feature_names=(),
            sha256="0" * 64,
        ),
        "tree",
        0,
        0,
        1,
        np.array([0, 1], dtype=np.int64),
        {"citrees": terminal, "partykit": terminal},
        {"citrees": (True, True), "partykit": (True, True)},
    )

    assert row["split_decision_agreement"] == 1.0
    assert np.isnan(float(row["root_agreement_given_both_split"]))
    assert row["root_comparison"] == "both_no_split"
    assert not bool(row["both_native_rankings_informative"])
    assert np.isnan(float(row["native_ranking_kendall_tau_b"]))
    assert np.isnan(float(row["native_top_k_with_ties_jaccard"]))


def test_repeat_summary_uses_same_estimand_for_point_and_interval() -> None:
    rows: list[dict[str, object]] = []
    for repeat, value in ((0, 0.0), (0, 0.0), (1, 10.0)):
        row: dict[str, object] = {
            "task": "classification",
            "dataset": "repeat_weighting",
            "model_family": "tree",
            **{metric: float("nan") for metric in SUMMARY_METRICS},
        }
        row["repeat"] = repeat
        row["prediction_agreement"] = value
        rows.append(row)

    summary = summarize_behavior(
        pd.DataFrame(rows),
        BehaviorSettings(
            n_splits=2,
            n_repeats=2,
            n_resamples=39,
            n_trees=5,
            importance_permutations=1,
            summary_resamples=99,
        ),
        base_seed=7,
    )
    agreement = summary.loc[summary["metric"].eq("prediction_agreement")].iloc[0]

    assert agreement["estimate"] == 5.0
    assert agreement["n_fold_values"] == 3
    assert agreement["n_repeats"] == 2
    assert agreement["bootstrap_resamples"] == 99


def test_all_missing_partykit_importance_is_structural_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fit_with_terminal_partykit_forest(
        method: behavior.Method,
        task: behavior.Task,
        family: behavior.ModelFamily,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        *,
        settings: BehaviorSettings,
        seed: int,
    ) -> ModelBehavior:
        result = _mock_fit_method(
            method,
            task,
            family,
            X_train,
            y_train,
            X_test,
            settings=settings,
            seed=seed,
        )
        if method == "partykit" and family == "forest":
            return ModelBehavior(
                root_feature=None,
                feature_values=np.full_like(result.feature_values, np.nan),
                predictions=result.predictions,
                probabilities=result.probabilities,
                classes=result.classes,
            )
        return result

    monkeypatch.setattr(behavior, "load_behavior_datasets", _mock_datasets)
    monkeypatch.setattr(behavior, "_fit_method", fit_with_terminal_partykit_forest)

    results = run_behavior("smoke", base_seed=7)
    features = results["behavior_feature_raw"]
    partykit_forest = features[
        features["model_family"].eq("forest") & features["method"].eq("partykit")
    ]
    folds = results["behavior_fold_summary"]
    forest_folds = folds[folds["model_family"].eq("forest")]

    assert partykit_forest["feature_value_missing"].all()
    assert partykit_forest["ranking_value"].eq(0.0).all()
    assert partykit_forest["rank"].eq(0).all()
    assert not forest_folds["partykit_native_ranking_informative"].any()
    assert not forest_folds["both_native_rankings_informative"].any()
    assert (
        forest_folds[["native_ranking_kendall_tau_b", "native_top_k_with_ties_jaccard"]]
        .isna()
        .all(axis=None)
    )


def test_writer_records_controls_sources_and_artifact_hashes(
    mocked_results: dict[str, pd.DataFrame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        behavior,
        "get_r_runtime_versions",
        lambda: {"r": "R fixture", "partykit": "partykit fixture"},
    )
    write_results(
        mocked_results,
        tmp_path,
        profile="smoke",
        base_seed=7,
        elapsed_seconds=1.25,
    )

    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="ascii"))
    assert receipt["analysis"] == "behavior"
    assert receipt["schema_version"] == 4
    assert receipt["profile"] == "smoke"
    assert receipt["base_seed"] == 7
    assert receipt["controls"]["partykit_selector"]["test_distribution"] == list(
        PARTYKIT_TESTTYPE
    )
    assert receipt["controls"]["forest"]["bootstrap"] == "n_out_of_n_with_replacement"
    assert (
        receipt["controls"]["root_comparison"]["root_identity"]
        == "conditional_on_both_trees_splitting"
    )
    assert (
        receipt["controls"]["summary"]["estimate"]
        == "unweighted_mean_of_partition_repeat_means"
    )
    assert (
        receipt["controls"]["citrees_performance_advantage"]["interpretation"]
        == "positive_values_favor_citrees"
    )
    assert set(receipt["tables"]) == set(BEHAVIOR_RESULT_SCHEMAS)
    assert "paper/jss/replication/behavior.py" in receipt["source_sha256"]
    assert "paper/benchmark/pipeline/r_methods.py" in receipt["source_sha256"]
    assert "citrees/_forest.py" in receipt["source_sha256"]
    assert "citrees/_permutation.py" in receipt["source_sha256"]
    assert "citrees/_selector.py" in receipt["source_sha256"]
    assert "citrees/_tree.py" in receipt["source_sha256"]
    assert "uv.lock" in receipt["source_sha256"]
    assert len(receipt["artifacts"]) == 8
    for artifact, metadata in receipt["artifacts"].items():
        artifact_path = tmp_path / artifact
        assert metadata["bytes"] == artifact_path.stat().st_size
        assert (
            metadata["sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        )


@pytest.mark.skipif(not R_AVAILABLE, reason="R and rpy2 are required")
def test_real_smoke_returns_same_fit_r_behavior() -> None:
    results = run_behavior("smoke")

    validate_behavior_results(
        results,
        _settings("smoke"),
        base_seed=behavior.BASE_SEED,
    )
    fits = results["behavior_fit_raw"]
    assert fits[["native_output_reproducible", "predictions_reproducible"]].all(
        axis=None
    )
    features = results["behavior_feature_raw"]
    partykit_forests = features[
        features["model_family"].eq("forest") & features["method"].eq("partykit")
    ]
    assert not np.isinf(
        partykit_forests["feature_value"].to_numpy(dtype=np.float64)
    ).any()
    probabilities = results["behavior_probability_raw"]
    probability_sums = probabilities.groupby(
        ["dataset", "model_family", "method", "repeat", "fold", "sample_id"]
    )["probability"].sum()
    assert np.allclose(probability_sums, 1.0)
