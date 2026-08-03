"""Tests for ranking and metric artifact contracts."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper.benchmark.config.constants import (
    CLF_DOWNSTREAM_MODELS,
    PIPELINE_ARTIFACT_VERSION,
    REG_DOWNSTREAM_MODELS,
)
from paper.benchmark.pipeline.types import DatasetIdentity, ExperimentConfig, MethodConfig
from paper.benchmark.pipeline.validation import (
    ArtifactValidationError,
    validate_artifact_provenance,
    validate_metrics_artifact,
    validate_metrics_ranking_payload,
    validate_ranking_artifact,
)

pytestmark = pytest.mark.paper


def _config(method: str = "rf") -> ExperimentConfig:
    return ExperimentConfig(
        method=MethodConfig(method, params=(("n_estimators", 100),)),
        dataset="fixture",
        seed=2,
        task="classification",
        dataset_identity=DatasetIdentity("d" * 64, n_samples=50, n_features=4),
    )


def _common(config: ExperimentConfig, *, n_features: int = 4) -> dict[str, object]:
    versions = {
        "citrees": "0.1.0",
        "numpy": "2.4.6",
        "python": "3.12.7",
        "rpy2": "3.6.7",
        "sklearn": "1.9.0",
    }
    if config.method.name.startswith("r_"):
        versions.update({"r": "R version 4.5.2", "partykit": "1.2.24"})
    return {
        "artifact_version": PIPELINE_ARTIFACT_VERSION,
        "artifact_prefix": "repairs/run-001",
        "aws_account_id": "123456789012",
        "campaign_sha256": "e" * 64,
        "container_image": "repository@sha256:" + "a" * 64,
        "created_at_utc": "2026-08-03T12:00:00+00:00",
        "dataset": config.dataset,
        "dataset_sha256": config.dataset_identity.sha256,
        "git_sha": "a" * 40,
        "hardware": {"logical_cpus": 32},
        "library_versions": versions,
        "method": config.method.label,
        "method_base": config.method.name,
        "method_id": config.method.label,
        "method_params_json": json.dumps(
            config.method.params_dict,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "manifest_sha256": "b" * 64,
        "n_features": n_features,
        "n_samples": 50,
        "seed": config.seed,
        "task": config.task,
    }


def _rankings(config: ExperimentConfig | None = None) -> pd.DataFrame:
    config = config or _config()
    common = _common(config)
    return pd.DataFrame(
        [
            {
                **common,
                "feature_ranking": [0, 1, 2, 3],
                "fold_idx": fold,
                "fold_random_state": config.seed * 1000 + fold,
                "selection_cpus": -1,
            }
            for fold in range(5)
        ]
    )


def _metrics(
    config: ExperimentConfig | None = None,
    *,
    k_values: tuple[int, ...] = (4,),
) -> pd.DataFrame:
    config = config or _config()
    common = _common(config)
    return pd.DataFrame(
        [
            {
                **common,
                "accuracy": 0.8,
                "auc": 0.85,
                "balanced_accuracy": 0.79,
                "downstream_model": model,
                "evaluation_cpus": -1,
                "f1": 0.8,
                "f1_macro": 0.79,
                "fold_idx": fold,
                "fold_random_state": config.seed * 1000 + fold,
                "k": k,
                "n_features_selected": k,
                "ranking_artifact_version": PIPELINE_ARTIFACT_VERSION,
                "ranking_artifact_prefix": "repairs/run-001",
                "ranking_aws_account_id": "123456789012",
                "ranking_container_image": "repository@sha256:" + "a" * 64,
                "ranking_dataset_sha256": config.dataset_identity.sha256,
                "ranking_git_sha": "a" * 40,
                "ranking_manifest_sha256": "b" * 64,
                "ranking_payload_sha256": "c" * 64,
                "roc_auc": 0.85,
            }
            for fold, k, model in product(range(5), k_values, CLF_DOWNSTREAM_MODELS)
        ]
    )


def _regression_config() -> ExperimentConfig:
    return ExperimentConfig(
        method=MethodConfig("rf", params=(("n_estimators", 100),)),
        dataset="fixture",
        seed=2,
        task="regression",
        dataset_identity=DatasetIdentity("d" * 64, n_samples=50, n_features=4),
    )


def _regression_metrics() -> pd.DataFrame:
    config = _regression_config()
    common = _common(config)
    return pd.DataFrame(
        [
            {
                **common,
                "downstream_model": model,
                "evaluation_cpus": -1,
                "fold_idx": fold,
                "fold_random_state": config.seed * 1000 + fold,
                "k": 4,
                "mae": 0.4,
                "mse": 0.25,
                "n_features_selected": 4,
                "r2": 0.7,
                "ranking_artifact_version": PIPELINE_ARTIFACT_VERSION,
                "ranking_artifact_prefix": "repairs/run-001",
                "ranking_aws_account_id": "123456789012",
                "ranking_container_image": "repository@sha256:" + "a" * 64,
                "ranking_dataset_sha256": config.dataset_identity.sha256,
                "ranking_git_sha": "a" * 40,
                "ranking_manifest_sha256": "b" * 64,
                "ranking_payload_sha256": "c" * 64,
                "rmse": 0.5,
            }
            for fold, model in product(range(5), REG_DOWNSTREAM_MODELS)
        ]
    )


def test_valid_ranking_artifact_returns_feature_count() -> None:
    config = _config()
    assert validate_ranking_artifact(_rankings(config), config) == 4


def test_artifact_provenance_must_match_configured_run() -> None:
    config = _config()
    expected = {
        "artifact_prefix": "repairs/run-001",
        "aws_account_id": "123456789012",
        "campaign_sha256": "e" * 64,
        "container_image": "repository@sha256:" + "a" * 64,
        "git_sha": "a" * 40,
        "manifest_sha256": "b" * 64,
    }

    validate_artifact_provenance(_rankings(config), expected)
    validate_artifact_provenance(
        _metrics(config),
        expected,
        include_ranking_reference=True,
    )

    with pytest.raises(ArtifactValidationError, match="manifest_sha256"):
        validate_artifact_provenance(
            _rankings(config).assign(manifest_sha256="c" * 64),
            expected,
        )


def test_ranking_artifact_validates_after_parquet_round_trip(tmp_path: Path) -> None:
    """Nested provenance and ranking arrays must survive real serialization."""
    config = _config()
    path = tmp_path / "rankings.parquet"
    _rankings(config).to_parquet(path, index=False)

    restored = pd.read_parquet(path)

    assert validate_ranking_artifact(restored, config) == 4


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda frame: frame.iloc[:-1], "must contain 5 folds"),
        (lambda frame: frame.drop(columns=["git_sha"]), "missing columns"),
        (lambda frame: frame.assign(artifact_version=2), "artifact_version"),
        (lambda frame: frame.assign(dataset="other"), "dataset"),
        (lambda frame: frame.assign(dataset_sha256="e" * 64), "dataset_sha256"),
        (lambda frame: frame.assign(n_features=5), "does not match manifest"),
        (lambda frame: frame.assign(n_samples=49), "does not match manifest"),
        (lambda frame: frame.assign(n_features=4.5), "positive integers"),
        (lambda frame: frame.assign(n_samples=50.5), "positive integers"),
        (lambda frame: frame.assign(n_features=True), "positive integers"),
        (lambda frame: frame.assign(n_samples=True), "positive integers"),
        (lambda frame: frame.assign(fold_idx=[0, 1, 2, 3, 3]), "fold_idx"),
        (lambda frame: frame.assign(fold_random_state=0), "fold_random_state"),
        (lambda frame: frame.assign(artifact_prefix="../escape"), "artifact_prefix"),
        (lambda frame: frame.assign(aws_account_id="unknown"), "aws_account_id"),
        (lambda frame: frame.assign(container_image="unknown"), "container_image"),
        (lambda frame: frame.assign(git_sha="unknown"), "git_sha"),
        (lambda frame: frame.assign(manifest_sha256="unknown"), "manifest_sha256"),
    ],
)
def test_ranking_artifact_rejects_schema_identity_and_fold_errors(
    mutation,
    match: str,
) -> None:
    config = _config()
    with pytest.raises(ArtifactValidationError, match=match):
        validate_ranking_artifact(mutation(_rankings(config)), config)


@pytest.mark.parametrize(
    "ranking",
    [
        [0, 1, 2],
        [0.0, 1.0, 2.0, 3.0],
        [0, 1, 1, 3],
        [-1, 0, 1, 2],
        [0, 1, 2, 4],
    ],
)
def test_ranking_artifact_rejects_malformed_feature_permutations(
    ranking: list[int] | list[float],
) -> None:
    config = _config()
    frame = _rankings(config)
    frame.at[0, "feature_ranking"] = ranking

    with pytest.raises(ArtifactValidationError, match="ranking"):
        validate_ranking_artifact(frame, config)


def test_r_artifact_requires_r_partykit_and_rpy2_versions() -> None:
    config = _config("r_ctree")
    frame = _rankings(config)
    frame["library_versions"] = [{"python": "3.12.7", "rpy2": "3.6.7"} for _ in range(len(frame))]

    with pytest.raises(ArtifactValidationError, match="runtime versions"):
        validate_ranking_artifact(frame, config)


def test_valid_metrics_artifact_covers_every_fold_k_model_cell() -> None:
    config = _config()
    validate_metrics_artifact(_metrics(config), config)
    validate_metrics_ranking_payload(_metrics(config), "c" * 64)


def test_metrics_are_bound_to_the_exact_ranking_payload() -> None:
    with pytest.raises(ArtifactValidationError, match="does not match"):
        validate_metrics_ranking_payload(_metrics(), "d" * 64)


def test_metrics_artifact_validates_after_parquet_round_trip(tmp_path: Path) -> None:
    """Metric cells and nested provenance must survive real serialization."""
    config = _config()
    path = tmp_path / "metrics.parquet"
    _metrics(config).to_parquet(path, index=False)

    restored = pd.read_parquet(path)

    validate_metrics_artifact(restored, config)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda frame: frame.iloc[:-1], "exactly one row"),
        (lambda frame: frame.drop(columns=["accuracy"]), "missing columns"),
        (lambda frame: frame.assign(n_features_selected=1), "must equal k"),
        (lambda frame: frame.assign(fold_random_state=0), "fold_random_state"),
        (lambda frame: frame.assign(ranking_artifact_version=2), "ranking version"),
        (
            lambda frame: frame.assign(ranking_container_image="unknown"),
            "ranking_container_image",
        ),
        (lambda frame: frame.assign(ranking_git_sha="unknown"), "ranking_git_sha"),
        (
            lambda frame: frame.assign(ranking_manifest_sha256="unknown"),
            "ranking_manifest_sha256",
        ),
        (
            lambda frame: frame.assign(ranking_payload_sha256="unknown"),
            "ranking_payload_sha256",
        ),
        (
            lambda frame: frame.assign(ranking_dataset_sha256="e" * 64),
            "ranking_dataset_sha256",
        ),
        (lambda frame: frame.assign(n_samples=49), "does not match manifest"),
        (lambda frame: frame.assign(n_features=4.5), "positive integers"),
        (lambda frame: frame.assign(n_samples=True), "positive integers"),
        (lambda frame: frame.assign(accuracy=np.nan), "non-finite accuracy"),
    ],
)
def test_metrics_artifact_rejects_incomplete_or_invalid_outputs(
    mutation,
    match: str,
) -> None:
    config = _config()
    with pytest.raises(ArtifactValidationError, match=match):
        validate_metrics_artifact(mutation(_metrics(config)), config)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("accuracy", -0.01),
        ("accuracy", 1.01),
        ("auc", -0.01),
        ("balanced_accuracy", 1.01),
        ("f1", -0.01),
        ("f1_macro", 1.01),
        ("roc_auc", 1.01),
    ],
)
def test_classification_metrics_are_probabilities(column: str, value: float) -> None:
    config = _config()
    with pytest.raises(ArtifactValidationError, match=r"\[0, 1\]"):
        validate_metrics_artifact(_metrics(config).assign(**{column: value}), config)


def test_auc_aliases_must_be_identical() -> None:
    config = _config()
    with pytest.raises(ArtifactValidationError, match="auc and roc_auc"):
        validate_metrics_artifact(_metrics(config).assign(auc=0.5), config)


@pytest.mark.parametrize("column", ["mae", "mse", "rmse"])
def test_regression_losses_are_non_negative(column: str) -> None:
    config = _regression_config()
    with pytest.raises(ArtifactValidationError, match="non-negative"):
        validate_metrics_artifact(
            _regression_metrics().assign(**{column: -0.1}),
            config,
        )


def test_regression_r2_and_rmse_mse_identity_are_validated() -> None:
    config = _regression_config()
    validate_metrics_artifact(_regression_metrics(), config)

    with pytest.raises(ArtifactValidationError, match="r2"):
        validate_metrics_artifact(_regression_metrics().assign(r2=1.1), config)
    with pytest.raises(ArtifactValidationError, match="rmse and mse"):
        validate_metrics_artifact(_regression_metrics().assign(rmse=0.6), config)
