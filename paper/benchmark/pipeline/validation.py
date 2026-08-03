"""Validation contracts for benchmark ranking and metric artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from paper.benchmark.config.constants import (
    CLF_DOWNSTREAM_MODELS,
    N_SPLITS,
    PIPELINE_ARTIFACT_VERSION,
    REG_DOWNSTREAM_MODELS,
)
from paper.benchmark.pipeline.types import ExperimentConfig

_COMMON_COLUMNS = {
    "artifact_version",
    "artifact_prefix",
    "aws_account_id",
    "campaign_sha256",
    "container_image",
    "created_at_utc",
    "dataset",
    "dataset_sha256",
    "git_sha",
    "hardware",
    "library_versions",
    "method",
    "method_base",
    "method_id",
    "method_params_json",
    "manifest_sha256",
    "n_features",
    "n_samples",
    "seed",
    "task",
}

_RANKING_COLUMNS = _COMMON_COLUMNS | {
    "feature_ranking",
    "fold_idx",
    "fold_random_state",
    "selection_cpus",
}

_METRIC_COLUMNS = _COMMON_COLUMNS | {
    "downstream_model",
    "evaluation_cpus",
    "fold_idx",
    "fold_random_state",
    "k",
    "n_features_selected",
    "ranking_artifact_version",
    "ranking_artifact_prefix",
    "ranking_aws_account_id",
    "ranking_container_image",
    "ranking_dataset_sha256",
    "ranking_git_sha",
    "ranking_manifest_sha256",
    "ranking_payload_sha256",
}

_CLASSIFICATION_METRICS = {
    "accuracy",
    "auc",
    "balanced_accuracy",
    "f1",
    "f1_macro",
    "roc_auc",
}
_REGRESSION_METRICS = {"mae", "mse", "r2", "rmse"}
_ACCOUNT_ID_PATTERN = re.compile(r"^[0-9]{12}$")
_ASSIGNMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_CONTROL_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_GIT_SHA_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_IMAGE_DIGEST_PATTERN = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_RECEIPT_FIELDS = {
    "artifact_prefix",
    "artifact_version",
    "assignment_id",
    "attempt",
    "aws_account_id",
    "campaign_sha256",
    "container_image",
    "created_at_utc",
    "dataset",
    "dataset_sha256",
    "git_sha",
    "lease_seconds",
    "manifest_sha256",
    "method_base",
    "method_id",
    "method_params_json",
    "n_features",
    "n_samples",
    "seed",
    "stage",
    "status",
    "task",
    "request_id",
    "worker_id",
}
_FAILURE_RECEIPT_FIELDS = _ATTEMPT_RECEIPT_FIELDS - {"lease_seconds"} | {
    "elapsed_seconds",
    "error",
    "error_type",
    "hardware",
    "hostname",
    "library_versions",
    "traceback",
}


class ArtifactValidationError(ValueError):
    """Raised when an artifact violates the benchmark data contract."""


def _require_exact_receipt_fields(
    receipt: Mapping[str, Any],
    expected_fields: set[str],
    receipt_type: str,
) -> None:
    if set(receipt) != expected_fields:
        missing = sorted(expected_fields - set(receipt))
        extra = sorted(set(receipt) - expected_fields)
        raise ArtifactValidationError(
            f"{receipt_type} receipt fields differ from the contract: "
            f"missing={missing}, extra={extra}"
        )


def _require_positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ArtifactValidationError(f"{field} must be a positive integer")
    return value


def _require_control_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _CONTROL_ID_PATTERN.fullmatch(value):
        raise ArtifactValidationError(f"{field} must contain 32 lowercase hexadecimal characters")
    return value


def _require_utc_timestamp(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ArtifactValidationError(f"{field} must be a non-empty UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ArtifactValidationError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ArtifactValidationError(f"{field} must include a UTC offset")


def _require_control_receipt_identity(
    receipt: Mapping[str, Any],
    config: ExperimentConfig,
    *,
    stage: str,
    assignment_id: str,
    attempt: int,
    request_id: str,
    worker_id: str,
    expected_provenance: Mapping[str, str],
) -> None:
    if stage not in {"rankings", "metrics"}:
        raise ArtifactValidationError(f"invalid receipt stage {stage!r}")
    if not _ASSIGNMENT_ID_PATTERN.fullmatch(assignment_id):
        raise ArtifactValidationError(
            "assignment_id must contain 32 lowercase hexadecimal characters"
        )
    _require_positive_int(attempt, "attempt")
    request_id = _require_control_id(request_id, "request_id")
    worker_id = _require_control_id(worker_id, "worker_id")
    provenance = validate_expected_provenance(expected_provenance)
    expected_assignment_id = derive_assignment_id(
        config,
        stage=stage,
        attempt=attempt,
        expected_provenance=provenance,
    )
    if assignment_id != expected_assignment_id:
        raise ArtifactValidationError(
            f"assignment_id does not match cell attempt {expected_assignment_id}"
        )
    expected = {
        "artifact_prefix": provenance["artifact_prefix"],
        "artifact_version": PIPELINE_ARTIFACT_VERSION,
        "assignment_id": assignment_id,
        "attempt": attempt,
        "aws_account_id": provenance["aws_account_id"],
        "campaign_sha256": provenance["campaign_sha256"],
        "container_image": provenance["container_image"],
        "dataset": config.dataset,
        "dataset_sha256": config.dataset_identity.sha256,
        "git_sha": provenance["git_sha"],
        "manifest_sha256": provenance["manifest_sha256"],
        "method_base": config.method.name,
        "method_id": config.method.label,
        "method_params_json": json.dumps(
            config.method.params_dict,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "n_features": config.dataset_identity.n_features,
        "n_samples": config.dataset_identity.n_samples,
        "seed": config.seed,
        "stage": stage,
        "task": config.task,
        "request_id": request_id,
        "worker_id": worker_id,
    }
    for field, expected_value in expected.items():
        if receipt[field] != expected_value:
            raise ArtifactValidationError(
                f"receipt {field} does not match expected value {expected_value!r}"
            )
    _require_utc_timestamp(receipt["created_at_utc"], "created_at_utc")


def validate_attempt_receipt(
    receipt: Mapping[str, Any],
    config: ExperimentConfig,
    *,
    stage: str,
    assignment_id: str,
    attempt: int,
    request_id: str,
    worker_id: str,
    expected_provenance: Mapping[str, str],
) -> None:
    """Validate one immutable record of a leased cell attempt."""
    _require_exact_receipt_fields(receipt, _ATTEMPT_RECEIPT_FIELDS, "attempt")
    _require_control_receipt_identity(
        receipt,
        config,
        stage=stage,
        assignment_id=assignment_id,
        attempt=attempt,
        request_id=request_id,
        worker_id=worker_id,
        expected_provenance=expected_provenance,
    )
    if receipt["status"] != "assigned":
        raise ArtifactValidationError("attempt receipt status must be 'assigned'")
    _require_positive_int(receipt["lease_seconds"], "lease_seconds")


def validate_failure_receipt(
    receipt: Mapping[str, Any],
    config: ExperimentConfig,
    *,
    stage: str,
    assignment_id: str,
    attempt: int,
    request_id: str,
    worker_id: str,
    expected_provenance: Mapping[str, str],
) -> None:
    """Validate one complete immutable failed-attempt receipt."""
    _require_exact_receipt_fields(receipt, _FAILURE_RECEIPT_FIELDS, "failure")
    _require_control_receipt_identity(
        receipt,
        config,
        stage=stage,
        assignment_id=assignment_id,
        attempt=attempt,
        request_id=request_id,
        worker_id=worker_id,
        expected_provenance=expected_provenance,
    )
    if receipt["status"] != "failed":
        raise ArtifactValidationError("failure receipt status must be 'failed'")
    for field in ("error", "error_type", "hostname"):
        if not isinstance(receipt[field], str) or not receipt[field].strip():
            raise ArtifactValidationError(f"failure receipt {field} must be non-empty")
    traceback_text = receipt["traceback"]
    if traceback_text is not None and (
        not isinstance(traceback_text, str) or not traceback_text.strip()
    ):
        raise ArtifactValidationError(
            "failure receipt traceback must be null or a non-empty string"
        )
    elapsed_seconds = receipt["elapsed_seconds"]
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not np.isfinite(float(elapsed_seconds))
        or float(elapsed_seconds) < 0
    ):
        raise ArtifactValidationError(
            "failure receipt elapsed_seconds must be a finite non-negative number"
        )
    for field in ("hardware", "library_versions"):
        if not isinstance(receipt[field], Mapping) or not receipt[field]:
            raise ArtifactValidationError(f"failure receipt {field} must be a non-empty mapping")


def validate_expected_provenance(
    provenance: Mapping[str, str],
) -> dict[str, str]:
    """Validate the immutable provenance expected for one distributed run."""
    required = {
        "artifact_prefix",
        "aws_account_id",
        "campaign_sha256",
        "container_image",
        "git_sha",
        "manifest_sha256",
    }
    if set(provenance) != required:
        raise ArtifactValidationError(
            f"expected provenance keys must be exactly {sorted(required)}"
        )

    from paper.benchmark.adapters.store import _normalize_artifact_prefix

    validated = dict(provenance)
    try:
        validated["artifact_prefix"] = _normalize_artifact_prefix(validated["artifact_prefix"])
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"artifact_prefix is invalid: {exc}") from exc
    if not _ACCOUNT_ID_PATTERN.fullmatch(validated["aws_account_id"]):
        raise ArtifactValidationError("aws_account_id must contain 12 decimal digits")
    if not _SHA256_PATTERN.fullmatch(validated["campaign_sha256"]):
        raise ArtifactValidationError("campaign_sha256 must be 64 lowercase hexadecimal characters")
    if not _IMAGE_DIGEST_PATTERN.fullmatch(validated["container_image"]):
        raise ArtifactValidationError("container_image must be an immutable digest URI")
    if not _GIT_SHA_PATTERN.fullmatch(validated["git_sha"]):
        raise ArtifactValidationError("git_sha must be a full hexadecimal commit ID")
    if not _SHA256_PATTERN.fullmatch(validated["manifest_sha256"]):
        raise ArtifactValidationError("manifest_sha256 must be 64 lowercase hexadecimal characters")
    return validated


def derive_assignment_id(
    config: ExperimentConfig,
    *,
    stage: str,
    attempt: int,
    expected_provenance: Mapping[str, str],
) -> str:
    """Derive the assignment identity from one immutable campaign cell attempt."""
    if stage not in {"rankings", "metrics"}:
        raise ArtifactValidationError(f"invalid assignment stage {stage!r}")
    _require_positive_int(attempt, "attempt")
    provenance = validate_expected_provenance(expected_provenance)
    payload = json.dumps(
        {
            **provenance,
            "attempt": attempt,
            "dataset": config.dataset,
            "dataset_sha256": config.dataset_identity.sha256,
            "method_id": config.method.label,
            "seed": config.seed,
            "stage": stage,
            "task": config.task,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def validate_artifact_provenance(
    frame: pd.DataFrame,
    expected: Mapping[str, str],
    *,
    include_ranking_reference: bool = False,
) -> None:
    """Require artifact provenance to match the configured run exactly."""
    validated = validate_expected_provenance(expected)
    for column, expected_value in validated.items():
        if column not in frame:
            raise ArtifactValidationError(f"artifact is missing provenance column {column}")
        if not frame[column].eq(expected_value).all():
            raise ArtifactValidationError(
                f"artifact {column} does not match configured value {expected_value!r}"
            )

    if include_ranking_reference:
        ranking_columns = {
            "artifact_prefix": "ranking_artifact_prefix",
            "aws_account_id": "ranking_aws_account_id",
            "container_image": "ranking_container_image",
            "git_sha": "ranking_git_sha",
            "manifest_sha256": "ranking_manifest_sha256",
        }
        for source, column in ranking_columns.items():
            expected_value = validated[source]
            if column not in frame or not frame[column].eq(expected_value).all():
                raise ArtifactValidationError(
                    f"metrics artifact {column} does not match configured value {expected_value!r}"
                )


def _require_columns(frame: pd.DataFrame, required: set[str], stage: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ArtifactValidationError(f"{stage} artifact is missing columns: {sorted(missing)}")


def _positive_integer_values(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return one artifact column after strict positive-integer validation."""
    values = frame[column]
    valid = values.map(
        lambda value: (
            isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            and int(value) > 0
        )
    )
    if not valid.all():
        raise ArtifactValidationError(f"artifact {column} must contain positive integers")
    return values.astype(int)


def _require_constant_identity(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> int:
    expected = {
        "artifact_version": PIPELINE_ARTIFACT_VERSION,
        "dataset": config.dataset,
        "dataset_sha256": config.dataset_identity.sha256,
        "method": config.method.label,
        "method_base": config.method.name,
        "method_id": config.method.label,
        "method_params_json": json.dumps(
            config.method.params_dict,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "seed": config.seed,
        "task": config.task,
    }
    for column, value in expected.items():
        if not frame[column].eq(value).all():
            raise ArtifactValidationError(
                f"artifact {column} does not match config value {value!r}"
            )

    feature_counts = _positive_integer_values(frame, "n_features")
    unique_feature_counts = set(feature_counts)
    if len(unique_feature_counts) != 1:
        raise ArtifactValidationError("artifact contains inconsistent n_features values")
    observed_n_features = unique_feature_counts.pop()
    if observed_n_features != config.dataset_identity.n_features:
        raise ArtifactValidationError(
            f"artifact n_features={observed_n_features} does not match "
            f"manifest={config.dataset_identity.n_features}"
        )

    sample_counts = _positive_integer_values(frame, "n_samples")
    if sample_counts.nunique() != 1:
        raise ArtifactValidationError("artifact contains inconsistent n_samples values")
    observed_n_samples = int(sample_counts.iloc[0])
    if observed_n_samples != config.dataset_identity.n_samples:
        raise ArtifactValidationError(
            f"artifact n_samples={observed_n_samples} does not match "
            f"manifest={config.dataset_identity.n_samples}"
        )

    for column in (
        "artifact_prefix",
        "aws_account_id",
        "campaign_sha256",
        "container_image",
        "created_at_utc",
        "git_sha",
        "manifest_sha256",
    ):
        if (
            not frame[column]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise ArtifactValidationError(f"artifact {column} must be a non-empty string")
        if frame[column].nunique(dropna=False) != 1:
            raise ArtifactValidationError(f"artifact contains inconsistent {column} values")

    from paper.benchmark.adapters.store import _normalize_artifact_prefix

    try:
        _normalize_artifact_prefix(str(frame["artifact_prefix"].iloc[0]))
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"artifact_prefix is invalid: {exc}") from exc
    if not _ACCOUNT_ID_PATTERN.fullmatch(str(frame["aws_account_id"].iloc[0])):
        raise ArtifactValidationError("artifact aws_account_id must contain 12 decimal digits")
    if not _SHA256_PATTERN.fullmatch(str(frame["campaign_sha256"].iloc[0])):
        raise ArtifactValidationError(
            "artifact campaign_sha256 must be 64 lowercase hexadecimal characters"
        )
    if not _IMAGE_DIGEST_PATTERN.fullmatch(str(frame["container_image"].iloc[0])):
        raise ArtifactValidationError("artifact container_image must be an immutable digest URI")
    if not _GIT_SHA_PATTERN.fullmatch(str(frame["git_sha"].iloc[0])):
        raise ArtifactValidationError("artifact git_sha must be a full hexadecimal commit ID")
    if not _SHA256_PATTERN.fullmatch(str(frame["manifest_sha256"].iloc[0])):
        raise ArtifactValidationError(
            "artifact manifest_sha256 must be 64 lowercase hexadecimal characters"
        )

    for column in ("hardware", "library_versions"):
        if not frame[column].map(lambda value: isinstance(value, Mapping) and bool(value)).all():
            raise ArtifactValidationError(f"artifact {column} must be a non-empty mapping")

    if config.method.name in {"r_ctree", "r_cforest"}:
        required_versions = {"partykit", "r", "rpy2"}
        for versions in frame["library_versions"]:
            missing = required_versions - set(versions)
            if missing:
                raise ArtifactValidationError(
                    f"R artifact is missing runtime versions: {sorted(missing)}"
                )

    return observed_n_features


def _validated_fold_ids(frame: pd.DataFrame) -> np.ndarray:
    folds = pd.to_numeric(frame["fold_idx"], errors="coerce")
    if folds.isna().any() or not np.equal(folds, np.floor(folds)).all():
        raise ArtifactValidationError("fold_idx must contain integers")
    return folds.astype(int).to_numpy()


def validate_ranking_artifact(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> int:
    """Validate one complete Stage 1 artifact and return its feature count."""
    if len(frame) != N_SPLITS:
        raise ArtifactValidationError(
            f"rankings artifact must contain {N_SPLITS} folds, got {len(frame)}"
        )
    _require_columns(frame, _RANKING_COLUMNS, "rankings")
    observed_n_features = _require_constant_identity(
        frame,
        config,
    )

    folds = _validated_fold_ids(frame)
    if sorted(folds.tolist()) != list(range(N_SPLITS)):
        raise ArtifactValidationError(
            f"rankings fold_idx values must be 0..{N_SPLITS - 1}, got {sorted(folds.tolist())}"
        )

    expected_random_states = config.seed * 1000 + folds
    observed_random_states = pd.to_numeric(frame["fold_random_state"], errors="coerce")
    if observed_random_states.isna().any() or not np.array_equal(
        observed_random_states.astype(int).to_numpy(),
        expected_random_states,
    ):
        raise ArtifactValidationError("fold_random_state does not match seed and fold_idx")

    expected_indices = np.arange(observed_n_features)
    for fold_idx, value in zip(folds, frame["feature_ranking"], strict=True):
        ranking = np.asarray(value)
        if ranking.ndim != 1 or ranking.shape[0] != observed_n_features:
            raise ArtifactValidationError(
                f"fold {fold_idx} ranking has shape {ranking.shape}; "
                f"expected ({observed_n_features},)"
            )
        if not np.issubdtype(ranking.dtype, np.integer):
            raise ArtifactValidationError(f"fold {fold_idx} ranking is not integer-valued")
        if not np.array_equal(np.sort(ranking), expected_indices):
            raise ArtifactValidationError(
                f"fold {fold_idx} ranking is not a complete feature permutation"
            )

    return observed_n_features


def validate_metrics_artifact(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    """Validate one complete Stage 2 artifact."""
    metric_columns = (
        _CLASSIFICATION_METRICS if config.task == "classification" else _REGRESSION_METRICS
    )
    _require_columns(frame, _METRIC_COLUMNS | metric_columns, "metrics")
    _require_constant_identity(
        frame,
        config,
    )

    from paper.benchmark.pipeline.stage2 import get_requested_evaluation_k_values

    required_k_values: Sequence[int] = get_requested_evaluation_k_values(
        config.dataset_identity.n_features
    )
    folds = _validated_fold_ids(frame)
    models = CLF_DOWNSTREAM_MODELS if config.task == "classification" else REG_DOWNSTREAM_MODELS
    expected_cells = set(product(range(N_SPLITS), required_k_values, models))
    observed_cells = list(
        zip(
            folds,
            pd.to_numeric(frame["k"], errors="coerce"),
            frame["downstream_model"],
            strict=True,
        )
    )
    if len(observed_cells) != len(expected_cells) or set(observed_cells) != expected_cells:
        raise ArtifactValidationError(
            "metrics artifact does not contain exactly one row per fold, k, and model"
        )

    selected = pd.to_numeric(frame["n_features_selected"], errors="coerce")
    k_values = pd.to_numeric(frame["k"], errors="coerce")
    if selected.isna().any() or k_values.isna().any() or not np.array_equal(selected, k_values):
        raise ArtifactValidationError("n_features_selected must equal k")

    expected_random_states = config.seed * 1000 + folds
    observed_random_states = pd.to_numeric(frame["fold_random_state"], errors="coerce")
    if observed_random_states.isna().any() or not np.array_equal(
        observed_random_states.astype(int).to_numpy(),
        expected_random_states,
    ):
        raise ArtifactValidationError("fold_random_state does not match seed and fold_idx")

    if not (
        pd.to_numeric(frame["ranking_artifact_version"], errors="coerce")
        == PIPELINE_ARTIFACT_VERSION
    ).all():
        raise ArtifactValidationError("metrics artifact references the wrong ranking version")
    for column in (
        "ranking_artifact_prefix",
        "ranking_aws_account_id",
        "ranking_container_image",
        "ranking_dataset_sha256",
        "ranking_git_sha",
        "ranking_manifest_sha256",
        "ranking_payload_sha256",
    ):
        if (
            not frame[column]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise ArtifactValidationError(f"metrics artifact {column} must be non-empty")
        if frame[column].nunique(dropna=False) != 1:
            raise ArtifactValidationError(f"metrics artifact contains inconsistent {column} values")

    try:
        from paper.benchmark.adapters.store import _normalize_artifact_prefix

        _normalize_artifact_prefix(str(frame["ranking_artifact_prefix"].iloc[0]))
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"ranking_artifact_prefix is invalid: {exc}") from exc
    if not _ACCOUNT_ID_PATTERN.fullmatch(str(frame["ranking_aws_account_id"].iloc[0])):
        raise ArtifactValidationError(
            "metrics artifact ranking_aws_account_id must contain 12 decimal digits"
        )
    if not _IMAGE_DIGEST_PATTERN.fullmatch(str(frame["ranking_container_image"].iloc[0])):
        raise ArtifactValidationError(
            "metrics artifact ranking_container_image must be an immutable digest URI"
        )
    ranking_dataset_sha256 = str(frame["ranking_dataset_sha256"].iloc[0])
    if ranking_dataset_sha256 != config.dataset_identity.sha256:
        raise ArtifactValidationError(
            "metrics artifact ranking_dataset_sha256 does not match the manifest dataset"
        )
    if not _GIT_SHA_PATTERN.fullmatch(str(frame["ranking_git_sha"].iloc[0])):
        raise ArtifactValidationError(
            "metrics artifact ranking_git_sha must be a full hexadecimal commit ID"
        )
    if not _SHA256_PATTERN.fullmatch(str(frame["ranking_manifest_sha256"].iloc[0])):
        raise ArtifactValidationError(
            "metrics artifact ranking_manifest_sha256 must be 64 lowercase hexadecimal characters"
        )
    if not _SHA256_PATTERN.fullmatch(str(frame["ranking_payload_sha256"].iloc[0])):
        raise ArtifactValidationError(
            "metrics artifact ranking_payload_sha256 must be 64 lowercase hexadecimal characters"
        )

    numeric_metrics: dict[str, np.ndarray] = {}
    for column in metric_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ArtifactValidationError(f"metrics artifact contains non-finite {column} values")
        numeric_metrics[column] = values

    if config.task == "classification":
        for column, values in numeric_metrics.items():
            if ((values < 0.0) | (values > 1.0)).any():
                raise ArtifactValidationError(f"classification metric {column} must be in [0, 1]")
        if not np.array_equal(numeric_metrics["auc"], numeric_metrics["roc_auc"]):
            raise ArtifactValidationError("classification metrics auc and roc_auc must be equal")
        return

    for column in ("mae", "mse", "rmse"):
        if (numeric_metrics[column] < 0.0).any():
            raise ArtifactValidationError(f"regression metric {column} must be non-negative")
    if (numeric_metrics["r2"] > 1.0).any():
        raise ArtifactValidationError("regression metric r2 must be <= 1")
    tolerance = 64 * np.finfo(np.float64).eps
    if not np.isclose(
        np.square(numeric_metrics["rmse"]),
        numeric_metrics["mse"],
        rtol=tolerance,
        atol=tolerance,
    ).all():
        raise ArtifactValidationError("regression metrics rmse and mse are inconsistent")


def validate_metrics_ranking_payload(
    frame: pd.DataFrame,
    expected_sha256: str,
) -> None:
    """Require metrics to reference the exact ranking object consumed."""
    if not _SHA256_PATTERN.fullmatch(expected_sha256):
        raise ArtifactValidationError(
            "expected ranking payload SHA-256 must be 64 lowercase hexadecimal characters"
        )
    if "ranking_payload_sha256" not in frame:
        raise ArtifactValidationError("metrics artifact is missing ranking_payload_sha256")
    if not frame["ranking_payload_sha256"].eq(expected_sha256).all():
        raise ArtifactValidationError(
            "metrics artifact ranking_payload_sha256 does not match the ranking object"
        )
