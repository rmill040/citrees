"""Tests for isolated, immutable S3 benchmark artifact storage."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from pandas.testing import assert_frame_equal

from paper.benchmark.adapters.store import ArtifactReadError, S3Store, get_s3_bucket
from paper.benchmark.config.constants import PIPELINE_ARTIFACT_VERSION
from paper.benchmark.pipeline.types import DatasetIdentity, ExperimentConfig, MethodConfig
from paper.benchmark.pipeline.validation import derive_assignment_id

BUCKET = "citrees-test-bucket"
ARTIFACT_PREFIX = "repairs/r-baselines-v2/run-001"
REQUEST_ID = "1" * 32
WORKER_ID = "2" * 32


def _config(
    *,
    task: str = "classification",
    dataset: str = "wine",
    seed: int = 3,
) -> ExperimentConfig:
    method = MethodConfig(
        name="r_ctree",
        params=(("alpha", 0.05), ("testtype", "Bonferroni")),
    )
    return ExperimentConfig(
        method=method,
        dataset=dataset,
        seed=seed,
        task=task,  # type: ignore[arg-type]
        dataset_identity=DatasetIdentity("d" * 64, n_samples=50, n_features=4),
    )


def _store(*, validate_uploads: bool = False) -> tuple[S3Store, MagicMock]:
    store = S3Store(
        bucket=BUCKET,
        artifact_prefix=ARTIFACT_PREFIX,
        validate_uploads=validate_uploads,
    )
    client = MagicMock()
    store._client = client
    return store, client


def _expected_provenance() -> dict[str, str]:
    return {
        "artifact_prefix": ARTIFACT_PREFIX,
        "aws_account_id": "123456789012",
        "campaign_sha256": "e" * 64,
        "canonical_manifest_sha256": "c" * 64,
        "container_image": "repository@sha256:" + "a" * 64,
        "gate_receipt_sha256": "d" * 64,
        "git_sha": "a" * 40,
        "manifest_sha256": "b" * 64,
        "runtime_contract_sha256": "f" * 64,
    }


def _assignment_id(config: ExperimentConfig, *, attempt: int = 1) -> str:
    return derive_assignment_id(
        config,
        stage="rankings",
        attempt=attempt,
        expected_provenance=_expected_provenance(),
    )


def _attempt_receipt(
    config: ExperimentConfig,
    assignment_id: str,
    *,
    attempt: int = 1,
) -> dict[str, object]:
    return {
        **_expected_provenance(),
        "artifact_version": PIPELINE_ARTIFACT_VERSION,
        "assignment_id": assignment_id,
        "attempt": attempt,
        "created_at_utc": "2026-08-03T12:00:00+00:00",
        "dataset": config.dataset,
        "dataset_sha256": config.dataset_identity.sha256,
        "lease_seconds": 900,
        "method_base": config.method.name,
        "method_id": config.method.label,
        "method_params_json": '{"alpha":0.05,"testtype":"Bonferroni"}',
        "n_features": config.dataset_identity.n_features,
        "n_samples": config.dataset_identity.n_samples,
        "seed": config.seed,
        "stage": "rankings",
        "status": "assigned",
        "task": config.task,
        "request_id": REQUEST_ID,
        "worker_id": WORKER_ID,
    }


def _failure_receipt(
    config: ExperimentConfig,
    assignment_id: str,
    *,
    attempt: int = 1,
) -> dict[str, object]:
    receipt = _attempt_receipt(config, assignment_id, attempt=attempt)
    receipt.pop("lease_seconds")
    receipt.update(
        {
            "elapsed_seconds": 1.25,
            "error": "fit failed",
            "error_type": "RuntimeError",
            "hardware": {"logical_cpus": 8},
            "hostname": "worker-1",
            "library_versions": {"python": "3.12.7"},
            "status": "failed",
            "traceback": "RuntimeError: fit failed",
        }
    )
    return receipt


def _client_error(
    code: str,
    operation: str,
    *,
    status: int | None = None,
) -> ClientError:
    response: dict[str, Any] = {
        "Error": {
            "Code": code,
            "Message": f"{operation} failed",
        }
    }
    if status is not None:
        response["ResponseMetadata"] = {"HTTPStatusCode": status}
    return ClientError(response, operation)


def _parquet_payload(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("raw_prefix", "expected"),
    [
        ("repairs/r-baselines-v2/run-001", "repairs/r-baselines-v2/run-001"),
        ("  repairs/r-baselines-v2/run-001  ", "repairs/r-baselines-v2/run-001"),
        ("repairs/r-baselines-v2/run-001/", "repairs/r-baselines-v2/run-001"),
        ("repairs/r-baselines-v2/run-001///", "repairs/r-baselines-v2/run-001"),
        ("run-001", "run-001"),
    ],
)
def test_artifact_prefix_is_normalized(raw_prefix: str, expected: str) -> None:
    store = S3Store(bucket=BUCKET, artifact_prefix=raw_prefix)

    assert store.artifact_prefix == expected


@pytest.mark.parametrize(
    "artifact_prefix",
    [
        "",
        " ",
        "/repairs/run-001",
        "//repairs/run-001",
        ".",
        "..",
        "./repairs",
        "../repairs",
        "repairs/.",
        "repairs/..",
        "repairs/../run-001",
        "repairs/./run-001",
        "repairs//run-001",
        "s3://bucket/repairs",
        "C:/repairs/run-001",
        "C:\\repairs\\run-001",
        "repairs\\run-001",
        "repairs/\x00/run-001",
        "repairs/\n/run-001",
    ],
)
def test_invalid_artifact_prefix_is_rejected(artifact_prefix: str) -> None:
    with pytest.raises(ValueError, match="artifact_prefix"):
        S3Store(bucket=BUCKET, artifact_prefix=artifact_prefix)


@pytest.mark.parametrize("artifact_prefix", [None, Path("repairs/run-001"), 123])
def test_non_string_artifact_prefix_is_rejected(artifact_prefix: object) -> None:
    with pytest.raises(TypeError, match="artifact_prefix must be a string"):
        S3Store(
            bucket=BUCKET,
            artifact_prefix=artifact_prefix,  # type: ignore[arg-type]
        )


def test_from_env_requires_and_normalizes_artifact_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("S3_BUCKET", BUCKET)
    monkeypatch.setenv("CITREES_ARTIFACT_PREFIX", "  repairs/run-002/ ")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")

    store = S3Store.from_env(validate_uploads=True)

    assert store.bucket == BUCKET
    assert store.artifact_prefix == "repairs/run-002"
    assert store.region == "us-west-2"
    assert store.validate_uploads is True


def test_from_env_rejects_missing_artifact_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S3_BUCKET", BUCKET)
    monkeypatch.delenv("CITREES_ARTIFACT_PREFIX", raising=False)

    with pytest.raises(RuntimeError, match="CITREES_ARTIFACT_PREFIX is required"):
        S3Store.from_env()


def test_data_bucket_lookup_is_independent_of_artifact_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("S3_BUCKET", BUCKET)
    monkeypatch.setenv("CITREES_ARTIFACT_PREFIX", ARTIFACT_PREFIX)

    assert get_s3_bucket() == BUCKET


def test_pickle_round_trip_preserves_normalized_artifact_prefix() -> None:
    store = S3Store(
        bucket=BUCKET,
        artifact_prefix=f" {ARTIFACT_PREFIX}/ ",
        region="us-west-2",
        validate_uploads=True,
    )

    restored = pickle.loads(pickle.dumps(store))

    assert restored.bucket == BUCKET
    assert restored.artifact_prefix == ARTIFACT_PREFIX
    assert restored.region == "us-west-2"
    assert restored.validate_uploads is True
    assert restored._client is None


@pytest.mark.parametrize("stage", ["rankings", "metrics"])
def test_key_and_path_include_artifact_prefix(stage: str) -> None:
    store, _client = _store()
    config = _config()
    expected_key = (
        f"{ARTIFACT_PREFIX}/{stage}/classification/wine/{config.method.label}_seed3.parquet"
    )

    assert store.artifact_key(stage, config) == expected_key  # type: ignore[arg-type]
    assert store._path(stage, config) == f"s3://{BUCKET}/{expected_key}"  # type: ignore[arg-type]


def test_exists_uses_prefixed_key() -> None:
    store, client = _store()
    config = _config()

    assert store.exists("rankings", config) is True

    client.head_object.assert_called_once_with(
        Bucket=BUCKET,
        Key=(f"{ARTIFACT_PREFIX}/rankings/classification/wine/{config.method.label}_seed3.parquet"),
    )


@pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound"])
def test_exists_returns_false_only_for_missing_objects(code: str) -> None:
    store, client = _store()
    client.head_object.side_effect = _client_error(code, "HeadObject", status=404)

    assert store.exists("rankings", _config()) is False


def test_exists_propagates_unexpected_s3_errors() -> None:
    store, client = _store()
    error = _client_error("AccessDenied", "HeadObject", status=403)
    client.head_object.side_effect = error

    with pytest.raises(ClientError) as exc_info:
        store.exists("rankings", _config())

    assert exc_info.value is error


@pytest.mark.parametrize("stage", ["rankings", "metrics"])
def test_save_is_prefixed_immutable_and_round_trips_payload(stage: str) -> None:
    store, client = _store()
    config = _config()
    frame = pd.DataFrame({"fold_idx": [0, 1], "score": [0.25, 0.75]})
    expected_key = (
        f"{ARTIFACT_PREFIX}/{stage}/classification/wine/{config.method.label}_seed3.parquet"
    )

    path = store.save(stage, config, frame)  # type: ignore[arg-type]

    call = client.put_object.call_args
    assert call.kwargs["Bucket"] == BUCKET
    assert call.kwargs["Key"] == expected_key
    assert call.kwargs["IfNoneMatch"] == "*"
    assert_frame_equal(pd.read_parquet(io.BytesIO(call.kwargs["Body"])), frame)
    assert path == f"s3://{BUCKET}/{expected_key}"


def test_save_validation_reads_exact_prefixed_object_bytes() -> None:
    store, client = _store(validate_uploads=True)
    config = _config()
    frame = pd.DataFrame({"value": [1, 2, 3]})
    expected_key = (
        f"{ARTIFACT_PREFIX}/rankings/classification/wine/{config.method.label}_seed3.parquet"
    )

    def get_object(**kwargs: object) -> dict[str, io.BytesIO]:
        assert kwargs == {"Bucket": BUCKET, "Key": expected_key}
        payload = client.put_object.call_args.kwargs["Body"]
        return {"Body": io.BytesIO(payload)}

    client.get_object.side_effect = get_object

    store.save("rankings", config, frame)

    client.get_object.assert_called_once_with(Bucket=BUCKET, Key=expected_key)


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("PreconditionFailed", 412),
        ("412", 400),
        ("Unknown", 412),
    ],
)
def test_save_converts_precondition_failure_to_file_exists(
    code: str,
    status: int,
) -> None:
    store, client = _store()
    config = _config()
    client.put_object.side_effect = _client_error(code, "PutObject", status=status)
    client.get_object.return_value = {"Body": io.BytesIO(b"different payload")}

    with pytest.raises(
        FileExistsError, match="Refusing to overwrite existing artifact"
    ) as exc_info:
        store.save("metrics", config, pd.DataFrame({"value": [1]}))

    assert (
        f"s3://{BUCKET}/{ARTIFACT_PREFIX}/metrics/classification/wine/"
        f"{config.method.label}_seed3.parquet"
    ) in str(exc_info.value)


def test_save_propagates_non_precondition_s3_errors() -> None:
    store, client = _store()
    error = _client_error("AccessDenied", "PutObject", status=403)
    client.put_object.side_effect = error
    client.get_object.side_effect = _client_error("NoSuchKey", "GetObject", status=404)

    with pytest.raises(ClientError) as exc_info:
        store.save("rankings", _config(), pd.DataFrame({"value": [1]}))

    assert exc_info.value is error


def test_save_reconciles_exact_payload_after_ambiguous_commit() -> None:
    store, client = _store(validate_uploads=True)
    config = _config()
    frame = pd.DataFrame({"value": [1, 2, 3]})
    stored: dict[str, bytes] = {}

    def commit_then_disconnect(**kwargs: object) -> None:
        stored[str(kwargs["Key"])] = bytes(kwargs["Body"])  # type: ignore[arg-type]
        raise EndpointConnectionError(endpoint_url="https://s3.test")

    client.put_object.side_effect = commit_then_disconnect
    client.get_object.side_effect = lambda **kwargs: {
        "Body": io.BytesIO(stored[str(kwargs["Key"])])
    }

    path = store.save("rankings", config, frame)

    assert path == store._path("rankings", config)
    assert_frame_equal(store.load("rankings", config), frame)


def test_save_write_guard_fences_a_lost_assignment() -> None:
    store, client = _store()
    store.bind_write_guard(lambda: False)

    with pytest.raises(RuntimeError, match="lease was lost"):
        store.save("rankings", _config(), pd.DataFrame({"value": [1]}))

    client.put_object.assert_not_called()


def test_attempt_receipt_is_json_prefixed_and_immutable() -> None:
    store, client = _store()
    config = _config()
    assignment_id = _assignment_id(config)
    receipt = _attempt_receipt(config, assignment_id)
    expected_key = (
        f"{ARTIFACT_PREFIX}/attempts/rankings/classification/wine/"
        f"{config.method.label}_seed3/{assignment_id}.json"
    )

    path = store.save_attempt(
        "rankings",
        config,
        assignment_id,
        receipt,
        expected_provenance=_expected_provenance(),
    )

    call = client.put_object.call_args
    assert call.kwargs["Bucket"] == BUCKET
    assert call.kwargs["Key"] == expected_key
    assert call.kwargs["ContentType"] == "application/json"
    assert call.kwargs["IfNoneMatch"] == "*"
    assert json.loads(call.kwargs["Body"]) == receipt
    assert path == f"s3://{BUCKET}/{expected_key}"


def test_failure_receipt_is_json_prefixed_and_immutable() -> None:
    store, client = _store()
    config = _config()
    assignment_id = _assignment_id(config)
    receipt = _failure_receipt(config, assignment_id)
    expected_key = (
        f"{ARTIFACT_PREFIX}/failures/rankings/classification/wine/"
        f"{config.method.label}_seed3/{assignment_id}.json"
    )

    path = store.save_failure(
        "rankings",
        config,
        assignment_id,
        receipt,
        expected_provenance=_expected_provenance(),
    )

    call = client.put_object.call_args
    assert call.kwargs["Bucket"] == BUCKET
    assert call.kwargs["Key"] == expected_key
    assert call.kwargs["ContentType"] == "application/json"
    assert call.kwargs["IfNoneMatch"] == "*"
    assert json.loads(call.kwargs["Body"]) == receipt
    assert path == f"s3://{BUCKET}/{expected_key}"


def test_failure_receipt_rejects_invalid_assignment_id() -> None:
    store, _client = _store()
    config = _config()
    assignment_id = "../escape"

    with pytest.raises(ValueError, match="assignment_id"):
        store.save_failure(
            "rankings",
            config,
            assignment_id,
            _failure_receipt(config, assignment_id),
            expected_provenance=_expected_provenance(),
        )


def test_failure_receipt_refuses_overwrite() -> None:
    store, client = _store()
    config = _config()
    assignment_id = _assignment_id(config)
    client.put_object.side_effect = _client_error(
        "PreconditionFailed",
        "PutObject",
        status=412,
    )

    with pytest.raises(FileExistsError, match="control receipt"):
        store.save_failure(
            "rankings",
            config,
            assignment_id,
            _failure_receipt(config, assignment_id),
            expected_provenance=_expected_provenance(),
        )


def test_failure_receipt_rejects_incomplete_payload() -> None:
    store, _client = _store()
    config = _config()
    assignment_id = _assignment_id(config)
    receipt = _failure_receipt(config, assignment_id)
    receipt.pop("traceback")

    with pytest.raises(ValueError, match="fields differ from the contract"):
        store.save_failure(
            "rankings",
            config,
            assignment_id,
            receipt,
            expected_provenance=_expected_provenance(),
        )


def test_load_failure_uses_exact_assignment_key() -> None:
    store, client = _store()
    config = _config()
    assignment_id = _assignment_id(config)
    receipt = _failure_receipt(config, assignment_id)
    expected_key = (
        f"{ARTIFACT_PREFIX}/failures/rankings/classification/wine/"
        f"{config.method.label}_seed3/{assignment_id}.json"
    )
    client.get_object.return_value = {"Body": io.BytesIO(json.dumps(receipt).encode("utf-8"))}

    assert store.load_failure("rankings", config, assignment_id) == receipt
    client.get_object.assert_called_once_with(Bucket=BUCKET, Key=expected_key)

    client.get_object.reset_mock()
    client.get_object.side_effect = _client_error("NoSuchKey", "GetObject", status=404)
    with pytest.raises(FileNotFoundError, match="Control receipt not found"):
        store.load_failure("rankings", config, assignment_id)


def test_load_attempt_uses_exact_assignment_key() -> None:
    store, client = _store()
    config = _config()
    assignment_id = _assignment_id(config)
    receipt = _attempt_receipt(config, assignment_id)
    expected_key = (
        f"{ARTIFACT_PREFIX}/attempts/rankings/classification/wine/"
        f"{config.method.label}_seed3/{assignment_id}.json"
    )
    client.get_object.return_value = {"Body": io.BytesIO(json.dumps(receipt).encode("utf-8"))}

    assert store.load_attempt("rankings", config, assignment_id) == receipt
    client.get_object.assert_called_once_with(Bucket=BUCKET, Key=expected_key)


@pytest.mark.parametrize("stage", ["rankings", "metrics"])
def test_load_uses_prefixed_key_and_returns_frame(stage: str) -> None:
    store, client = _store()
    config = _config()
    frame = pd.DataFrame({"fold_idx": [0, 1], "score": [0.4, 0.6]})
    client.get_object.return_value = {"Body": io.BytesIO(_parquet_payload(frame))}
    expected_key = (
        f"{ARTIFACT_PREFIX}/{stage}/classification/wine/{config.method.label}_seed3.parquet"
    )

    loaded = store.load(stage, config)  # type: ignore[arg-type]

    assert_frame_equal(loaded, frame)
    client.get_object.assert_called_once_with(Bucket=BUCKET, Key=expected_key)


def test_load_with_payload_sha256_hashes_exact_parquet_bytes() -> None:
    store, client = _store()
    config = _config()
    frame = pd.DataFrame({"fold_idx": [0, 1], "score": [0.4, 0.6]})
    payload = _parquet_payload(frame)
    client.get_object.return_value = {"Body": io.BytesIO(payload)}

    loaded = store.load_with_payload_sha256("rankings", config)

    assert_frame_equal(loaded.frame, frame)
    assert loaded.payload_sha256 == hashlib.sha256(payload).hexdigest()


def test_load_converts_missing_object_to_file_not_found() -> None:
    store, client = _store()
    config = _config()
    client.get_object.side_effect = _client_error("NoSuchKey", "GetObject", status=404)

    with pytest.raises(FileNotFoundError, match="Artifact not found") as exc_info:
        store.load("rankings", config)

    assert (
        f"s3://{BUCKET}/{ARTIFACT_PREFIX}/rankings/classification/wine/"
        f"{config.method.label}_seed3.parquet"
    ) in str(exc_info.value)


def test_load_propagates_unexpected_s3_errors() -> None:
    store, client = _store()
    error = _client_error("AccessDenied", "GetObject", status=403)
    client.get_object.side_effect = error

    with pytest.raises(ClientError) as exc_info:
        store.load("rankings", _config())

    assert exc_info.value is error


def test_load_classifies_invalid_parquet_payload() -> None:
    store, client = _store()
    client.get_object.return_value = {"Body": io.BytesIO(b"not parquet")}

    with pytest.raises(ArtifactReadError, match="not readable Parquet"):
        store.load("rankings", _config())


@pytest.mark.parametrize("stage", ["rankings", "metrics"])
def test_list_completed_is_isolated_to_prefixed_stage_and_task(stage: str) -> None:
    store, client = _store()
    config = _config()
    stage_prefix = f"{ARTIFACT_PREFIX}/{stage}/classification/"
    valid_key = f"{stage_prefix}wine/{config.method.label}_seed3.parquet"
    second_key = f"{stage_prefix}glass/r_cforest__abc123_seed0.parquet"
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": valid_key},
                {"Key": second_key},
                {"Key": valid_key},
                {"Key": (f"{stage}/classification/wine/{config.method.label}_seed3.parquet")},
                {
                    "Key": (
                        f"repairs/other-run/{stage}/classification/wine/"
                        f"{config.method.label}_seed3.parquet"
                    )
                },
                {
                    "Key": (
                        f"{ARTIFACT_PREFIX}/{stage}/regression/wine/"
                        f"{config.method.label}_seed3.parquet"
                    )
                },
                {"Key": f"{stage_prefix}wine/nested/r_ctree__abc_seed1.parquet"},
                {"Key": f"{stage_prefix}wine/r_ctree__abc_seedx.parquet"},
                {"Key": f"{stage_prefix}wine/r_ctree__abc_seed1.csv"},
                {"Key": f"{stage_prefix}/r_ctree__abc_seed1.parquet"},
                {"Key": stage_prefix},
                {},
            ]
        },
        {},
    ]
    client.get_paginator.return_value = paginator

    completed = store.list_completed(stage, "classification")  # type: ignore[arg-type]

    assert completed == {
        ("classification", "wine", config.method.label, 3),
        ("classification", "glass", "r_cforest__abc123", 0),
    }
    client.get_paginator.assert_called_once_with("list_objects_v2")
    paginator.paginate.assert_called_once_with(Bucket=BUCKET, Prefix=stage_prefix)


@pytest.mark.parametrize("stage", ["rankings", "metrics"])
def test_list_stage_keys_preserves_every_key_for_strict_reconciliation(stage: str) -> None:
    store, client = _store()
    prefix = f"{ARTIFACT_PREFIX}/{stage}/"
    valid = f"{prefix}classification/wine/rf__abc_seed0.parquet"
    malformed = f"{prefix}classification/wine/nested/object.parquet"
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": malformed},
                {"Key": valid},
                {"Key": valid},
                {"Key": f"{ARTIFACT_PREFIX}/other/object"},
                {},
            ]
        }
    ]
    client.get_paginator.return_value = paginator

    assert store.list_stage_keys(stage) == tuple(sorted((valid, malformed)))  # type: ignore[arg-type]
    paginator.paginate.assert_called_once_with(Bucket=BUCKET, Prefix=prefix)
