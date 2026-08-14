"""Tests for provenance-safe canonical ranking materialization."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pandas as pd
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from pandas.testing import assert_frame_equal

from paper.benchmark.config.constants import PIPELINE_ARTIFACT_VERSION
from paper.benchmark.pipeline.manifest import (
    ManifestCell,
    RerunManifest,
    compute_campaign_sha256,
)
from paper.benchmark.pipeline.materialize import (
    RankingMaterializationError,
    RankingSource,
    materialize_canonical_rankings,
)
from paper.benchmark.pipeline.types import DatasetIdentity, ExperimentConfig, MethodConfig
from paper.benchmark.pipeline.validation import (
    validate_artifact_provenance,
    validate_ranking_artifact,
)

pytestmark = pytest.mark.paper

TARGET_BUCKET = "citrees-target-bucket"
TARGET_PREFIX = "canonical/stage2-inputs/run-001"
TARGET_MANIFEST_SHA256 = "b" * 64
TARGET_CANONICAL_MANIFEST_SHA256 = "c" * 64
TARGET_RUNTIME_SHA256 = "f" * 64
SOURCE_ACCOUNT_ID = "210987654321"
TARGET_ACCOUNT_ID = "123456789012"


def _client_error(code: str, operation: str, *, status: int) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": f"{operation} failed"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        operation,
    )


class _VersionedMemoryS3:
    """Small version-aware S3 fake with conditional immutable writes."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.get_calls: list[dict[str, Any]] = []
        self.put_calls: list[dict[str, Any]] = []
        self._next_version = 1

    def seed(
        self,
        *,
        bucket: str,
        key: str,
        payload: bytes,
        version_id: str,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self.objects.setdefault((bucket, key), []).append(
            {
                "payload": payload,
                "version_id": version_id,
                "metadata": dict(metadata or {}),
            }
        )

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        self.get_calls.append(dict(kwargs))
        versions = self.objects.get((str(kwargs["Bucket"]), str(kwargs["Key"])), [])
        requested_version = kwargs.get("VersionId")
        selected: dict[str, Any] | None
        if requested_version is None:
            if not versions:
                raise _client_error("NoSuchKey", "GetObject", status=404)
            selected = versions[-1]
        else:
            selected = next(
                (version for version in versions if version["version_id"] == requested_version),
                None,
            )
            if selected is None:
                raise _client_error("NoSuchVersion", "GetObject", status=404)
        return {
            "Body": io.BytesIO(selected["payload"]),
            "Metadata": dict(selected["metadata"]),
            "VersionId": selected["version_id"],
        }

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.put_calls.append(dict(kwargs))
        identity = (str(kwargs["Bucket"]), str(kwargs["Key"]))
        if kwargs.get("IfNoneMatch") != "*":
            raise AssertionError("materialization writes must be conditional")
        if self.objects.get(identity):
            raise _client_error("PreconditionFailed", "PutObject", status=412)
        version_id = f"target-version-{self._next_version}"
        self._next_version += 1
        self.seed(
            bucket=identity[0],
            key=identity[1],
            payload=bytes(kwargs["Body"]),
            version_id=version_id,
            metadata=kwargs.get("Metadata"),
        )
        return {"VersionId": version_id}


class _CommitThenDisconnectS3(_VersionedMemoryS3):
    """Commit the first ranking write, then lose the client response."""

    def __init__(self) -> None:
        super().__init__()
        self.disconnected = False

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        response = super().put_object(**kwargs)
        if kwargs["ContentType"] == "application/vnd.apache.parquet" and not self.disconnected:
            self.disconnected = True
            raise EndpointConnectionError(endpoint_url="https://s3.test")
        return response


class _DeniedWriteS3(_VersionedMemoryS3):
    """Reject every target write with a non-ambiguous permission error."""

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.put_calls.append(dict(kwargs))
        raise _client_error("AccessDenied", "PutObject", status=403)


def _config(seed: int) -> ExperimentConfig:
    return ExperimentConfig(
        method=MethodConfig("rf", params=(("n_estimators", 100),)),
        dataset="fixture",
        seed=seed,
        task="classification",
        dataset_identity=DatasetIdentity("d" * 64, n_samples=50, n_features=4),
    )


def _manifest(
    *,
    required_seeds: tuple[int, ...] = (0,),
    include_blacklisted: bool = False,
) -> RerunManifest:
    cells = [
        ManifestCell(
            config=_config(seed),
            target_aws_account_id=TARGET_ACCOUNT_ID,
            dataset_source="real",
            rerun_reason="canonical_stage2_input",
            historically_omitted=False,
            stage1_required=False,
            stage2_required=True,
        )
        for seed in required_seeds
    ]
    if include_blacklisted:
        cells.append(
            ManifestCell(
                config=_config(max(required_seeds, default=-1) + 1),
                target_aws_account_id=TARGET_ACCOUNT_ID,
                dataset_source="real",
                rerun_reason="blacklisted",
                historically_omitted=False,
                stage1_required=False,
                stage2_required=False,
            )
        )
    frozen_cells = tuple(cells)
    return RerunManifest(
        sha256=TARGET_MANIFEST_SHA256,
        campaign_sha256=compute_campaign_sha256(
            frozen_cells,
            runtime_contract_sha256=TARGET_RUNTIME_SHA256,
        ),
        runtime_contract_sha256=TARGET_RUNTIME_SHA256,
        cells=frozen_cells,
    )


def _provenance(
    *,
    artifact_prefix: str,
    aws_account_id: str,
    campaign_sha256: str,
    manifest_sha256: str,
    runtime_contract_sha256: str,
) -> dict[str, str]:
    return {
        "artifact_prefix": artifact_prefix,
        "aws_account_id": aws_account_id,
        "campaign_sha256": campaign_sha256,
        "canonical_manifest_sha256": TARGET_CANONICAL_MANIFEST_SHA256,
        "container_image": "repository@sha256:" + "a" * 64,
        "gate_receipt_sha256": "e" * 64,
        "git_sha": "1" * 40,
        "manifest_sha256": manifest_sha256,
        "runtime_contract_sha256": runtime_contract_sha256,
    }


def _target_provenance(manifest: RerunManifest) -> dict[str, str]:
    return _provenance(
        artifact_prefix=TARGET_PREFIX,
        aws_account_id=TARGET_ACCOUNT_ID,
        campaign_sha256=manifest.campaign_sha256,
        manifest_sha256=manifest.sha256,
        runtime_contract_sha256=manifest.runtime_contract_sha256,
    )


def _source_provenance() -> dict[str, str]:
    return _provenance(
        artifact_prefix="repairs/source-run",
        aws_account_id=SOURCE_ACCOUNT_ID,
        campaign_sha256="2" * 64,
        manifest_sha256="3" * 64,
        runtime_contract_sha256="4" * 64,
    )


def _rankings(
    config: ExperimentConfig,
    provenance: Mapping[str, str],
) -> pd.DataFrame:
    common = {
        "artifact_version": PIPELINE_ARTIFACT_VERSION,
        **provenance,
        "created_at_utc": "2026-08-10T12:00:00+00:00",
        "dataset": config.dataset,
        "dataset_sha256": config.dataset_identity.sha256,
        "dataset_source": "real",
        "dataset_type": "tabular",
        "elapsed_seconds": 12.5,
        "hardware": {"logical_cpus": 4, "cpu_affinity": [0, 1, 2, 3]},
        "library_versions": {"citrees": "0.1.0", "python": "3.12.7"},
        "method": config.method.label,
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
        "task": config.task,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "feature_ranking": [fold, *[value for value in range(4) if value != fold]]
                if fold < 4
                else [0, 1, 2, 3],
                "fold_cpu_affinity": [0, 1, 2, 3],
                "fold_idx": fold,
                "fold_random_state": config.seed * 1000 + fold,
            }
            for fold in range(5)
        ]
    )


def _parquet_payload(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _source(
    *,
    client: _VersionedMemoryS3,
    config: ExperimentConfig,
    version_id: str = "source-version-1",
) -> tuple[RankingSource, pd.DataFrame, bytes]:
    provenance = _source_provenance()
    key = (
        f"{provenance['artifact_prefix']}/rankings/{config.task}/{config.dataset}/"
        f"{config.method.label}_seed{config.seed}.parquet"
    )
    frame = _rankings(config, provenance)
    payload = _parquet_payload(frame)
    bucket = "citrees-source-bucket"
    client.seed(
        bucket=bucket,
        key=key,
        payload=payload,
        version_id=version_id,
    )
    return (
        RankingSource(
            cell_key=config.key,
            source_aws_account_id=SOURCE_ACCOUNT_ID,
            bucket=bucket,
            key=key,
            version_id=version_id,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            expected_provenance=provenance,
        ),
        frame,
        payload,
    )


def _target_key(manifest: RerunManifest, config: ExperimentConfig) -> str:
    provenance = _target_provenance(manifest)
    return (
        f"{provenance['artifact_prefix']}/rankings/{config.task}/{config.dataset}/"
        f"{config.method.label}_seed{config.seed}.parquet"
    )


def test_materializes_exact_inventory_with_lineage_and_idempotent_receipt() -> None:
    client = _VersionedMemoryS3()
    manifest = _manifest()
    config = manifest.cells[0].config
    source, source_frame, source_payload = _source(client=client, config=config)

    first = materialize_canonical_rankings(
        sources=[source],
        target_manifest=manifest,
        target_provenance=_target_provenance(manifest),
        target_bucket=TARGET_BUCKET,
        s3_client=client,
    )
    writes_after_first = len(client.put_calls)
    second = materialize_canonical_rankings(
        sources=[source],
        target_manifest=manifest,
        target_provenance=_target_provenance(manifest),
        target_bucket=TARGET_BUCKET,
        s3_client=client,
    )

    assert first == second
    assert len(client.put_calls) == writes_after_first + 2
    assert first.receipt["entry_count"] == 1
    assert first.receipt_payload_sha256 == hashlib.sha256(first.receipt_payload).hexdigest()
    assert json.loads(first.receipt_payload) == first.receipt
    assert first.receipt_key == f"{TARGET_PREFIX}/materialization/rankings-receipt.json"
    entry = first.receipt["entries"][0]
    assert entry["source"] == {
        "aws_account_id": SOURCE_ACCOUNT_ID,
        "bucket": source.bucket,
        "key": source.key,
        "payload_sha256": hashlib.sha256(source_payload).hexdigest(),
        "provenance": _source_provenance(),
        "version_id": source.version_id,
    }

    target = entry["target"]
    response = client.get_object(
        Bucket=target["bucket"],
        Key=target["key"],
        VersionId=target["version_id"],
    )
    target_payload = response["Body"].read()
    assert hashlib.sha256(target_payload).hexdigest() == target["payload_sha256"]
    target_frame = pd.read_parquet(io.BytesIO(target_payload))
    validate_ranking_artifact(target_frame, config)
    validate_artifact_provenance(target_frame, _target_provenance(manifest))

    active_provenance = set(_target_provenance(manifest))
    preserved_columns = [
        column for column in source_frame.columns if column not in active_provenance
    ]
    assert_frame_equal(
        target_frame[preserved_columns],
        source_frame[preserved_columns],
        check_dtype=False,
    )
    assert target_frame["source_bucket"].unique().tolist() == [source.bucket]
    assert target_frame["source_key"].unique().tolist() == [source.key]
    assert target_frame["source_version_id"].unique().tolist() == [source.version_id]
    assert target_frame["source_payload_sha256"].unique().tolist() == [source.payload_sha256]
    assert target_frame["source_campaign_sha256"].unique().tolist() == ["2" * 64]

    target_writes = [call for call in client.put_calls if call["Key"] == target["key"]]
    assert len(target_writes) == 2
    assert target_writes[0]["Body"] == target_writes[1]["Body"]
    assert target_writes[0]["IfNoneMatch"] == "*"
    assert len(client.objects[(TARGET_BUCKET, target["key"])]) == 1
    assert len(client.objects[(TARGET_BUCKET, first.receipt_key)]) == 1


def test_rejects_missing_duplicate_and_nonrequired_sources_before_s3_reads() -> None:
    client = _VersionedMemoryS3()
    manifest = _manifest(include_blacklisted=True)
    required, _frame, _payload = _source(client=client, config=manifest.cells[0].config)
    blacklisted, _frame, _payload = _source(
        client=client,
        config=manifest.cells[1].config,
        version_id="source-version-2",
    )

    with pytest.raises(RankingMaterializationError, match="missing"):
        materialize_canonical_rankings(
            sources=[],
            target_manifest=manifest,
            target_provenance=_target_provenance(manifest),
            target_bucket=TARGET_BUCKET,
            s3_client=client,
        )
    with pytest.raises(RankingMaterializationError, match="duplicate"):
        materialize_canonical_rankings(
            sources=[required, required],
            target_manifest=manifest,
            target_provenance=_target_provenance(manifest),
            target_bucket=TARGET_BUCKET,
            s3_client=client,
        )
    with pytest.raises(RankingMaterializationError, match="non-required"):
        materialize_canonical_rankings(
            sources=[required, blacklisted],
            target_manifest=manifest,
            target_provenance=_target_provenance(manifest),
            target_bucket=TARGET_BUCKET,
            s3_client=client,
        )
    assert client.get_calls == []


def test_reads_only_requested_source_version_and_rejects_payload_drift() -> None:
    client = _VersionedMemoryS3()
    manifest = _manifest()
    source, _frame, _payload = _source(client=client, config=manifest.cells[0].config)
    client.seed(
        bucket=source.bucket,
        key=source.key,
        payload=b"newer but unapproved payload",
        version_id="source-version-2",
    )

    materialize_canonical_rankings(
        sources=[source],
        target_manifest=manifest,
        target_provenance=_target_provenance(manifest),
        target_bucket=TARGET_BUCKET,
        s3_client=client,
    )

    source_reads = [
        call
        for call in client.get_calls
        if call["Bucket"] == source.bucket and call["Key"] == source.key
    ]
    assert source_reads == [
        {"Bucket": source.bucket, "Key": source.key, "VersionId": source.version_id}
    ]

    drifted = RankingSource(
        cell_key=source.cell_key,
        source_aws_account_id=source.source_aws_account_id,
        bucket=source.bucket,
        key=source.key,
        version_id=source.version_id,
        payload_sha256="0" * 64,
        expected_provenance=source.expected_provenance,
    )
    with pytest.raises(RankingMaterializationError, match="payload SHA-256"):
        materialize_canonical_rankings(
            sources=[drifted],
            target_manifest=manifest,
            target_provenance=_target_provenance(manifest),
            target_bucket="another-target-bucket",
            s3_client=client,
        )


def test_missing_requested_source_version_never_falls_back_to_current() -> None:
    client = _VersionedMemoryS3()
    manifest = _manifest()
    source, _frame, _payload = _source(client=client, config=manifest.cells[0].config)
    missing_version = replace(source, version_id="missing-version")

    with pytest.raises(RankingMaterializationError, match="NoSuchVersion"):
        materialize_canonical_rankings(
            sources=[missing_version],
            target_manifest=manifest,
            target_provenance=_target_provenance(manifest),
            target_bucket=TARGET_BUCKET,
            s3_client=client,
        )

    assert client.get_calls == [
        {
            "Bucket": source.bucket,
            "Key": source.key,
            "VersionId": "missing-version",
        }
    ]
    assert client.put_calls == []


def test_source_provenance_mismatch_stops_before_target_writes() -> None:
    client = _VersionedMemoryS3()
    manifest = _manifest()
    source, _frame, _payload = _source(client=client, config=manifest.cells[0].config)
    wrong_provenance = {
        **source.expected_provenance,
        "manifest_sha256": "9" * 64,
    }

    with pytest.raises(RankingMaterializationError, match="source ranking validation"):
        materialize_canonical_rankings(
            sources=[replace(source, expected_provenance=wrong_provenance)],
            target_manifest=manifest,
            target_provenance=_target_provenance(manifest),
            target_bucket=TARGET_BUCKET,
            s3_client=client,
        )

    assert client.put_calls == []


def test_malformed_source_ranking_stops_before_target_writes() -> None:
    client = _VersionedMemoryS3()
    manifest = _manifest()
    source, frame, _payload = _source(client=client, config=manifest.cells[0].config)
    malformed = frame.copy(deep=True)
    malformed.at[0, "feature_ranking"] = [0, 0, 1, 2]
    malformed_payload = _parquet_payload(malformed)
    malformed_version = "malformed-source-version"
    client.seed(
        bucket=source.bucket,
        key=source.key,
        payload=malformed_payload,
        version_id=malformed_version,
    )

    with pytest.raises(RankingMaterializationError, match="source ranking validation"):
        materialize_canonical_rankings(
            sources=[
                replace(
                    source,
                    version_id=malformed_version,
                    payload_sha256=hashlib.sha256(malformed_payload).hexdigest(),
                )
            ],
            target_manifest=manifest,
            target_provenance=_target_provenance(manifest),
            target_bucket=TARGET_BUCKET,
            s3_client=client,
        )

    assert client.put_calls == []


def test_existing_different_target_bytes_are_never_overwritten_or_receipted() -> None:
    client = _VersionedMemoryS3()
    manifest = _manifest()
    config = manifest.cells[0].config
    source, _frame, _payload = _source(client=client, config=config)
    target_key = _target_key(manifest, config)
    client.seed(
        bucket=TARGET_BUCKET,
        key=target_key,
        payload=b"different target bytes",
        version_id="existing-target-version",
        metadata={"record-kind": "canonical-ranking", "sha256": "0" * 64},
    )

    with pytest.raises(FileExistsError, match="stored bytes differ"):
        materialize_canonical_rankings(
            sources=[source],
            target_manifest=manifest,
            target_provenance=_target_provenance(manifest),
            target_bucket=TARGET_BUCKET,
            s3_client=client,
        )

    assert client.objects[(TARGET_BUCKET, target_key)][0]["payload"] == b"different target bytes"
    receipt_key = f"{TARGET_PREFIX}/materialization/rankings-receipt.json"
    assert (TARGET_BUCKET, receipt_key) not in client.objects


def test_ambiguous_transport_commit_is_accepted_only_after_exact_readback() -> None:
    client = _CommitThenDisconnectS3()
    manifest = _manifest()
    source, _frame, _payload = _source(client=client, config=manifest.cells[0].config)

    result = materialize_canonical_rankings(
        sources=[source],
        target_manifest=manifest,
        target_provenance=_target_provenance(manifest),
        target_bucket=TARGET_BUCKET,
        s3_client=client,
    )

    target = result.receipt["entries"][0]["target"]
    assert target["version_id"] == "target-version-1"
    assert client.disconnected is True


def test_nonambiguous_write_error_propagates_without_target_fallback() -> None:
    client = _DeniedWriteS3()
    manifest = _manifest()
    source, _frame, _payload = _source(client=client, config=manifest.cells[0].config)

    with pytest.raises(ClientError, match="AccessDenied"):
        materialize_canonical_rankings(
            sources=[source],
            target_manifest=manifest,
            target_provenance=_target_provenance(manifest),
            target_bucket=TARGET_BUCKET,
            s3_client=client,
        )

    assert client.get_calls == [
        {
            "Bucket": source.bucket,
            "Key": source.key,
            "VersionId": source.version_id,
        }
    ]


def test_unversioned_source_records_the_observed_current_version() -> None:
    client = _VersionedMemoryS3()
    manifest = _manifest()
    source, _frame, _payload = _source(client=client, config=manifest.cells[0].config)

    result = materialize_canonical_rankings(
        sources=[replace(source, version_id=None)],
        target_manifest=manifest,
        target_provenance=_target_provenance(manifest),
        target_bucket=TARGET_BUCKET,
        s3_client=client,
    )

    assert result.receipt["entries"][0]["source"]["version_id"] == "source-version-1"
    assert client.get_calls[0] == {"Bucket": source.bucket, "Key": source.key}


def test_receipt_order_is_canonical_independent_of_source_sequence() -> None:
    manifest = _manifest(required_seeds=(0, 2))

    first_client = _VersionedMemoryS3()
    first_sources = [
        _source(
            client=first_client,
            config=cell.config,
            version_id=f"source-version-{cell.config.seed}",
        )[0]
        for cell in manifest.cells
    ]
    first = materialize_canonical_rankings(
        sources=list(reversed(first_sources)),
        target_manifest=manifest,
        target_provenance=_target_provenance(manifest),
        target_bucket=TARGET_BUCKET,
        s3_client=first_client,
    )

    second_client = _VersionedMemoryS3()
    second_sources = [
        _source(
            client=second_client,
            config=cell.config,
            version_id=f"source-version-{cell.config.seed}",
        )[0]
        for cell in manifest.cells
    ]
    second = materialize_canonical_rankings(
        sources=second_sources,
        target_manifest=manifest,
        target_provenance=_target_provenance(manifest),
        target_bucket=TARGET_BUCKET,
        s3_client=second_client,
    )

    assert [entry["cell"]["seed"] for entry in first.receipt["entries"]] == [0, 2]
    assert first.receipt_payload == second.receipt_payload
