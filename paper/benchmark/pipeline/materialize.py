"""Provenance-safe canonicalization of Stage 1 ranking artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from pandas.testing import assert_frame_equal

from paper.benchmark.pipeline.manifest import RerunManifest, validate_canonical_campaign
from paper.benchmark.pipeline.types import CellKey, ExperimentConfig
from paper.benchmark.pipeline.validation import (
    ArtifactValidationError,
    validate_artifact_provenance,
    validate_expected_provenance,
    validate_ranking_artifact,
)

_MATERIALIZATION_SCHEMA_VERSION = 1
_ACTIVE_PROVENANCE_COLUMNS = (
    "artifact_prefix",
    "aws_account_id",
    "campaign_sha256",
    "canonical_manifest_sha256",
    "container_image",
    "gate_receipt_sha256",
    "git_sha",
    "manifest_sha256",
    "runtime_contract_sha256",
)
_SOURCE_LINEAGE_COLUMNS = (
    "source_aws_account_id",
    "source_bucket",
    "source_key",
    "source_version_id",
    "source_payload_sha256",
    "source_artifact_prefix",
    "source_campaign_sha256",
    "source_canonical_manifest_sha256",
    "source_container_image",
    "source_gate_receipt_sha256",
    "source_git_sha",
    "source_manifest_sha256",
    "source_runtime_contract_sha256",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_AWS_ACCOUNT_ID_PATTERN = re.compile(r"^[0-9]{12}$")
_BUCKET_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")


class RankingMaterializationError(ValueError):
    """Raised when canonical ranking materialization violates its contract."""


class S3MaterializationClient(Protocol):
    """S3 operations required by canonical ranking materialization."""

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        """Read one exact S3 object or object version."""
        ...

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        """Conditionally write one immutable S3 object."""
        ...


@dataclass(frozen=True)
class RankingSource:
    """Explicit immutable source binding for one Stage 2-required cell."""

    cell_key: CellKey
    source_aws_account_id: str
    bucket: str
    key: str
    version_id: str | None
    payload_sha256: str
    expected_provenance: Mapping[str, str]


@dataclass(frozen=True)
class MaterializationResult:
    """Canonical receipt and its immutable S3 publication identity."""

    receipt: dict[str, Any]
    receipt_payload: bytes
    receipt_payload_sha256: str
    receipt_key: str
    receipt_version_id: str | None


@dataclass(frozen=True)
class _ReadObject:
    payload: bytes
    version_id: str | None
    metadata: dict[str, str]


@dataclass(frozen=True)
class _WriteResult:
    version_id: str | None
    payload_sha256: str


def _cell_record(cell_key: CellKey) -> dict[str, str | int]:
    task, dataset, method_id, seed = cell_key
    return {
        "task": task,
        "dataset": dataset,
        "method_id": method_id,
        "seed": seed,
    }


def _cell_text(cell_key: CellKey) -> str:
    task, dataset, method_id, seed = cell_key
    return f"{task}/{dataset}/{method_id}/seed{seed}"


def _validate_bucket(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _BUCKET_PATTERN.fullmatch(value):
        raise RankingMaterializationError(f"{field} must be a canonical S3 bucket name")
    if ".." in value or ".-" in value or "-." in value:
        raise RankingMaterializationError(f"{field} must be a canonical S3 bucket name")
    return value


def _validate_version_id(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RankingMaterializationError(f"{field} must be null or a non-empty string")
    if value != value.strip() or any(ord(character) < 32 for character in value):
        raise RankingMaterializationError(f"{field} contains invalid characters")
    return value


def _validate_sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise RankingMaterializationError(
            f"{field} must contain 64 lowercase hexadecimal characters"
        )
    return value


def _ranking_key(artifact_prefix: str, config: ExperimentConfig) -> str:
    return (
        f"{artifact_prefix}/rankings/{config.task}/{config.dataset}/"
        f"{config.method.label}_seed{config.seed}.parquet"
    )


def _validate_target_scope(
    target_manifest: RerunManifest,
    target_provenance: Mapping[str, str],
    target_bucket: str,
) -> dict[str, str]:
    _validate_bucket(target_bucket, field="target_bucket")
    try:
        validate_canonical_campaign(target_manifest)
        provenance = validate_expected_provenance(target_provenance)
    except (ArtifactValidationError, ValueError) as exc:
        raise RankingMaterializationError(f"invalid target contract: {exc}") from exc

    if provenance["manifest_sha256"] != target_manifest.sha256:
        raise RankingMaterializationError(
            "target manifest SHA-256 does not match target provenance"
        )
    if provenance["campaign_sha256"] != target_manifest.campaign_sha256:
        raise RankingMaterializationError("target campaign SHA-256 does not match target manifest")
    if provenance["runtime_contract_sha256"] != target_manifest.runtime_contract_sha256:
        raise RankingMaterializationError(
            "target runtime contract SHA-256 does not match target manifest"
        )
    if target_manifest.account_ids != (provenance["aws_account_id"],):
        raise RankingMaterializationError(
            "target manifest account binding does not match target provenance"
        )
    return provenance


def _validate_source(
    source: RankingSource,
    config: ExperimentConfig,
) -> dict[str, str]:
    if source.cell_key != config.key:
        raise RankingMaterializationError(
            f"source cell identity does not match manifest cell {_cell_text(config.key)}"
        )
    if not _AWS_ACCOUNT_ID_PATTERN.fullmatch(source.source_aws_account_id):
        raise RankingMaterializationError("source_aws_account_id must contain 12 decimal digits")
    _validate_bucket(source.bucket, field="source bucket")
    _validate_version_id(source.version_id, field="source version_id")
    _validate_sha256(source.payload_sha256, field="source payload SHA-256")
    try:
        provenance = validate_expected_provenance(source.expected_provenance)
    except ArtifactValidationError as exc:
        raise RankingMaterializationError(f"invalid source provenance: {exc}") from exc
    if provenance["aws_account_id"] != source.source_aws_account_id:
        raise RankingMaterializationError(
            "source account does not match expected source provenance"
        )
    expected_key = _ranking_key(provenance["artifact_prefix"], config)
    if source.key != expected_key:
        raise RankingMaterializationError(
            f"source key does not match its cell and provenance: {source.key!r} != {expected_key!r}"
        )
    return provenance


def _validate_inventory(
    sources: Sequence[RankingSource],
    target_manifest: RerunManifest,
) -> tuple[tuple[RankingSource, ExperimentConfig], ...]:
    cells_by_key = {cell.identity: cell for cell in target_manifest.cells}
    required_by_key = {
        cell.identity: cell for cell in target_manifest.cells if cell.stage2_required
    }
    if not required_by_key:
        raise RankingMaterializationError(
            "target manifest contains no Stage 2-required ranking cells"
        )

    sources_by_key: dict[CellKey, RankingSource] = {}
    duplicates: set[CellKey] = set()
    for source in sources:
        if not isinstance(source, RankingSource):
            raise RankingMaterializationError("sources must contain RankingSource entries")
        if source.cell_key in sources_by_key:
            duplicates.add(source.cell_key)
        else:
            sources_by_key[source.cell_key] = source
    if duplicates:
        raise RankingMaterializationError(
            "duplicate source cells: "
            + ", ".join(_cell_text(cell_key) for cell_key in sorted(duplicates))
        )

    unknown = set(sources_by_key) - set(cells_by_key)
    if unknown:
        raise RankingMaterializationError(
            "source cells are absent from the target manifest: "
            + ", ".join(_cell_text(cell_key) for cell_key in sorted(unknown))
        )
    non_required = set(sources_by_key) - set(required_by_key)
    if non_required:
        raise RankingMaterializationError(
            "source cells are non-required or blacklisted for Stage 2: "
            + ", ".join(_cell_text(cell_key) for cell_key in sorted(non_required))
        )
    missing = set(required_by_key) - set(sources_by_key)
    if missing:
        raise RankingMaterializationError(
            "missing Stage 2-required source cells: "
            + ", ".join(_cell_text(cell_key) for cell_key in sorted(missing))
        )

    return tuple(
        (sources_by_key[cell_key], required_by_key[cell_key].config)
        for cell_key in sorted(required_by_key)
    )


def _read_object(
    s3_client: S3MaterializationClient,
    *,
    bucket: str,
    key: str,
    version_id: str | None,
) -> _ReadObject:
    request: dict[str, str] = {"Bucket": bucket, "Key": key}
    if version_id is not None:
        request["VersionId"] = version_id
    try:
        response = s3_client.get_object(**request)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "unknown")
        location = f"s3://{bucket}/{key}"
        suffix = f"?versionId={version_id}" if version_id is not None else ""
        raise RankingMaterializationError(
            f"unable to read exact source or target object {location}{suffix}: {code}"
        ) from exc

    body = response.get("Body")
    if body is None or not hasattr(body, "read"):
        raise RankingMaterializationError(f"S3 object has no readable body: s3://{bucket}/{key}")
    payload = body.read()
    if not isinstance(payload, bytes):
        raise RankingMaterializationError(f"S3 object body is not bytes: s3://{bucket}/{key}")
    observed_version = _validate_version_id(
        response.get("VersionId"),
        field=f"S3 object VersionId for s3://{bucket}/{key}",
    )
    if version_id is not None and observed_version != version_id:
        raise RankingMaterializationError(
            f"S3 returned VersionId {observed_version!r} instead of requested {version_id!r}"
        )
    metadata_raw = response.get("Metadata", {})
    if not isinstance(metadata_raw, Mapping) or not all(
        isinstance(key_value, str) and isinstance(item, str)
        for key_value, item in metadata_raw.items()
    ):
        raise RankingMaterializationError(f"S3 object metadata is malformed: s3://{bucket}/{key}")
    return _ReadObject(
        payload=payload,
        version_id=observed_version,
        metadata=dict(metadata_raw),
    )


def _load_source_ranking(
    source: RankingSource,
    config: ExperimentConfig,
    source_provenance: Mapping[str, str],
    s3_client: S3MaterializationClient,
) -> tuple[pd.DataFrame, str | None]:
    loaded = _read_object(
        s3_client,
        bucket=source.bucket,
        key=source.key,
        version_id=source.version_id,
    )
    observed_sha256 = hashlib.sha256(loaded.payload).hexdigest()
    if observed_sha256 != source.payload_sha256:
        raise RankingMaterializationError(
            f"source payload SHA-256 mismatch for {_cell_text(config.key)}: "
            f"expected {source.payload_sha256}, observed {observed_sha256}"
        )
    try:
        frame = pd.read_parquet(io.BytesIO(loaded.payload))
    except Exception as exc:
        raise RankingMaterializationError(
            f"source ranking is not readable Parquet for {_cell_text(config.key)}"
        ) from exc
    reserved = sorted(set(frame.columns) & set(_SOURCE_LINEAGE_COLUMNS))
    if reserved:
        raise RankingMaterializationError(
            f"source ranking already contains reserved lineage columns: {reserved}"
        )
    try:
        validate_ranking_artifact(frame, config)
        validate_artifact_provenance(frame, source_provenance)
    except ArtifactValidationError as exc:
        raise RankingMaterializationError(
            f"source ranking validation failed for {_cell_text(config.key)}: {exc}"
        ) from exc
    return frame, loaded.version_id


def _materialize_frame(
    source_frame: pd.DataFrame,
    *,
    config: ExperimentConfig,
    source: RankingSource,
    source_version_id: str | None,
    source_provenance: Mapping[str, str],
    target_provenance: Mapping[str, str],
) -> pd.DataFrame:
    target_frame = source_frame.copy(deep=True)
    for column in _ACTIVE_PROVENANCE_COLUMNS:
        target_frame[column] = target_provenance[column]

    lineage = {
        "source_aws_account_id": source.source_aws_account_id,
        "source_bucket": source.bucket,
        "source_key": source.key,
        "source_version_id": source_version_id,
        "source_payload_sha256": source.payload_sha256,
        "source_artifact_prefix": source_provenance["artifact_prefix"],
        "source_campaign_sha256": source_provenance["campaign_sha256"],
        "source_canonical_manifest_sha256": source_provenance["canonical_manifest_sha256"],
        "source_container_image": source_provenance["container_image"],
        "source_gate_receipt_sha256": source_provenance["gate_receipt_sha256"],
        "source_git_sha": source_provenance["git_sha"],
        "source_manifest_sha256": source_provenance["manifest_sha256"],
        "source_runtime_contract_sha256": source_provenance["runtime_contract_sha256"],
    }
    for column in _SOURCE_LINEAGE_COLUMNS:
        target_frame[column] = lineage[column]

    preserved_columns = [
        column for column in source_frame.columns if column not in _ACTIVE_PROVENANCE_COLUMNS
    ]
    try:
        assert_frame_equal(
            target_frame[preserved_columns],
            source_frame[preserved_columns],
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as exc:
        raise RankingMaterializationError(
            f"scientific ranking content changed for {_cell_text(config.key)}"
        ) from exc

    try:
        validate_ranking_artifact(target_frame, config)
        validate_artifact_provenance(target_frame, target_provenance)
    except ArtifactValidationError as exc:
        raise RankingMaterializationError(
            f"target ranking validation failed for {_cell_text(config.key)}: {exc}"
        ) from exc
    return target_frame


def _serialize_parquet(frame: pd.DataFrame) -> tuple[bytes, pa.Table]:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    buffer = io.BytesIO()
    pq.write_table(
        table,
        buffer,
        compression="zstd",
        compression_level=9,
        data_page_version="2.0",
        use_compliant_nested_type=True,
        use_dictionary=True,
        version="2.6",
        write_statistics=True,
    )
    return buffer.getvalue(), table


def _is_precondition_failure(exc: ClientError) -> bool:
    code = str(exc.response.get("Error", {}).get("Code", ""))
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"PreconditionFailed", "412"} or status == 412


def _verify_readback(
    observed: _ReadObject,
    *,
    expected_payload: bytes,
    expected_metadata: Mapping[str, str],
    location: str,
) -> None:
    if observed.payload != expected_payload:
        raise FileExistsError(f"stored bytes differ from canonical payload: {location}")
    if observed.metadata != dict(expected_metadata):
        raise FileExistsError(f"stored metadata differs from canonical metadata: {location}")


def _put_immutable(
    s3_client: S3MaterializationClient,
    *,
    bucket: str,
    key: str,
    payload: bytes,
    content_type: str,
    record_kind: str,
) -> _WriteResult:
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    metadata = {"record-kind": record_kind, "sha256": payload_sha256}
    location = f"s3://{bucket}/{key}"
    try:
        response = s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType=content_type,
            IfNoneMatch="*",
            Metadata=metadata,
        )
    except ClientError as exc:
        if not _is_precondition_failure(exc):
            raise
        existing = _read_object(
            s3_client,
            bucket=bucket,
            key=key,
            version_id=None,
        )
        _verify_readback(
            existing,
            expected_payload=payload,
            expected_metadata=metadata,
            location=location,
        )
        return _WriteResult(
            version_id=existing.version_id,
            payload_sha256=payload_sha256,
        )
    except (ConnectionClosedError, EndpointConnectionError, ReadTimeoutError) as exc:
        try:
            existing = _read_object(
                s3_client,
                bucket=bucket,
                key=key,
                version_id=None,
            )
        except Exception:
            raise exc from None
        _verify_readback(
            existing,
            expected_payload=payload,
            expected_metadata=metadata,
            location=location,
        )
        return _WriteResult(
            version_id=existing.version_id,
            payload_sha256=payload_sha256,
        )

    version_id = _validate_version_id(
        response.get("VersionId"),
        field=f"target VersionId for {location}",
    )
    observed = _read_object(
        s3_client,
        bucket=bucket,
        key=key,
        version_id=version_id,
    )
    _verify_readback(
        observed,
        expected_payload=payload,
        expected_metadata=metadata,
        location=location,
    )
    return _WriteResult(
        version_id=observed.version_id,
        payload_sha256=payload_sha256,
    )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def materialize_canonical_rankings(
    *,
    sources: Sequence[RankingSource],
    target_manifest: RerunManifest,
    target_provenance: Mapping[str, str],
    target_bucket: str,
    s3_client: S3MaterializationClient,
) -> MaterializationResult:
    """Materialize one exact canonical ranking namespace for Stage 2.

    Every Stage 2-required manifest cell must have exactly one explicitly
    versioned or unversioned source binding. Source bytes and provenance are
    validated before any target write. Target writes are conditional,
    byte-verified, and idempotent only when the existing object is exact.
    """
    provenance = _validate_target_scope(
        target_manifest,
        target_provenance,
        target_bucket,
    )
    inventory = _validate_inventory(sources, target_manifest)
    prepared: list[
        tuple[
            RankingSource,
            ExperimentConfig,
            dict[str, str],
            str | None,
            bytes,
            str,
        ]
    ] = []

    for source, config in inventory:
        source_provenance = _validate_source(source, config)
        target_key = _ranking_key(provenance["artifact_prefix"], config)
        if source.bucket == target_bucket and source.key == target_key:
            raise RankingMaterializationError(
                f"source and target locations are identical for {_cell_text(config.key)}"
            )
        source_frame, source_version_id = _load_source_ranking(
            source,
            config,
            source_provenance,
            s3_client,
        )
        target_frame = _materialize_frame(
            source_frame,
            config=config,
            source=source,
            source_version_id=source_version_id,
            source_provenance=source_provenance,
            target_provenance=provenance,
        )
        target_payload, target_table = _serialize_parquet(target_frame)
        try:
            restored_table = pq.read_table(io.BytesIO(target_payload))
        except Exception as exc:
            raise RankingMaterializationError(
                f"serialized target ranking is not readable for {_cell_text(config.key)}"
            ) from exc
        if not restored_table.equals(target_table, check_metadata=False):
            raise RankingMaterializationError(
                f"serialized target ranking changed materialized content for "
                f"{_cell_text(config.key)}"
            )
        restored = restored_table.to_pandas()
        try:
            validate_ranking_artifact(restored, config)
            validate_artifact_provenance(restored, provenance)
        except ArtifactValidationError as exc:
            raise RankingMaterializationError(
                f"serialized target ranking failed readback for {_cell_text(config.key)}"
            ) from exc
        prepared.append(
            (
                source,
                config,
                source_provenance,
                source_version_id,
                target_payload,
                target_key,
            )
        )

    entries: list[dict[str, Any]] = []
    for (
        source,
        config,
        source_provenance,
        source_version_id,
        target_payload,
        target_key,
    ) in prepared:
        write = _put_immutable(
            s3_client,
            bucket=target_bucket,
            key=target_key,
            payload=target_payload,
            content_type="application/vnd.apache.parquet",
            record_kind="canonical-ranking",
        )
        entries.append(
            {
                "cell": _cell_record(config.key),
                "source": {
                    "aws_account_id": source.source_aws_account_id,
                    "bucket": source.bucket,
                    "key": source.key,
                    "payload_sha256": source.payload_sha256,
                    "provenance": dict(source_provenance),
                    "version_id": source_version_id,
                },
                "target": {
                    "bucket": target_bucket,
                    "key": target_key,
                    "payload_sha256": write.payload_sha256,
                    "version_id": write.version_id,
                },
            }
        )

    receipt: dict[str, Any] = {
        "schema_version": _MATERIALIZATION_SCHEMA_VERSION,
        "stage": "rankings",
        "entry_count": len(entries),
        "target_bucket": target_bucket,
        "target_manifest_sha256": target_manifest.sha256,
        "target_provenance": dict(provenance),
        "entries": entries,
    }
    receipt_payload = _canonical_json_bytes(receipt)
    receipt_key = f"{provenance['artifact_prefix']}/materialization/rankings-receipt.json"
    receipt_write = _put_immutable(
        s3_client,
        bucket=target_bucket,
        key=receipt_key,
        payload=receipt_payload,
        content_type="application/json",
        record_kind="canonical-ranking-materialization-receipt",
    )
    return MaterializationResult(
        receipt=receipt,
        receipt_payload=receipt_payload,
        receipt_payload_sha256=receipt_write.payload_sha256,
        receipt_key=receipt_key,
        receipt_version_id=receipt_write.version_id,
    )
