"""Store protocol and implementations for experiment artifacts.

Provides a clean abstraction for storing and retrieving experiment results,
with S3 as the primary backend.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import boto3
import pandas as pd
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from paper.benchmark.pipeline.types import CellKey, ExperimentConfig, StageType, TaskType

# Module-level S3 client cache
_S3_CLIENTS: dict[str | None, Any] = {}
_S3_CLIENT_LOCK = threading.Lock()
_ASSIGNMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_METHOD_LABEL_PATTERN = re.compile(r"^[a-z][a-z0-9_]*__[0-9a-f]{16}$")
ReceiptType = Literal["attempts", "failures"]


class ArtifactReadError(ValueError):
    """Raised when an artifact object is not readable Parquet."""


@dataclass(frozen=True)
class ControlReceipt:
    """One structurally parsed immutable control-plane receipt."""

    key: str
    dataset: str
    method_label: str
    seed: int
    assignment_id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class LoadedArtifact:
    """One parsed artifact and the SHA-256 of its exact stored bytes."""

    frame: pd.DataFrame
    payload_sha256: str


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r} is not allowed")


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _parse_json_object(payload: bytes, *, location: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactReadError(f"Control receipt is not valid JSON: {location}") from exc
    if not isinstance(value, dict):
        raise ArtifactReadError(f"Control receipt must contain a JSON object: {location}")
    return value


def get_s3_client(*, region_name: str | None = None) -> Any:
    """Create and cache an S3 client with retry config (thread-safe)."""
    with _S3_CLIENT_LOCK:
        if region_name not in _S3_CLIENTS:
            retry_config = BotoConfig(
                retries={"max_attempts": 5, "mode": "adaptive"},
                connect_timeout=10,
                read_timeout=30,
            )
            kwargs: dict[str, Any] = {"config": retry_config}
            if region_name:
                kwargs["region_name"] = region_name
            _S3_CLIENTS[region_name] = boto3.client("s3", **kwargs)
        return _S3_CLIENTS[region_name]


def get_s3_bucket() -> str:
    """Return the S3 bucket name from environment (required for S3-backed runs)."""
    bucket = os.environ.get("S3_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError(
            "S3_BUCKET is required but not set. "
            "Set it explicitly (e.g., `export S3_BUCKET=...`) or use the infra setup "
            "that exports S3_BUCKET on workers."
        )
    return bucket


def _normalize_artifact_prefix(artifact_prefix: str) -> str:
    """Validate and normalize a relative S3 prefix for experiment artifacts."""
    if not isinstance(artifact_prefix, str):
        raise TypeError("artifact_prefix must be a string")

    normalized = artifact_prefix.strip()
    if not normalized:
        raise ValueError("artifact_prefix must be non-empty")
    if normalized.startswith("/"):
        raise ValueError("artifact_prefix must be relative, not absolute")
    if "\\" in normalized:
        raise ValueError("artifact_prefix must use forward slashes")
    if len(normalized) >= 2 and normalized[0].isalpha() and normalized[1] == ":":
        raise ValueError("artifact_prefix must not use a drive-qualified path")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError("artifact_prefix must not contain control characters")

    normalized = normalized.rstrip("/")
    parts = normalized.split("/")
    if not normalized or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("artifact_prefix must not contain empty, '.' or '..' path segments")
    return normalized


class Store(Protocol):
    """Protocol for experiment artifact storage.

    Defines the interface for storing and retrieving experiment results.
    Implementations can use S3, local filesystem, or other backends.
    """

    def exists(self, stage: StageType, config: ExperimentConfig) -> bool:
        """Check if an artifact exists.

        Parameters
        ----------
        stage : StageType
            Either "rankings" or "metrics".
        config : ExperimentConfig
            Experiment configuration.

        Returns
        -------
        bool
            True if the artifact exists.
        """
        ...

    def save(self, stage: StageType, config: ExperimentConfig, df: pd.DataFrame) -> str:
        """Save an artifact.

        Parameters
        ----------
        stage : StageType
            Either "rankings" or "metrics".
        config : ExperimentConfig
            Experiment configuration.
        df : pd.DataFrame
            Data to save.

        Returns
        -------
        str
            Path where the artifact was saved.
        """
        ...

    def load(self, stage: StageType, config: ExperimentConfig) -> pd.DataFrame:
        """Load an artifact.

        Parameters
        ----------
        stage : StageType
            Either "rankings" or "metrics".
        config : ExperimentConfig
            Experiment configuration.

        Returns
        -------
        pd.DataFrame
            Loaded data.

        Raises
        ------
        FileNotFoundError
            If the artifact does not exist.
        """
        ...

    def load_with_payload_sha256(
        self,
        stage: StageType,
        config: ExperimentConfig,
    ) -> LoadedArtifact:
        """Load an artifact with the digest of its exact stored bytes."""
        ...

    def list_completed(self, stage: StageType, task: TaskType) -> set[CellKey]:
        """List completed artifacts for a stage.

        Parameters
        ----------
        stage : StageType
            Either "rankings" or "metrics".
        task : TaskType
            Task type.

        Returns
        -------
        set[CellKey]
            Set of task-inclusive experiment cell keys.
        """
        ...


class S3Store:
    """S3-backed artifact store.

    Stores experiment artifacts in S3 as parquet files with the structure:
    s3://{bucket}/{artifact_prefix}/{stage}/{task}/{dataset}/{method}_seed{seed}.parquet

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    artifact_prefix : str
        Non-empty relative S3 prefix reserved for experiment artifacts.
    region : str, default "us-east-1"
        AWS region.
    validate_uploads : bool, default False
        If True, verify uploads by checking ContentLength.

    Examples
    --------
    >>> store = S3Store(
    ...     bucket="citrees-123456789012",
    ...     artifact_prefix="repairs/r-baselines-v2/run-001",
    ... )
    >>> completed = store.list_completed("rankings", "classification")
    >>> len(completed)
    420
    """

    def __init__(
        self,
        bucket: str,
        *,
        artifact_prefix: str,
        region: str = "us-east-1",
        validate_uploads: bool = False,
    ):
        self.bucket = bucket
        self.artifact_prefix = _normalize_artifact_prefix(artifact_prefix)
        self.region = region
        self.validate_uploads = validate_uploads
        self._client: Any = None
        self._client_lock = threading.Lock()
        self._write_guard: Callable[[], bool] | None = None

    @property
    def client(self) -> Any:
        """Lazy-initialized S3 client with retry config."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    retry_config = BotoConfig(
                        retries={"max_attempts": 5, "mode": "adaptive"},
                        connect_timeout=10,
                        read_timeout=30,
                    )
                    self._client = boto3.client(
                        "s3",
                        region_name=self.region,
                        config=retry_config,
                    )
        return self._client

    def __getstate__(self) -> dict[str, Any]:
        """Support pickling (exclude non-serializable fields)."""
        return {
            "bucket": self.bucket,
            "artifact_prefix": self.artifact_prefix,
            "region": self.region,
            "validate_uploads": self.validate_uploads,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state after unpickling (reinitialize lock and client)."""
        self.bucket = state["bucket"]
        self.artifact_prefix = _normalize_artifact_prefix(state["artifact_prefix"])
        self.region = state["region"]
        self.validate_uploads = state["validate_uploads"]
        self._client = None
        self._client_lock = threading.Lock()
        self._write_guard = None

    def bind_write_guard(self, guard: Callable[[], bool]) -> None:
        """Require one active assignment lease before artifact writes."""
        self._write_guard = guard

    def clear_write_guard(self) -> None:
        """Remove the assignment write fence after one cell terminates."""
        self._write_guard = None

    def _require_write_permission(self) -> None:
        if self._write_guard is not None and not self._write_guard():
            raise RuntimeError("assignment lease was lost before artifact persistence")

    def _stage_prefix(self, stage: StageType, task: TaskType) -> str:
        """Build the namespace prefix for one artifact stage and task."""
        return f"{self.artifact_prefix}/{stage}/{task}/"

    def artifact_key(self, stage: StageType, config: ExperimentConfig) -> str:
        """Build the canonical S3 key for an artifact."""
        return (
            f"{self._stage_prefix(stage, config.task)}{config.dataset}/"
            f"{config.method.label}_seed{config.seed}.parquet"
        )

    def _path(self, stage: StageType, config: ExperimentConfig) -> str:
        """Build full S3 path for an artifact."""
        return f"s3://{self.bucket}/{self.artifact_key(stage, config)}"

    def _failure_key(
        self,
        stage: StageType,
        config: ExperimentConfig,
        assignment_id: str,
    ) -> str:
        """Build an immutable key for one failed assignment."""
        return self._control_receipt_key("failures", stage, config, assignment_id)

    def _attempt_key(
        self,
        stage: StageType,
        config: ExperimentConfig,
        assignment_id: str,
    ) -> str:
        """Build an immutable key for one leased assignment."""
        return self._control_receipt_key("attempts", stage, config, assignment_id)

    def _control_receipt_key(
        self,
        receipt_type: ReceiptType,
        stage: StageType,
        config: ExperimentConfig,
        assignment_id: str,
    ) -> str:
        """Build a canonical immutable control-receipt key."""
        if not _ASSIGNMENT_ID_PATTERN.fullmatch(assignment_id):
            raise ValueError("assignment_id must be 32 lowercase hexadecimal characters")
        return (
            f"{self.artifact_prefix}/{receipt_type}/{stage}/{config.task}/{config.dataset}/"
            f"{config.method.label}_seed{config.seed}/{assignment_id}.json"
        )

    def _control_receipt_prefix(
        self,
        receipt_type: ReceiptType,
        stage: StageType,
        task: TaskType,
    ) -> str:
        """Return the listing prefix for one control-receipt scope."""
        return f"{self.artifact_prefix}/{receipt_type}/{stage}/{task}/"

    def _parse_control_receipt_key(
        self,
        key: str,
        *,
        receipt_type: ReceiptType,
        stage: StageType,
        task: TaskType,
    ) -> tuple[str, str, int, str]:
        """Parse one key or fail on malformed campaign state."""
        prefix = self._control_receipt_prefix(receipt_type, stage, task)
        if not key.startswith(prefix):
            raise ArtifactReadError(f"Control receipt is outside its expected prefix: {key}")
        parts = key[len(prefix) :].split("/")
        if len(parts) != 3:
            raise ArtifactReadError(f"Control receipt key is malformed: {key}")
        dataset, method_seed, filename = parts
        if not dataset or dataset in {".", ".."} or "_seed" not in method_seed:
            raise ArtifactReadError(f"Control receipt key is malformed: {key}")
        method_label, seed_text = method_seed.rsplit("_seed", 1)
        if not _METHOD_LABEL_PATTERN.fullmatch(method_label):
            raise ArtifactReadError(f"Control receipt method identity is malformed: {key}")
        try:
            seed = int(seed_text)
        except ValueError as exc:
            raise ArtifactReadError(f"Control receipt seed is malformed: {key}") from exc
        if str(seed) != seed_text or seed < 0:
            raise ArtifactReadError(f"Control receipt seed is malformed: {key}")
        if not filename.endswith(".json"):
            raise ArtifactReadError(f"Control receipt filename is malformed: {key}")
        assignment_id = filename.removesuffix(".json")
        if not _ASSIGNMENT_ID_PATTERN.fullmatch(assignment_id):
            raise ArtifactReadError(f"Control receipt assignment identity is malformed: {key}")
        return dataset, method_label, seed, assignment_id

    def exists(self, stage: StageType, config: ExperimentConfig) -> bool:
        """Check if an artifact exists in S3."""
        key = self.artifact_key(stage, config)
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def save(self, stage: StageType, config: ExperimentConfig, df: pd.DataFrame) -> str:
        """Save a DataFrame to S3 as parquet."""
        self._require_write_permission()
        key = self.artifact_key(stage, config)
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        payload = buffer.getvalue()
        self._put_immutable_bytes(
            key,
            payload,
            conflict_message=f"Refusing to overwrite existing artifact: {self._path(stage, config)}",
        )

        return self._path(stage, config)

    def load(self, stage: StageType, config: ExperimentConfig) -> pd.DataFrame:
        """Load a DataFrame from S3."""
        return self.load_with_payload_sha256(stage, config).frame

    def load_with_payload_sha256(
        self,
        stage: StageType,
        config: ExperimentConfig,
    ) -> LoadedArtifact:
        """Load a DataFrame and hash the exact Parquet object bytes."""
        key = self.artifact_key(stage, config)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(f"Artifact not found: {self._path(stage, config)}") from e
            raise
        payload = response["Body"].read()
        try:
            frame = pd.read_parquet(io.BytesIO(payload))
        except Exception as exc:
            raise ArtifactReadError(
                f"Artifact is not readable Parquet: {self._path(stage, config)}"
            ) from exc
        return LoadedArtifact(
            frame=frame,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
        )

    def _put_immutable_json(
        self,
        key: str,
        receipt: dict[str, Any],
    ) -> str:
        """Persist one canonical JSON object without allowing replacement."""
        payload = json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self._put_immutable_bytes(
            key,
            payload,
            content_type="application/json",
            conflict_message=(
                f"Refusing to overwrite existing control receipt: s3://{self.bucket}/{key}"
            ),
        )
        return f"s3://{self.bucket}/{key}"

    def _read_object_bytes(self, key: str) -> bytes:
        """Read exact object bytes or report an absent key."""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(f"Object not found: s3://{self.bucket}/{key}") from exc
            raise
        return response["Body"].read()

    def _put_immutable_bytes(
        self,
        key: str,
        payload: bytes,
        *,
        conflict_message: str,
        content_type: str | None = None,
    ) -> None:
        """Write once and reconcile exact bytes after every ambiguous response."""
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": key,
            "Body": payload,
            "IfNoneMatch": "*",
        }
        if content_type is not None:
            kwargs["ContentType"] = content_type

        write_error: Exception | None = None
        try:
            self.client.put_object(**kwargs)
        except Exception as exc:
            write_error = exc

        if write_error is not None or self.validate_uploads:
            try:
                observed = self._read_object_bytes(key)
            except FileNotFoundError:
                if write_error is not None:
                    raise write_error from None
                raise RuntimeError(
                    f"S3 upload validation failed because the object is absent: "
                    f"s3://{self.bucket}/{key}"
                ) from None
            if observed == payload:
                return
            if write_error is not None:
                raise FileExistsError(conflict_message) from write_error
            raise RuntimeError(
                f"S3 upload validation failed because stored bytes differ: s3://{self.bucket}/{key}"
            )

    def save_attempt(
        self,
        stage: StageType,
        config: ExperimentConfig,
        assignment_id: str,
        receipt: dict[str, Any],
        *,
        expected_provenance: dict[str, str],
    ) -> str:
        """Validate and persist one immutable assignment-attempt receipt."""
        from paper.benchmark.pipeline.validation import validate_attempt_receipt

        attempt = cast(int, receipt.get("attempt"))
        request_id = cast(str, receipt.get("request_id"))
        worker_id = cast(str, receipt.get("worker_id"))
        validate_attempt_receipt(
            receipt,
            config,
            stage=stage,
            assignment_id=assignment_id,
            attempt=attempt,
            request_id=request_id,
            worker_id=worker_id,
            expected_provenance=expected_provenance,
        )
        return self._put_immutable_json(
            self._attempt_key(stage, config, assignment_id),
            receipt,
        )

    def save_failure(
        self,
        stage: StageType,
        config: ExperimentConfig,
        assignment_id: str,
        receipt: dict[str, Any],
        *,
        expected_provenance: dict[str, str],
    ) -> str:
        """Validate and persist one immutable failed-attempt receipt."""
        from paper.benchmark.pipeline.validation import validate_failure_receipt

        attempt = cast(int, receipt.get("attempt"))
        request_id = cast(str, receipt.get("request_id"))
        worker_id = cast(str, receipt.get("worker_id"))
        validate_failure_receipt(
            receipt,
            config,
            stage=stage,
            assignment_id=assignment_id,
            attempt=attempt,
            request_id=request_id,
            worker_id=worker_id,
            expected_provenance=expected_provenance,
        )
        return self._put_immutable_json(
            self._failure_key(stage, config, assignment_id),
            receipt,
        )

    def _load_control_receipt(self, key: str) -> dict[str, Any]:
        """Load one immutable JSON control receipt."""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(
                    f"Control receipt not found: s3://{self.bucket}/{key}"
                ) from exc
            raise
        return _parse_json_object(
            response["Body"].read(),
            location=f"s3://{self.bucket}/{key}",
        )

    def load_failure(
        self,
        stage: StageType,
        config: ExperimentConfig,
        assignment_id: str,
    ) -> dict[str, Any]:
        """Load one failed-attempt receipt."""
        return self._load_control_receipt(self._failure_key(stage, config, assignment_id))

    def load_attempt(
        self,
        stage: StageType,
        config: ExperimentConfig,
        assignment_id: str,
    ) -> dict[str, Any]:
        """Load one assignment-attempt receipt."""
        return self._load_control_receipt(self._attempt_key(stage, config, assignment_id))

    def list_control_receipts(
        self,
        receipt_type: ReceiptType,
        stage: StageType,
        task: TaskType,
    ) -> tuple[ControlReceipt, ...]:
        """Load every immutable control receipt in one campaign scope."""
        prefix = self._control_receipt_prefix(receipt_type, stage, task)
        paginator = self.client.get_paginator("list_objects_v2")
        keys: set[str] = set()
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj.get("Key")
                if not isinstance(key, str):
                    raise ArtifactReadError("Control receipt listing contained a non-string key")
                keys.add(key)

        receipts: list[ControlReceipt] = []
        for key in sorted(keys):
            dataset, method_label, seed, assignment_id = self._parse_control_receipt_key(
                key,
                receipt_type=receipt_type,
                stage=stage,
                task=task,
            )
            receipts.append(
                ControlReceipt(
                    key=key,
                    dataset=dataset,
                    method_label=method_label,
                    seed=seed,
                    assignment_id=assignment_id,
                    payload=self._load_control_receipt(key),
                )
            )
        return tuple(receipts)

    def list_completed(self, stage: StageType, task: TaskType) -> set[CellKey]:
        """List completed artifacts for a stage.

        Parses object keys into task-inclusive experiment cell keys.
        """
        prefix = self._stage_prefix(stage, task)
        paginator = self.client.get_paginator("list_objects_v2")
        completed: set[CellKey] = set()

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj.get("Key", "")
                if not key.startswith(prefix):
                    continue
                relative_key = key[len(prefix) :]
                parts = relative_key.split("/")
                if len(parts) != 2:
                    continue
                dataset, filename = parts
                if not dataset:
                    continue
                if not filename.endswith(".parquet"):
                    continue
                method_seed = filename[: -len(".parquet")]
                if "_seed" not in method_seed:
                    continue
                method_label, seed_str = method_seed.rsplit("_seed", 1)
                try:
                    seed = int(seed_str)
                except ValueError:
                    continue
                completed.add((task, dataset, method_label, seed))

        return completed

    def list_stage_keys(self, stage: StageType) -> tuple[str, ...]:
        """List every object key under one artifact-stage namespace."""
        prefix = f"{self.artifact_prefix}/{stage}/"
        paginator = self.client.get_paginator("list_objects_v2")
        keys: set[str] = set()

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj.get("Key")
                if isinstance(key, str) and key.startswith(prefix):
                    keys.add(key)
        return tuple(sorted(keys))

    @classmethod
    def from_env(cls, validate_uploads: bool = False) -> S3Store:
        """Create store from environment variables.

        Uses S3_BUCKET, CITREES_ARTIFACT_PREFIX, and AWS_DEFAULT_REGION environment
        variables.

        Parameters
        ----------
        validate_uploads : bool, default False
            If True, verify uploads.

        Returns
        -------
        S3Store
            Configured store.

        Raises
        ------
        RuntimeError
            If S3_BUCKET is not set.
        """
        bucket = os.environ.get("S3_BUCKET", "").strip()
        if not bucket:
            raise RuntimeError(
                "S3_BUCKET is required but not set. Set it explicitly or use the infra setup."
            )
        artifact_prefix = os.environ.get("CITREES_ARTIFACT_PREFIX", "").strip()
        if not artifact_prefix:
            raise RuntimeError(
                "CITREES_ARTIFACT_PREFIX is required but not set. "
                "Set it to a non-empty, isolated artifact namespace."
            )
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        return cls(
            bucket=bucket,
            artifact_prefix=artifact_prefix,
            region=region,
            validate_uploads=validate_uploads,
        )


class IgnoreExistsStore:
    """Store wrapper that returns False from exists() for specified stages.

    Used with --force to bypass per-task skip checks while preserving
    cross-stage dependency checks (e.g., Stage 2 still verifies rankings exist).

    Parameters
    ----------
    store : S3Store
        The underlying store to delegate to.
    ignore_stages : frozenset[str]
        Stage names for which exists() should return False.
    """

    def __init__(self, store: S3Store, ignore_stages: frozenset[str]) -> None:
        self._store = store
        self._ignore_stages = ignore_stages

    def exists(self, stage: StageType, config: ExperimentConfig) -> bool:
        """Return False for ignored stages, delegate otherwise."""
        if stage in self._ignore_stages:
            return False
        return self._store.exists(stage, config)

    def save(self, stage: StageType, config: ExperimentConfig, df: pd.DataFrame) -> str:
        """Delegate to inner store."""
        return self._store.save(stage, config, df)

    def load(self, stage: StageType, config: ExperimentConfig) -> pd.DataFrame:
        """Delegate to inner store."""
        return self._store.load(stage, config)

    def load_with_payload_sha256(
        self,
        stage: StageType,
        config: ExperimentConfig,
    ) -> LoadedArtifact:
        """Delegate exact-byte loading to the inner store."""
        return self._store.load_with_payload_sha256(stage, config)

    def list_completed(self, stage: StageType, task: TaskType) -> set[CellKey]:
        """Delegate to inner store."""
        return self._store.list_completed(stage, task)

    def __getstate__(self) -> dict[str, Any]:
        """Support pickling."""
        return {"store": self._store, "ignore_stages": self._ignore_stages}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state after unpickling."""
        self._store = state["store"]
        self._ignore_stages = state["ignore_stages"]
