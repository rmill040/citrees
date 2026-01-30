"""Store protocol and implementations for experiment artifacts.

Provides a clean abstraction for storing and retrieving experiment results,
with S3 as the primary backend.
"""

from __future__ import annotations

import io
import os
import threading
from typing import Any, Protocol

import boto3
import pandas as pd
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from paper.scripts.pipeline.types import ExperimentConfig, StageType, TaskType

# Module-level S3 client cache
_S3_CLIENTS: dict[str | None, Any] = {}
_S3_CLIENT_LOCK = threading.Lock()


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
            "Set it explicitly (e.g., `export S3_BUCKET=...`) or use the Ray cluster setup "
            "that exports S3_BUCKET on the head/workers."
        )
    return bucket


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

    def list_completed(self, stage: StageType, task: TaskType) -> set[tuple[str, str, int]]:
        """List completed artifacts for a stage.

        Parameters
        ----------
        stage : StageType
            Either "rankings" or "metrics".
        task : TaskType
            Task type.

        Returns
        -------
        set[tuple[str, str, int]]
            Set of (method_label, dataset, seed) tuples.
        """
        ...


class S3Store:
    """S3-backed artifact store.

    Stores experiment artifacts in S3 as parquet files with the structure:
    s3://{bucket}/{stage}/{task}/{dataset}/{method}_seed{seed}.parquet

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    region : str, default "us-east-1"
        AWS region.
    validate_uploads : bool, default False
        If True, verify uploads by checking ContentLength.

    Examples
    --------
    >>> store = S3Store(bucket="citrees-123456789012")
    >>> completed = store.list_completed("rankings", "classification")
    >>> len(completed)
    420
    """

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        validate_uploads: bool = False,
    ):
        self.bucket = bucket
        self.region = region
        self.validate_uploads = validate_uploads
        self._client: Any = None
        self._client_lock = threading.Lock()

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
        """Support pickling for Ray serialization (exclude non-serializable fields)."""
        return {
            "bucket": self.bucket,
            "region": self.region,
            "validate_uploads": self.validate_uploads,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state after unpickling (reinitialize lock and client)."""
        self.bucket = state["bucket"]
        self.region = state["region"]
        self.validate_uploads = state["validate_uploads"]
        self._client = None
        self._client_lock = threading.Lock()

    def _key(self, stage: StageType, config: ExperimentConfig) -> str:
        """Build S3 key for an artifact."""
        return (
            f"{stage}/{config.task}/{config.dataset}/"
            f"{config.method.label}_seed{config.seed}.parquet"
        )

    def _path(self, stage: StageType, config: ExperimentConfig) -> str:
        """Build full S3 path for an artifact."""
        return f"s3://{self.bucket}/{self._key(stage, config)}"

    def exists(self, stage: StageType, config: ExperimentConfig) -> bool:
        """Check if an artifact exists in S3."""
        key = self._key(stage, config)
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
        key = self._key(stage, config)
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        payload = buffer.getvalue()
        expected_bytes = len(payload)

        self.client.put_object(Bucket=self.bucket, Key=key, Body=payload)

        if self.validate_uploads:
            head = self.client.head_object(Bucket=self.bucket, Key=key)
            uploaded_bytes = int(head.get("ContentLength") or 0)
            if uploaded_bytes != expected_bytes:
                raise RuntimeError(
                    f"S3 upload validation failed for {key}: "
                    f"expected {expected_bytes} bytes, got {uploaded_bytes}"
                )

        return self._path(stage, config)

    def load(self, stage: StageType, config: ExperimentConfig) -> pd.DataFrame:
        """Load a DataFrame from S3."""
        key = self._key(stage, config)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return pd.read_parquet(io.BytesIO(response["Body"].read()))
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(f"Artifact not found: {self._path(stage, config)}") from e
            raise

    def list_completed(self, stage: StageType, task: TaskType) -> set[tuple[str, str, int]]:
        """List completed artifacts for a stage.

        Parses object keys to extract (method_label, dataset, seed) tuples.
        """
        prefix = f"{stage}/{task}/"
        paginator = self.client.get_paginator("list_objects_v2")
        completed: set[tuple[str, str, int]] = set()

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj.get("Key", "")
                parts = key.split("/")
                if len(parts) < 4:
                    continue
                dataset = parts[2]
                filename = parts[3]
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
                completed.add((method_label, dataset, seed))

        return completed

    @classmethod
    def from_env(cls, validate_uploads: bool = False) -> S3Store:
        """Create store from environment variables.

        Uses S3_BUCKET and AWS_DEFAULT_REGION environment variables.

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
                "S3_BUCKET is required but not set. Set it explicitly or use the Ray cluster setup."
            )
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        return cls(bucket=bucket, region=region, validate_uploads=validate_uploads)

    @classmethod
    def from_config(cls) -> S3Store:
        """Create store from configuration file.

        Returns
        -------
        S3Store
            Configured store using settings from config.yaml.
        """
        from paper.scripts.config import load_config

        config = load_config()
        bucket = config.s3_bucket
        if not bucket:
            # Derive bucket name from account ID
            sts = boto3.client("sts", region_name=config.aws_region)
            account_id = sts.get_caller_identity()["Account"]
            bucket = f"citrees-{account_id}"

        return cls(
            bucket=bucket,
            region=config.aws_region,
            validate_uploads=config.experiment.s3_validate_uploads,
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

    def list_completed(self, stage: StageType, task: TaskType) -> set[tuple[str, str, int]]:
        """Delegate to inner store."""
        return self._store.list_completed(stage, task)

    def __getstate__(self) -> dict[str, Any]:
        """Support pickling for Ray serialization."""
        return {"store": self._store, "ignore_stages": self._ignore_stages}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state after unpickling."""
        self._store = state["store"]
        self._ignore_stages = state["ignore_stages"]
