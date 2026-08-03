"""Dataset loading, discovery, and S3 caching.

This module handles all dataset-related operations:
- Dataset discovery (listing available datasets)
- Dataset loading (parquet to numpy arrays)
- S3 caching (download datasets from S3 to local cache)
- Dataset metadata extraction
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import threading
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from botocore.exceptions import ClientError
from loguru import logger

from paper.benchmark.adapters.store import get_s3_bucket as _get_s3_bucket
from paper.benchmark.adapters.store import get_s3_client as _get_s3_client
from paper.benchmark.pipeline.types import DatasetIdentity
from paper.benchmark.utils.env import get_repo_root

TaskType = Literal["classification", "regression"]
DataSource = Literal["real", "synthetic"]
_DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def get_data_dir(task: TaskType, source: DataSource = "real") -> Path:
    """Return the location of dataset directory for the given task type and source.

    Directory structure:
        paper/data/classification/real/      - real classification datasets
        paper/data/classification/synthetic/ - synthetic classification datasets
        paper/data/regression/real/          - real regression datasets
        paper/data/regression/synthetic/     - synthetic regression datasets
    """
    return get_repo_root() / "paper" / "data" / task / source


def get_data_cache_dir() -> Path:
    """Return the local cache directory for S3 datasets.

    On workers, datasets are downloaded from S3 to this cache directory.
    Uses /tmp/citrees-data for fast local access.
    """
    cache_dir = Path("/tmp/citrees-data")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_dataset_s3_key(
    name: str,
    task: TaskType,
    source: DataSource,
    identity: DatasetIdentity,
) -> str:
    """Return the name- and content-bound S3 key for one dataset."""
    if not _DATASET_NAME_PATTERN.fullmatch(name):
        raise ValueError("dataset name must use a safe canonical name")
    if task not in {"classification", "regression"}:
        raise ValueError(f"invalid dataset task {task!r}")
    if source not in {"real", "synthetic"}:
        raise ValueError(f"invalid dataset source {source!r}")
    return f"data/{task}/{source}/{name}/sha256/{identity.sha256}.parquet"


def get_dataset_prefix(task: TaskType) -> str:
    """Return the filename prefix for datasets (clf_ or reg_)."""
    return "clf_" if task == "classification" else "reg_"


def _infer_source(name: str) -> DataSource:
    """Infer dataset source from name."""
    return "synthetic" if "synthetic" in name else "real"


def get_datasets(
    task: TaskType,
    *,
    source: Literal["real", "synthetic", "all"] = "all",
    data_dir: Path | None = None,
) -> list[str]:
    """List dataset names for the given task type and source.

    Scans the local filesystem. Distributed runs obtain their exact dataset
    inventory and content hashes from the immutable rerun manifest.

    Parameters
    ----------
    task : TaskType
        Either "classification" or "regression".
    source : {"real", "synthetic", "all"}, default "all"
        Which datasets to include.
    data_dir : Path, optional
        Override base data directory (mainly for testing).

    Returns
    -------
    list[str]
        Sorted list of dataset names (without prefix).
    """
    prefix = get_dataset_prefix(task)
    datasets: list[str] = []

    sources: list[DataSource] = ["real", "synthetic"] if source == "all" else [source]  # type: ignore[list-item]
    for src in sources:
        search_dir = data_dir / task / src if data_dir is not None else get_data_dir(task, src)
        if not search_dir.exists():
            continue
        for f in search_dir.glob(f"{prefix}*.parquet"):
            name = f.stem.replace(prefix, "")
            datasets.append(name)

    return sorted(datasets)


def get_dataset_file_identity(path: Path) -> DatasetIdentity:
    """Hash and inspect one Parquet dataset file."""
    return get_dataset_payload_identity(path.read_bytes())


def get_dataset_payload_identity(payload: bytes) -> DatasetIdentity:
    """Return the byte hash and logical dimensions of Parquet dataset bytes."""
    try:
        parquet = pq.ParquetFile(io.BytesIO(payload))
    except Exception as exc:
        raise ValueError("dataset bytes are not readable Parquet") from exc

    columns = parquet.schema_arrow.names
    if columns.count("y") != 1:
        raise ValueError("dataset Parquet must contain exactly one target column named 'y'")
    return DatasetIdentity(
        sha256=hashlib.sha256(payload).hexdigest(),
        n_samples=int(parquet.metadata.num_rows),
        n_features=len(columns) - 1,
    )


def validate_dataset_payload(
    payload: bytes,
    expected: DatasetIdentity,
    *,
    location: str,
) -> None:
    """Require dataset bytes to match the manifest-bound identity."""
    observed = get_dataset_payload_identity(payload)
    if observed != expected:
        raise ValueError(
            f"dataset identity mismatch at {location}: expected {expected}, observed {observed}"
        )


def validate_dataset_path(path: Path, expected: DatasetIdentity) -> None:
    """Require a dataset file to match the manifest-bound identity."""
    validate_dataset_payload(path.read_bytes(), expected, location=str(path))


def get_dataset_s3_payload(
    name: str,
    task: TaskType,
    source: DataSource,
    identity: DatasetIdentity,
    *,
    bucket: str | None = None,
    client: Any | None = None,
    region_name: str | None = None,
) -> bytes:
    """Download and validate one content-addressed dataset object."""
    resolved_bucket = _get_s3_bucket() if bucket is None else bucket
    resolved_client = _get_s3_client(region_name=region_name) if client is None else client
    s3_key = get_dataset_s3_key(name, task, source, identity)
    s3_path = f"s3://{resolved_bucket}/{s3_key}"
    try:
        response = resolved_client.get_object(Bucket=resolved_bucket, Key=s3_key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            raise FileNotFoundError(f"Dataset not found in S3: {s3_path}") from exc
        raise RuntimeError(f"Failed to download dataset from S3: {s3_path}") from exc
    payload = response["Body"].read()
    validate_dataset_payload(payload, identity, location=s3_path)
    return payload


def get_dataset_identity(
    name: str,
    task: TaskType,
    *,
    source: DataSource | None = None,
    data_dir: Path | None = None,
) -> DatasetIdentity:
    """Return the byte hash and dimensions of a local named dataset."""
    path = get_dataset_path(name, task, source=source, data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found locally: {path}")
    return get_dataset_file_identity(path)


def get_dataset_path(
    name: str,
    task: TaskType,
    *,
    source: DataSource | None = None,
    data_dir: Path | None = None,
) -> Path:
    """Return the parquet file path for a dataset.

    Parameters
    ----------
    name : str
        Dataset name (without prefix).
    task : TaskType
        Either "classification" or "regression".
    source : {"real", "synthetic"}, optional
        If None, infers from name: "synthetic" if name contains "synthetic", else "real".
    data_dir : Path, optional
        Override base data directory (mainly for testing).

    Returns
    -------
    Path
        Path to the parquet file.
    """
    prefix = get_dataset_prefix(task)
    resolved_source = source if source is not None else _infer_source(name)

    if data_dir is not None:
        search_dir = data_dir / task / resolved_source
    else:
        search_dir = get_data_dir(task, resolved_source)

    return search_dir / f"{prefix}{name}.parquet"


def _resolve_dataset_path(
    name: str,
    task: TaskType,
    source: DataSource,
    identity: DatasetIdentity,
    data_dir: Path | None = None,
) -> Path:
    """Resolve a local named file or exact content-addressed cache path."""
    path = get_dataset_path(name, task, source=source, data_dir=data_dir)
    if path.exists():
        return path

    if data_dir is None:
        return ensure_dataset_cached(name, task, source, identity)

    raise FileNotFoundError(f"Dataset not found: {path}")


def _load_parquet_to_arrays(payload: bytes, task: TaskType) -> tuple[np.ndarray, np.ndarray]:
    """Load verified Parquet bytes into ``(X, y)`` arrays."""
    df = pd.read_parquet(io.BytesIO(payload))
    y = df.pop("y").values
    y = y.astype(np.int64) if task == "classification" else y.astype(np.float64)
    X = df.values.astype(np.float64)
    return X, y


def ensure_dataset_cached(
    name: str,
    task: TaskType,
    source: DataSource,
    identity: DatasetIdentity,
    *,
    region_name: str | None = None,
) -> Path:
    """Download and verify an exact content-addressed dataset.

    Uses file-based locking to prevent redundant downloads when multiple
    workers request the same dataset concurrently.

    Parameters
    ----------
    identity : DatasetIdentity
        Exact manifest-bound byte hash and dimensions.
    region_name : str, optional
        AWS region name for S3 client.

    Returns
    -------
    Path
        Path to the cached parquet file.

    Raises
    ------
    RuntimeError
        If S3_BUCKET is not set or download fails.
    """
    import fcntl

    cache_dir = get_data_cache_dir()
    local_dir = cache_dir / "sha256"
    local_dir.mkdir(parents=True, exist_ok=True)
    cache_path = local_dir / f"{identity.sha256}.parquet"

    if cache_path.exists():
        validate_dataset_path(cache_path, identity)
        return cache_path

    lock_path = cache_path.with_suffix(".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)

        if cache_path.exists():
            validate_dataset_path(cache_path, identity)
            return cache_path

        s3_key = get_dataset_s3_key(name, task, source, identity)
        bucket = _get_s3_bucket()
        s3_path = f"s3://{bucket}/{s3_key}"

        logger.info(f"Downloading dataset from S3: {s3_path} -> {cache_path}")
        content = get_dataset_s3_payload(
            name,
            task,
            source,
            identity,
            bucket=bucket,
            region_name=region_name,
        )

        tmp_path = cache_path.with_suffix(f".tmp.{os.getpid()}.{threading.current_thread().ident}")
        tmp_path.write_bytes(content)
        tmp_path.rename(cache_path)

        logger.info(f"Successfully cached dataset: {cache_path} ({len(content)} bytes)")

    validate_dataset_path(cache_path, identity)
    return cache_path


def load_dataset(
    name: str,
    task: TaskType,
    *,
    identity: DatasetIdentity,
    source: DataSource | None = None,
    data_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a manifest-bound dataset into ``(X, y)`` arrays.

    Resolution order:
    1. Local path (for local development or custom data_dir)
    2. S3 download to /tmp/citrees-data/ cache (for workers)

    Parameters
    ----------
    name : str
        Dataset name (without prefix).
    task : TaskType
        Either "classification" or "regression".
    source : {"real", "synthetic"}, optional
        If None, infers from name: "synthetic" if name contains "synthetic", else "real".
    data_dir : Path, optional
        Override base data directory (mainly for testing).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X, y) arrays where X is float64 and y is int64 (classification) or float64 (regression).
    """
    resolved_source = source if source is not None else _infer_source(name)
    path = _resolve_dataset_path(name, task, resolved_source, identity, data_dir)
    payload = path.read_bytes()
    validate_dataset_payload(payload, identity, location=str(path))
    X, y = _load_parquet_to_arrays(payload, task)
    if X.shape != (identity.n_samples, identity.n_features):
        raise ValueError(
            f"loaded dataset shape {X.shape} does not match manifest "
            f"({identity.n_samples}, {identity.n_features})"
        )
    return X, y


def get_dataset_shape(
    name: str,
    task: TaskType,
    *,
    source: DataSource | None = None,
    data_dir: Path | None = None,
) -> tuple[int, int]:
    """Return dimensions from the verified local dataset identity."""
    identity = get_dataset_identity(name, task, source=source, data_dir=data_dir)
    return identity.n_samples, identity.n_features


def get_dataset_metadata(
    name: str,
    task: TaskType,
    *,
    identity: DatasetIdentity | None = None,
    source: DataSource | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Extract metadata from a dataset's parquet schema.

    Returns a dict with:
    - source: "real" or "synthetic"
    - dataset_type: extracted type (e.g., "standard", "bias", "nonlinear") for synthetic
    - n_informative: number of informative features (synthetic only)
    - config: full generation config (synthetic only)
    """
    inferred_source = _infer_source(name)
    resolved_source = source if source is not None else inferred_source
    expected_identity = identity or get_dataset_identity(
        name,
        task,
        source=resolved_source,
        data_dir=data_dir,
    )
    path = _resolve_dataset_path(name, task, resolved_source, expected_identity, data_dir)
    payload = path.read_bytes()
    validate_dataset_payload(payload, expected_identity, location=str(path))

    metadata: dict[str, Any] = {
        "dataset_source": inferred_source,
        "dataset_type": None,
        "dataset_family": None,
        "n_informative": None,
    }

    try:
        pf = pq.ParquetFile(io.BytesIO(payload))
        schema_meta = pf.schema_arrow.metadata or {}

        # Check if synthetic
        is_synthetic = schema_meta.get(b"synthetic", b"false").decode() == "true"
        if is_synthetic:
            metadata["dataset_source"] = "synthetic"

            # Extract dataset type from name
            if "bias" in name:
                metadata["dataset_type"] = "bias"
            elif "friedman" in name:
                metadata["dataset_type"] = "friedman"
            elif "nonlinear" in name:
                metadata["dataset_type"] = "nonlinear"
            elif "heteroscedastic" in name:
                metadata["dataset_type"] = "heteroscedastic"
            elif "corr_noise" in name:
                metadata["dataset_type"] = "confounder"
            elif "redundant" in name:
                metadata["dataset_type"] = "redundant"
            elif "toeplitz" in name:
                metadata["dataset_type"] = "toeplitz"
            elif "weak" in name:
                metadata["dataset_type"] = "weak_signal"
            else:
                metadata["dataset_type"] = "standard"

            # Extract informative indices
            import json

            info_json = schema_meta.get(b"informative_indices")
            if info_json:
                info_indices = json.loads(info_json.decode())
                metadata["n_informative"] = len(info_indices)

            # Extract family from config
            config_json = schema_meta.get(b"config")
            if config_json:
                config_dict = json.loads(config_json.decode())
                # Order matches synthetic_analysis.py:dataset_type_from_config
                if config_dict.get("toeplitz_rho", 0) > 0:
                    metadata["dataset_family"] = "toeplitz"
                elif config_dict.get("weak_signal"):
                    metadata["dataset_family"] = "weak_signal"
                elif config_dict.get("friedman_variant") is not None:
                    metadata["dataset_family"] = "friedman"
                elif config_dict.get("nonlinear"):
                    metadata["dataset_family"] = "nonlinear"
                elif config_dict.get("heteroscedastic"):
                    metadata["dataset_family"] = "heteroscedastic"
                elif config_dict.get("n_high_cardinality_noise", 0) > 0:
                    metadata["dataset_family"] = "bias"
                elif config_dict.get("n_correlated_noise", 0) > 0:
                    metadata["dataset_family"] = "confounder"
                elif config_dict.get("n_redundant", 0) > 0:
                    metadata["dataset_family"] = "redundant"
                else:
                    metadata["dataset_family"] = "standard"
        else:
            metadata["dataset_source"] = "real"
            metadata["dataset_type"] = "real"
            if name.startswith("openml_"):
                metadata["dataset_family"] = "openml"
            elif name.startswith("uci_"):
                metadata["dataset_family"] = "uci"
            else:
                metadata["dataset_family"] = "other"

    except Exception as e:
        logger.warning(f"Failed to parse metadata for dataset {name!r}: {e}")

    return metadata


def get_cv_splitter(task: TaskType, n_splits: int, seed: int):
    """Return a cross-validation splitter for the given task type.

    Parameters
    ----------
    task : TaskType
        Either "classification" or "regression".
    n_splits : int
        Number of CV folds.
    seed : int
        Random seed for shuffling.

    Returns
    -------
    StratifiedKFold or KFold
        CV splitter instance.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    if task == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
