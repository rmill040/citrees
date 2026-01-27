"""Dataset loading, discovery, and S3 caching.

This module handles all dataset-related operations:
- Dataset discovery (listing available datasets)
- Dataset loading (parquet to numpy arrays)
- S3 caching (download datasets from S3 to local cache)
- Dataset metadata extraction
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from loguru import logger

import pyarrow.parquet as pq

from paper.scripts.utils.env import get_repo_root

TaskType = Literal["classification", "regression"]
DataSource = Literal["real", "synthetic"]


def get_data_dir(task_type: TaskType, source: DataSource = "real") -> Path:
    """Return the location of dataset directory for the given task type and source.

    Directory structure:
        paper/data/classification/real/      - real classification datasets
        paper/data/classification/synthetic/ - synthetic classification datasets
        paper/data/regression/real/          - real regression datasets
        paper/data/regression/synthetic/     - synthetic regression datasets
    """
    return get_repo_root() / "paper" / "data" / task_type / source


def get_data_cache_dir() -> Path:
    """Return the local cache directory for S3 datasets.

    On workers, datasets are downloaded from S3 to this cache directory.
    Uses /tmp/citrees-data for fast local access.
    """
    cache_dir = Path("/tmp/citrees-data")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_data_s3_prefix(task_type: TaskType, source: DataSource) -> str:
    """Return the S3 prefix for datasets."""
    return f"data/{task_type}/{source}"


def get_dataset_prefix(task_type: TaskType) -> str:
    """Return the filename prefix for datasets (clf_ or reg_)."""
    return "clf_" if task_type == "classification" else "reg_"


def _infer_source(name: str) -> DataSource:
    """Infer dataset source from name."""
    return "synthetic" if "synthetic" in name else "real"


def get_datasets(
    task_type: TaskType,
    *,
    source: Literal["real", "synthetic", "all"] = "all",
    data_dir: Path | None = None,
) -> list[str]:
    """List dataset names for the given task type and source.

    Parameters
    ----------
    task_type : TaskType
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
    prefix = get_dataset_prefix(task_type)
    datasets: list[str] = []

    sources: list[DataSource] = ["real", "synthetic"] if source == "all" else [source]  # type: ignore[list-item]
    for src in sources:
        if data_dir is not None:
            search_dir = data_dir / task_type / src
        else:
            search_dir = get_data_dir(task_type, src)
        if not search_dir.exists():
            continue
        for f in search_dir.glob(f"{prefix}*.parquet"):
            name = f.stem.replace(prefix, "")
            datasets.append(name)

    return sorted(datasets)


def get_dataset_path(
    name: str,
    task_type: TaskType,
    *,
    source: DataSource | None = None,
    data_dir: Path | None = None,
) -> Path:
    """Return the parquet file path for a dataset.

    Parameters
    ----------
    name : str
        Dataset name (without prefix).
    task_type : TaskType
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
    prefix = get_dataset_prefix(task_type)
    resolved_source = source if source is not None else _infer_source(name)

    if data_dir is not None:
        search_dir = data_dir / task_type / resolved_source
    else:
        search_dir = get_data_dir(task_type, resolved_source)

    return search_dir / f"{prefix}{name}.parquet"


def _resolve_dataset_path(
    name: str,
    task_type: TaskType,
    source: DataSource,
    data_dir: Path | None = None,
) -> Path:
    """Resolve dataset path, falling back to S3 cache if needed.

    Tries local path first, then downloads from S3 to cache.
    """
    path = get_dataset_path(name, task_type, source=source, data_dir=data_dir)
    if path.exists():
        return path

    # Fall back to S3 cache only when data_dir is not specified
    if data_dir is None:
        return ensure_dataset_cached(name, task_type, source)

    raise FileNotFoundError(f"Dataset not found: {path}")


def _load_parquet_to_arrays(path: Path, task_type: TaskType) -> tuple[np.ndarray, np.ndarray]:
    """Load a parquet file into (X, y) numpy arrays."""
    df = pd.read_parquet(path)
    y = df.pop("y").values
    y = y.astype(np.int64) if task_type == "classification" else y.astype(np.float64)
    X = df.values.astype(np.float64)
    return X, y


from paper.scripts.adapters.store import get_s3_bucket as _get_s3_bucket
from paper.scripts.adapters.store import get_s3_client as _get_s3_client


def ensure_dataset_cached(
    name: str,
    task_type: TaskType,
    source: DataSource,
    *,
    region_name: str | None = None,
) -> Path:
    """Download dataset from S3 to local cache if not already present.

    Uses file-based locking to prevent redundant downloads when multiple
    workers request the same dataset concurrently.

    Parameters
    ----------
    name : str
        Dataset name (without prefix).
    task_type : TaskType
        Either "classification" or "regression".
    source : DataSource
        Either "real" or "synthetic".
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
    prefix = get_dataset_prefix(task_type)
    filename = f"{prefix}{name}.parquet"

    # Create subdirectory structure matching S3: data/{task_type}/{source}/
    local_dir = cache_dir / task_type / source
    local_dir.mkdir(parents=True, exist_ok=True)
    cache_path = local_dir / filename

    # Fast path: already cached
    if cache_path.exists():
        return cache_path

    # Slow path: acquire lock, check again, download if needed
    lock_path = cache_path.with_suffix(".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)

        # Check again after acquiring lock (another worker may have downloaded)
        if cache_path.exists():
            return cache_path

        # Download from S3
        bucket = _get_s3_bucket()
        s3_key = f"{get_data_s3_prefix(task_type, source)}/{filename}"
        s3_path = f"s3://{bucket}/{s3_key}"

        logger.info(f"Downloading dataset from S3: {s3_path} -> {cache_path}")
        try:
            client = _get_s3_client(region_name=region_name)
            response = client.get_object(Bucket=bucket, Key=s3_key)
            content = response["Body"].read()

            # Write atomically to avoid partial files
            tmp_path = cache_path.with_suffix(
                f".tmp.{os.getpid()}.{threading.current_thread().ident}"
            )
            tmp_path.write_bytes(content)
            tmp_path.rename(cache_path)

            logger.info(f"Successfully cached dataset: {cache_path} ({len(content)} bytes)")
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(
                    f"Dataset not found in S3: {s3_path}. "
                    f"Upload datasets with: aws s3 sync paper/data/ s3://{bucket}/data/"
                ) from e
            raise RuntimeError(f"Failed to download dataset from S3: {s3_path}") from e

    return cache_path


def load_dataset(
    name: str,
    task_type: TaskType,
    *,
    source: DataSource | None = None,
    data_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset into (X, y) numpy arrays.

    Resolution order:
    1. Local path (for local development or custom data_dir)
    2. S3 download to /tmp/citrees-data/ cache (for Ray workers)

    Parameters
    ----------
    name : str
        Dataset name (without prefix).
    task_type : TaskType
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
    path = _resolve_dataset_path(name, task_type, resolved_source, data_dir)
    return _load_parquet_to_arrays(path, task_type)


def get_dataset_shape(
    name: str,
    task_type: TaskType,
    *,
    source: DataSource | None = None,
    data_dir: Path | None = None,
) -> tuple[int, int]:
    """Get (n_samples, n_features) from parquet metadata without loading full dataset."""
    resolved_source = source if source is not None else _infer_source(name)
    path = _resolve_dataset_path(name, task_type, resolved_source, data_dir)

    try:
        pf = pq.ParquetFile(path)
        n_samples = pf.metadata.num_rows
        feature_columns = [c for c in pf.schema_arrow.names if c != "y"]
        return int(n_samples), int(len(feature_columns))
    except Exception:
        pass

    X, _y = load_dataset(name, task_type, source=resolved_source, data_dir=data_dir)
    return int(X.shape[0]), int(X.shape[1])


def get_dataset_metadata(
    name: str,
    task_type: TaskType,
    *,
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
    path = _resolve_dataset_path(name, task_type, resolved_source, data_dir)

    metadata: dict[str, Any] = {
        "dataset_source": inferred_source,
        "dataset_type": None,
        "dataset_family": None,
        "n_informative": None,
    }

    try:
        pf = pq.ParquetFile(path)
        schema_meta = pf.schema_arrow.metadata or {}

        # Check if synthetic
        is_synthetic = schema_meta.get(b"synthetic", b"false").decode() == "true"
        if is_synthetic:
            metadata["dataset_source"] = "synthetic"

            # Extract dataset type from name
            if "bias" in name:
                metadata["dataset_type"] = "bias"
            elif "nonlinear" in name:
                metadata["dataset_type"] = "nonlinear"
            elif "corr_noise" in name:
                metadata["dataset_type"] = "corr_noise"
            elif "corr_blocks" in name or "corr" in name and "noise" not in name:
                metadata["dataset_type"] = "correlated"
            elif "redundant" in name:
                metadata["dataset_type"] = "redundant"
            elif "toeplitz" in name:
                metadata["dataset_type"] = "toeplitz"
            elif "weak" in name:
                metadata["dataset_type"] = "weak_signal"
            else:
                metadata["dataset_type"] = "standard"

            # Extract informative indices
            info_json = schema_meta.get(b"informative_indices")
            if info_json:
                import json

                info_indices = json.loads(info_json.decode())
                metadata["n_informative"] = len(info_indices)

            # Extract family from config
            config_json = schema_meta.get(b"config")
            if config_json:
                import json

                config_dict = json.loads(config_json.decode())
                if config_dict.get("n_high_cardinality_noise", 0) > 0:
                    metadata["dataset_family"] = "high_cardinality"
                elif config_dict.get("nonlinear"):
                    metadata["dataset_family"] = "friedman"
                elif config_dict.get("toeplitz_rho", 0) > 0:
                    metadata["dataset_family"] = "toeplitz"
                elif config_dict.get("n_correlated_noise", 0) > 0:
                    metadata["dataset_family"] = "confounded"
                elif config_dict.get("weak_signal"):
                    metadata["dataset_family"] = "weak"
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


def get_cv_splitter(
    task_type: TaskType, n_splits: int, seed: int
) -> "KFold | StratifiedKFold":
    """Return a cross-validation splitter for the given task type.

    Parameters
    ----------
    task_type : TaskType
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

    if task_type == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
