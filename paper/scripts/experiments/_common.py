"""Shared helpers for Ray experiment scripts.

This module centralizes:
- Dataset discovery/loading (parquet in `paper/data/`)
- S3 IO helpers (existence checks + parquet upload/download)
- Artifact path building (rankings/metrics)

The goal is to keep Stage 1/2 scripts focused on computation logic while ensuring consistent IO behavior.
"""

from __future__ import annotations

import io
import os
import subprocess
import threading
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from loguru import logger

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None

TaskType = Literal["classification", "regression"]

_S3_CLIENTS: dict[str | None, Any] = {}
_S3_CLIENT_LOCK = threading.Lock()


DataSource = Literal["real", "synthetic"]


def get_repo_root() -> Path:
    """Return the repository root path.

    Resolution order:
    1) CITREES_REPO_ROOT env var (set by init_ray for local mode workers)
    2) /home/ubuntu/citrees (Ray EC2 cluster setup)
    3) Relative to this file (development fallback)
    """
    env_root = os.environ.get("CITREES_REPO_ROOT")
    if env_root:
        return Path(env_root)
    if Path("/home/ubuntu/citrees").exists():
        return Path("/home/ubuntu/citrees")
    return Path(__file__).resolve().parents[3]


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


def utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def get_library_versions() -> dict[str, str]:
    """Return versions of key libraries for reproducibility tracking."""
    versions: dict[str, str] = {}

    with suppress(Exception):
        import sklearn

        versions["sklearn"] = sklearn.__version__

    with suppress(Exception):
        versions["numpy"] = np.__version__

    with suppress(Exception):
        import numba

        versions["numba"] = numba.__version__

    with suppress(Exception):
        import citrees

        versions["citrees"] = getattr(citrees, "__version__", "unknown")

    return versions


def get_git_sha() -> str:
    """Return the git SHA for provenance (best effort).

    Resolution order:
    1) `GIT_SHA` env var (recommended; works on remote workers without a `.git/` checkout).
    2) `git rev-parse HEAD` from the repo root (best effort, local/dev convenience).
    3) `"unknown"`
    """
    value = os.environ.get("GIT_SHA")
    if value:
        return value.strip()

    repo_root = get_repo_root()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        return "unknown"


def parse_csv_list(value: str | None) -> list[str] | None:
    """Parse comma-separated values into a list of stripped strings."""
    if value is None:
        return None
    items = [v.strip() for v in value.split(",")]
    return [v for v in items if v]


def parse_csv_ints(value: str | None) -> list[int] | None:
    """Parse comma-separated values into a list of ints."""
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return [int(v) for v in items]


def get_dataset_prefix(task_type: TaskType) -> str:
    return "clf_" if task_type == "classification" else "reg_"


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

    # Infer source from name if not provided
    if source is None:
        source = "synthetic" if "synthetic" in name else "real"

    if data_dir is not None:
        search_dir = data_dir / task_type / source
    else:
        search_dir = get_data_dir(task_type, source)

    return search_dir / f"{prefix}{name}.parquet"


def _load_parquet_to_arrays(path: Path, task_type: TaskType) -> tuple[np.ndarray, np.ndarray]:
    """Load a parquet file into (X, y) numpy arrays."""
    df = pd.read_parquet(path)
    y = df.pop("y").values
    y = y.astype(np.int64) if task_type == "classification" else y.astype(np.float64)
    X = df.values.astype(np.float64)
    return X, y


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
        bucket = get_s3_bucket()
        s3_key = f"{get_data_s3_prefix(task_type, source)}/{filename}"
        s3_path = f"s3://{bucket}/{s3_key}"

        logger.info(f"Downloading dataset from S3: {s3_path} -> {cache_path}")
        try:
            client = get_s3_client(region_name=region_name)
            response = client.get_object(Bucket=bucket, Key=s3_key)
            content = response["Body"].read()

            # Write atomically to avoid partial files
            # Use pid + tid to avoid collisions when multiple workers download concurrently
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
    # Infer source if not provided
    if source is None:
        source = "synthetic" if "synthetic" in name else "real"

    # Try local path first
    path = get_dataset_path(name, task_type, source=source, data_dir=data_dir)
    if path.exists():
        return _load_parquet_to_arrays(path, task_type)

    # Fall back to S3 cache (only when data_dir is not specified)
    if data_dir is None:
        cache_path = ensure_dataset_cached(name, task_type, source)
        return _load_parquet_to_arrays(cache_path, task_type)

    # data_dir was specified but file doesn't exist
    raise FileNotFoundError(f"Dataset not found: {path}")


def get_dataset_shape(
    name: str,
    task_type: TaskType,
    *,
    source: DataSource | None = None,
    data_dir: Path | None = None,
) -> tuple[int, int]:
    """Get (n_samples, n_features) from parquet metadata without loading full dataset."""
    # Infer source if not provided
    if source is None:
        source = "synthetic" if "synthetic" in name else "real"

    # Resolve path: local first, then S3 cache
    path = get_dataset_path(name, task_type, source=source, data_dir=data_dir)
    if not path.exists() and data_dir is None:
        path = ensure_dataset_cached(name, task_type, source)

    if pq is not None:
        try:
            pf = pq.ParquetFile(path)
            n_samples = pf.metadata.num_rows
            feature_columns = [c for c in pf.schema_arrow.names if c != "y"]
            return int(n_samples), int(len(feature_columns))
        except Exception:
            pass

    X, _y = load_dataset(name, task_type, source=source, data_dir=data_dir)
    return int(X.shape[0]), int(X.shape[1])


def rankings_s3_path(
    task_type: TaskType, dataset: str, method_id: str, seed: int, *, bucket: str | None = None
) -> str:
    bucket = bucket or get_s3_bucket()
    return f"s3://{bucket}/rankings/{task_type}/{dataset}/{method_id}_seed{seed}.parquet"


def metrics_s3_path(
    task_type: TaskType, dataset: str, method_id: str, seed: int, *, bucket: str | None = None
) -> str:
    bucket = bucket or get_s3_bucket()
    return f"s3://{bucket}/metrics/{task_type}/{dataset}/{method_id}_seed{seed}.parquet"


def _split_s3_path(s3_path: str) -> tuple[str, str]:
    parts = s3_path.replace("s3://", "").split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 path: {s3_path!r}")
    return parts[0], parts[1]


def get_s3_client(*, region_name: str | None = None):
    """Create and cache an S3 client with retry config (safe for Ray serialization)."""
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


def list_s3_completed(
    stage: str,
    task_type: TaskType,
    *,
    region_name: str | None = None,
) -> set[tuple[str, str, int]]:
    """List completed artifacts for a stage in S3.

    Returns a set of (method_id, dataset, seed) tuples parsed from object keys.
    Expected key format: {stage}/{task_type}/{dataset}/{method_id}_seed{seed}.parquet
    """
    bucket = get_s3_bucket()
    prefix = f"{stage}/{task_type}/"
    client = get_s3_client(region_name=region_name)
    paginator = client.get_paginator("list_objects_v2")
    completed: set[tuple[str, str, int]] = set()

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key") or ""
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
            method_id, seed_str = method_seed.rsplit("_seed", 1)
            try:
                seed = int(seed_str)
            except ValueError:
                continue
            completed.add((method_id, dataset, seed))

    return completed


def s3_file_exists(s3_path: str, *, region_name: str | None = None) -> bool:
    """Return True if the object exists; only 'not found' returns False; everything else raises."""
    bucket, key = _split_s3_path(s3_path)
    try:
        get_s3_client(region_name=region_name).head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def download_parquet_from_s3(s3_path: str, *, region_name: str | None = None) -> pd.DataFrame:
    bucket, key = _split_s3_path(s3_path)
    response = get_s3_client(region_name=region_name).get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(response["Body"].read()))


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
    # Infer source if not provided
    inferred_source: DataSource = "synthetic" if "synthetic" in name else "real"
    resolved_source = source if source is not None else inferred_source

    # Resolve path: local first, then S3 cache
    path = get_dataset_path(name, task_type, source=resolved_source, data_dir=data_dir)
    if not path.exists() and data_dir is None:
        path = ensure_dataset_cached(name, task_type, resolved_source)

    metadata: dict[str, Any] = {
        "dataset_source": inferred_source,
        "dataset_type": None,
        "dataset_family": None,
        "n_informative": None,
    }

    if pq is None:
        return metadata

    try:
        pf = pq.ParquetFile(path)
        schema_meta = pf.schema_arrow.metadata or {}

        # Check if synthetic
        is_synthetic = schema_meta.get(b"synthetic", b"false").decode() == "true"
        if is_synthetic:
            metadata["dataset_source"] = "synthetic"

            # Extract dataset type from name
            # Examples: synthetic_bias_noise10_levels50, synthetic_nonlinear_p50_n500
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
                # Family based on generation parameters
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
            # Extract family from dataset name for real datasets
            # e.g., openml_xxx, uci_xxx
            if name.startswith("openml_"):
                metadata["dataset_family"] = "openml"
            elif name.startswith("uci_"):
                metadata["dataset_family"] = "uci"
            else:
                metadata["dataset_family"] = "other"

    except Exception as e:
        logger.warning(f"Failed to parse metadata for dataset {name!r}: {e}")

    return metadata


def upload_parquet_to_s3(
    rows: list[dict[str, Any]],
    s3_path: str,
    *,
    region_name: str | None = None,
    validate: bool = False,
) -> None:
    df = pd.DataFrame(rows)
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    payload = buffer.getvalue()
    expected_bytes = len(payload)
    bucket, key = _split_s3_path(s3_path)
    client = get_s3_client(region_name=region_name)
    client.put_object(Bucket=bucket, Key=key, Body=payload)
    if validate:
        head = client.head_object(Bucket=bucket, Key=key)
        uploaded_bytes = int(head.get("ContentLength") or 0)
        if uploaded_bytes <= 0 or uploaded_bytes != expected_bytes:
            raise RuntimeError(
                f"S3 upload validation failed for {s3_path!r}: expected {expected_bytes} bytes, got {uploaded_bytes}"
            )
