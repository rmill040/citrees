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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import boto3
from botocore.exceptions import ClientError
import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None

TaskType = Literal["classification", "regression"]

_S3_CLIENTS: dict[str | None, Any] = {}


DataSource = Literal["real", "synthetic"]

def get_repo_root() -> Path:
    """Return the repository root path.

    In the Ray EC2 setup we clone to `~/citrees` (i.e., `/home/ubuntu/citrees`).
    Locally, resolve relative to this file.
    """
    if Path("/home/ubuntu/citrees").exists():
        return Path("/home/ubuntu/citrees")
    return Path(__file__).resolve().parents[3]


def get_data_dir(task_type: TaskType, source: DataSource = "real") -> Path:
    """Return the location of dataset directory for the given task type and source.

    Directory structure:
        paper/data/classification/real/      - real classification datasets
        paper/data/classification/synthetic/ - synthetic classification datasets
        paper/data/regression/real/          - real regression datasets
        paper/data/regression/synthetic/     - (future)
    """
    return get_repo_root() / "paper" / "data" / task_type / source


def get_s3_bucket() -> str:
    """Return the S3 bucket name from environment (required)."""
    return os.environ["S3_BUCKET"]


def utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


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
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
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


def load_dataset(
    name: str,
    task_type: TaskType,
    *,
    source: DataSource | None = None,
    data_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset into (X, y) numpy arrays."""
    path = get_dataset_path(name, task_type, source=source, data_dir=data_dir)
    df = pd.read_parquet(path)
    y = df.pop("y").values
    if task_type == "classification":
        y = y.astype(np.int64)
    else:
        y = y.astype(np.float64)
    X = df.values.astype(np.float64)
    return X, y


def get_dataset_shape(
    name: str,
    task_type: TaskType,
    *,
    source: DataSource | None = None,
    data_dir: Path | None = None,
) -> tuple[int, int]:
    """Get (n_samples, n_features) from parquet metadata without loading full dataset."""
    path = get_dataset_path(name, task_type, source=source, data_dir=data_dir)

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


def rankings_s3_path(task_type: TaskType, dataset: str, method_id: str, seed: int, *, bucket: str | None = None) -> str:
    bucket = bucket or get_s3_bucket()
    return f"s3://{bucket}/rankings/{task_type}/{dataset}/{method_id}_seed{seed}.parquet"


def metrics_s3_path(task_type: TaskType, dataset: str, method_id: str, seed: int, *, bucket: str | None = None) -> str:
    bucket = bucket or get_s3_bucket()
    return f"s3://{bucket}/metrics/{task_type}/{dataset}/{method_id}_seed{seed}.parquet"


def _split_s3_path(s3_path: str) -> tuple[str, str]:
    parts = s3_path.replace("s3://", "").split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 path: {s3_path!r}")
    return parts[0], parts[1]


def get_s3_client(*, region_name: str | None = None):
    """Create and cache an S3 client (safe for Ray serialization)."""
    if region_name not in _S3_CLIENTS:
        kwargs = {"region_name": region_name} if region_name else {}
        _S3_CLIENTS[region_name] = boto3.client("s3", **kwargs)
    return _S3_CLIENTS[region_name]


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
