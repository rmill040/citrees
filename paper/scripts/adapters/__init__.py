"""Infrastructure adapters for experiment execution.

This module provides adapters for external services:
- store: Store protocol + S3Store for artifact storage
- runner: Runner protocol + RayRunner for distributed execution
- data: Dataset loading, discovery, and S3 caching

These adapters implement the ports defined by the pipeline module,
allowing different backends (S3, local, Ray, sequential) to be swapped.
"""

from paper.scripts.adapters.data import (
    ensure_dataset_cached,
    get_data_cache_dir,
    get_data_dir,
    get_data_s3_prefix,
    get_dataset_metadata,
    get_dataset_path,
    get_dataset_prefix,
    get_dataset_shape,
    get_datasets,
    get_repo_root,
    load_dataset,
)
from paper.scripts.adapters.runner import LocalRunner, RayRunner, Runner
from paper.scripts.adapters.store import (
    IgnoreExistsStore,
    S3Store,
    Store,
    get_s3_bucket,
    get_s3_client,
)

__all__ = [
    # Store
    "Store",
    "S3Store",
    "IgnoreExistsStore",
    "get_s3_client",
    "get_s3_bucket",
    # Runner
    "Runner",
    "RayRunner",
    "LocalRunner",
    # Data
    "get_repo_root",
    "get_data_dir",
    "get_data_cache_dir",
    "get_data_s3_prefix",
    "get_dataset_prefix",
    "get_datasets",
    "get_dataset_path",
    "get_dataset_shape",
    "get_dataset_metadata",
    "load_dataset",
    "ensure_dataset_cached",
]
