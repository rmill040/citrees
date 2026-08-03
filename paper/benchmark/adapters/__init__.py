"""Infrastructure adapters for experiment execution.

This module provides adapters for external services:
- store: Store protocol + S3Store for artifact storage
- runner: Runner protocol + LocalRunner for sequential execution
- data: Dataset loading, discovery, and S3 caching

For distributed execution, use the API server (paper.benchmark.api.server)
and pull workers (paper.benchmark.api.worker).
"""

from paper.benchmark.adapters.data import (
    ensure_dataset_cached,
    get_data_cache_dir,
    get_data_dir,
    get_dataset_file_identity,
    get_dataset_identity,
    get_dataset_metadata,
    get_dataset_path,
    get_dataset_payload_identity,
    get_dataset_prefix,
    get_dataset_s3_key,
    get_dataset_s3_payload,
    get_dataset_shape,
    get_datasets,
    get_repo_root,
    load_dataset,
    validate_dataset_path,
    validate_dataset_payload,
)
from paper.benchmark.adapters.runner import LocalRunner, Runner
from paper.benchmark.adapters.store import (
    ArtifactReadError,
    IgnoreExistsStore,
    S3Store,
    Store,
    get_s3_bucket,
    get_s3_client,
)

__all__ = [
    # Store
    "ArtifactReadError",
    "Store",
    "S3Store",
    "IgnoreExistsStore",
    "get_s3_client",
    "get_s3_bucket",
    # Runner
    "Runner",
    "LocalRunner",
    # Data
    "get_repo_root",
    "get_data_dir",
    "get_data_cache_dir",
    "get_dataset_prefix",
    "get_dataset_s3_key",
    "get_dataset_s3_payload",
    "get_datasets",
    "get_dataset_path",
    "get_dataset_identity",
    "get_dataset_file_identity",
    "get_dataset_payload_identity",
    "get_dataset_shape",
    "get_dataset_metadata",
    "load_dataset",
    "ensure_dataset_cached",
    "validate_dataset_path",
    "validate_dataset_payload",
]
