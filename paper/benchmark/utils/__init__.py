"""Shared utilities for paper experiments.

This module provides pure utility functions:
- env: Environment helpers (git SHA, library versions, timestamps)
- metrics: Evaluation metrics (precision@k, recall@k, etc.)
"""

from paper.benchmark.utils.env import (
    get_git_sha,
    get_library_versions,
    get_repo_root,
    utc_now_iso,
)
from paper.benchmark.utils.metrics import (
    f1_at_k,
    jaccard_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    # Environment
    "get_repo_root",
    "utc_now_iso",
    "get_library_versions",
    "get_git_sha",
    # Metrics
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "jaccard_at_k",
]
