"""Environment utilities for provenance tracking.

This module provides utilities for tracking execution environment:
- Git SHA for code version
- Library versions for reproducibility
- Timestamps for audit trails
"""

from __future__ import annotations

import os
import subprocess
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


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
