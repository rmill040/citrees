"""Environment utilities for provenance tracking.

This module provides utilities for tracking execution environment:
- Git SHA for code version
- Library versions for reproducibility
- Timestamps for audit trails
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path


def get_repo_root() -> Path:
    """Return the repository root path.

    Resolution order:
    1) CITREES_REPO_ROOT env var
    2) Relative to this file (development fallback)
    """
    env_root = os.environ.get("CITREES_REPO_ROOT")
    if env_root:
        return Path(env_root)
    return Path(__file__).resolve().parents[3]


def utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def get_library_versions() -> dict[str, str]:
    """Return versions of key libraries for reproducibility tracking."""
    versions = {"python": platform.python_version()}
    distributions = {
        "boruta": "Boruta",
        "catboost": "catboost",
        "citrees": "citrees",
        "dcor": "dcor",
        "lightgbm": "lightgbm",
        "mrmr_selection": "mrmr-selection",
        "numba": "numba",
        "numpy": "numpy",
        "pandas": "pandas",
        "pyarrow": "pyarrow",
        "rpy2": "rpy2",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
    }
    for key, distribution in distributions.items():
        with suppress(importlib.metadata.PackageNotFoundError):
            versions[key] = importlib.metadata.version(distribution)

    return versions


def get_hardware_metadata() -> dict[str, str | int]:
    """Return host metadata needed to interpret runtime measurements."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "logical_cpus": os.cpu_count() or 1,
        "ec2_instance_type": os.environ.get("EC2_INSTANCE_TYPE", "unknown"),
    }


def get_container_image() -> str:
    """Return the immutable image identity supplied to a distributed worker."""
    return os.environ.get("CITREES_IMAGE_URI", "unknown").strip() or "unknown"


def get_benchmark_scope() -> dict[str, str]:
    """Return the required immutable scope for a benchmark artifact."""
    from paper.benchmark.adapters.store import _normalize_artifact_prefix
    from paper.benchmark.infra.aws import get_aws_account_id
    from paper.benchmark.pipeline.manifest import validate_manifest_sha256

    artifact_prefix = _normalize_artifact_prefix(os.environ.get("CITREES_ARTIFACT_PREFIX", ""))
    manifest_sha256 = validate_manifest_sha256(os.environ.get("CITREES_MANIFEST_SHA256", ""))
    campaign_sha256 = validate_manifest_sha256(os.environ.get("CITREES_CAMPAIGN_SHA256", ""))
    canonical_manifest_sha256 = validate_manifest_sha256(
        os.environ.get("CITREES_CANONICAL_MANIFEST_SHA256", "")
    )
    gate_receipt_sha256 = validate_manifest_sha256(
        os.environ.get("CITREES_GATE_RECEIPT_SHA256", "")
    )
    runtime_contract_sha256 = validate_manifest_sha256(
        os.environ.get("CITREES_RUNTIME_CONTRACT_SHA256", "")
    )
    aws_account_id = get_aws_account_id()
    return {
        "artifact_prefix": artifact_prefix,
        "campaign_sha256": campaign_sha256,
        "canonical_manifest_sha256": canonical_manifest_sha256,
        "gate_receipt_sha256": gate_receipt_sha256,
        "manifest_sha256": manifest_sha256,
        "runtime_contract_sha256": runtime_contract_sha256,
        "aws_account_id": aws_account_id,
    }


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
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"
