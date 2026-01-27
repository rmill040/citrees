"""Core types for the experiment infrastructure.

This module provides clean, frozen dataclasses for experiment configuration
and results. All types are immutable and hashable for use in sets and dicts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

TaskType = Literal["classification", "regression"]
StatusType = Literal["done", "failed", "skipped", "no_rankings"]
StageType = Literal["rankings", "metrics"]


@dataclass(frozen=True)
class MethodConfig:
    """Configuration for a feature selection method.

    Immutable configuration identifying a specific method variant.
    The label property provides a stable identifier for filenames.

    Parameters
    ----------
    name : str
        Base method name (e.g., "cit", "rf", "boruta").
    params : tuple[tuple[str, Any], ...], optional
        Model hyperparameters to override defaults.

    Examples
    --------
    >>> mc = MethodConfig("mc")
    >>> mc.label
    'mc__99914b932bd37a50'

    >>> mc = MethodConfig("cit", params=(("feature_muting", True),))
    >>> mc.label
    'cit__816e78cf71d84843'
    """

    name: str
    params: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        """Stable identifier for filenames and reporting.

        Always returns "{name}__{hash}" format for consistency.
        The hash is the first 16 characters of the MD5 digest of the
        JSON-encoded params dict (empty dict {} if no params).
        """
        payload = json.dumps(dict(self.params), sort_keys=True, default=str)
        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]
        return f"{self.name}__{digest}"

    @property
    def params_dict(self) -> dict[str, Any]:
        """Return params as a mutable dict."""
        return dict(self.params)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a single experiment run.

    Identifies a unique (method, dataset, seed, task) combination.
    Immutable and hashable for use as dict keys and set members.

    Parameters
    ----------
    method : MethodConfig
        Feature selection method configuration.
    dataset : str
        Dataset name (without prefix).
    seed : int
        Random seed index.
    task : TaskType
        Task type: "classification" or "regression".
    """

    method: MethodConfig
    dataset: str
    seed: int
    task: TaskType

    @property
    def key(self) -> tuple[str, str, int]:
        """Tuple key for S3 lookups: (method_label, dataset, seed)."""
        return (self.method.label, self.dataset, self.seed)

    def __str__(self) -> str:
        return f"{self.method.label}/{self.dataset}/seed{self.seed}"


@dataclass
class Result:
    """Result of a single experiment execution.

    Mutable dataclass capturing execution status and outputs.

    Parameters
    ----------
    config : ExperimentConfig
        The experiment configuration that was executed.
    status : StatusType
        Execution status: "done", "failed", "skipped", or "no_rankings".
    elapsed_seconds : float
        Wall time for execution.
    error : str, optional
        Error message if status is "failed".
    error_type : str, optional
        Exception type name if status is "failed".
    traceback : str, optional
        Full traceback if status is "failed".
    data : pd.DataFrame, optional
        Result data (rankings or metrics).
    s3_path : str, optional
        S3 path where result was saved.
    hostname : str, optional
        Hostname where task executed.
    """

    config: ExperimentConfig
    status: StatusType
    elapsed_seconds: float = 0.0
    error: str | None = None
    error_type: str | None = None
    traceback: str | None = None
    data: pd.DataFrame | None = None
    s3_path: str | None = None
    hostname: str | None = None

    @property
    def is_success(self) -> bool:
        """True if status is 'done'."""
        return self.status == "done"

    @property
    def is_failure(self) -> bool:
        """True if status is 'failed'."""
        return self.status == "failed"

    @property
    def is_skipped(self) -> bool:
        """True if status is 'skipped'."""
        return self.status == "skipped"

    @property
    def is_no_rankings(self) -> bool:
        """True if status is 'no_rankings'."""
        return self.status == "no_rankings"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for logging and compatibility."""
        return {
            "status": self.status,
            "method": self.config.method.label,
            "dataset": self.config.dataset,
            "seed": self.config.seed,
            "elapsed_seconds": self.elapsed_seconds,
            "s3_path": self.s3_path,
            "error": self.error,
            "error_type": self.error_type,
            "hostname": self.hostname,
        }
