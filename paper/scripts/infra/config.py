"""Experiment configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class ExperimentConfig:
    """Experiment configuration for Ray scheduling and S3 settings."""

    type: Literal["classification", "regression"] = "classification"
    n_seeds: int = 10
    stale_timeout_minutes: int = 30
    # Ray CPU scheduling for experiment tasks
    selection_cpus_default: int = 1
    selection_cpus_threaded: int = 8
    selection_cpus_cif: int = 16
    selection_cpus_cif_large: int = 32
    selection_cif_large_threshold: int = 10_000_000
    selection_cpus_overrides: dict[str, int] = field(default_factory=dict)
    # Ray memory scheduling for Stage 1 (in GB)
    selection_memory_gb_default: float = 4.0
    selection_memory_gb_overrides: dict[str, float] = field(default_factory=dict)
    # Stage 2 (evaluation) CPU scheduling
    evaluation_cpus_default: int = 1
    evaluation_cpus_overrides: dict[str, int] = field(default_factory=dict)
    # Ray memory scheduling for Stage 2 (in GB)
    evaluation_memory_gb_default: float = 2.0
    evaluation_memory_gb_overrides: dict[str, float] = field(default_factory=dict)
    # S3 robustness
    s3_validate_uploads: bool = True


@dataclass
class Config:
    """Experiment configuration."""

    aws_region: str = "us-east-1"
    s3_bucket: str | None = None
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def load_config(path: Path | None = None) -> Config:
    """Load configuration from YAML file.

    Parameters
    ----------
    path : Path | None
        Path to config file. If None, looks for config.yaml first,
        then falls back to config.example.yaml in same directory.

    Returns
    -------
    Config
        Loaded configuration.
    """
    if path is None:
        config_dir = Path(__file__).parent
        path = config_dir / "config.yaml"
        if not path.exists():
            path = config_dir / "config.example.yaml"

    if not path.exists():
        return Config()

    data = yaml.safe_load(path.read_text())
    if not data:
        return Config()

    config = Config(
        aws_region=data.get("aws_region", "us-east-1"),
        s3_bucket=data.get("s3_bucket"),
    )

    if exp_data := data.get("experiment"):
        cpu_overrides = exp_data.get("selection_cpus_overrides") or {}
        eval_cpu_overrides = exp_data.get("evaluation_cpus_overrides") or {}
        mem_overrides = exp_data.get("selection_memory_gb_overrides") or {}
        eval_mem_overrides = exp_data.get("evaluation_memory_gb_overrides") or {}
        config.experiment = ExperimentConfig(
            type=exp_data.get("type", "classification"),
            n_seeds=exp_data.get("n_seeds", 10),
            stale_timeout_minutes=exp_data.get("stale_timeout_minutes", 30),
            selection_cpus_default=exp_data.get("selection_cpus_default", 1),
            selection_cpus_threaded=exp_data.get("selection_cpus_threaded", 8),
            selection_cpus_cif=exp_data.get("selection_cpus_cif", 16),
            selection_cpus_cif_large=exp_data.get("selection_cpus_cif_large", 32),
            selection_cif_large_threshold=exp_data.get("selection_cif_large_threshold", 10_000_000),
            selection_cpus_overrides={str(k): int(v) for k, v in cpu_overrides.items()},
            selection_memory_gb_default=float(exp_data.get("selection_memory_gb_default", 4.0)),
            selection_memory_gb_overrides={str(k): float(v) for k, v in mem_overrides.items()},
            evaluation_cpus_default=exp_data.get("evaluation_cpus_default", 1),
            evaluation_cpus_overrides={str(k): int(v) for k, v in eval_cpu_overrides.items()},
            evaluation_memory_gb_default=float(exp_data.get("evaluation_memory_gb_default", 2.0)),
            evaluation_memory_gb_overrides={
                str(k): float(v) for k, v in eval_mem_overrides.items()
            },
            s3_validate_uploads=bool(exp_data.get("s3_validate_uploads", True)),
        )

    return config
