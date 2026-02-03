"""Experiment configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    n_seeds: int = 5
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
        Path to config file. If None, looks for config.yaml first in the
        infra directory, then falls back to config.example.yaml.

    Returns
    -------
    Config
        Loaded configuration.
    """
    if path is None:
        # Look in the infra directory for config files
        infra_dir = Path(__file__).parent.parent / "infra"
        path = infra_dir / "config.yaml"
        if not path.exists():
            path = infra_dir / "config.example.yaml"

    if not path.exists():
        return Config()

    data = yaml.safe_load(path.read_text())
    if not data:
        return Config()

    s3_bucket = data.get("s3_bucket")
    config = Config(
        aws_region=data.get("aws_region", "us-east-1"),
        s3_bucket=s3_bucket,
    )

    if exp_data := data.get("experiment"):
        config.experiment = ExperimentConfig(
            n_seeds=exp_data.get("n_seeds", 5),
            s3_validate_uploads=bool(exp_data.get("s3_validate_uploads", True)),
        )

    return config
