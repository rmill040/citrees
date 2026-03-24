"""Configuration management for citrees experiments.

This module provides:
- settings: Config dataclass and load_config() function
- constants: Static constants (seeds, timeouts, downstream models)
"""

from paper.scripts.config.constants import (
    AWS_REGION,
    CLF_DOWNSTREAM_MODELS,
    DEFAULT_PARAMS,
    N_SEEDS,
    N_SPLITS,
    OPENML_IDS,
    RANDOM_STATE,
    REG_DOWNSTREAM_MODELS,
    S3_BUCKET,
)
from paper.scripts.config.settings import Config, ExperimentConfig, load_config

__all__ = [
    # Settings
    "Config",
    "ExperimentConfig",
    "load_config",
    # Constants
    "RANDOM_STATE",
    "N_SEEDS",
    "N_SPLITS",
    "S3_BUCKET",
    "AWS_REGION",
    "CLF_DOWNSTREAM_MODELS",
    "REG_DOWNSTREAM_MODELS",
    "OPENML_IDS",
    "DEFAULT_PARAMS",
]
