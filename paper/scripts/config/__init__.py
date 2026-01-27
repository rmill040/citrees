"""Configuration management for citrees experiments.

This module provides:
- settings: Config dataclass and load_config() function
- constants: Static constants (seeds, timeouts, downstream models)
"""

from paper.scripts.config.constants import (
    AWS_REGION,
    CLF_DOWNSTREAM_MODELS,
    DEFAULT_PARAMS,
    EVAL_STALE_TIMEOUT_MINUTES,
    EVALUATION_K_VALUES,
    N_SEEDS,
    N_SPLITS,
    OPENML_IDS,
    RANDOM_STATE,
    REG_DOWNSTREAM_MODELS,
    S3_BUCKET,
    STALE_TIMEOUT_MINUTES,
)
from paper.scripts.config.settings import Config, SchedulingConfig, load_config

__all__ = [
    # Settings
    "Config",
    "SchedulingConfig",
    "load_config",
    # Constants
    "RANDOM_STATE",
    "N_SEEDS",
    "N_SPLITS",
    "STALE_TIMEOUT_MINUTES",
    "EVAL_STALE_TIMEOUT_MINUTES",
    "S3_BUCKET",
    "AWS_REGION",
    "CLF_DOWNSTREAM_MODELS",
    "REG_DOWNSTREAM_MODELS",
    "EVALUATION_K_VALUES",
    "OPENML_IDS",
    "DEFAULT_PARAMS",
]
