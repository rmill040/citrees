"""Constants for citrees benchmark experiments."""

from paper.benchmark.config.constants import (
    AWS_REGION,
    CLF_DOWNSTREAM_MODELS,
    DEFAULT_PARAMS,
    EVALUATION_K_VALUES,
    N_SEEDS,
    N_SPLITS,
    OPENML_IDS,
    RANDOM_STATE,
    REG_DOWNSTREAM_MODELS,
    S3_BUCKET,
)

__all__ = [
    "RANDOM_STATE",
    "N_SEEDS",
    "N_SPLITS",
    "S3_BUCKET",
    "AWS_REGION",
    "CLF_DOWNSTREAM_MODELS",
    "REG_DOWNSTREAM_MODELS",
    "EVALUATION_K_VALUES",
    "OPENML_IDS",
    "DEFAULT_PARAMS",
]
