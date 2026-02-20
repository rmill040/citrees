#!/usr/bin/env python3
"""Generate synthetic datasets for distributed experiments.

Each dataset tests ONE phenomenon where ground-truth informative features
are needed to make causal claims. Real datasets (24 clf + 8 reg) carry
the main benchmarking story; these synthetics supplement with controlled
experiments.

Classification (8 datasets):
  toeplitz       — Correlated features (Strobl et al. unbiased selection)
  corr_noise     — Confounded noise (variable importance bias)
  nonlinear      — Friedman #1 binarized (RDC vs linear selectors)
  weak_signal    — Low separation + label noise (Type I error control)
  bias           — High-cardinality noise (known ctree advantage)
  redundant      — Multicollinearity (feature muting validation)
  standard_easy  — Ground-truth ranking (validates metrics pipeline)
  standard_hard  — Ground-truth ranking (high-p, small-n, low separation)

Regression (8 datasets):
  toeplitz       — Correlated features
  corr_noise     — Confounded noise
  friedman1      — Nonlinear (Friedman #1, standard benchmark)
  weak_signal    — Low SNR
  heteroscedastic — Non-constant variance
  redundant      — Multicollinearity
  standard_easy  — Ground-truth ranking
  standard_hard  — Ground-truth ranking

Usage:
    uv run python paper/scripts/data_generation/generate_synthetic_datasets.py
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from sklearn.datasets import make_classification, make_friedman1, make_regression

RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "1718"))
CLF_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "classification" / "synthetic"
REG_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "regression" / "synthetic"


@dataclass
class SyntheticConfig:
    """Configuration for a synthetic dataset."""

    name: str
    n_samples: int
    n_features: int
    n_informative: int
    task: Literal["classification", "regression"] = "classification"

    # Classification-specific
    n_redundant: int = 0
    n_clusters_per_class: int = 2
    class_sep: float = 1.0
    flip_y: float = 0.0

    # Regression-specific
    noise: float = 1.0
    friedman_variant: int | None = None
    heteroscedastic: bool = False
    heteroscedastic_scale: float = 2.0

    # Shared
    seed: int = RANDOM_STATE
    nonlinear: bool = False
    n_correlated_noise: int = 0
    correlated_noise_strength: float = 0.9
    toeplitz_rho: float = 0.0
    weak_signal: bool = False  # Metadata tag only — does not affect generation
    n_high_cardinality_noise: int = 0
    high_cardinality_levels: int = 100


# =============================================================================
# Generators
# =============================================================================


def _shuffle_columns(
    X: np.ndarray, n_informative: int, seed: int, n_redundant: int = 0
) -> tuple[np.ndarray, list[int], list[int]]:
    """Shuffle columns and return new informative/redundant index lists."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(X.shape[1])
    X = X[:, perm]
    inv = np.argsort(perm)
    informative = [int(inv[i]) for i in range(n_informative)]
    redundant = (
        [int(inv[i]) for i in range(n_informative, n_informative + n_redundant)]
        if n_redundant
        else []
    )
    return X, informative, redundant


def generate_standard_clf(
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    X, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_informative=config.n_informative,
        n_redundant=config.n_redundant,
        n_clusters_per_class=config.n_clusters_per_class,
        class_sep=config.class_sep,
        flip_y=config.flip_y,
        random_state=config.seed,
        shuffle=False,
    )
    X, info, redun = _shuffle_columns(X, config.n_informative, config.seed, config.n_redundant)
    return X, y, info, redun


def generate_nonlinear_clf(
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Friedman #1 binarized at the median."""
    X, y_cont = make_friedman1(
        n_samples=config.n_samples,
        n_features=config.n_features,
        noise=1.0,
        random_state=config.seed,
    )
    y = (y_cont >= np.median(y_cont)).astype(int)
    X, info, _ = _shuffle_columns(X, 5, config.seed)
    return X, y, info, []


def generate_toeplitz(
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Toeplitz covariance: cov(i,j) = rho^|i-j|."""
    rng = np.random.RandomState(config.seed)
    p = config.n_features
    idx = np.arange(p)
    cov = config.toeplitz_rho ** np.abs(np.subtract.outer(idx, idx))
    cov += np.eye(p) * 1e-6
    X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=config.n_samples)

    beta = rng.normal(size=config.n_informative)
    signal = X[:, : config.n_informative] @ beta

    if config.task == "classification":
        signal += rng.normal(scale=1.0, size=config.n_samples)
        y = (signal >= np.median(signal)).astype(int)
    else:
        y = signal + rng.normal(scale=config.noise, size=config.n_samples)

    perm = rng.permutation(p)
    X = X[:, perm]
    inv = np.argsort(perm)
    info = [int(inv[i]) for i in range(config.n_informative)]
    return X, y, info, []


def generate_standard_reg(
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    X, y = make_regression(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_informative=config.n_informative,
        noise=config.noise,
        random_state=config.seed,
        shuffle=False,
    )
    X, info, _ = _shuffle_columns(X, config.n_informative, config.seed)
    return X, y, info, []


def generate_friedman_reg(
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    X, y = make_friedman1(
        n_samples=config.n_samples,
        n_features=config.n_features,
        noise=config.noise,
        random_state=config.seed,
    )
    X, info, _ = _shuffle_columns(X, 5, config.seed)
    return X, y, info, []


def generate_heteroscedastic_reg(
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """noise_scale = 1 + heteroscedastic_scale * |x_0|."""
    rng = np.random.RandomState(config.seed)
    X = rng.randn(config.n_samples, config.n_features)
    beta = rng.normal(size=config.n_informative)
    signal = X[:, : config.n_informative] @ beta
    noise_scale = 1 + config.heteroscedastic_scale * np.abs(X[:, 0])
    y = signal + rng.randn(config.n_samples) * noise_scale * config.noise
    X, info, _ = _shuffle_columns(X, config.n_informative, config.seed)
    return X, y, info, []


def generate_redundant(
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Informative features + linear combinations of them."""
    rng = np.random.RandomState(config.seed)
    n_base = config.n_features - config.n_redundant
    X_base = rng.randn(config.n_samples, n_base)

    informative = X_base[:, : config.n_informative]
    redundant_cols = []
    for _ in range(config.n_redundant):
        w = rng.randn(config.n_informative)
        redundant_cols.append((informative @ w + rng.randn(config.n_samples) * 0.1).reshape(-1, 1))
    X = np.hstack([X_base, *redundant_cols])

    beta = rng.normal(size=config.n_informative)
    if config.task == "classification":
        signal = X[:, : config.n_informative] @ beta
        y = (signal >= np.median(signal)).astype(int)
    else:
        y = X[:, : config.n_informative] @ beta + rng.normal(
            scale=config.noise, size=config.n_samples
        )

    perm = rng.permutation(X.shape[1])
    X = X[:, perm]
    inv = np.argsort(perm)
    info = [int(inv[i]) for i in range(config.n_informative)]
    redun = [int(inv[i]) for i in range(n_base, X.shape[1])]
    return X, y, info, redun


def add_high_cardinality_noise(
    X: np.ndarray, n_noise: int, n_levels: int, seed: int
) -> tuple[np.ndarray, list[int]]:
    rng = np.random.RandomState(seed)
    noise = rng.randint(0, n_levels, size=(X.shape[0], n_noise)).astype(float)
    noise_indices = list(range(X.shape[1], X.shape[1] + n_noise))
    return np.hstack([X, noise]), noise_indices


def add_correlated_noise(
    X: np.ndarray, informative_indices: list[int], n_noise: int, corr: float, seed: int
) -> tuple[np.ndarray, list[int]]:
    rng = np.random.RandomState(seed)
    cols = []
    for i in range(n_noise):
        base_idx = informative_indices[i % len(informative_indices)]
        cols.append(
            (corr * X[:, base_idx] + rng.randn(X.shape[0]) * np.sqrt(1 - corr**2)).reshape(-1, 1)
        )
    noise_indices = list(range(X.shape[1], X.shape[1] + n_noise))
    return np.hstack([X, *cols]), noise_indices


# =============================================================================
# Dispatch + serialization
# =============================================================================


def _generate_base(config: SyntheticConfig) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    if config.toeplitz_rho > 0:
        return generate_toeplitz(config)
    if config.n_redundant > 0:
        return generate_redundant(config)
    if config.task == "classification":
        if config.nonlinear:
            return generate_nonlinear_clf(config)
        return generate_standard_clf(config)
    # regression
    if config.friedman_variant is not None:
        return generate_friedman_reg(config)
    if config.heteroscedastic:
        return generate_heteroscedastic_reg(config)
    return generate_standard_reg(config)


def generate_dataset(config: SyntheticConfig) -> pa.Table:
    """Generate dataset and return as PyArrow table with metadata."""
    X, y, informative_indices, redundant_indices = _generate_base(config)

    noise_indices: list[int] = []
    corr_noise_indices: list[int] = []

    if config.n_high_cardinality_noise > 0:
        X, noise_indices = add_high_cardinality_noise(
            X, config.n_high_cardinality_noise, config.high_cardinality_levels, config.seed + 1
        )

    if config.n_correlated_noise > 0:
        X, corr_noise_indices = add_correlated_noise(
            X,
            informative_indices,
            config.n_correlated_noise,
            config.correlated_noise_strength,
            config.seed + 3,
        )

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["y"] = y.astype(np.float64 if config.task == "regression" else np.int64)

    metadata = {
        "synthetic": "true",
        "task": config.task,
        "config": json.dumps(asdict(config)),
        "informative_indices": json.dumps(informative_indices),
        "redundant_indices": json.dumps(redundant_indices),
        "noise_indices": json.dumps(noise_indices),
        "correlated_noise_indices": json.dumps(corr_noise_indices),
        "n_features_final": str(X.shape[1]),
    }

    table = pa.Table.from_pandas(df)
    return table.replace_schema_metadata({k.encode(): v.encode() for k, v in metadata.items()})


# =============================================================================
# Dataset definitions — one per paper claim
# =============================================================================


def get_classification_configs() -> list[SyntheticConfig]:
    return [
        # Correlated features — Strobl et al. unbiased selection
        SyntheticConfig(
            name="synthetic_toeplitz_p100_k10_n1000_r0.95",
            n_samples=1000,
            n_features=100,
            n_informative=10,
            toeplitz_rho=0.95,
        ),
        # Confounded noise — variable importance bias
        SyntheticConfig(
            name="synthetic_corr_noise_p100_k10_n1000_noise20_r0.9",
            n_samples=1000,
            n_features=100,
            n_informative=10,
            class_sep=1.0,
            n_correlated_noise=20,
            correlated_noise_strength=0.9,
        ),
        # Nonlinear — RDC should beat MC
        SyntheticConfig(
            name="synthetic_nonlinear_p100_n1000",
            n_samples=1000,
            n_features=100,
            n_informative=5,
            nonlinear=True,
        ),
        # Weak signal — Type I error control
        SyntheticConfig(
            name="synthetic_weak_p100_k10_n1000_sep0.1_flip0.1",
            n_samples=1000,
            n_features=100,
            n_informative=10,
            class_sep=0.1,
            flip_y=0.1,
            weak_signal=True,
        ),
        # High-cardinality bias — known ctree advantage
        SyntheticConfig(
            name="synthetic_bias_noise50_levels500",
            n_samples=1000,
            n_features=50,
            n_informative=10,
            class_sep=1.0,
            n_high_cardinality_noise=50,
            high_cardinality_levels=500,
        ),
        # Redundant features — feature muting
        SyntheticConfig(
            name="synthetic_redundant20",
            n_samples=1000,
            n_features=50,
            n_informative=10,
            n_redundant=20,
            class_sep=1.0,
        ),
        # Ground truth easy — validates metrics pipeline
        SyntheticConfig(
            name="synthetic_p100_k10_n1000_sep2.0",
            n_samples=1000,
            n_features=100,
            n_informative=10,
            class_sep=2.0,
        ),
        # Ground truth hard — high-p, small-n, low separation
        SyntheticConfig(
            name="synthetic_p1000_k5_n200_sep0.5",
            n_samples=200,
            n_features=1000,
            n_informative=5,
            class_sep=0.5,
        ),
    ]


def get_regression_configs() -> list[SyntheticConfig]:
    return [
        # Correlated features
        SyntheticConfig(
            name="synthetic_toeplitz_p100_k10_n1000_r0.95",
            n_samples=1000,
            n_features=100,
            n_informative=10,
            task="regression",
            toeplitz_rho=0.95,
        ),
        # Confounded noise
        SyntheticConfig(
            name="synthetic_corr_noise_p100_k10_n1000_noise20_r0.9",
            n_samples=1000,
            n_features=100,
            n_informative=10,
            task="regression",
            n_correlated_noise=20,
            correlated_noise_strength=0.9,
        ),
        # Nonlinear — Friedman #1
        SyntheticConfig(
            name="synthetic_friedman1_p100_n1000_noise1.0",
            n_samples=1000,
            n_features=100,
            n_informative=5,
            task="regression",
            friedman_variant=1,
            noise=1.0,
        ),
        # Weak signal — low SNR
        SyntheticConfig(
            name="synthetic_weak_p100_k10_n1000_noise50.0",
            n_samples=1000,
            n_features=100,
            n_informative=10,
            task="regression",
            noise=50.0,
            weak_signal=True,
        ),
        # Heteroscedastic — non-constant variance
        SyntheticConfig(
            name="synthetic_heteroscedastic_scale4.0_noise5.0",
            n_samples=1000,
            n_features=50,
            n_informative=10,
            task="regression",
            heteroscedastic=True,
            heteroscedastic_scale=4.0,
            noise=5.0,
        ),
        # Redundant features
        SyntheticConfig(
            name="synthetic_redundant20",
            n_samples=1000,
            n_features=50,
            n_informative=10,
            n_redundant=20,
            task="regression",
        ),
        # Ground truth easy
        SyntheticConfig(
            name="synthetic_p100_k10_n1000_noise1.0",
            n_samples=1000,
            n_features=100,
            n_informative=10,
            task="regression",
            noise=1.0,
        ),
        # Ground truth hard
        SyntheticConfig(
            name="synthetic_p500_k5_n200_noise10.0",
            n_samples=200,
            n_features=500,
            n_informative=5,
            task="regression",
            noise=10.0,
        ),
    ]


def main() -> None:
    CLF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clf_configs = get_classification_configs()
    logger.info(f"Generating {len(clf_configs)} classification datasets...")
    for config in clf_configs:
        table = generate_dataset(config)
        path = CLF_OUTPUT_DIR / f"clf_{config.name}.parquet"
        pq.write_table(table, path)
        logger.info(f"  {config.name} ({path.stat().st_size / 1024:.0f} KB)")

    reg_configs = get_regression_configs()
    logger.info(f"Generating {len(reg_configs)} regression datasets...")
    for config in reg_configs:
        table = generate_dataset(config)
        path = REG_OUTPUT_DIR / f"reg_{config.name}.parquet"
        pq.write_table(table, path)
        logger.info(f"  {config.name} ({path.stat().st_size / 1024:.0f} KB)")

    logger.info(
        f"Done: {len(clf_configs)} clf + {len(reg_configs)} reg = {len(clf_configs) + len(reg_configs)} total"
    )


if __name__ == "__main__":
    main()
