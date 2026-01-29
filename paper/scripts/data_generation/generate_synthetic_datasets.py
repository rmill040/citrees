#!/usr/bin/env python3
"""Generate synthetic datasets for distributed experiments.

Creates synthetic classification and regression datasets with known informative features.
Metadata (generation params + ground truth) is stored IN the parquet schema.

Classification dataset types:
1. STANDARD: Varying n_features, n_informative, class_sep, n_samples
2. SELECTION BIAS: High-cardinality noise features to demonstrate bias
3. NONLINEAR: Friedman #1 function to test RDC vs linear selectors
4. CORRELATED: Correlated feature blocks to test conditional importance
5. REDUNDANT: Linear combinations of informative features (multicollinearity)
6. CORR_NOISE: Correlated noise features (confounders)
7. TOEPLITZ: Toeplitz covariance structure (global correlation)
8. WEAK_SIGNAL: Low class_sep + label noise (hard signal)

Regression dataset types:
1. STANDARD: Linear regression with varying dimensionality and noise
2. FRIEDMAN: Nonlinear relationships (Friedman #1, #2, #3)
3. CORRELATED: Correlated feature blocks
4. REDUNDANT: Multicollinearity
5. HETEROSCEDASTIC: Non-constant variance
6. TOEPLITZ: Global correlation structure
7. WEAK_SIGNAL: Low SNR (high noise)

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
from sklearn.datasets import (
    make_classification,
    make_friedman1,
    make_friedman2,
    make_friedman3,
    make_regression,
)

# Configurable via environment variable for reproducibility
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
    noise: float = 1.0  # Gaussian noise std for regression
    friedman_variant: int | None = None  # 1, 2, or 3 for Friedman functions
    heteroscedastic: bool = False  # Non-constant variance
    heteroscedastic_scale: float = 2.0  # Max noise multiplier

    # Shared
    seed: int = RANDOM_STATE
    nonlinear: bool = False
    n_correlated_blocks: int = 0
    correlation_strength: float = 0.9
    n_correlated_noise: int = 0
    correlated_noise_strength: float = 0.9
    toeplitz_rho: float = 0.0
    weak_signal: bool = False
    n_high_cardinality_noise: int = 0
    high_cardinality_levels: int = 100


def generate_standard_dataset(
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Generate standard synthetic classification dataset."""
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

    # Shuffle columns but track informative indices
    rng = np.random.RandomState(config.seed)
    perm = rng.permutation(config.n_features)
    X = X[:, perm]

    inv_perm = np.argsort(perm)
    informative_indices = [int(inv_perm[i]) for i in range(config.n_informative)]
    if config.n_redundant > 0:
        redundant_indices = [
            int(inv_perm[i])
            for i in range(config.n_informative, config.n_informative + config.n_redundant)
        ]
    else:
        redundant_indices = []

    return X, y, informative_indices, redundant_indices


def generate_nonlinear_dataset(
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Generate dataset with nonlinear relationships (Friedman #1)."""
    n_informative = 5  # Friedman1 always has 5 informative

    X, y_cont = make_friedman1(
        n_samples=config.n_samples,
        n_features=config.n_features,
        noise=1.0,
        random_state=config.seed,
    )

    # Convert to binary
    y = (y_cont >= np.median(y_cont)).astype(int)

    # Shuffle columns
    rng = np.random.RandomState(config.seed)
    perm = rng.permutation(config.n_features)
    X = X[:, perm]

    inv_perm = np.argsort(perm)
    informative_indices = [int(inv_perm[i]) for i in range(n_informative)]

    return X, y, informative_indices, []


def generate_toeplitz_dataset(
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Generate dataset with Toeplitz correlation structure."""
    rng = np.random.RandomState(config.seed)
    p = config.n_features
    rho = config.toeplitz_rho

    idx = np.arange(p)
    cov = rho ** np.abs(np.subtract.outer(idx, idx))
    cov += np.eye(p) * 1e-6
    X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=config.n_samples)

    # Linear signal on first n_informative features
    beta = rng.normal(size=config.n_informative)
    signal = X[:, : config.n_informative] @ beta
    signal += rng.normal(scale=1.0, size=config.n_samples)
    y = (signal >= np.median(signal)).astype(int)

    # Shuffle columns but track informative indices
    perm = rng.permutation(p)
    X = X[:, perm]
    inv_perm = np.argsort(perm)
    informative_indices = [int(inv_perm[i]) for i in range(config.n_informative)]

    return X, y, informative_indices, []


# =============================================================================
# REGRESSION GENERATORS
# =============================================================================


def generate_standard_regression(
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Generate standard linear regression dataset.

    y = X @ beta + noise, where beta has n_informative non-zero coefficients.
    """
    X, y = make_regression(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_informative=config.n_informative,
        noise=config.noise,
        random_state=config.seed,
        shuffle=False,
    )

    # Shuffle columns but track informative indices
    rng = np.random.RandomState(config.seed)
    perm = rng.permutation(config.n_features)
    X = X[:, perm]

    inv_perm = np.argsort(perm)
    informative_indices = [int(inv_perm[i]) for i in range(config.n_informative)]

    return X, y, informative_indices, []


def generate_friedman_regression(
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Generate nonlinear regression using Friedman functions.

    Friedman #1: y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4 + noise
                 5 informative features
    Friedman #2: y = sqrt(x0^2 + (x1*x2 - 1/(x1*x3))^2) + noise
                 4 informative features
    Friedman #3: y = atan((x1*x2 - 1/(x1*x3)) / x0) + noise
                 4 informative features
    """
    variant = config.friedman_variant or 1

    if variant == 1:
        X, y = make_friedman1(
            n_samples=config.n_samples,
            n_features=config.n_features,
            noise=config.noise,
            random_state=config.seed,
        )
        n_informative = 5  # Friedman1 always has 5 informative
    elif variant == 2:
        X, y = make_friedman2(
            n_samples=config.n_samples,
            noise=config.noise,
            random_state=config.seed,
        )
        # Friedman2 only generates 4 features by default, need to add noise features
        n_informative = 4
        if config.n_features > 4:
            rng = np.random.RandomState(config.seed)
            noise_features = rng.randn(config.n_samples, config.n_features - 4)
            X = np.hstack([X, noise_features])
    elif variant == 3:
        X, y = make_friedman3(
            n_samples=config.n_samples,
            noise=config.noise,
            random_state=config.seed,
        )
        # Friedman3 only generates 4 features by default, need to add noise features
        n_informative = 4
        if config.n_features > 4:
            rng = np.random.RandomState(config.seed)
            noise_features = rng.randn(config.n_samples, config.n_features - 4)
            X = np.hstack([X, noise_features])
    else:
        raise ValueError(f"friedman_variant must be 1, 2, or 3, got {variant}")

    # Shuffle columns but track informative indices
    rng = np.random.RandomState(config.seed)
    perm = rng.permutation(X.shape[1])
    X = X[:, perm]

    inv_perm = np.argsort(perm)
    informative_indices = [int(inv_perm[i]) for i in range(n_informative)]

    return X, y, informative_indices, []


def generate_toeplitz_regression(
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Generate regression dataset with Toeplitz correlation structure."""
    rng = np.random.RandomState(config.seed)
    p = config.n_features
    rho = config.toeplitz_rho

    idx = np.arange(p)
    cov = rho ** np.abs(np.subtract.outer(idx, idx))
    cov += np.eye(p) * 1e-6
    X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=config.n_samples)

    # Linear signal on first n_informative features
    beta = rng.normal(size=config.n_informative)
    y = X[:, : config.n_informative] @ beta
    y += rng.normal(scale=config.noise, size=config.n_samples)

    # Shuffle columns but track informative indices
    perm = rng.permutation(p)
    X = X[:, perm]
    inv_perm = np.argsort(perm)
    informative_indices = [int(inv_perm[i]) for i in range(config.n_informative)]

    return X, y, informative_indices, []


def generate_heteroscedastic_regression(
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Generate regression with heteroscedastic (non-constant variance) noise.

    noise_scale = 1 + heteroscedastic_scale * |x_0|
    This creates variance that grows with the first feature.
    """
    rng = np.random.RandomState(config.seed)

    # Generate base X
    X = rng.randn(config.n_samples, config.n_features)

    # Linear signal on first n_informative features
    beta = rng.normal(size=config.n_informative)
    signal = X[:, : config.n_informative] @ beta

    # Heteroscedastic noise: variance increases with |x_0|
    noise_scale = 1 + config.heteroscedastic_scale * np.abs(X[:, 0])
    noise = rng.randn(config.n_samples) * noise_scale * config.noise
    y = signal + noise

    # Shuffle columns but track informative indices
    perm = rng.permutation(config.n_features)
    X = X[:, perm]
    inv_perm = np.argsort(perm)
    informative_indices = [int(inv_perm[i]) for i in range(config.n_informative)]

    return X, y, informative_indices, []


def generate_redundant_regression(
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Generate regression with redundant features (multicollinearity).

    Creates n_redundant features that are linear combinations of informative features.
    """
    rng = np.random.RandomState(config.seed)

    # Generate base features
    n_base = config.n_features - config.n_redundant
    X_base = rng.randn(config.n_samples, n_base)

    # Create redundant features as linear combinations of informative features
    if config.n_redundant > 0 and config.n_informative > 0:
        informative_features = X_base[:, : config.n_informative]
        redundant = []
        for _ in range(config.n_redundant):
            # Random linear combination with small noise
            weights = rng.randn(config.n_informative)
            combination = informative_features @ weights
            combination += rng.randn(config.n_samples) * 0.1  # Small noise
            redundant.append(combination.reshape(-1, 1))
        X_redundant = np.hstack(redundant)
        X = np.hstack([X_base, X_redundant])
    else:
        X = X_base

    # Linear signal on first n_informative features
    beta = rng.normal(size=config.n_informative)
    y = X[:, : config.n_informative] @ beta
    y += rng.normal(scale=config.noise, size=config.n_samples)

    # Shuffle columns but track informative and redundant indices
    perm = rng.permutation(X.shape[1])
    X = X[:, perm]
    inv_perm = np.argsort(perm)
    informative_indices = [int(inv_perm[i]) for i in range(config.n_informative)]
    redundant_indices = (
        [int(inv_perm[i]) for i in range(n_base, X.shape[1])] if config.n_redundant > 0 else []
    )

    return X, y, informative_indices, redundant_indices


def add_high_cardinality_noise(
    X: np.ndarray, n_noise: int, n_levels: int, seed: int
) -> tuple[np.ndarray, list[int]]:
    """Add high-cardinality categorical noise (selection bias test)."""
    rng = np.random.RandomState(seed)
    noise = rng.randint(0, n_levels, size=(X.shape[0], n_noise)).astype(float)
    X_aug = np.hstack([X, noise])
    noise_indices = list(range(X.shape[1], X_aug.shape[1]))
    return X_aug, noise_indices


def add_correlated_blocks(
    X: np.ndarray, informative_indices: list[int], n_blocks: int, corr: float, seed: int
) -> tuple[np.ndarray, list[int]]:
    """Add features correlated with informative features."""
    rng = np.random.RandomState(seed)
    correlated = []
    for idx in informative_indices[:n_blocks]:
        noise = rng.randn(X.shape[0]) * np.sqrt(1 - corr**2)
        correlated.append((corr * X[:, idx] + noise).reshape(-1, 1))

    if correlated:
        X_aug = np.hstack([X, np.hstack(correlated)])
        corr_indices = list(range(X.shape[1], X_aug.shape[1]))
    else:
        X_aug = X
        corr_indices = []

    return X_aug, corr_indices


def add_correlated_noise(
    X: np.ndarray, informative_indices: list[int], n_noise: int, corr: float, seed: int
) -> tuple[np.ndarray, list[int]]:
    """Add correlated noise features (confounders)."""
    if n_noise <= 0 or not informative_indices:
        return X, []

    rng = np.random.RandomState(seed)
    correlated = []
    for i in range(n_noise):
        base_idx = informative_indices[i % len(informative_indices)]
        noise = rng.randn(X.shape[0]) * np.sqrt(1 - corr**2)
        correlated.append((corr * X[:, base_idx] + noise).reshape(-1, 1))

    X_aug = np.hstack([X, np.hstack(correlated)])
    corr_indices = list(range(X.shape[1], X_aug.shape[1]))

    return X_aug, corr_indices


def generate_dataset(config: SyntheticConfig) -> tuple[pa.Table, dict]:
    """Generate dataset and return as PyArrow table with metadata."""
    # Generate base data based on task
    if config.task == "regression":
        X, y, informative_indices, redundant_indices = _generate_regression_base(config)
    else:  # classification
        X, y, informative_indices, redundant_indices = _generate_classification_base(config)

    noise_indices = []
    corr_block_indices: list[int] = []
    corr_noise_indices: list[int] = []

    # Add high-cardinality noise
    if config.n_high_cardinality_noise > 0:
        X, noise_indices = add_high_cardinality_noise(
            X, config.n_high_cardinality_noise, config.high_cardinality_levels, config.seed + 1
        )

    # Add correlated features
    if config.n_correlated_blocks > 0:
        X, corr_block_indices = add_correlated_blocks(
            X,
            informative_indices,
            config.n_correlated_blocks,
            config.correlation_strength,
            config.seed + 2,
        )

    # Add correlated noise (confounders)
    if config.n_correlated_noise > 0:
        X, corr_noise_indices = add_correlated_noise(
            X,
            informative_indices,
            config.n_correlated_noise,
            config.correlated_noise_strength,
            config.seed + 3,
        )

    correlated_indices = corr_block_indices + corr_noise_indices

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if config.task == "regression":
        df["y"] = y.astype(np.float64)
    else:
        df["y"] = y.astype(np.int64)

    # Metadata to store in parquet
    metadata = {
        "synthetic": "true",
        "task": config.task,
        "config": json.dumps(asdict(config)),
        "informative_indices": json.dumps(informative_indices),
        "redundant_indices": json.dumps(redundant_indices),
        "noise_indices": json.dumps(noise_indices),
        "correlated_indices": json.dumps(correlated_indices),
        "correlated_noise_indices": json.dumps(corr_noise_indices),
        "n_features_final": str(X.shape[1]),
    }

    # Convert to PyArrow with metadata
    table = pa.Table.from_pandas(df)
    table = table.replace_schema_metadata({k.encode(): v.encode() for k, v in metadata.items()})

    return table, metadata


def _generate_classification_base(
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Generate base classification data based on config."""
    if config.toeplitz_rho > 0:
        return generate_toeplitz_dataset(config)
    elif config.nonlinear:
        return generate_nonlinear_dataset(config)
    else:
        return generate_standard_dataset(config)


def _generate_regression_base(
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Generate base regression data based on config."""
    if config.friedman_variant is not None:
        return generate_friedman_regression(config)
    elif config.toeplitz_rho > 0:
        return generate_toeplitz_regression(config)
    elif config.heteroscedastic:
        return generate_heteroscedastic_regression(config)
    elif config.n_redundant > 0:
        return generate_redundant_regression(config)
    else:
        return generate_standard_regression(config)


def get_all_configs() -> list[SyntheticConfig]:
    """Generate all synthetic dataset configurations.

    Uses single RANDOM_STATE for all datasets. Variance in experiments
    comes from method seeds (N_SEEDS in the distributed pipeline).
    """
    configs = []

    # =========================================================================
    # 1. STANDARD: Varying dimensionality, signal, sample size
    # =========================================================================
    for n_features in [50, 100, 500, 1000]:
        for n_informative in [5, 10, 20]:
            if n_informative >= n_features:
                continue
            for n_samples in [200, 500, 1000]:
                for class_sep in [0.5, 1.0, 2.0]:
                    configs.append(
                        SyntheticConfig(
                            name=f"synthetic_p{n_features}_k{n_informative}_n{n_samples}_sep{class_sep}",
                            n_samples=n_samples,
                            n_features=n_features,
                            n_informative=n_informative,
                            class_sep=class_sep,
                            seed=RANDOM_STATE,
                        )
                    )

    # =========================================================================
    # 2. SELECTION BIAS: High-cardinality noise
    # =========================================================================
    for n_noise in [10, 20, 50]:
        for n_levels in [50, 100, 500]:
            configs.append(
                SyntheticConfig(
                    name=f"synthetic_bias_noise{n_noise}_levels{n_levels}",
                    n_samples=1000,
                    n_features=50,
                    n_informative=10,
                    class_sep=1.0,
                    seed=RANDOM_STATE,
                    n_high_cardinality_noise=n_noise,
                    high_cardinality_levels=n_levels,
                )
            )

    # =========================================================================
    # 3. NONLINEAR: Friedman #1 (tests RDC vs linear)
    # =========================================================================
    for n_features in [50, 100, 500]:
        for n_samples in [500, 1000]:
            configs.append(
                SyntheticConfig(
                    name=f"synthetic_nonlinear_p{n_features}_n{n_samples}",
                    n_samples=n_samples,
                    n_features=n_features,
                    n_informative=5,  # Friedman1 always 5
                    seed=RANDOM_STATE,
                    nonlinear=True,
                )
            )

    # =========================================================================
    # 4. CORRELATED: Tests conditional importance
    # =========================================================================
    for n_corr in [5, 10]:
        for corr_strength in [0.7, 0.9, 0.95]:
            configs.append(
                SyntheticConfig(
                    name=f"synthetic_corr_blocks{n_corr}_r{corr_strength}",
                    n_samples=1000,
                    n_features=50,
                    n_informative=10,
                    class_sep=1.0,
                    seed=RANDOM_STATE,
                    n_correlated_blocks=n_corr,
                    correlation_strength=corr_strength,
                )
            )

    # =========================================================================
    # 5. REDUNDANT: Linear combinations of informative features
    # Tests how methods handle multicollinearity
    # =========================================================================
    for n_redundant in [5, 10, 20]:
        configs.append(
            SyntheticConfig(
                name=f"synthetic_redundant{n_redundant}",
                n_samples=1000,
                n_features=50,
                n_informative=10,
                n_redundant=n_redundant,
                class_sep=1.0,
                seed=RANDOM_STATE,
            )
        )

    # =========================================================================
    # 6. CORRELATED NOISE: Confounders correlated with informative features
    # =========================================================================
    for n_corr in [10, 20]:
        for corr_strength in [0.7, 0.9]:
            configs.append(
                SyntheticConfig(
                    name=f"synthetic_corr_noise_p100_k10_n1000_noise{n_corr}_r{corr_strength}",
                    n_samples=1000,
                    n_features=100,
                    n_informative=10,
                    class_sep=1.0,
                    seed=RANDOM_STATE,
                    n_correlated_noise=n_corr,
                    correlated_noise_strength=corr_strength,
                )
            )

    # =========================================================================
    # 7. TOEPLITZ: Global correlation structure
    # =========================================================================
    for n_features in [50, 100]:
        for n_informative in [5, 10]:
            for n_samples in [500, 1000]:
                for rho in [0.7, 0.9, 0.95]:
                    configs.append(
                        SyntheticConfig(
                            name=f"synthetic_toeplitz_p{n_features}_k{n_informative}_n{n_samples}_r{rho}",
                            n_samples=n_samples,
                            n_features=n_features,
                            n_informative=n_informative,
                            seed=RANDOM_STATE,
                            toeplitz_rho=rho,
                        )
                    )

    # =========================================================================
    # 8. WEAK SIGNAL: Low separation + label noise
    # =========================================================================
    for class_sep in [0.1, 0.2, 0.3]:
        for flip_y in [0.05, 0.1, 0.2]:
            configs.append(
                SyntheticConfig(
                    name=f"synthetic_weak_p100_k10_n1000_sep{class_sep}_flip{flip_y}",
                    n_samples=1000,
                    n_features=100,
                    n_informative=10,
                    class_sep=class_sep,
                    flip_y=flip_y,
                    seed=RANDOM_STATE,
                    weak_signal=True,
                )
            )

    return configs


def get_all_regression_configs() -> list[SyntheticConfig]:
    """Generate all synthetic regression dataset configurations.

    Uses single RANDOM_STATE for all datasets. Variance in experiments
    comes from method seeds (N_SEEDS in the distributed pipeline).
    """
    configs = []

    # =========================================================================
    # 1. STANDARD: Varying dimensionality, signal, sample size
    # Linear regression: y = X @ beta + noise
    # =========================================================================
    for n_features in [50, 100, 500]:
        for n_informative in [5, 10, 20]:
            if n_informative >= n_features:
                continue
            for n_samples in [200, 500, 1000]:
                for noise in [1.0, 5.0, 10.0]:
                    configs.append(
                        SyntheticConfig(
                            name=f"synthetic_p{n_features}_k{n_informative}_n{n_samples}_noise{noise}",
                            n_samples=n_samples,
                            n_features=n_features,
                            n_informative=n_informative,
                            task="regression",
                            noise=noise,
                            seed=RANDOM_STATE,
                        )
                    )

    # =========================================================================
    # 2. FRIEDMAN: Nonlinear relationships
    # Tests RDC/DC vs linear selectors (PC)
    # =========================================================================
    for variant in [1, 2, 3]:
        for n_features in [20, 50, 100]:
            for n_samples in [500, 1000]:
                for noise in [1.0, 5.0]:
                    configs.append(
                        SyntheticConfig(
                            name=f"synthetic_friedman{variant}_p{n_features}_n{n_samples}_noise{noise}",
                            n_samples=n_samples,
                            n_features=n_features,
                            n_informative=5 if variant == 1 else 4,  # Friedman1 has 5, others 4
                            task="regression",
                            friedman_variant=variant,
                            noise=noise,
                            seed=RANDOM_STATE,
                        )
                    )

    # =========================================================================
    # 3. CORRELATED: Correlated feature blocks
    # Tests conditional importance
    # =========================================================================
    for n_corr in [5, 10]:
        for corr_strength in [0.7, 0.9, 0.95]:
            configs.append(
                SyntheticConfig(
                    name=f"synthetic_corr_blocks{n_corr}_r{corr_strength}",
                    n_samples=1000,
                    n_features=50,
                    n_informative=10,
                    task="regression",
                    n_correlated_blocks=n_corr,
                    correlation_strength=corr_strength,
                    noise=1.0,
                    seed=RANDOM_STATE,
                )
            )

    # =========================================================================
    # 4. REDUNDANT: Multicollinearity
    # Tests how methods handle linear dependencies
    # =========================================================================
    for n_redundant in [5, 10, 20]:
        configs.append(
            SyntheticConfig(
                name=f"synthetic_redundant{n_redundant}",
                n_samples=1000,
                n_features=50,
                n_informative=10,
                n_redundant=n_redundant,
                task="regression",
                noise=1.0,
                seed=RANDOM_STATE,
            )
        )

    # =========================================================================
    # 5. HETEROSCEDASTIC: Non-constant variance
    # Tests robustness to heteroscedasticity
    # =========================================================================
    for scale in [1.0, 2.0, 4.0]:
        for noise in [1.0, 5.0]:
            configs.append(
                SyntheticConfig(
                    name=f"synthetic_heteroscedastic_scale{scale}_noise{noise}",
                    n_samples=1000,
                    n_features=50,
                    n_informative=10,
                    task="regression",
                    heteroscedastic=True,
                    heteroscedastic_scale=scale,
                    noise=noise,
                    seed=RANDOM_STATE,
                )
            )

    # =========================================================================
    # 6. TOEPLITZ: Global correlation structure
    # Tests feature selection under correlated design
    # =========================================================================
    for n_features in [50, 100]:
        for n_informative in [5, 10]:
            for n_samples in [500, 1000]:
                for rho in [0.7, 0.9, 0.95]:
                    configs.append(
                        SyntheticConfig(
                            name=f"synthetic_toeplitz_p{n_features}_k{n_informative}_n{n_samples}_r{rho}",
                            n_samples=n_samples,
                            n_features=n_features,
                            n_informative=n_informative,
                            task="regression",
                            toeplitz_rho=rho,
                            noise=1.0,
                            seed=RANDOM_STATE,
                        )
                    )

    # =========================================================================
    # 7. WEAK SIGNAL: Low SNR (high noise)
    # Tests detection of weak signals
    # =========================================================================
    for noise in [10.0, 20.0, 50.0]:
        configs.append(
            SyntheticConfig(
                name=f"synthetic_weak_p100_k10_n1000_noise{noise}",
                n_samples=1000,
                n_features=100,
                n_informative=10,
                task="regression",
                noise=noise,
                weak_signal=True,
                seed=RANDOM_STATE,
            )
        )

    # =========================================================================
    # 8. CORRELATED NOISE: Confounders correlated with informative features
    # =========================================================================
    for n_corr in [10, 20]:
        for corr_strength in [0.7, 0.9]:
            configs.append(
                SyntheticConfig(
                    name=f"synthetic_corr_noise_p100_k10_n1000_noise{n_corr}_r{corr_strength}",
                    n_samples=1000,
                    n_features=100,
                    n_informative=10,
                    task="regression",
                    n_correlated_noise=n_corr,
                    correlated_noise_strength=corr_strength,
                    noise=1.0,
                    seed=RANDOM_STATE,
                )
            )

    return configs


def main() -> None:
    """Generate all synthetic classification and regression datasets."""
    # Generate classification datasets
    CLF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clf_configs = get_all_configs()
    logger.info(f"Generating {len(clf_configs)} synthetic classification datasets...")

    for i, config in enumerate(clf_configs):
        table, _ = generate_dataset(config)

        # Save with clf_ prefix so server discovers it
        filepath = CLF_OUTPUT_DIR / f"clf_{config.name}.parquet"
        pq.write_table(table, filepath)

        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i + 1}/{len(clf_configs)}")

    logger.info(f"Generated {len(clf_configs)} classification datasets in {CLF_OUTPUT_DIR}")

    # Classification summary
    _log_classification_summary(clf_configs)

    # Generate regression datasets
    REG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reg_configs = get_all_regression_configs()
    logger.info(f"Generating {len(reg_configs)} synthetic regression datasets...")

    for i, config in enumerate(reg_configs):
        table, _ = generate_dataset(config)

        # Save with reg_ prefix so server discovers it
        filepath = REG_OUTPUT_DIR / f"reg_{config.name}.parquet"
        pq.write_table(table, filepath)

        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i + 1}/{len(reg_configs)}")

    logger.info(f"Generated {len(reg_configs)} regression datasets in {REG_OUTPUT_DIR}")

    # Regression summary
    _log_regression_summary(reg_configs)


def _log_classification_summary(configs: list[SyntheticConfig]) -> None:
    """Log summary statistics for classification datasets."""
    standard = sum(
        1
        for c in configs
        if not c.nonlinear
        and not c.weak_signal
        and c.toeplitz_rho == 0.0
        and c.n_high_cardinality_noise == 0
        and c.n_correlated_blocks == 0
        and c.n_correlated_noise == 0
        and c.n_redundant == 0
    )
    bias = sum(1 for c in configs if c.n_high_cardinality_noise > 0)
    nonlinear = sum(1 for c in configs if c.nonlinear)
    correlated = sum(1 for c in configs if c.n_correlated_blocks > 0)
    correlated_noise = sum(1 for c in configs if c.n_correlated_noise > 0)
    redundant = sum(1 for c in configs if c.n_redundant > 0)
    toeplitz = sum(1 for c in configs if c.toeplitz_rho > 0)
    weak_signal = sum(1 for c in configs if c.weak_signal)

    logger.info("Classification dataset summary:")
    logger.info(f"  Standard: {standard}")
    logger.info(f"  Bias: {bias}")
    logger.info(f"  Nonlinear: {nonlinear}")
    logger.info(f"  Correlated: {correlated}")
    logger.info(f"  Correlated noise: {correlated_noise}")
    logger.info(f"  Redundant: {redundant}")
    logger.info(f"  Toeplitz: {toeplitz}")
    logger.info(f"  Weak signal: {weak_signal}")


def _log_regression_summary(configs: list[SyntheticConfig]) -> None:
    """Log summary statistics for regression datasets."""
    standard = sum(
        1
        for c in configs
        if c.friedman_variant is None
        and not c.weak_signal
        and not c.heteroscedastic
        and c.toeplitz_rho == 0.0
        and c.n_correlated_blocks == 0
        and c.n_correlated_noise == 0
        and c.n_redundant == 0
    )
    friedman = sum(1 for c in configs if c.friedman_variant is not None)
    correlated = sum(1 for c in configs if c.n_correlated_blocks > 0)
    redundant = sum(1 for c in configs if c.n_redundant > 0)
    heteroscedastic = sum(1 for c in configs if c.heteroscedastic)
    toeplitz = sum(1 for c in configs if c.toeplitz_rho > 0)
    weak_signal = sum(1 for c in configs if c.weak_signal)
    correlated_noise = sum(1 for c in configs if c.n_correlated_noise > 0)

    logger.info("Regression dataset summary:")
    logger.info(f"  Standard: {standard}")
    logger.info(f"  Friedman: {friedman}")
    logger.info(f"  Correlated: {correlated}")
    logger.info(f"  Redundant: {redundant}")
    logger.info(f"  Heteroscedastic: {heteroscedastic}")
    logger.info(f"  Toeplitz: {toeplitz}")
    logger.info(f"  Weak signal: {weak_signal}")
    logger.info(f"  Correlated noise: {correlated_noise}")


if __name__ == "__main__":
    main()
