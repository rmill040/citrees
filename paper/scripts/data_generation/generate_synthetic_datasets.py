#!/usr/bin/env python3
"""Generate synthetic datasets for distributed experiments.

Creates synthetic classification datasets with known informative features.
Metadata (generation params + ground truth) is stored IN the parquet schema.

Dataset types:
1. STANDARD: Varying n_features, n_informative, class_sep, n_samples
2. SELECTION BIAS: High-cardinality noise features to demonstrate bias
3. NONLINEAR: Friedman #1 function to test RDC vs linear selectors
4. CORRELATED: Correlated feature blocks to test conditional importance
5. REDUNDANT: Linear combinations of informative features (multicollinearity)
6. CORR_NOISE: Correlated noise features (confounders)
7. TOEPLITZ: Toeplitz covariance structure (global correlation)
8. WEAK_SIGNAL: Low class_sep + label noise (hard signal)

Usage:
    uv run python paper/scripts/data_generation/generate_synthetic_datasets.py
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from sklearn.datasets import make_classification, make_friedman1

# Configurable via environment variable for reproducibility
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "1718"))
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class SyntheticConfig:
    """Configuration for a synthetic dataset."""

    name: str
    n_samples: int
    n_features: int
    n_informative: int
    n_redundant: int = 0
    n_clusters_per_class: int = 2
    class_sep: float = 1.0
    flip_y: float = 0.0
    seed: int = RANDOM_STATE
    # Selection bias
    n_high_cardinality_noise: int = 0
    high_cardinality_levels: int = 100
    # Nonlinear
    nonlinear: bool = False
    # Correlated features
    n_correlated_blocks: int = 0
    correlation_strength: float = 0.9
    # Correlated noise (confounders)
    n_correlated_noise: int = 0
    correlated_noise_strength: float = 0.9
    # Toeplitz correlation
    toeplitz_rho: float = 0.0
    # Weak signal
    weak_signal: bool = False


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
    # Generate base data
    if config.toeplitz_rho > 0:
        X, y, informative_indices, redundant_indices = generate_toeplitz_dataset(config)
    elif config.nonlinear:
        X, y, informative_indices, redundant_indices = generate_nonlinear_dataset(config)
    else:
        X, y, informative_indices, redundant_indices = generate_standard_dataset(config)

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
            X, informative_indices, config.n_correlated_blocks, config.correlation_strength, config.seed + 2
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
    df["y"] = y.astype(np.int64)

    # Metadata to store in parquet
    metadata = {
        "synthetic": "true",
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


def main() -> None:
    """Generate all synthetic datasets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = get_all_configs()
    logger.info(f"Generating {len(configs)} synthetic datasets...")

    for i, config in enumerate(configs):
        table, _ = generate_dataset(config)

        # Save with clf_ prefix so server discovers it
        filepath = OUTPUT_DIR / f"clf_{config.name}.parquet"
        pq.write_table(table, filepath)

        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i + 1}/{len(configs)}")

    logger.info(f"Generated {len(configs)} datasets in {OUTPUT_DIR}")

    # Summary
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

    logger.info(f"  Standard: {standard}")
    logger.info(f"  Bias: {bias}")
    logger.info(f"  Nonlinear: {nonlinear}")
    logger.info(f"  Correlated: {correlated}")
    logger.info(f"  Correlated noise: {correlated_noise}")
    logger.info(f"  Redundant: {redundant}")
    logger.info(f"  Toeplitz: {toeplitz}")
    logger.info(f"  Weak signal: {weak_signal}")


if __name__ == "__main__":
    main()
