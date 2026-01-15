#!/usr/bin/env python3
"""Generate synthetic datasets for distributed experiments.

Creates synthetic classification datasets with known informative features.
Metadata (generation params + ground truth) is stored IN the parquet schema.

Dataset types:
1. STANDARD: Varying n_features, n_informative, class_sep, n_samples
2. SELECTION BIAS: High-cardinality noise features to demonstrate bias
3. NONLINEAR: Friedman #1 function to test RDC vs linear selectors
4. CORRELATED: Correlated feature blocks to test conditional importance

Usage:
    uv run python paper/scripts/generate_synthetic_datasets.py
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
OUTPUT_DIR = Path(__file__).parent.parent / "data"


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


def generate_standard_dataset(config: SyntheticConfig) -> tuple[np.ndarray, np.ndarray, list[int]]:
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

    return X, y, informative_indices


def generate_nonlinear_dataset(config: SyntheticConfig) -> tuple[np.ndarray, np.ndarray, list[int]]:
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

    return X, y, informative_indices


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


def generate_dataset(config: SyntheticConfig) -> tuple[pa.Table, dict]:
    """Generate dataset and return as PyArrow table with metadata."""
    # Generate base data
    if config.nonlinear:
        X, y, informative_indices = generate_nonlinear_dataset(config)
    else:
        X, y, informative_indices = generate_standard_dataset(config)

    noise_indices = []
    corr_indices = []

    # Add high-cardinality noise
    if config.n_high_cardinality_noise > 0:
        X, noise_indices = add_high_cardinality_noise(
            X, config.n_high_cardinality_noise, config.high_cardinality_levels, config.seed + 1
        )

    # Add correlated features
    if config.n_correlated_blocks > 0:
        X, corr_indices = add_correlated_blocks(
            X, informative_indices, config.n_correlated_blocks, config.correlation_strength, config.seed + 2
        )

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["y"] = y.astype(np.int64)

    # Metadata to store in parquet
    metadata = {
        "synthetic": "true",
        "config": json.dumps(asdict(config)),
        "informative_indices": json.dumps(informative_indices),
        "noise_indices": json.dumps(noise_indices),
        "correlated_indices": json.dumps(corr_indices),
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
    standard = sum(1 for c in configs if not c.nonlinear and c.n_high_cardinality_noise == 0 and c.n_correlated_blocks == 0 and c.n_redundant == 0)
    bias = sum(1 for c in configs if c.n_high_cardinality_noise > 0)
    nonlinear = sum(1 for c in configs if c.nonlinear)
    correlated = sum(1 for c in configs if c.n_correlated_blocks > 0)
    redundant = sum(1 for c in configs if c.n_redundant > 0)

    logger.info(f"  Standard: {standard}")
    logger.info(f"  Bias: {bias}")
    logger.info(f"  Nonlinear: {nonlinear}")
    logger.info(f"  Correlated: {correlated}")
    logger.info(f"  Redundant: {redundant}")


if __name__ == "__main__":
    main()
