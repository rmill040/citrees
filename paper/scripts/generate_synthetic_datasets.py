"""Generate synthetic datasets for the citrees paper experiments.

This script creates synthetic classification datasets as parquet files that
integrate with the existing server/worker pipeline. Each dataset has an
accompanying JSON file storing ground truth (informative feature indices).

Dataset types:
1. SELECTION BIAS: High-cardinality noise features to demonstrate bias
2. STANDARD: Varying n_features, n_informative, class_sep, n_samples
3. NONLINEAR: Nonlinear relationships to test RDC vs MC selectors
4. CORRELATED: Correlated feature blocks to test conditional importance

Usage:
    uv run python paper/scripts/generate_synthetic_datasets.py
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_friedman1

RANDOM_STATE = 1718
OUTPUT_DIR = Path(__file__).parent.parent / "data"


@dataclass
class SyntheticDatasetConfig:
    """Configuration for a synthetic dataset."""

    name: str
    n_samples: int
    n_features: int
    n_informative: int
    n_redundant: int = 0
    n_clusters_per_class: int = 2
    class_sep: float = 1.0
    flip_y: float = 0.0
    random_state: int = RANDOM_STATE
    # For selection bias experiments
    n_high_cardinality_noise: int = 0
    high_cardinality_levels: int = 100
    # For nonlinear experiments
    nonlinear: bool = False
    # For correlated feature experiments
    n_correlated_blocks: int = 0
    correlation_strength: float = 0.9


def generate_standard_dataset(
    config: SyntheticDatasetConfig,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Generate a standard synthetic classification dataset.

    Returns X, y, and list of informative feature indices.
    """
    X, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_informative=config.n_informative,
        n_redundant=config.n_redundant,
        n_clusters_per_class=config.n_clusters_per_class,
        class_sep=config.class_sep,
        flip_y=config.flip_y,
        random_state=config.random_state,
        shuffle=False,  # Keep informative features at indices 0..n_informative-1
    )

    # Informative features are at the start
    informative_indices = list(range(config.n_informative))

    # Now shuffle columns to hide the informative features
    prng = np.random.RandomState(config.random_state)
    perm = prng.permutation(config.n_features)
    X = X[:, perm]

    # Map original informative indices to new positions
    inv_perm = np.argsort(perm)
    informative_indices = [int(inv_perm[i]) for i in range(config.n_informative)]

    return X, y, informative_indices


def add_high_cardinality_noise(
    X: np.ndarray, n_noise_features: int, n_levels: int, random_state: int
) -> tuple[np.ndarray, list[int]]:
    """Add high-cardinality categorical noise features (encoded as integers).

    These features have many unique values but NO relationship to y. CART/RF will spuriously select
    these due to selection bias. citrees should correctly ignore them.

    Returns augmented X and indices of the noise features.
    """
    prng = np.random.RandomState(random_state)
    n_samples = X.shape[0]

    # Generate random categorical features with many levels
    noise_features = prng.randint(0, n_levels, size=(n_samples, n_noise_features)).astype(float)

    # Append to X
    X_augmented = np.hstack([X, noise_features])

    # Noise feature indices are at the end
    noise_indices = list(range(X.shape[1], X_augmented.shape[1]))

    return X_augmented, noise_indices


def add_correlated_blocks(
    X: np.ndarray,
    informative_indices: list[int],
    n_blocks: int,
    correlation: float,
    random_state: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Add features that are correlated with informative features.

    This tests whether methods correctly handle correlated features. Standard permutation importance
    inflates importance of correlated features.

    Returns augmented X, correlated feature indices, and updated informative indices.
    """
    prng = np.random.RandomState(random_state)
    n_samples = X.shape[0]

    correlated_features = []
    for i, idx in enumerate(informative_indices[:n_blocks]):
        # Create a feature correlated with informative feature
        noise = prng.randn(n_samples) * np.sqrt(1 - correlation**2)
        correlated = correlation * X[:, idx] + noise
        correlated_features.append(correlated.reshape(-1, 1))

    if correlated_features:
        correlated_array = np.hstack(correlated_features)
        X_augmented = np.hstack([X, correlated_array])
        correlated_indices = list(range(X.shape[1], X_augmented.shape[1]))
    else:
        X_augmented = X
        correlated_indices = []

    return X_augmented, correlated_indices, informative_indices


def generate_nonlinear_dataset(
    config: SyntheticDatasetConfig,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Generate dataset with nonlinear relationships.

    Uses Friedman #1 function which has nonlinear relationships. Linear methods (Pearson, MC) will
    underperform; RDC should excel.
    """
    # Friedman1 has 5 truly informative features
    n_informative = 5

    X, y_cont = make_friedman1(
        n_samples=config.n_samples,
        n_features=config.n_features,
        noise=1.0,
        random_state=config.random_state,
    )

    # Convert to binary classification
    y = (y_cont >= np.median(y_cont)).astype(int)

    # Informative features are indices 0-4 in Friedman1
    informative_indices = list(range(n_informative))

    # Shuffle columns
    prng = np.random.RandomState(config.random_state)
    perm = prng.permutation(config.n_features)
    X = X[:, perm]

    inv_perm = np.argsort(perm)
    informative_indices = [int(inv_perm[i]) for i in range(n_informative)]

    return X, y, informative_indices


def generate_dataset(config: SyntheticDatasetConfig) -> tuple[pd.DataFrame, dict]:
    """Generate a complete synthetic dataset with ground truth metadata."""

    # Generate base dataset
    if config.nonlinear:
        X, y, informative_indices = generate_nonlinear_dataset(config)
    else:
        X, y, informative_indices = generate_standard_dataset(config)

    metadata = {
        "name": config.name,
        "n_samples": config.n_samples,
        "n_features": X.shape[1],
        "n_informative": len(informative_indices),
        "informative_indices": informative_indices,
        "noise_indices": [],
        "correlated_indices": [],
        "class_sep": config.class_sep,
        "nonlinear": config.nonlinear,
    }

    # Add high-cardinality noise features (selection bias test)
    if config.n_high_cardinality_noise > 0:
        X, noise_indices = add_high_cardinality_noise(
            X,
            config.n_high_cardinality_noise,
            config.high_cardinality_levels,
            config.random_state + 1,
        )
        metadata["noise_indices"] = noise_indices
        metadata["n_features"] = X.shape[1]
        metadata["high_cardinality_levels"] = config.high_cardinality_levels

    # Add correlated features
    if config.n_correlated_blocks > 0:
        X, correlated_indices, informative_indices = add_correlated_blocks(
            X,
            informative_indices,
            config.n_correlated_blocks,
            config.correlation_strength,
            config.random_state + 2,
        )
        metadata["correlated_indices"] = correlated_indices
        metadata["n_features"] = X.shape[1]
        metadata["correlation_strength"] = config.correlation_strength

    # Create DataFrame
    feature_cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df["y"] = y

    return df, metadata


def generate_all_datasets() -> None:
    """Generate all synthetic datasets for the paper."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = []

    # ==========================================================================
    # 1. SELECTION BIAS DATASETS
    # High-cardinality noise features that CART/RF will spuriously select
    # ==========================================================================
    for seed in range(3):  # 3 replicates
        for n_noise in [10, 20, 50]:
            for n_levels in [50, 100, 500]:
                configs.append(
                    SyntheticDatasetConfig(
                        name=f"syn_bias_noise{n_noise}_levels{n_levels}_seed{seed}",
                        n_samples=1000,
                        n_features=50,  # 50 real features
                        n_informative=10,
                        class_sep=1.0,
                        random_state=RANDOM_STATE + seed,
                        n_high_cardinality_noise=n_noise,
                        high_cardinality_levels=n_levels,
                    )
                )

    # ==========================================================================
    # 2. STANDARD SYNTHETIC DATASETS
    # Varying dimensionality, signal strength, sample size
    # ==========================================================================
    for seed in range(3):
        for n_features in [50, 100, 500]:
            for n_informative in [5, 10, 20]:
                if n_informative >= n_features:
                    continue
                for n_samples in [500, 1000, 2000]:
                    for class_sep in [0.5, 1.0, 2.0]:
                        configs.append(
                            SyntheticDatasetConfig(
                                name=f"syn_p{n_features}_k{n_informative}_n{n_samples}_sep{class_sep}_seed{seed}",
                                n_samples=n_samples,
                                n_features=n_features,
                                n_informative=n_informative,
                                class_sep=class_sep,
                                random_state=RANDOM_STATE + seed,
                            )
                        )

    # ==========================================================================
    # 3. NONLINEAR DATASETS (Friedman #1)
    # Tests RDC vs linear selectors (MC, PC)
    # ==========================================================================
    for seed in range(3):
        for n_features in [50, 100, 500]:
            for n_samples in [500, 1000, 2000]:
                configs.append(
                    SyntheticDatasetConfig(
                        name=f"syn_nonlinear_p{n_features}_n{n_samples}_seed{seed}",
                        n_samples=n_samples,
                        n_features=n_features,
                        n_informative=5,  # Friedman1 has 5 informative
                        random_state=RANDOM_STATE + seed,
                        nonlinear=True,
                    )
                )

    # ==========================================================================
    # 4. CORRELATED FEATURE DATASETS
    # Tests conditional importance / handling of correlated features
    # ==========================================================================
    for seed in range(3):
        for n_correlated in [5, 10]:
            for correlation in [0.7, 0.9, 0.95]:
                configs.append(
                    SyntheticDatasetConfig(
                        name=f"syn_corr_blocks{n_correlated}_r{correlation}_seed{seed}",
                        n_samples=1000,
                        n_features=50,
                        n_informative=10,
                        class_sep=1.0,
                        random_state=RANDOM_STATE + seed,
                        n_correlated_blocks=n_correlated,
                        correlation_strength=correlation,
                    )
                )

    print(f"Generating {len(configs)} synthetic datasets...")

    all_metadata = {}

    for i, config in enumerate(configs):
        print(f"[{i + 1}/{len(configs)}] {config.name}")

        df, metadata = generate_dataset(config)

        # Save parquet
        parquet_path = OUTPUT_DIR / f"clf_{config.name}.snappy.parquet"
        df.to_parquet(parquet_path, compression="snappy", index=False)

        # Collect metadata
        all_metadata[config.name] = metadata

    # Save all metadata to single JSON file
    metadata_path = OUTPUT_DIR / "synthetic_ground_truth.json"
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\nGenerated {len(configs)} datasets")
    print(f"Metadata saved to {metadata_path}")

    # Print summary
    print("\n=== DATASET SUMMARY ===")
    print(f"Selection bias datasets: {sum(1 for c in configs if c.n_high_cardinality_noise > 0)}")
    print(
        f"Standard datasets: {sum(1 for c in configs if not c.nonlinear and c.n_high_cardinality_noise == 0 and c.n_correlated_blocks == 0)}"
    )
    print(f"Nonlinear datasets: {sum(1 for c in configs if c.nonlinear)}")
    print(f"Correlated datasets: {sum(1 for c in configs if c.n_correlated_blocks > 0)}")


if __name__ == "__main__":
    generate_all_datasets()
