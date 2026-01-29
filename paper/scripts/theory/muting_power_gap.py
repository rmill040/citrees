#!/usr/bin/env python3
"""
Empirical validation of feature muting power theory.

This script validates the theoretical predictions from theoretical_predictions.py
by running Monte Carlo simulations of the gated effect model and comparing
empirical rejection rates with theoretical power calculations.

Usage:
    # Quick test (few configs, few sims)
    uv run python paper/scripts/theory/muting_power_gap.py --quick

    # Full validation
    uv run python paper/scripts/theory/muting_power_gap.py

    # Custom parameters
    uv run python paper/scripts/theory/muting_power_gap.py --n 2000 --p 0.05 --n-sims 500
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from paper.scripts.theory.theoretical_predictions import (
    find_gap_region,
    gate_power,
    root_power,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SimConfig:
    """Configuration for a single simulation."""

    n: int
    p: float
    alpha: float
    n_resamples: int
    seed: int


@dataclass
class SimResult:
    """Results from a single simulation."""

    # Config
    n: int
    p: float
    alpha: float
    n_resamples: int
    seed: int

    # Data characteristics
    n_gate_actual: int

    # Empirical statistics
    root_correlation: float
    gate_correlation: float

    # Permutation test results
    root_pvalue: float
    gate_pvalue: float
    root_rejects: bool
    gate_rejects: bool

    # Theoretical predictions
    root_power_theory: float
    gate_power_theory: float
    in_gap_region: bool


def generate_gated_data(
    n: int,
    p: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data from the gated effect model.

    Model:
        X0, X1 ~ N(0, 1) independent
        Z = 1{X0 > c} where P(Z=1) = p
        Y = 1{X1 > 0} if Z=1, else Bernoulli(0.5)

    Returns
    -------
    X0, X1, Y, Z : np.ndarray
        Feature X0 (gate), feature X1 (signal), response Y, gate indicator Z
    """
    X0 = rng.standard_normal(n)
    X1 = rng.standard_normal(n)

    # Gate threshold for P(Z=1) = p
    c = stats.norm.ppf(1 - p)
    Z = (c < X0).astype(np.int64)

    # Response: deterministic in gate, random outside
    eps = rng.integers(0, 2, n)
    Y = np.where(Z == 1, (X1 > 0).astype(np.int64), eps)

    return X0, X1, Y, Z


def permutation_test_correlation(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int,
    rng: np.random.Generator,
) -> float:
    """
    Permutation test for correlation (two-sided).

    Uses +1 correction (Phipson & Smyth, 2010).

    Returns
    -------
    float
        p-value
    """
    n = len(x)
    if n < 4:
        return 1.0

    # Observed statistic (absolute correlation)
    r_obs = np.abs(np.corrcoef(x, y)[0, 1])
    if np.isnan(r_obs):
        return 1.0

    # Permutation distribution
    count = 0
    y_perm = y.copy()

    for _ in range(n_resamples):
        rng.shuffle(y_perm)
        r_perm = np.abs(np.corrcoef(x, y_perm)[0, 1])
        if not np.isnan(r_perm) and r_perm >= r_obs:
            count += 1

    # +1 correction
    pvalue = (count + 1) / (n_resamples + 1)
    return pvalue


def run_single_simulation(config: SimConfig) -> SimResult:
    """Run a single simulation of the gated model."""
    rng = np.random.default_rng(config.seed)

    # Generate data
    X0, X1, Y, Z = generate_gated_data(config.n, config.p, rng)
    n_gate = Z.sum()

    # Compute correlations
    root_corr = np.corrcoef(X1, Y)[0, 1] if len(X1) >= 4 else np.nan

    if n_gate >= 4:
        X1_gate = X1[Z == 1]
        Y_gate = Y[Z == 1]
        gate_corr = np.corrcoef(X1_gate, Y_gate)[0, 1]
    else:
        gate_corr = np.nan

    # Permutation tests
    root_pval = permutation_test_correlation(X1, Y, config.n_resamples, rng)

    if n_gate >= 4:
        gate_pval = permutation_test_correlation(X1_gate, Y_gate, config.n_resamples, rng)
    else:
        gate_pval = 1.0

    # Theoretical predictions
    root_power_th = root_power(config.p, config.n, config.alpha)
    gate_power_th = gate_power(config.p, config.n, config.alpha)

    # Check if in gap region
    gap = find_gap_region(config.n, config.alpha)
    in_gap = gap.is_valid and gap.p_min <= config.p <= gap.p_max

    return SimResult(
        n=config.n,
        p=config.p,
        alpha=config.alpha,
        n_resamples=config.n_resamples,
        seed=config.seed,
        n_gate_actual=n_gate,
        root_correlation=root_corr,
        gate_correlation=gate_corr,
        root_pvalue=root_pval,
        gate_pvalue=gate_pval,
        root_rejects=root_pval < config.alpha,
        gate_rejects=gate_pval < config.alpha,
        root_power_theory=root_power_th,
        gate_power_theory=gate_power_th,
        in_gap_region=in_gap,
    )


def run_simulation_batch(
    n_values: list[int],
    p_values: list[float],
    alpha: float,
    n_resamples: int,
    n_sims: int,
    start_seed: int = 42,
) -> pd.DataFrame:
    """
    Run batch of simulations across parameter grid.

    Returns DataFrame with all results.
    """
    configs = []
    for n, p, sim_idx in product(n_values, p_values, range(n_sims)):
        seed = start_seed + sim_idx + hash((n, p)) % 10000
        configs.append(SimConfig(n, p, alpha, n_resamples, seed))

    logger.info(f"Running {len(configs)} simulations...")

    results = []
    for i, config in enumerate(configs):
        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i + 1}/{len(configs)}")
        result = run_single_simulation(config)
        results.append(asdict(result))

    return pd.DataFrame(results)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize simulation results by (n, p) configuration.

    Computes empirical rejection rates and compares to theory.
    """
    summary = (
        df.groupby(["n", "p"])
        .agg(
            n_sims=("seed", "count"),
            n_gate_mean=("n_gate_actual", "mean"),
            n_gate_std=("n_gate_actual", "std"),
            root_corr_mean=("root_correlation", "mean"),
            gate_corr_mean=("gate_correlation", "mean"),
            root_reject_rate=("root_rejects", "mean"),
            gate_reject_rate=("gate_rejects", "mean"),
            root_power_theory=("root_power_theory", "first"),
            gate_power_theory=("gate_power_theory", "first"),
            in_gap_region=("in_gap_region", "first"),
        )
        .reset_index()
    )

    # Compute discrepancies
    summary["root_discrepancy"] = summary["root_reject_rate"] - summary["root_power_theory"]
    summary["gate_discrepancy"] = summary["gate_reject_rate"] - summary["gate_power_theory"]

    return summary


def print_summary(summary: pd.DataFrame) -> None:
    """Print formatted summary table."""
    print("\n" + "=" * 100)
    print("FEATURE MUTING POWER GAP VALIDATION")
    print("=" * 100)

    print("\n--- Empirical vs Theoretical Power ---\n")
    print(
        f"{'n':>6} | {'p':>6} | {'n_gate':>8} | "
        f"{'Root(emp)':>10} | {'Root(th)':>10} | {'Diff':>8} | "
        f"{'Gate(emp)':>10} | {'Gate(th)':>10} | {'Diff':>8} | {'Gap?':>5}"
    )
    print("-" * 100)

    for _, row in summary.iterrows():
        gap_str = "YES" if row["in_gap_region"] else ""
        print(
            f"{row['n']:>6} | {row['p']:>6.3f} | {row['n_gate_mean']:>8.1f} | "
            f"{row['root_reject_rate']:>10.4f} | {row['root_power_theory']:>10.4f} | "
            f"{row['root_discrepancy']:>+8.4f} | "
            f"{row['gate_reject_rate']:>10.4f} | {row['gate_power_theory']:>10.4f} | "
            f"{row['gate_discrepancy']:>+8.4f} | {gap_str:>5}"
        )

    # Summary statistics
    print("\n--- Validation Summary ---")
    print(f"Max |root discrepancy|: {summary['root_discrepancy'].abs().max():.4f}")
    print(f"Max |gate discrepancy|: {summary['gate_discrepancy'].abs().max():.4f}")
    print(f"Mean |root discrepancy|: {summary['root_discrepancy'].abs().mean():.4f}")
    print(f"Mean |gate discrepancy|: {summary['gate_discrepancy'].abs().mean():.4f}")

    # Gap region analysis
    gap_rows = summary[summary["in_gap_region"]]
    if len(gap_rows) > 0:
        print(f"\n--- Gap Region Configurations ({len(gap_rows)} found) ---")
        print("In gap region: root power LOW, gate power HIGH")
        for _, row in gap_rows.iterrows():
            print(
                f"  n={row['n']}, p={row['p']:.3f}: "
                f"root={row['root_reject_rate']:.3f} (th:{row['root_power_theory']:.3f}), "
                f"gate={row['gate_reject_rate']:.3f} (th:{row['gate_power_theory']:.3f})"
            )


def main():
    parser = argparse.ArgumentParser(description="Feature muting power gap validation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/results/cache"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=None,
        help="Sample sizes to test",
    )
    parser.add_argument(
        "--p",
        type=float,
        nargs="+",
        default=None,
        help="Gate probabilities to test",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=999,
        help="Number of permutation resamples",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=200,
        help="Number of simulations per configuration",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with reduced parameters",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Set defaults based on mode
    if args.quick:
        n_values = args.n or [500, 2000]
        p_values = args.p or [0.02, 0.05, 0.1]
        n_sims = 50
        n_resamples = 199
        logger.info("Running in QUICK mode")
    else:
        n_values = args.n or [500, 1000, 2000, 5000]
        p_values = args.p or [0.01, 0.02, 0.03, 0.05, 0.1, 0.2]
        n_sims = args.n_sims
        n_resamples = args.n_resamples

    logger.info(f"Sample sizes: {n_values}")
    logger.info(f"Gate probabilities: {p_values}")
    logger.info(f"Simulations per config: {n_sims}")
    logger.info(f"Permutation resamples: {n_resamples}")

    # Run simulations
    df = run_simulation_batch(
        n_values=n_values,
        p_values=p_values,
        alpha=args.alpha,
        n_resamples=n_resamples,
        n_sims=n_sims,
        start_seed=args.seed,
    )

    # Save raw results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "muting_power_gap_raw.parquet"
    df.to_parquet(raw_path)
    logger.info(f"Saved raw results to {raw_path}")

    # Compute and save summary
    summary = summarize_results(df)
    summary_path = args.output_dir / "muting_power_gap_summary.parquet"
    summary.to_parquet(summary_path)
    logger.info(f"Saved summary to {summary_path}")

    # Print results
    print_summary(summary)

    # Validation check
    max_discrepancy = max(
        summary["root_discrepancy"].abs().max(),
        summary["gate_discrepancy"].abs().max(),
    )
    if max_discrepancy < 0.10:
        logger.info("VALIDATION PASSED: All discrepancies < 0.10")
    else:
        logger.warning(f"VALIDATION WARNING: Max discrepancy = {max_discrepancy:.4f}")


if __name__ == "__main__":
    main()
