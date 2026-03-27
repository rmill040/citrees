"""Empirical validation: batched vs sequential adaptive stopping.

TODO: Report these results in the paper's sequential stopping section.
      Key finding: K=32 batched stopping preserves Type I error (4.61% vs 4.55%
      sequential, 100K sims), enabling parallel permutation testing with adaptive
      early stopping.

Compares Type I error, power, and stopping times for:
  1. Sequential (per-permutation) stopping  — current citrees implementation
  2. Batched stopping with various batch sizes K

Uses the CORRECT H0 simulation dynamics from sequential_stopping_analysis.py:
  - p = F(T_0) ~ Uniform(0, 1)
  - Each permutation exceeds with probability p (fixed per test, random across tests)
  - L_n | p ~ Binomial(n, p)
  - Marginally: L_n ~ Uniform{0, ..., n}

Reference: sequential_stopping_analysis.py (existing validation infrastructure)
"""

import time
from math import ceil

import numpy as np
from scipy.special import betainc


# =============================================================================
# Core functions (matching citrees implementation exactly)
# =============================================================================


def beta_cdf(x: float, a: float, b: float) -> float:
    """Beta CDF: P(X < x) for X ~ Beta(a, b). Matches citrees._sequential._beta_cdf."""
    return float(betainc(a, b, x))


def check_stopping(
    extreme_count: int, n: int, alpha: float, confidence: float
) -> str | None:
    """Check the two-sided adaptive stopping criterion.

    Returns "reject", "accept", or None (continue).
    """
    a = 1.0 + extreme_count
    b = 1.0 + n - extreme_count
    prob_sig = beta_cdf(alpha, a, b)

    if prob_sig >= confidence:
        return "reject"
    if (1.0 - prob_sig) >= confidence:
        return "accept"
    return None


# =============================================================================
# Sequential simulation (per-permutation checking) — baseline
# =============================================================================


def simulate_sequential(
    n_max: int = 1000,
    alpha: float = 0.05,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[str, int, int, float]:
    """Simulate sequential adaptive stopping (check every permutation).

    Returns: (outcome, n_stop, L_final, p)
    """
    rng = np.random.default_rng(seed)
    min_n = ceil(1 / alpha)
    p = rng.random()  # True p-value under H0
    L_n = 0

    for i in range(1, n_max + 1):
        if rng.random() < p:
            L_n += 1

        if i >= min_n:
            decision = check_stopping(L_n, i, alpha, confidence)
            if decision is not None:
                return decision, i, L_n, p

    return "max_reached", n_max, L_n, p


# =============================================================================
# Batched simulation (check every K permutations)
# =============================================================================


def simulate_batched(
    batch_size: int,
    n_max: int = 1000,
    alpha: float = 0.05,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[str, int, int, float]:
    """Simulate batched adaptive stopping (check every K permutations).

    CRITICAL: Uses the SAME random sequence as sequential (same seed, same rng calls)
    so the only difference is WHEN we check the stopping criterion.

    Returns: (outcome, n_stop, L_final, p)
    """
    rng = np.random.default_rng(seed)
    min_n = ceil(1 / alpha)
    p = rng.random()  # True p-value under H0
    L_n = 0

    for i in range(1, n_max + 1):
        if rng.random() < p:
            L_n += 1

        # Only check stopping at batch boundaries (and after min_resamples)
        if i >= min_n and i % batch_size == 0:
            decision = check_stopping(L_n, i, alpha, confidence)
            if decision is not None:
                return decision, i, L_n, p

    return "max_reached", n_max, L_n, p


# =============================================================================
# Run comparison
# =============================================================================


def run_comparison(
    n_sims: int = 100_000,
    n_max: int = 2000,
    batch_sizes: list[int] | None = None,
    alpha: float = 0.05,
    confidence: float = 0.95,
) -> dict:
    """Run full comparison of sequential vs batched stopping.

    Returns dict with results for each method.
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32, 64, 128]

    all_results = {}

    for K in batch_sizes:
        label = f"K={K}" if K > 1 else "sequential"
        t0 = time.time()

        outcomes = {"reject": 0, "accept": 0, "max_reached": 0}
        stop_times = []
        reject_p_values = []  # true p for false rejections

        for i in range(n_sims):
            if K == 1:
                outcome, n_stop, L_final, p = simulate_sequential(
                    n_max=n_max, alpha=alpha, confidence=confidence, seed=i
                )
            else:
                outcome, n_stop, L_final, p = simulate_batched(
                    batch_size=K, n_max=n_max, alpha=alpha, confidence=confidence, seed=i
                )

            outcomes[outcome] += 1
            stop_times.append(n_stop)
            if outcome == "reject":
                reject_p_values.append(p)

        elapsed = time.time() - t0
        type1 = outcomes["reject"] / n_sims

        all_results[label] = {
            "K": K,
            "outcomes": outcomes,
            "type1_error": type1,
            "mean_stop": np.mean(stop_times),
            "median_stop": np.median(stop_times),
            "reject_p_values": reject_p_values,
            "elapsed": elapsed,
        }

    return all_results


def print_results(results: dict, n_sims: int, alpha: float = 0.05) -> None:
    """Print comparison table."""
    print("=" * 90)
    print("BATCHED vs SEQUENTIAL ADAPTIVE STOPPING — TYPE I ERROR COMPARISON")
    print("=" * 90)
    print(f"\nSimulations: {n_sims:,}  |  alpha: {alpha}  |  confidence: 0.95")
    print(f"H0 dynamics: p ~ Uniform(0,1), L_n|p ~ Binomial(n,p)")
    print()

    # Standard error for binomial proportion at alpha=0.05
    se = np.sqrt(alpha * (1 - alpha) / n_sims)
    ci_lo = alpha - 1.96 * se
    ci_hi = alpha + 1.96 * se
    print(f"95% CI for true alpha=0.05: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"Standard error: {se:.5f}")
    print()

    header = f"{'Method':<14} {'Type I':<10} {'Reject':<10} {'Accept':<10} {'MaxReach':<10} {'Mean Stop':<12} {'Med Stop':<10} {'Time(s)':<8}"
    print(header)
    print("-" * len(header))

    for label, r in results.items():
        print(
            f"{label:<14} "
            f"{r['type1_error']:<10.5f} "
            f"{r['outcomes']['reject']:<10} "
            f"{r['outcomes']['accept']:<10} "
            f"{r['outcomes']['max_reached']:<10} "
            f"{r['mean_stop']:<12.1f} "
            f"{r['median_stop']:<10.1f} "
            f"{r['elapsed']:<8.1f}"
        )

    print()

    # Detailed comparison vs sequential
    seq_type1 = results["sequential"]["type1_error"]
    print("Comparison vs sequential (K=1):")
    print(f"{'Method':<14} {'Type I':<10} {'Delta':<12} {'Relative':<12} {'Within 95% CI?':<15}")
    print("-" * 63)

    for label, r in results.items():
        delta = r["type1_error"] - seq_type1
        relative = delta / seq_type1 * 100 if seq_type1 > 0 else 0
        within_ci = "YES" if ci_lo <= r["type1_error"] <= ci_hi else "no"
        print(
            f"{label:<14} "
            f"{r['type1_error']:<10.5f} "
            f"{delta:<+12.5f} "
            f"{relative:<+12.2f}% "
            f"{within_ci:<15}"
        )

    print()

    # False rejection analysis — what p values caused false rejections?
    print("False rejection analysis (true p values that caused rejection):")
    print(f"{'Method':<14} {'N reject':<10} {'Mean p':<10} {'Max p':<10} {'p > 0.05':<10}")
    print("-" * 54)

    for label, r in results.items():
        rp = r["reject_p_values"]
        if rp:
            rp_arr = np.array(rp)
            print(
                f"{label:<14} "
                f"{len(rp):<10} "
                f"{np.mean(rp_arr):<10.4f} "
                f"{np.max(rp_arr):<10.4f} "
                f"{np.sum(rp_arr > 0.05):<10}"
            )
        else:
            print(f"{label:<14} {'0':<10} {'N/A':<10} {'N/A':<10} {'0':<10}")


def print_path_divergence_analysis(n_sims: int = 100_000, n_max: int = 2000) -> None:
    """Analyze the specific scenario we're worried about: sequential accepts but batched rejects."""
    print("\n" + "=" * 90)
    print("PATH DIVERGENCE ANALYSIS")
    print("=" * 90)
    print("\nCounting cases where sequential ACCEPTS but batched REJECTS (the dangerous scenario)")
    print()

    batch_sizes = [4, 8, 16, 32, 64, 128]
    alpha = 0.05
    confidence = 0.95

    header = f"{'Batch K':<10} {'Seq->Accept, Bat->Reject':<25} {'Rate':<12} {'Seq->Reject, Bat->Accept':<25} {'Rate':<12}"
    print(header)
    print("-" * len(header))

    for K in batch_sizes:
        accept_then_reject = 0  # DANGEROUS: sequential accepts but batched rejects
        reject_then_accept = 0  # SAFE: sequential rejects but batched accepts (conservative)

        for i in range(n_sims):
            seq_outcome, _, _, _ = simulate_sequential(n_max=n_max, alpha=alpha, confidence=confidence, seed=i)
            bat_outcome, _, _, _ = simulate_batched(batch_size=K, n_max=n_max, alpha=alpha, confidence=confidence, seed=i)

            if seq_outcome == "accept" and bat_outcome == "reject":
                accept_then_reject += 1
            if seq_outcome == "reject" and bat_outcome == "accept":
                reject_then_accept += 1

        print(
            f"K={K:<7} "
            f"{accept_then_reject:<25} "
            f"{accept_then_reject / n_sims:<12.6f} "
            f"{reject_then_accept:<25} "
            f"{reject_then_accept / n_sims:<12.6f}"
        )


# =============================================================================
# Sensitivity analysis: Type I error at various alpha levels
# =============================================================================


def sensitivity_analysis(
    n_sims: int = 50_000,
    n_max: int = 2000,
    batch_sizes: list[int] | None = None,
    alphas: list[float] | None = None,
) -> None:
    """Test batched stopping across multiple alpha levels."""
    if batch_sizes is None:
        batch_sizes = [1, 32, 64]
    if alphas is None:
        alphas = [0.01, 0.05, 0.10]

    print("\n" + "=" * 90)
    print("SENSITIVITY ANALYSIS: TYPE I ERROR ACROSS ALPHA LEVELS")
    print("=" * 90)
    print(f"\nSimulations per cell: {n_sims:,}")
    print()

    header = f"{'Alpha':<8}"
    for K in batch_sizes:
        label = "seq" if K == 1 else f"K={K}"
        header += f" {label:<12}"
    print(header)
    print("-" * len(header))

    for alpha in alphas:
        row = f"{alpha:<8}"
        for K in batch_sizes:
            rejects = 0
            for i in range(n_sims):
                if K == 1:
                    outcome, _, _, _ = simulate_sequential(
                        n_max=n_max, alpha=alpha, confidence=0.95, seed=i
                    )
                else:
                    outcome, _, _, _ = simulate_batched(
                        batch_size=K, n_max=n_max, alpha=alpha, confidence=0.95, seed=i
                    )
                if outcome == "reject":
                    rejects += 1

            type1 = rejects / n_sims
            row += f" {type1:<12.5f}"
        print(row)


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    N_SIMS = 100_000
    N_MAX = 2000

    print("=" * 90)
    print("EMPIRICAL VALIDATION: BATCHED ADAPTIVE PERMUTATION TESTING")
    print("=" * 90)
    print(f"\nThis validates that checking the Beta CDF stopping criterion every")
    print(f"K permutations (instead of every 1) preserves Type I error control.")
    print(f"\nUsing {N_SIMS:,} simulations, n_max={N_MAX}")
    print()

    # 1. Main comparison
    results = run_comparison(n_sims=N_SIMS, n_max=N_MAX, batch_sizes=[1, 4, 8, 16, 32, 64, 128])
    print_results(results, N_SIMS)

    # 2. Path divergence analysis (the specific scenario we're worried about)
    print_path_divergence_analysis(n_sims=N_SIMS, n_max=N_MAX)

    # 3. Sensitivity across alpha levels
    sensitivity_analysis(n_sims=50_000, batch_sizes=[1, 32, 64], alphas=[0.01, 0.05, 0.10])

    # 4. Final verdict
    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)

    seq_type1 = results["sequential"]["type1_error"]
    k32_type1 = results["K=32"]["type1_error"]
    se = np.sqrt(0.05 * 0.95 / N_SIMS)

    print(f"\n  Sequential Type I error:  {seq_type1:.5f}")
    print(f"  Batched K=32 Type I error: {k32_type1:.5f}")
    print(f"  Difference:                {k32_type1 - seq_type1:+.5f}")
    print(f"  Binomial SE at alpha=0.05: {se:.5f}")
    print(f"  Both within 95% CI of 0.05? [{0.05 - 1.96*se:.4f}, {0.05 + 1.96*se:.4f}]")

    if k32_type1 <= 0.05 + 1.96 * se:
        print("\n  CONCLUSION: Batched stopping at K=32 preserves Type I error control.")
    else:
        print("\n  WARNING: Batched stopping may inflate Type I error. Investigate further.")
