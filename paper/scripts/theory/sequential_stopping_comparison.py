"""Compare citrees Beta CDF vs Fischer-Ramdas - NJIT OPTIMIZED VERSION.

Fair comparison: both stopping rules compiled with numba @njit.
"""

import time
from math import ceil, exp, lgamma, log

import numpy as np
from numba import njit

# =============================================================================
# NJIT-compiled implementations
# =============================================================================


@njit(cache=True, fastmath=True)
def beta_cdf_njit(x: float, a: float, b: float) -> float:
    """Beta CDF using continued fraction expansion (Lentz's algorithm).

    This is citrees' actual implementation.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    if a <= 0 or b <= 0:
        return 0.5

    if x > (a + 1) / (a + b + 2):
        return 1.0 - beta_cdf_njit(1 - x, b, a)

    max_iter = 200
    eps = 1e-10

    log_prefix = a * log(x) + b * log(1 - x) - log(a)
    log_prefix += lgamma(a + b) - lgamma(a) - lgamma(b)

    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    result = d

    for m in range(1, max_iter):
        m2 = 2 * m

        num = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        result *= d * c

        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        delta = d * c
        result *= delta

        if abs(delta - 1.0) < eps:
            break

    return exp(log_prefix) * result


@njit(cache=True, fastmath=True)
def binomial_cdf_njit(k: int, n: int, p: float) -> float:
    """Binomial CDF: P(X <= k) for X ~ Binomial(n, p).

    Uses the relationship: BinomialCDF(k; n, p) = 1 - BetaCDF(p; k+1, n-k)
    This is mathematically exact and allows us to reuse the Beta CDF.
    """
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0

    # BinomialCDF(k; n, p) = 1 - I_p(k+1, n-k) = 1 - BetaCDF(p; k+1, n-k)
    return 1.0 - beta_cdf_njit(p, k + 1, n - k)


@njit(cache=True, fastmath=True)
def citrees_check_stopping(
    extreme_count: int,
    n: int,
    alpha: float,
    confidence: float,
) -> tuple:
    """citrees Beta CDF adaptive stopping check.

    Returns: (should_stop, is_significant)
    """
    a = 1.0 + extreme_count
    b = 1.0 + n - extreme_count
    prob_sig = beta_cdf_njit(alpha, a, b)

    if prob_sig >= confidence:
        return True, True  # Stop, significant
    if (1.0 - prob_sig) >= confidence:
        return True, False  # Stop, non-significant
    return False, False  # Continue


@njit(cache=True, fastmath=True)
def fischer_ramdas_check_stopping(
    extreme_count: int,
    n: int,
    alpha: float,
    c: float,
) -> tuple:
    """Fischer-Ramdas wealth-based stopping check.

    Wealth W_t = (1 - BinomialCDF(L_t; n, c)) / c
    Reject when W_t >= 1/alpha

    Returns: (should_stop, is_significant)
    """
    if c <= 0:
        return False, False

    binom_cdf = binomial_cdf_njit(extreme_count, n, c)
    wealth = (1.0 - binom_cdf) / c

    threshold = 1.0 / alpha
    if wealth >= threshold:
        return True, True  # Stop, significant

    # Futility: if wealth is very low
    if wealth < alpha and n >= int(ceil(1.0 / alpha)):
        return True, False  # Stop, non-significant

    return False, False  # Continue


@njit(cache=True, fastmath=True)
def pearson_corr_njit(x: np.ndarray, y: np.ndarray) -> float:
    """Absolute Pearson correlation - njit compiled."""
    n = len(x)
    if n < 2:
        return 0.0

    mean_x = 0.0
    mean_y = 0.0
    for i in range(n):
        mean_x += x[i]
        mean_y += y[i]
    mean_x /= n
    mean_y /= n

    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for i in range(n):
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy

    if var_x < 1e-10 or var_y < 1e-10:
        return 0.0

    corr = cov / (var_x**0.5 * var_y**0.5)
    return abs(corr)


@njit(cache=True, fastmath=True)
def run_single_test_citrees(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    confidence: float,
    max_resamples: int,
    seed: int,
) -> tuple:
    """Single permutation test with citrees stopping. Returns (rejected, n_perms)."""
    np.random.seed(seed)
    min_resamples = int(ceil(1.0 / alpha))

    theta = pearson_corr_njit(x, y)
    y_ = y.copy()
    extreme_count = 0

    for i in range(max_resamples):
        np.random.shuffle(y_)
        theta_p = pearson_corr_njit(x, y_)
        if theta_p >= theta:
            extreme_count += 1

        n = i + 1
        if n >= min_resamples:
            should_stop, is_sig = citrees_check_stopping(extreme_count, n, alpha, confidence)
            if should_stop:
                return is_sig, n

    pvalue = (extreme_count + 1) / (max_resamples + 1)
    return pvalue < alpha, max_resamples


@njit(cache=True, fastmath=True)
def run_single_test_fischer(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    c: float,
    max_resamples: int,
    seed: int,
) -> tuple:
    """Single permutation test with Fischer-Ramdas stopping. Returns (rejected, n_perms)."""
    np.random.seed(seed)
    min_resamples = int(ceil(1.0 / alpha))

    theta = pearson_corr_njit(x, y)
    y_ = y.copy()
    extreme_count = 0

    for i in range(max_resamples):
        np.random.shuffle(y_)
        theta_p = pearson_corr_njit(x, y_)
        if theta_p >= theta:
            extreme_count += 1

        n = i + 1
        if n >= min_resamples:
            should_stop, is_sig = fischer_ramdas_check_stopping(extreme_count, n, alpha, c)
            if should_stop:
                return is_sig, n

    pvalue = (extreme_count + 1) / (max_resamples + 1)
    return pvalue < alpha, max_resamples


# =============================================================================
# Experiment runners (not njit - orchestration code)
# =============================================================================


def warmup_jit():
    """Warm up JIT compilation."""
    print("Warming up JIT compilation...")
    x = np.random.randn(50)
    y = np.random.randn(50)
    for _ in range(10):
        run_single_test_citrees(x, y, 0.05, 0.95, 100, 42)
        run_single_test_fischer(x, y, 0.05, 0.05, 100, 42)
    print("JIT warmup complete.\n")


def run_cdf_timing_benchmark(n_iterations: int = 100000):
    """Benchmark CDF computation - both njit compiled."""
    print(f"{'=' * 60}")
    print("EXPERIMENT: CDF Computation Timing (both njit)")
    print(f"{'=' * 60}")

    # Generate test cases
    rng = np.random.default_rng(42)
    ns = rng.integers(20, 500, n_iterations)
    ks = np.array([rng.integers(0, n) for n in ns])

    # Warmup
    for _ in range(1000):
        beta_cdf_njit(0.05, 10.0, 50.0)
        binomial_cdf_njit(5, 50, 0.05)

    # Benchmark Beta CDF
    start = time.perf_counter()
    for k, n in zip(ks, ns, strict=False):
        a = 1.0 + k
        b = 1.0 + n - k
        _ = beta_cdf_njit(0.05, a, b)
    beta_time = time.perf_counter() - start

    # Benchmark Binomial CDF (via Beta)
    start = time.perf_counter()
    for k, n in zip(ks, ns, strict=False):
        _ = binomial_cdf_njit(k, n, 0.05)
    binom_time = time.perf_counter() - start

    print(f"\n{'Method':<35} {'Total (s)':<15} {'Per call (μs)':<15}")
    print("-" * 65)
    print(
        f"{'Beta CDF (citrees stopping)':<35} {beta_time:<15.3f} {beta_time / n_iterations * 1e6:<15.2f}"
    )
    print(
        f"{'Binomial CDF (F-R stopping)':<35} {binom_time:<15.3f} {binom_time / n_iterations * 1e6:<15.2f}"
    )

    return beta_time, binom_time


def run_type1_error_experiment(
    n_simulations: int = 10000,
    n_samples: int = 100,
    alpha: float = 0.05,
    confidence: float = 0.95,
    max_resamples: int = 1000,
    seed: int = 42,
):
    """Compare Type I error under null hypothesis."""
    print(f"\n{'=' * 60}")
    print("EXPERIMENT: Type I Error (Null Hypothesis)")
    print(f"{'=' * 60}")
    print(f"Simulations: {n_simulations}, Samples: {n_samples}, Alpha: {alpha}")

    rng = np.random.default_rng(seed)

    citrees_rejections = 0
    fischer_rejections = 0
    citrees_perms = []
    fischer_perms = []

    # Time the full experiments
    start_citrees = time.perf_counter()
    for sim in range(n_simulations):
        x = rng.standard_normal(n_samples)
        y = rng.standard_normal(n_samples)
        rejected, n_perms = run_single_test_citrees(
            x, y, alpha, confidence, max_resamples, seed + sim
        )
        citrees_rejections += rejected
        citrees_perms.append(n_perms)
    citrees_total_time = time.perf_counter() - start_citrees

    start_fischer = time.perf_counter()
    rng = np.random.default_rng(seed)  # Reset for same data
    for sim in range(n_simulations):
        x = rng.standard_normal(n_samples)
        y = rng.standard_normal(n_samples)
        rejected, n_perms = run_single_test_fischer(x, y, alpha, alpha, max_resamples, seed + sim)
        fischer_rejections += rejected
        fischer_perms.append(n_perms)
    fischer_total_time = time.perf_counter() - start_fischer

    citrees_type1 = citrees_rejections / n_simulations
    fischer_type1 = fischer_rejections / n_simulations

    print(f"\n{'Metric':<30} {'citrees':<20} {'Fischer-Ramdas':<20}")
    print("-" * 70)
    print(f"{'Type I Error':<30} {citrees_type1:<20.4f} {fischer_type1:<20.4f}")
    print(
        f"{'Mean Permutations':<30} {np.mean(citrees_perms):<20.1f} {np.mean(fischer_perms):<20.1f}"
    )
    print(
        f"{'Median Permutations':<30} {np.median(citrees_perms):<20.1f} {np.median(fischer_perms):<20.1f}"
    )
    print(f"{'Total Time (s)':<30} {citrees_total_time:<20.2f} {fischer_total_time:<20.2f}")
    print(
        f"{'Time per test (ms)':<30} {citrees_total_time / n_simulations * 1000:<20.3f} {fischer_total_time / n_simulations * 1000:<20.3f}"
    )

    return {
        "citrees_type1": citrees_type1,
        "fischer_type1": fischer_type1,
        "citrees_perms": citrees_perms,
        "fischer_perms": fischer_perms,
        "citrees_time": citrees_total_time,
        "fischer_time": fischer_total_time,
    }


def run_power_experiment(
    n_simulations: int = 5000,
    n_samples: int = 100,
    effect_sizes: list[float] | None = None,
    alpha: float = 0.05,
    confidence: float = 0.95,
    max_resamples: int = 1000,
    seed: int = 42,
):
    """Compare power under alternative hypothesis."""
    if effect_sizes is None:
        effect_sizes = [0.1, 0.2, 0.3, 0.5]

    print(f"\n{'=' * 60}")
    print("EXPERIMENT: Power (Alternative Hypothesis)")
    print(f"{'=' * 60}")

    print(f"\n{'Effect':<10} {'citrees':<12} {'F-R':<12} {'citrees perms':<15} {'F-R perms':<15}")
    print("-" * 65)

    for effect in effect_sizes:
        rng = np.random.default_rng(seed)

        citrees_rej = 0
        fischer_rej = 0
        citrees_perms = []
        fischer_perms = []

        for sim in range(n_simulations):
            x = rng.standard_normal(n_samples)
            noise = rng.standard_normal(n_samples)
            y = effect * x + np.sqrt(1 - effect**2) * noise

            rej, n_p = run_single_test_citrees(x, y, alpha, confidence, max_resamples, seed + sim)
            citrees_rej += rej
            citrees_perms.append(n_p)

            rej, n_p = run_single_test_fischer(x, y, alpha, alpha, max_resamples, seed + sim)
            fischer_rej += rej
            fischer_perms.append(n_p)

        print(
            f"{effect:<10.2f} {citrees_rej / n_simulations:<12.3f} {fischer_rej / n_simulations:<12.3f} "
            f"{np.mean(citrees_perms):<15.1f} {np.mean(fischer_perms):<15.1f}"
        )


if __name__ == "__main__":
    print("=" * 60)
    print("COMPARING STOPPING RULES - NJIT OPTIMIZED")
    print("=" * 60)

    warmup_jit()

    # CDF timing (both njit)
    run_cdf_timing_benchmark(n_iterations=100000)

    # Type I error
    results = run_type1_error_experiment(n_simulations=10000)

    # Power
    run_power_experiment(n_simulations=3000)

    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"\nType I Error: citrees={results['citrees_type1']:.4f}, F-R={results['fischer_type1']:.4f}"
    )
    print(
        f"Mean Perms (null): citrees={np.mean(results['citrees_perms']):.1f}, F-R={np.mean(results['fischer_perms']):.1f}"
    )
    print(
        f"Speedup: citrees is {results['fischer_time'] / results['citrees_time']:.2f}x faster overall"
    )
