"""
Theoretical predictions for feature muting power analysis.

This module implements exact power functions for correlation tests and
gap region calculations for the gated effect model.

The gated model is:
    X_0, X_1 ~ N(0, 1) independent
    Z = 1{X_0 > c} with P(Z=1) = p
    Y = 1{X_1 > 0} if Z=1, else Y = Bernoulli(0.5)

Key results:
    - Root correlation: rho_root = 0.798 * p
    - Gate correlation: rho_gate = 0.798 (constant)
    - Gap region: exists when p_min < p < p_max where root test fails but gate succeeds

References:
    - Fisher z-transformation: Fisher (1921)
    - Bias correction: Hotelling (1953)
    - Permutation CLT: Bolthausen (1984), Chen-Shao (2005)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import binom


# =============================================================================
# Constants
# =============================================================================

# Exact population correlation in the gated subset
# rho_gate = 2 / sqrt(2*pi) = sqrt(2/pi)
RHO_GATE: float = 2.0 / np.sqrt(2 * np.pi)  # ≈ 0.7978845608...

# Root correlation coefficient: rho_root = RHO_ROOT_COEF * p
RHO_ROOT_COEF: float = RHO_GATE


# =============================================================================
# Core Power Functions
# =============================================================================


def exact_r_critical(n: int, alpha: float) -> float:
    """
    Compute exact critical value for two-sided correlation test.

    Uses the t-distribution relationship:
        r_crit = t_crit / sqrt(t_crit^2 + n - 2)

    Parameters
    ----------
    n : int
        Sample size (must be >= 4)
    alpha : float
        Significance level (two-sided)

    Returns
    -------
    float
        Critical value |r| > r_crit implies rejection

    Examples
    --------
    >>> exact_r_critical(100, 0.05)  # doctest: +ELLIPSIS
    0.196...
    >>> exact_r_critical(1000, 0.05)  # doctest: +ELLIPSIS
    0.062...
    """
    if n < 4:
        return np.nan
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 2)
    return t_crit / np.sqrt(t_crit**2 + n - 2)


def power_correlation_test(
    rho: float,
    n: int,
    alpha: float,
    use_bias_correction: bool = True,
) -> float:
    """
    Compute exact power of two-sided correlation test.

    Uses Fisher z-transformation with optional bias correction.

    Parameters
    ----------
    rho : float
        Population correlation (-1 < rho < 1)
    n : int
        Sample size (must be >= 4)
    alpha : float
        Significance level (two-sided)
    use_bias_correction : bool
        Whether to apply the O(1/n) bias correction to Fisher z

    Returns
    -------
    float
        Power = P(reject H0 | rho)

    Notes
    -----
    The Fisher z-transformation is:
        z = arctanh(r) ~ N(zeta, 1/(n-3))

    With bias correction (Hotelling, 1953):
        zeta = arctanh(rho) + rho / (2(n-1))

    Without:
        zeta = arctanh(rho)

    Examples
    --------
    >>> # Type I error under H0
    >>> power_correlation_test(0, 1000, 0.05)  # doctest: +ELLIPSIS
    0.05...

    >>> # High power for large effect
    >>> power_correlation_test(0.5, 100, 0.05) > 0.99
    True
    """
    if n < 4:
        return np.nan

    # Clip rho to avoid numerical issues at boundaries
    rho = np.clip(rho, -0.9999, 0.9999)

    # Critical value in z-space
    r_crit = exact_r_critical(n, alpha)
    if np.isnan(r_crit):
        return np.nan
    z_crit = np.arctanh(r_crit)

    # Mean of z under alternative
    zeta = np.arctanh(rho)
    if use_bias_correction and n > 1:
        zeta += rho / (2 * (n - 1))

    # Standard deviation of z
    sigma_z = 1.0 / np.sqrt(n - 3)

    # Two-sided power: P(|z| > z_crit)
    # = P(z > z_crit) + P(z < -z_crit)
    power = 1 - ndtr((z_crit - zeta) / sigma_z) + ndtr((-z_crit - zeta) / sigma_z)

    return float(power)


def root_power(p: float, n: int, alpha: float) -> float:
    """
    Power of root-level test for X1 in gated model.

    The population correlation at the root is rho_root = 0.798 * p
    where p is the gate probability.

    Parameters
    ----------
    p : float
        Gate probability P(Z=1)
    n : int
        Total sample size
    alpha : float
        Significance level

    Returns
    -------
    float
        Power of detecting X1 at the root

    Examples
    --------
    >>> # Small p -> low power at root
    >>> root_power(0.05, 2000, 0.05) < 0.2
    True

    >>> # Large p -> higher power at root
    >>> root_power(0.3, 2000, 0.05) > 0.9
    True
    """
    rho_root = RHO_ROOT_COEF * p
    return power_correlation_test(rho_root, n, alpha)


def gate_power(p: float, n: int, alpha: float) -> float:
    """
    Power of gate-level test for X1 in gated model.

    The population correlation in the gate is rho_gate = 0.798 (constant).
    The sample size is n_gate = floor(n * p).

    Parameters
    ----------
    p : float
        Gate probability P(Z=1)
    n : int
        Total sample size
    alpha : float
        Significance level

    Returns
    -------
    float
        Power of detecting X1 in the gated subset

    Examples
    --------
    >>> # Even with small p, gate power can be high if np is large enough
    >>> gate_power(0.05, 2000, 0.05) > 0.99  # np = 100
    True

    >>> # Very small gate -> insufficient samples
    >>> gate_power(0.001, 1000, 0.05)  # np = 1
    0.0
    """
    n_gate = int(n * p)
    if n_gate < 4:
        return 0.0
    return power_correlation_test(RHO_GATE, n_gate, alpha)


# =============================================================================
# Gap Region Calculation
# =============================================================================


class GapRegion(NamedTuple):
    """
    Gap region boundaries and metadata.

    The gap region is where local muting succeeds (gate power >= beta_high)
    but global muting fails (root power <= beta_low).

    Attributes
    ----------
    p_min : float
        Lower boundary of gap region
    p_max : float
        Upper boundary of gap region
    n : int
        Sample size
    alpha : float
        Significance level
    beta_low : float
        Root power threshold (global muting "fails" if power <= beta_low)
    beta_high : float
        Gate power threshold (local muting "succeeds" if power >= beta_high)
    is_valid : bool
        Whether gap exists (p_min < p_max)
    """

    p_min: float
    p_max: float
    n: int
    alpha: float
    beta_low: float
    beta_high: float
    is_valid: bool

    @property
    def width(self) -> float:
        """Width of gap region (p_max - p_min)."""
        if not self.is_valid:
            return 0.0
        return self.p_max - self.p_min

    @property
    def ratio(self) -> float:
        """Ratio p_max / p_min (measure of gap strength)."""
        if not self.is_valid or self.p_min <= 0:
            return np.nan
        return self.p_max / self.p_min


def find_gap_region(
    n: int,
    alpha: float = 0.05,
    beta_low: float = 0.2,
    beta_high: float = 0.8,
    p_search_min: float = 1e-4,
    p_search_max: float = 0.5,
) -> GapRegion:
    """
    Find the gap region [p_min, p_max] where local muting succeeds
    but global muting fails.

    Parameters
    ----------
    n : int
        Sample size
    alpha : float
        Significance level
    beta_low : float
        Root power threshold (global muting "fails" if power <= beta_low)
    beta_high : float
        Gate power threshold (local muting "succeeds" if power >= beta_high)
    p_search_min, p_search_max : float
        Search bounds for p

    Returns
    -------
    GapRegion
        Named tuple with p_min, p_max, and validity flag

    Examples
    --------
    >>> gap = find_gap_region(2000, 0.05)
    >>> gap.is_valid
    True
    >>> 0.005 < gap.p_min < 0.01
    True
    >>> 0.02 < gap.p_max < 0.05
    True
    """
    # Find p_max: solve root_power(p_max) = beta_low
    def root_objective(p: float) -> float:
        return root_power(p, n, alpha) - beta_low

    try:
        val_min = root_objective(p_search_min)
        val_max = root_objective(p_search_max)

        if val_min > 0:
            # Power already above threshold at p_search_min
            p_max = p_search_min
        elif val_max < 0:
            # Power never reaches threshold
            p_max = np.nan
        else:
            p_max = brentq(root_objective, p_search_min, p_search_max)
    except (ValueError, RuntimeError):
        p_max = np.nan

    # Find p_min: solve gate_power(p_min) = beta_high
    def gate_objective(p: float) -> float:
        return gate_power(p, n, alpha) - beta_high

    try:
        val_min = gate_objective(p_search_min)
        val_max = gate_objective(p_search_max)

        if val_min > 0:
            # Power already above threshold at p_search_min
            p_min = p_search_min
        elif val_max < 0:
            # Power never reaches threshold
            p_min = np.nan
        else:
            p_min = brentq(gate_objective, p_search_min, p_search_max)
    except (ValueError, RuntimeError):
        p_min = np.nan

    # Check validity
    is_valid = not np.isnan(p_min) and not np.isnan(p_max) and p_min < p_max

    return GapRegion(
        p_min=p_min if not np.isnan(p_min) else 0.0,
        p_max=p_max if not np.isnan(p_max) else 0.0,
        n=n,
        alpha=alpha,
        beta_low=beta_low,
        beta_high=beta_high,
        is_valid=is_valid,
    )


def gap_region_table(
    n_values: list[int],
    alpha: float = 0.05,
    beta_low: float = 0.2,
    beta_high: float = 0.8,
) -> str:
    """
    Generate a formatted table of gap regions for different sample sizes.

    Parameters
    ----------
    n_values : list[int]
        Sample sizes to evaluate
    alpha, beta_low, beta_high : float
        Gap region parameters

    Returns
    -------
    str
        Formatted markdown table
    """
    lines = [
        "| n | p_min | p_max | Ratio | Width |",
        "|---:|------:|------:|------:|------:|",
    ]

    for n in n_values:
        gap = find_gap_region(n, alpha, beta_low, beta_high)
        if gap.is_valid:
            lines.append(
                f"| {n:,} | {gap.p_min:.4f} | {gap.p_max:.4f} | "
                f"{gap.ratio:.1f} | {gap.width:.4f} |"
            )
        else:
            lines.append(f"| {n:,} | --- | --- | --- | --- |")

    return "\n".join(lines)


def minimum_n_for_gap(
    alpha: float = 0.05,
    beta_low: float = 0.2,
    beta_high: float = 0.8,
) -> int:
    """
    Find minimum sample size for which a non-empty gap region exists.

    Uses binary search over n.

    Parameters
    ----------
    alpha : float
        Significance level
    beta_low : float
        Root power threshold
    beta_high : float
        Gate power threshold

    Returns
    -------
    int
        Minimum n such that gap region is non-empty

    Examples
    --------
    >>> n_min = minimum_n_for_gap(0.05, 0.2, 0.8)
    >>> 50 < n_min < 150
    True
    """
    n_low, n_high = 10, 10000

    while n_high - n_low > 1:
        n_mid = (n_low + n_high) // 2
        gap = find_gap_region(n_mid, alpha, beta_low, beta_high)

        if gap.is_valid:
            n_high = n_mid
        else:
            n_low = n_mid

    return n_high


# =============================================================================
# Sample Size Randomness Integration
# =============================================================================


def marginal_gate_power_exact(
    p: float,
    n: int,
    alpha: float,
    min_samples: int = 4,
) -> float:
    """
    Gate power marginalized over n_1 ~ Binomial(n, p).

    Computes the exact sum over all possible gate sizes,
    weighted by their binomial probabilities.

    Parameters
    ----------
    p : float
        Gate probability
    n : int
        Total sample size
    alpha : float
        Significance level
    min_samples : int
        Minimum samples required for correlation test

    Returns
    -------
    float
        E[power | n_1 ~ Binomial(n, p)]

    Notes
    -----
    This is generally lower than the deterministic gate_power(p, n, alpha)
    due to Jensen's inequality (power is concave in sample size).
    """
    total_power = 0.0
    total_prob = 0.0

    # Determine effective range (5 sigma around mean)
    mean_k = n * p
    std_k = np.sqrt(n * p * (1 - p))
    k_min = max(min_samples, int(mean_k - 5 * std_k))
    k_max = min(n, int(mean_k + 5 * std_k))

    for k in range(k_min, k_max + 1):
        prob_k = binom.pmf(k, n, p)
        if prob_k < 1e-15:
            continue
        power_k = power_correlation_test(RHO_GATE, k, alpha)
        if not np.isnan(power_k):
            total_power += prob_k * power_k
            total_prob += prob_k

    # Normalize by probability mass in range
    if total_prob > 1e-15:
        return total_power / total_prob
    return 0.0


def marginal_gate_power_approx(
    p: float,
    n: int,
    alpha: float,
    min_samples: int = 4,
) -> float:
    """
    Gate power using normal approximation for n_1.

    Faster than exact computation for large n.
    Valid when np >= 20 and n(1-p) >= 20.

    Parameters
    ----------
    p : float
        Gate probability
    n : int
        Total sample size
    alpha : float
        Significance level
    min_samples : int
        Minimum samples required for correlation test

    Returns
    -------
    float
        Approximate E[power | n_1 ~ N(np, np(1-p))]
    """
    mean_k = n * p
    std_k = np.sqrt(n * p * (1 - p))

    if std_k < 1e-10:
        # Degenerate case: n_1 is essentially deterministic
        return gate_power(p, n, alpha)

    def integrand(k: float) -> float:
        if k < min_samples:
            return 0.0
        power_k = power_correlation_test(RHO_GATE, int(k), alpha)
        if np.isnan(power_k):
            return 0.0
        pdf_k = stats.norm.pdf(k, mean_k, std_k)
        return power_k * pdf_k

    result, _ = quad(integrand, min_samples, n, limit=100)
    return result


# =============================================================================
# Soft Gate Extension
# =============================================================================


def soft_gate_prob(a: float, c: float) -> float:
    """
    Compute P(Z=1) for soft gate with parameters (a, c).

    The soft gate is: P(Z=1 | X_0) = sigmoid(a * (X_0 - c))
    where X_0 ~ N(0, 1).

    Parameters
    ----------
    a : float
        Sharpness parameter (larger = sharper gate)
    c : float
        Threshold parameter

    Returns
    -------
    float
        Marginal probability P(Z=1)

    Notes
    -----
    As a -> infinity, this approaches the hard gate 1{X_0 > c}
    with P(Z=1) = 1 - Phi(c).
    """

    def integrand(x: float) -> float:
        sigmoid = 1 / (1 + np.exp(-a * (x - c)))
        return sigmoid * stats.norm.pdf(x)

    result, _ = quad(integrand, -10, 10)
    return result


def calibrate_soft_gate(target_p: float, a: float) -> float:
    """
    Find threshold c such that P(Z=1) = target_p for given sharpness a.

    Parameters
    ----------
    target_p : float
        Target gate probability (0 < target_p < 1)
    a : float
        Sharpness parameter

    Returns
    -------
    float
        Threshold c such that soft_gate_prob(a, c) = target_p

    Examples
    --------
    >>> c = calibrate_soft_gate(0.05, 100)
    >>> abs(soft_gate_prob(100, c) - 0.05) < 0.001
    True
    """

    def objective(c: float) -> float:
        return soft_gate_prob(a, c) - target_p

    return brentq(objective, -5, 5)


# =============================================================================
# Noisy Gate Extension
# =============================================================================


def noisy_gate_true_positive_rate(
    p0: float,
    eta: float,
) -> float:
    """
    Compute true positive rate among Z=1 observations under noisy gate.

    The noisy gate is: Z = 1{X_0 > c} XOR Bernoulli(eta)

    Parameters
    ----------
    p0 : float
        Clean gate probability P(X_0 > c)
    eta : float
        Gate noise rate (flip probability), must be in [0, 0.5)

    Returns
    -------
    float
        P(X_0 > c | Z=1) = pi_TP

    Notes
    -----
    pi_TP measures the "purity" of the gated subset. When eta=0, pi_TP=1.
    As eta increases, the gated subset becomes contaminated with false positives.
    """
    # P(Z=1) = (1-eta)*p0 + eta*(1-p0)
    p_z1 = (1 - eta) * p0 + eta * (1 - p0)

    if p_z1 < 1e-15:
        return 0.0

    # P(X_0 > c, Z=1) = P(X_0 > c) * P(Z=1 | X_0 > c) = p0 * (1-eta)
    p_true_positive = p0 * (1 - eta)

    return p_true_positive / p_z1


def noisy_gate_correlations(
    p0: float,
    eta: float,
) -> dict[str, float]:
    """
    Compute root and gate correlations under noisy gate model.

    Parameters
    ----------
    p0 : float
        Clean gate probability
    eta : float
        Gate noise rate

    Returns
    -------
    dict
        Contains:
        - p0: clean gate probability
        - eta: noise rate
        - p_effective: P(Z=1) under noisy gate
        - pi_TP: true positive rate
        - rho_root: root correlation
        - rho_gate: gate correlation (reduced by pi_TP)
        - rho_ratio: rho_gate / rho_root
    """
    # Effective gate probability
    p = (1 - eta) * p0 + eta * (1 - p0)

    # Root correlation (unchanged formula, uses effective p)
    rho_root = RHO_ROOT_COEF * p

    # True positive rate
    pi_TP = noisy_gate_true_positive_rate(p0, eta)

    # Gate correlation (reduced by pi_TP)
    rho_gate = RHO_GATE * pi_TP

    return {
        "p0": p0,
        "eta": eta,
        "p_effective": p,
        "pi_TP": pi_TP,
        "rho_root": rho_root,
        "rho_gate": rho_gate,
        "rho_ratio": rho_gate / rho_root if rho_root > 0 else np.inf,
    }


# =============================================================================
# Multi-Feature Competition
# =============================================================================


def prob_feature_selected(
    mu: float,
    sigma: float,
    m: int,
) -> float:
    """
    Probability that signal feature (with mean mu) has the largest statistic
    among m features where m-1 are null.

    Assumes all statistics are approximately Gaussian with the same variance.

    Parameters
    ----------
    mu : float
        Mean of signal feature's statistic
    sigma : float
        Standard deviation of all statistics (assumed equal)
    m : int
        Total number of features

    Returns
    -------
    float
        P(T_1 > max_{j>=2} T_j)
    """
    if m == 1:
        return 1.0

    if sigma < 1e-15:
        return 1.0 if mu > 0 else 0.0

    # Integrate over distribution of T_1
    def integrand(t: float) -> float:
        # P(T_1 = t) * P(max null < t)
        pdf_t1 = stats.norm.pdf(t, mu, sigma)
        prob_max_null_below = stats.norm.cdf(t / sigma) ** (m - 1)
        return pdf_t1 * prob_max_null_below

    # Integrate from -inf to +inf (practically, 6 sigma range)
    t_min = mu - 6 * sigma
    t_max = mu + 6 * sigma

    result, _ = quad(integrand, t_min, t_max)
    return result


# =============================================================================
# Tree Depth Propagation
# =============================================================================


def critical_depth(
    p: float,
    n: int,
    alpha: float = 0.05,
    rho_gate: float = RHO_GATE,
) -> float:
    """
    Compute critical depth at which gated signal becomes undetectable.

    Assumes balanced splits where sample size halves at each depth.

    Parameters
    ----------
    p : float
        Gate probability
    n : int
        Initial sample size
    alpha : float
        Significance level
    rho_gate : float
        Population correlation in gated subset

    Returns
    -------
    float
        Maximum depth at which signal is detectable

    Notes
    -----
    At depth d, the sample size in the gate is approximately n*p / 2^d.
    The signal is detectable when sqrt(n_gate) * rho_gate > z_{1-alpha/2}.
    """
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # d < 2 * log2(rho_gate * sqrt(np) / z_crit)
    arg = rho_gate * np.sqrt(n * p) / z_crit

    if arg <= 1:
        return 0.0  # Not detectable even at root

    return 2 * np.log2(arg)


def power_at_depth(
    p: float,
    n: int,
    d: int,
    alpha: float = 0.05,
) -> dict[str, float | int | bool]:
    """
    Compute detection power at a given tree depth.

    Assumes balanced splits and that the gate is at depth d.

    Parameters
    ----------
    p : float
        Gate probability
    n : int
        Initial sample size
    d : int
        Depth of the gated node
    alpha : float
        Significance level

    Returns
    -------
    dict
        Contains depth, sample sizes, powers, and whether gap exists
    """
    # Sample size at depth d
    n_d = n // (2**d)

    # Sample size in gate at depth d
    n_gate = int(n_d * p)

    # Power for root (at full sample)
    power_root_full = root_power(p, n, alpha)

    # Power at depth d (reduced sample)
    power_depth_d = root_power(p, n_d, alpha) if n_d >= 4 else 0.0

    # Power for gate at depth d
    power_gate_at_d = power_correlation_test(RHO_GATE, n_gate, alpha) if n_gate >= 4 else 0.0

    return {
        "depth": d,
        "n_at_depth": n_d,
        "n_gate": n_gate,
        "power_root": power_root_full,
        "power_at_depth": power_depth_d,
        "power_gate": power_gate_at_d,
        "gap_exists": power_gate_at_d > 0.8 and power_depth_d < 0.2,
    }


def depth_propagation_table(
    p: float,
    n: int,
    max_depth: int = 10,
    alpha: float = 0.05,
) -> str:
    """
    Generate table showing power degradation with tree depth.

    Parameters
    ----------
    p : float
        Gate probability
    n : int
        Initial sample size
    max_depth : int
        Maximum depth to evaluate
    alpha : float
        Significance level

    Returns
    -------
    str
        Formatted markdown table
    """
    lines = [
        "| Depth | n_node | n_gate | Power@node | Power@gate | Gap? |",
        "|------:|-------:|-------:|-----------:|-----------:|:-----|",
    ]

    for d in range(max_depth + 1):
        result = power_at_depth(p, n, d, alpha)
        gap_str = "YES" if result["gap_exists"] else "no"
        lines.append(
            f"| {d:>5} | {result['n_at_depth']:>6} | {result['n_gate']:>6} | "
            f"{result['power_at_depth']:>10.4f} | {result['power_gate']:>10.4f} | "
            f"{gap_str:>4} |"
        )

    d_crit = critical_depth(p, n, alpha)
    lines.append(f"\nCritical depth: {d_crit:.1f}")

    return "\n".join(lines)


# =============================================================================
# Utility Functions
# =============================================================================


def compare_deterministic_vs_marginal(
    p: float,
    n: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    """
    Compare deterministic vs marginalized gate power.

    Returns
    -------
    dict
        Contains 'deterministic', 'marginal_exact', 'marginal_approx',
        and their differences.
    """
    det = gate_power(p, n, alpha)
    marg_exact = marginal_gate_power_exact(p, n, alpha)
    marg_approx = marginal_gate_power_approx(p, n, alpha)

    return {
        "p": p,
        "n": n,
        "n_gate_expected": int(n * p),
        "n_gate_std": np.sqrt(n * p * (1 - p)),
        "deterministic": det,
        "marginal_exact": marg_exact,
        "marginal_approx": marg_approx,
        "diff_exact": det - marg_exact,
        "diff_approx": det - marg_approx,
    }


def noisy_gate_gap_analysis(
    p0: float,
    eta_values: list[float] | None = None,
) -> str:
    """
    Analyze how gate noise affects the correlation gap.

    Parameters
    ----------
    p0 : float
        Clean gate probability
    eta_values : list[float], optional
        Noise rates to analyze (default: [0, 0.05, 0.1, 0.15, 0.2])

    Returns
    -------
    str
        Formatted markdown table
    """
    if eta_values is None:
        eta_values = [0, 0.05, 0.1, 0.15, 0.2]

    lines = [
        "| η (noise) | p_eff | π_TP | ρ_root | ρ_gate | Ratio |",
        "|----------:|------:|-----:|-------:|-------:|------:|",
    ]

    for eta in eta_values:
        result = noisy_gate_correlations(p0, eta)
        lines.append(
            f"| {eta:>9.2f} | {result['p_effective']:>5.3f} | "
            f"{result['pi_TP']:>4.2f} | {result['rho_root']:>6.4f} | "
            f"{result['rho_gate']:>6.4f} | {result['rho_ratio']:>5.1f} |"
        )

    return "\n".join(lines)


def soft_gate_calibration_table(
    target_p: float,
    a_values: list[float] | None = None,
) -> str:
    """
    Generate calibration table for soft gate parameters.

    Shows the threshold c needed for each sharpness a to achieve target_p.

    Parameters
    ----------
    target_p : float
        Target gate probability
    a_values : list[float], optional
        Sharpness values to evaluate (default: [1, 2, 5, 10, 50, 100])

    Returns
    -------
    str
        Formatted markdown table
    """
    if a_values is None:
        a_values = [1, 2, 5, 10, 50, 100]

    lines = [
        "| Sharpness a | Threshold c | Actual p |",
        "|------------:|------------:|---------:|",
    ]

    for a in a_values:
        c = calibrate_soft_gate(target_p, a)
        actual_p = soft_gate_prob(a, c)
        lines.append(f"| {a:>11.1f} | {c:>11.4f} | {actual_p:>8.4f} |")

    # Hard gate limit
    c_hard = stats.norm.ppf(1 - target_p)
    lines.append(f"| {'∞ (hard)':>11} | {c_hard:>11.4f} | {target_p:>8.4f} |")

    return "\n".join(lines)


def multi_feature_power_analysis(
    p: float,
    n: int,
    m_values: list[int] | None = None,
    alpha: float = 0.05,
) -> str:
    """
    Analyze power under multi-feature competition for gated model.

    Parameters
    ----------
    p : float
        Gate probability
    n : int
        Sample size
    m_values : list[int], optional
        Number of features to analyze (default: [1, 10, 50, 100, 500])
    alpha : float
        Significance level

    Returns
    -------
    str
        Formatted markdown table
    """
    if m_values is None:
        m_values = [1, 10, 50, 100, 500]

    rho_root = RHO_ROOT_COEF * p
    sigma = 1 / np.sqrt(n)  # Approximate std of correlation

    lines = [
        "| m features | P(select X_1) | Bonf threshold |",
        "|-----------:|--------------:|---------------:|",
    ]

    for m in m_values:
        p_select = prob_feature_selected(rho_root, sigma, m)
        bonf_thresh = alpha / m

        lines.append(f"| {m:>10} | {p_select:>13.4f} | {bonf_thresh:>14.5f} |")

    return "\n".join(lines)


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=== Gap Region Table ===")
    print(gap_region_table([100, 200, 500, 1000, 2000, 5000, 10000]))
    print()

    print("=== Minimum n for gap ===")
    n_min = minimum_n_for_gap()
    print(f"Minimum n for gap region: {n_min}")
    print()

    print("=== Depth Propagation (n=2000, p=0.05) ===")
    print(depth_propagation_table(0.05, 2000, max_depth=6))
    print()

    print("=== Noisy Gate Analysis (p0=0.05) ===")
    print(noisy_gate_gap_analysis(0.05))
    print()

    print("=== Soft Gate Calibration (target_p=0.05) ===")
    print(soft_gate_calibration_table(0.05))
    print()

    print("=== Multi-Feature Competition (n=2000, p=0.05) ===")
    print(multi_feature_power_analysis(0.05, 2000))
