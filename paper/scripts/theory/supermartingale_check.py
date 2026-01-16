"""Sanity checks for citrees' adaptive sequential permutation stopping statistic.

This script validates the key mathematical identity behind the "posterior-confidence" stopping rule used in
`early_stopping="adaptive"`:

- Under the standard continuous-null idealization, the process
    S_n := P(p* < alpha | L_n, n) = I_alpha(1+L_n, 1+n-L_n)
  is a Doob martingale (hence also a supermartingale), and
    W_n := S_n / alpha
  is a test martingale (e-process) with W_0 = 1.

We check this numerically by:
1) verifying the one-step martingale identity E[S_{n+1} | L_n, n] = S_n on a grid;
2) simulating the sequential stopping rule and confirming:
   - E[S_tau] ≈ alpha,
   - P(stop_sig) <= alpha / gamma (Markov/Ville bound).

Important: This script does NOT claim that the returned p-hat at a stopping time is a classical fixed-B p-value.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy.special import betainc


def _beta_cdf(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a,b), i.e., Beta(a,b) CDF at x."""
    return float(betainc(a, b, x))


def s_n(*, l_n: int, n: int, alpha: float) -> float:
    """Posterior-confidence score S_n = P(p* < alpha | L_n, n) under Beta(1,1) prior."""
    if not (0 <= l_n <= n):
        raise ValueError(f"Require 0 <= l_n <= n; got l_n={l_n}, n={n}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Require alpha in (0,1); got {alpha}.")
    return _beta_cdf(alpha, 1.0 + float(l_n), 1.0 + float(n - l_n))


def _expected_s_next(*, l_n: int, n: int, alpha: float) -> float:
    """Posterior-predictive expectation E[S_{n+1} | L_n, n] under the Beta-Binomial model."""
    # Posterior: p | (l_n,n) ~ Beta(1+l_n, 1+n-l_n). Predictive P(exceed) = E[p | data].
    p_exceed = (1.0 + float(l_n)) / (2.0 + float(n))
    return p_exceed * s_n(l_n=l_n + 1, n=n + 1, alpha=alpha) + (1.0 - p_exceed) * s_n(
        l_n=l_n, n=n + 1, alpha=alpha
    )


@dataclass(frozen=True)
class StoppingRun:
    outcome: str  # "stop_sig" | "stop_nonsig" | "max_reached"
    tau: int
    s_tau: float
    p_hat: float
    p_star: float


def _simulate_one(
    *,
    rng: np.random.Generator,
    alpha: float,
    gamma: float,
    n_max: int,
) -> StoppingRun:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Require alpha in (0,1); got {alpha}.")
    if not (0.0 < gamma < 1.0):
        raise ValueError(f"Require gamma in (0,1); got {gamma}.")
    if n_max <= 0:
        raise ValueError(f"Require n_max > 0; got {n_max}.")

    min_n = int(ceil(1.0 / alpha))
    n_max = max(n_max, min_n)

    # Continuous-null idealization: p_star ~ Uniform(0,1)
    p_star = float(rng.random())
    l_n = 0

    for n in range(1, n_max + 1):
        if rng.random() < p_star:
            l_n += 1

        if n < min_n:
            continue

        s = s_n(l_n=l_n, n=n, alpha=alpha)
        if s >= gamma:
            return StoppingRun(
                outcome="stop_sig",
                tau=n,
                s_tau=s,
                p_hat=float((l_n + 1) / (n + 1)),
                p_star=p_star,
            )
        if (1.0 - s) >= gamma:
            return StoppingRun(
                outcome="stop_nonsig",
                tau=n,
                s_tau=s,
                p_hat=float((l_n + 1) / (n + 1)),
                p_star=p_star,
            )

    s = s_n(l_n=l_n, n=n_max, alpha=alpha)
    return StoppingRun(
        outcome="max_reached",
        tau=n_max,
        s_tau=s,
        p_hat=float((l_n + 1) / (n_max + 1)),
        p_star=p_star,
    )


def main() -> None:
    alpha = 0.05
    gamma = 0.95

    # ---------------------------------------------------------------------
    # 1) One-step martingale identity check
    # ---------------------------------------------------------------------
    print("=" * 72)
    print("1) One-step martingale identity: E[S_{n+1} | L_n, n] == S_n")
    print("=" * 72)

    max_abs_err = 0.0
    worst = (0, 0)
    for n in [1, 2, 5, 10, 20, 50, 100, 200]:
        for l_n in range(n + 1):
            s = s_n(l_n=l_n, n=n, alpha=alpha)
            e = _expected_s_next(l_n=l_n, n=n, alpha=alpha)
            err = abs(e - s)
            if err > max_abs_err:
                max_abs_err = err
                worst = (l_n, n)

    print(f"alpha={alpha}, gamma={gamma}")
    print(f"max |E[S_{{n+1}}|L_n,n] - S_n| = {max_abs_err:.3e} at (L_n,n)={worst}")

    # ---------------------------------------------------------------------
    # 2) Stopping-time calibration check
    # ---------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("2) Stopping-time calibration under continuous-null idealization")
    print("=" * 72)

    rng = np.random.default_rng(0)
    n_sims = 10_000
    n_max = 2_000

    runs = [_simulate_one(rng=rng, alpha=alpha, gamma=gamma, n_max=n_max) for _ in range(n_sims)]

    s_tau = np.array([r.s_tau for r in runs], dtype=float)
    tau = np.array([r.tau for r in runs], dtype=int)
    outcome = np.array([r.outcome for r in runs], dtype=object)
    p_hat = np.array([r.p_hat for r in runs], dtype=float)

    stop_sig = outcome == "stop_sig"
    stop_nonsig = outcome == "stop_nonsig"
    max_reached = outcome == "max_reached"

    print(f"n_sims={n_sims}, n_max={n_max}")
    print(f"E[S_tau] = {s_tau.mean():.4f} (target alpha={alpha:.4f})")
    print(f"P(stop_sig) = {stop_sig.mean():.4f} (bound alpha/gamma={alpha/gamma:.4f})")
    print(f"P(stop_nonsig) = {stop_nonsig.mean():.4f}")
    print(f"P(max_reached) = {max_reached.mean():.4f}")
    print(f"mean(tau) = {tau.mean():.1f}, median(tau) = {float(np.median(tau)):.1f}")
    print(f"P(p_hat < alpha) = {float((p_hat < alpha).mean()):.4f}  (algorithm-returned p_hat)")


if __name__ == "__main__":
    main()

