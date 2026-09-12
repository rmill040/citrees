"""Exact null size of the implemented adaptive stopping rule via the Polya-urn recursion.

Used by the manuscripts; `tests/paper/test_adaptive_stopping_size.py` regenerates the
reported values.

Under exchangeability with no ties, the number of exceedances among the first m
permutations follows the Polya urn started from one white and one black ball:
P(k_m = k) = 1/(m+1), and given (m, k) the next indicator is 1 with probability
(k+1)/(m+2). The implemented rule checks the Beta(1+k, 1+m-k) posterior mass
below alpha at batch boundaries m in {s, 2s, ...} with m >= ceil(1/alpha), stops
when that mass is >= c or <= 1-c, and returns (k+1)/(m+1); the decision rejects
when the returned value is <= alpha. This module computes P(reject | H0) exactly.
"""

from __future__ import annotations

from fractions import Fraction
from math import ceil

from scipy.stats import beta


def checkpoints(alpha: float, s: int, B: int) -> list[int]:
    m_min = ceil(1.0 / alpha)
    pts = [m for m in range(s, B + 1, s) if m >= m_min]
    if not pts or pts[-1] != B:
        pts.append(B)
    return pts


def null_size(alpha: float, c: float, s: int, B: int, decision_alpha: float | None = None) -> float:
    """P(returned p-value <= decision_alpha) under H0, no ties (exact)."""
    dec = alpha if decision_alpha is None else decision_alpha
    B = max(B, ceil(1.0 / alpha))
    pts = checkpoints(alpha, s, B)
    # state: distribution over k after m permutations; start m=0, k=0 with prob 1
    dist = {0: Fraction(1)}
    m = 0
    reject = Fraction(0)
    for target in pts:
        # advance Polya urn from m to target
        for _ in range(target - m):
            new: dict[int, Fraction] = {}
            for k, pr in dist.items():
                p1 = Fraction(k + 1, m + 2)
                new[k + 1] = new.get(k + 1, Fraction(0)) + pr * p1
                new[k] = new.get(k, Fraction(0)) + pr * (1 - p1)
            dist = new
            m += 1
        # apply stopping rule at boundary m == target
        stopped: dict[int, Fraction] = {}
        for k, pr in dist.items():
            prob_sig = float(beta.cdf(alpha, 1 + k, 1 + m - k))
            stop = (prob_sig >= c) or ((1.0 - prob_sig) >= c) or (m == B)
            if stop:
                if (k + 1) / (m + 1) <= dec:
                    reject += pr
            else:
                stopped[k] = pr
        dist = stopped
        if not dist:
            break
    return float(reject)


if __name__ == "__main__":
    s = 32
    print("alpha  c     B      size")
    for alpha, c, B in [
        (0.05, 0.95, 20),
        (0.05, 0.95, 999),
        (0.05, 0.80, 999),
        (0.05, 0.95, 1537),
        (0.05 / 10, 0.95, 200),
        (0.05 / 50, 0.95, 1000),
        (0.05 / 200, 0.95, 4000),
    ]:
        print(f"{alpha:<7.4g}{c:<6}{B:<7}{null_size(alpha, c, s, B):.5f}")
