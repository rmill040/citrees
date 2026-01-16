"""Generate paper-ready outputs for citrees' adaptive sequential permutation stopping rule.

This script simulates the *null* behavior under the standard continuous-null idealization:

- Draw a single p* ~ Uniform(0, 1) per simulated test (rank/PIT argument).
- Conditional on p*, permutation exceedances are i.i.d. Bernoulli(p*).
- citrees' adaptive stopping uses the Beta(1+L_n, 1+n-L_n) posterior-confidence score
  S_n = I_alpha(1+L_n, 1+n-L_n) and stops when S_n >= gamma (confident significant) or
  1-S_n >= gamma (confident non-significant), after a minimum n >= ceil(1/alpha).
- The function returns the usual +1 Monte Carlo estimate p_hat = (L_n+1)/(n+1) evaluated at the stopping time.

Outputs:
- `paper/results/figures/sequential_stopping_calibration_data.parquet`
- `paper/results/figures/sequential_stopping_calibration.png`

Run:
  uv sync
  uv run python paper/scripts/theory/generate_sequential_stopping_calibration.py
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

_paper_dir = Path(__file__).resolve().parents[2]
# Ensure Matplotlib/fontconfig caches are writable in sandboxed environments.
_cache_root = Path(tempfile.gettempdir()) / "citrees-paper-cache"
_mpl_dir = _cache_root / "mplconfig"
_xdg_dir = _cache_root / "xdg-cache"
_mpl_dir.mkdir(parents=True, exist_ok=True)
_xdg_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_dir))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import betainc


@dataclass(frozen=True)
class SimulationResult:
    gamma: float
    n_sims: int
    n_max: int
    alpha: float
    reject_rate: float
    stop_sig_rate: float
    stop_nonsig_rate: float
    max_reached_rate: float
    mean_stop_n: float
    median_stop_n: float
    bound_stop_sig: float


def _simulate_one_gamma(
    *,
    rng: np.random.Generator,
    n_sims: int,
    n_max: int,
    alpha: float,
    gamma: float,
) -> SimulationResult:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1); got {alpha}")
    if not (0.0 < gamma < 1.0):
        raise ValueError(f"gamma must be in (0,1); got {gamma}")
    if n_sims <= 0:
        raise ValueError(f"n_sims must be positive; got {n_sims}")
    if n_max <= 0:
        raise ValueError(f"n_max must be positive; got {n_max}")

    min_resamples = int(np.ceil(1.0 / alpha))
    n_max = max(n_max, min_resamples)

    # Under the continuous-null idealization: p* ~ Uniform(0, 1)
    p_star = rng.random(n_sims)

    active = np.ones(n_sims, dtype=bool)
    l_n = np.zeros(n_sims, dtype=np.int64)
    stop_n = np.full(n_sims, n_max, dtype=np.int64)
    # reason codes: 0 = sig, 1 = nonsig, 2 = max_reached
    reason = np.full(n_sims, 2, dtype=np.int8)

    for n in range(1, n_max + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break

        # Exceedance update
        exceed = rng.random(idx.size) < p_star[idx]
        l_n[idx] += exceed.astype(np.int64)

        if n < min_resamples:
            continue

        a = 1.0 + l_n[idx].astype(np.float64)
        b = 1.0 + (n - l_n[idx]).astype(np.float64)
        prob_sig = betainc(a, b, alpha)  # I_alpha(a,b)

        stop_sig = prob_sig >= gamma
        stop_nonsig = (1.0 - prob_sig) >= gamma
        stop = stop_sig | stop_nonsig
        if not np.any(stop):
            continue

        stop_idx = idx[stop]
        stop_n[stop_idx] = n
        reason[stop_idx[stop_sig[stop]]] = 0
        reason[stop_idx[stop_nonsig[stop]]] = 1
        active[stop_idx] = False

    # p_hat returned by citrees at the stopping time (or at n_max if never stopped)
    p_hat = (l_n.astype(np.float64) + 1.0) / (stop_n.astype(np.float64) + 1.0)

    reject = p_hat < alpha
    reject_rate = float(np.mean(reject))

    stop_sig_rate = float(np.mean(reason == 0))
    stop_nonsig_rate = float(np.mean(reason == 1))
    max_reached_rate = float(np.mean(reason == 2))

    return SimulationResult(
        gamma=gamma,
        n_sims=n_sims,
        n_max=n_max,
        alpha=alpha,
        reject_rate=reject_rate,
        stop_sig_rate=stop_sig_rate,
        stop_nonsig_rate=stop_nonsig_rate,
        max_reached_rate=max_reached_rate,
        mean_stop_n=float(np.mean(stop_n)),
        median_stop_n=float(np.median(stop_n)),
        bound_stop_sig=float(alpha / gamma),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sims", type=int, default=50_000)
    parser.add_argument("--n-max", type=int, default=2_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--gammas",
        type=str,
        default="0.90,0.95,0.975,0.99",
        help="Comma-separated confidence thresholds in (0,1).",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    gammas = [float(x.strip()) for x in args.gammas.split(",") if x.strip()]
    if not gammas:
        raise ValueError("No gammas provided.")

    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, float | int]] = []
    for gamma in gammas:
        result = _simulate_one_gamma(
            rng=rng,
            n_sims=args.n_sims,
            n_max=args.n_max,
            alpha=args.alpha,
            gamma=gamma,
        )
        rows.append(result.__dict__)

    df = pd.DataFrame(rows).sort_values("gamma").reset_index(drop=True)

    paper_dir = Path(__file__).resolve().parents[2]
    out_dir = paper_dir / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = out_dir / "sequential_stopping_calibration_data.parquet"
    fig_path = out_dir / "sequential_stopping_calibration.png"

    df.to_parquet(data_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["gamma"], df["reject_rate"], marker="o", label=r"Pr($\hat p_\tau < \alpha$) (algorithm output)")
    ax.plot(df["gamma"], df["stop_sig_rate"], marker="o", label=r"Pr(stop due to $S_\tau \geq \gamma$)")
    ax.plot(df["gamma"], df["bound_stop_sig"], linestyle="--", label=r"Bound: $\alpha/\gamma$")
    ax.axhline(args.alpha, color="black", linewidth=1, linestyle=":", label=r"Target $\alpha$")
    ax.set_xlabel(r"Confidence threshold $\gamma$")
    ax.set_ylabel("Probability under $H_0$")
    ax.set_title("Adaptive sequential stopping calibration (null)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    with pd.option_context("display.max_columns", 100, "display.width", 120):
        print(df)
    print(f"\nWrote: {data_path}")
    print(f"Wrote: {fig_path}")


if __name__ == "__main__":
    main()
