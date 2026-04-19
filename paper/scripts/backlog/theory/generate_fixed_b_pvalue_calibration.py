"""Fixed-B permutation p-value calibration (Theorem 1 backstop).

Archived theory generator; superseded on the live paper path by
`generate_calibration_support_package.py`.

This script empirically checks that the +1 Monte Carlo permutation p-value is (super-)uniform under the null
(exchangeability target of the permutation scheme; in the paper this corresponds to the nodewise complete permutation null).

We use the regressor selector test `ptest_pc` (Pearson correlation permutation test) as an example, but the calibration
is a consequence of exchangeability (Theorem 1), not of this particular statistic.

Outputs
-------
- `paper/results/cache/fixedB_pvalue_calibration_data.parquet`
- `paper/results/figures/fixedB_pvalue_calibration.png`

Run
---
  uv sync --group paper
  UV_CACHE_DIR=$PWD/.uv-cache uv run python paper/scripts/backlog/theory/generate_fixed_b_pvalue_calibration.py
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_paper_dir = Path(__file__).resolve().parents[2]
_cache_root = Path(tempfile.gettempdir()) / "citrees-paper-cache"
_mpl_dir = _cache_root / "mplconfig"
_xdg_dir = _cache_root / "xdg-cache"
_mpl_dir.mkdir(parents=True, exist_ok=True)
_xdg_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_dir))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from matplotlib import pyplot as plt

from citrees._selector import ptest_pc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sims", type=int, default=50_000)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--n-resamples", type=int, default=199)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.n_sims <= 0 or args.n <= 1 or args.n_resamples <= 0:
        raise ValueError("Invalid arguments.")

    rng = np.random.default_rng(args.seed)

    pvals = np.empty(args.n_sims, dtype=np.float64)
    for i in range(args.n_sims):
        x = rng.standard_normal(args.n).astype(np.float64)
        y = rng.standard_normal(args.n).astype(np.float64)
        pvals[i] = ptest_pc(
            x=x,
            y=y,
            standardize=True,
            n_resamples=args.n_resamples,
            early_stopping=None,
            alpha=0.05,
            random_state=i,
        )

    # Empirical CDF at a small grid (for a quick super-uniform check)
    grid = np.array([0.01, 0.02, 0.05, 0.10, 0.20], dtype=np.float64)
    cdf = np.array([(pvals <= t).mean() for t in grid], dtype=np.float64)
    df_summary = pd.DataFrame(
        {"t": grid, "empirical_P(p<=t)": cdf, "t_minus_empirical": grid - cdf}
    )

    figures_dir = _paper_dir / "results" / "figures"
    cache_dir = _paper_dir / "results" / "cache"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / "fixedB_pvalue_calibration_data.parquet"
    fig_path = figures_dir / "fixedB_pvalue_calibration.png"

    df_out = pd.DataFrame({"p_value": pvals})
    df_out.to_parquet(data_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Histogram
    axes[0].hist(pvals, bins=min(50, args.n_resamples), color="#4C78A8", edgecolor="white")
    axes[0].set_title("Histogram of +1 permutation p-values (null)")
    axes[0].set_xlabel("p-value")
    axes[0].set_ylabel("count")

    # CDF
    p_sorted = np.sort(pvals)
    y_ecdf = np.arange(1, args.n_sims + 1) / args.n_sims
    axes[1].plot(p_sorted, y_ecdf, label="empirical CDF")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="black", label="Uniform(0,1)")
    axes[1].set_title("Empirical CDF vs Uniform")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("P(p ≤ t)")
    axes[1].legend(loc="best")

    fig.suptitle(
        f"Fixed-B p-value calibration (n={args.n}, B={args.n_resamples}, sims={args.n_sims})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    with pd.option_context("display.max_columns", 100, "display.width", 120):
        print(df_summary)
    print(f"\nWrote: {data_path}")
    print(f"Wrote: {fig_path}")


if __name__ == "__main__":
    main()
