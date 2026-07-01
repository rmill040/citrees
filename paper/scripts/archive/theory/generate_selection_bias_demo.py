"""Selection-bias demo: CART vs citrees Stage A (root).

Archived theory generator; not part of the current paper-facing rebuild path.

Goal
----
Show the classic CART variable-selection bias toward high-cardinality features under a complete global null, and
contrast with citrees' root Stage A, which uses fixed-`B` permutation p-values + Bonferroni.

This script does **not** modify any algorithm. It runs a controlled simulation and writes a table/figure.

Outputs
-------
- `paper/results/cache/selection_bias_demo_data.parquet`
- `paper/results/figures/selection_bias_demo.png`

Run
---
  uv sync --group paper
  uv run python paper/scripts/archive/theory/generate_selection_bias_demo.py
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_paper_dir = Path(__file__).resolve().parents[2]
# Ensure Matplotlib/fontconfig caches are writable in sandboxed environments.
_cache_root = Path(tempfile.gettempdir()) / "citrees-paper-cache"
_mpl_dir = _cache_root / "mplconfig"
_xdg_dir = _cache_root / "xdg-cache"
_mpl_dir.mkdir(parents=True, exist_ok=True)
_xdg_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_dir))
# Keep this script robust even if Numba can't compile in the current environment.
# (Permutation-test validity does not depend on JIT compilation.)
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from citrees._selector import ptest_mc


@dataclass(frozen=True)
class BiasDemoConfig:
    n_sims: int
    n: int
    n_levels: int
    alpha: float
    base_resamples: int
    seed: int


def _parse_args() -> BiasDemoConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sims", type=int, default=10000)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument(
        "--n-levels", type=int, default=50, help="Number of levels for the categorical feature."
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--base-resamples",
        type=int,
        default=50,
        help=(
            "Base permutations per test (R). Under Bonferroni with m features, citrees scales integer R to B=R*m "
            "per feature-test. We mirror that here for Stage A."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    return BiasDemoConfig(
        n_sims=args.n_sims,
        n=args.n,
        n_levels=args.n_levels,
        alpha=args.alpha,
        base_resamples=args.base_resamples,
        seed=args.seed,
    )


def main() -> None:
    cfg = _parse_args()
    rng = np.random.default_rng(cfg.seed)

    feature_names = ["continuous", "binary", f"categorical_{cfg.n_levels}lv"]
    m = len(feature_names)
    alpha_per_feature = cfg.alpha / m
    n_resamples = cfg.base_resamples * m  # mirrors citrees integer-resamples Bonferroni scaling

    cart_counts = np.zeros(m, dtype=np.int64)
    citrees_counts = np.zeros(m, dtype=np.int64)
    citrees_no_split = 0

    for i in range(cfg.n_sims):
        # Complete global null: y independent of all X
        y = rng.integers(0, 2, size=cfg.n, dtype=np.int64)
        x_cont = rng.standard_normal(cfg.n).astype(np.float64)
        x_bin = rng.integers(0, 2, size=cfg.n).astype(np.float64)
        x_cat = rng.integers(0, cfg.n_levels, size=cfg.n).astype(np.float64)
        X = np.column_stack([x_cont, x_bin, x_cat])

        # CART (max_depth=1 => root split only)
        cart = DecisionTreeClassifier(max_depth=1, random_state=i)
        cart.fit(X, y)
        root_feature = int(cart.tree_.feature[0])
        if root_feature >= 0:
            cart_counts[root_feature] += 1

        # citrees Stage A at root (fixed-B p-values + Bonferroni threshold)
        pvals = np.empty(m, dtype=np.float64)
        for j in range(m):
            pvals[j] = ptest_mc(
                x=X[:, j],
                y=y,
                n_classes=2,
                n_resamples=n_resamples,
                early_stopping=None,
                alpha=alpha_per_feature,
                random_state=i,
            )

        j_star = int(np.argmin(pvals))
        if pvals[j_star] < alpha_per_feature:
            citrees_counts[j_star] += 1
        else:
            citrees_no_split += 1

    # Summaries
    cart_rate = cart_counts / cfg.n_sims
    citrees_rate = citrees_counts / cfg.n_sims
    citrees_split_rate = 1.0 - (citrees_no_split / cfg.n_sims)

    citrees_cond = np.zeros_like(citrees_rate)
    if citrees_split_rate > 0:
        citrees_cond = citrees_counts / (cfg.n_sims - citrees_no_split)

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "cart_root_select_rate": cart_rate,
            "citrees_root_select_rate": citrees_rate,
            "citrees_root_select_rate_given_split": citrees_cond,
        }
    )

    meta = pd.DataFrame(
        [
            {
                "n_sims": cfg.n_sims,
                "n": cfg.n,
                "n_levels": cfg.n_levels,
                "alpha": cfg.alpha,
                "alpha_per_feature": alpha_per_feature,
                "base_resamples": cfg.base_resamples,
                "n_resamples_per_feature_test": n_resamples,
                "citrees_root_split_rate": citrees_split_rate,
            }
        ]
    )

    figures_dir = _paper_dir / "results" / "figures"
    cache_dir = _paper_dir / "results" / "cache"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / "selection_bias_demo_data.parquet"
    fig_path = figures_dir / "selection_bias_demo.png"

    # Store as a single parquet with two tables via a simple convention: concatenate with a marker column.
    df_out = df.copy()
    df_out.insert(0, "_table", "selection_rates")
    meta_out = meta.copy()
    meta_out.insert(0, "_table", "metadata")
    pd.concat([df_out, meta_out], ignore_index=True).to_parquet(data_path, index=False)

    # Figure: two panels (CART vs citrees)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    x = np.arange(m)

    axes[0].bar(x, df["cart_root_select_rate"], color="#4C78A8")
    axes[0].set_title("CART (root feature)")
    axes[0].set_xticks(x, feature_names, rotation=20, ha="right")
    axes[0].set_ylabel("Selection probability")

    axes[1].bar(x, df["citrees_root_select_rate_given_split"], color="#72B7B2")
    axes[1].set_title(f"citrees Stage A (given split)\nPr(split)≈{citrees_split_rate:.3f}")
    axes[1].set_xticks(x, feature_names, rotation=20, ha="right")

    fig.suptitle(
        f"Selection bias under complete global null (alpha={cfg.alpha}, Bonferroni alpha/m={alpha_per_feature:.4f})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    with pd.option_context("display.max_columns", 100, "display.width", 120):
        print(meta)
        print(df)
    print(f"\nWrote: {data_path}")
    print(f"Wrote: {fig_path}")


if __name__ == "__main__":
    main()
