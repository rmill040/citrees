"""Generate the paper's calibration support package under a complete global null.

This script empirically validates three levels of the citrees permutation
testing framework under the null hypothesis (independent X and y):

1. **Filter-level**: ptest_mc (classification) and ptest_pc (regression)
   produce (super-)uniform p-values under exchangeability.
2. **Root-level**: CIT with fixed-B, Bonferroni-corrected Stage A rejects
   the root at rate <= alpha.
3. **Adaptive stopping**: ptest_mc with adaptive early stopping is checked
   empirically against fixed-B null rejection.

Outputs
-------
- paper/results/figures/calibration_pvalue_ecdf.png
- paper/results/figures/calibration_root_split.png
- paper/results/tables/calibration_summary.csv

Run
---
  UV_CACHE_DIR=./scratch/.uv_cache uv run python \
      paper/scripts/theory/generate_calibration_support_package.py

  # Quick mode (fewer sims):
  UV_CACHE_DIR=./scratch/.uv_cache uv run python \
      paper/scripts/theory/generate_calibration_support_package.py --quick
"""

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
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from matplotlib import pyplot as plt

from citrees import ConditionalInferenceTreeClassifier
from citrees._selector import ptest_mc, ptest_pc


FIGURES_DIR = _paper_dir / "results" / "figures"
TABLES_DIR = _paper_dir / "results" / "tables"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Fewer sims for fast iteration")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Part 1: Filter-level p-value ECDF
# ---------------------------------------------------------------------------

def run_ptest_calibration(
    *,
    n_sims: int,
    n: int,
    p: int,
    B: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Run ptest_mc and ptest_pc under null, collect p-values."""
    rng = np.random.default_rng(seed)
    pvals_mc = np.empty(n_sims, dtype=np.float64)
    pvals_pc = np.empty(n_sims, dtype=np.float64)

    for i in range(n_sims):
        x = rng.standard_normal(n).astype(np.float64)

        # Classification null: random binary y, independent of x
        y_clf = rng.integers(0, 2, size=n).astype(np.int64)
        pvals_mc[i] = ptest_mc(
            x=x, y=y_clf, n_classes=2,
            n_resamples=B, early_stopping=None,
            alpha=0.05, random_state=seed + i,
        )

        # Regression null: random continuous y, independent of x
        y_reg = rng.standard_normal(n).astype(np.float64)
        pvals_pc[i] = ptest_pc(
            x=x, y=y_reg, standardize=True,
            n_resamples=B, early_stopping=None,
            alpha=0.05, random_state=seed + i,
        )

    return {"ptest_mc": pvals_mc, "ptest_pc": pvals_pc}


def run_ptest_adaptive_calibration(
    *,
    n_sims: int,
    n: int,
    B: int,
    seed: int,
) -> np.ndarray:
    """Run ptest_mc with adaptive stopping under null."""
    rng = np.random.default_rng(seed)
    pvals = np.empty(n_sims, dtype=np.float64)
    for i in range(n_sims):
        x = rng.standard_normal(n).astype(np.float64)
        y = rng.integers(0, 2, size=n).astype(np.int64)
        pvals[i] = ptest_mc(
            x=x, y=y, n_classes=2,
            n_resamples=B, early_stopping="adaptive",
            alpha=0.05, random_state=seed + i,
        )
    return pvals


# ---------------------------------------------------------------------------
# Part 2: CIT root split rate
# ---------------------------------------------------------------------------

def run_root_split_calibration(
    *,
    n_sims: int,
    n: int,
    p: int,
    B: int,
    alpha: float,
    seed: int,
) -> float:
    """Fraction of null datasets where CIT splits at the root."""
    rng = np.random.default_rng(seed)
    splits = 0
    for i in range(n_sims):
        X = rng.standard_normal((n, p))
        y = rng.integers(0, 2, size=n)
        tree = ConditionalInferenceTreeClassifier(
            selector="mc",
            alpha_selector=alpha,
            n_resamples_selector=B,
            early_stopping_selector=None,
            adjust_alpha_selector=True,
            random_state=i,
        )
        tree.fit(X, y)
        if tree._n_nodes > 1:
            splits += 1
    return splits / n_sims


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ecdf(results: dict[str, np.ndarray], title: str, outpath: Path) -> None:
    """Plot p-value ECDFs vs Uniform(0,1) diagonal."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"ptest_mc": "#4C78A8", "ptest_pc": "#E45756", "ptest_mc (adaptive)": "#72B7B2"}

    # Left: ECDF
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", color="black", lw=1, label="Uniform(0,1)", zorder=0)
    for label, pvals in results.items():
        p_sorted = np.sort(pvals)
        ecdf = np.arange(1, len(pvals) + 1) / len(pvals)
        ax.plot(p_sorted, ecdf, label=label, color=colors.get(label, "gray"), lw=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("P(p-value <= t)")
    ax.set_title("Empirical CDF")
    ax.legend(loc="lower right", fontsize=8)

    # Right: histogram of all combined
    ax2 = axes[1]
    for label, pvals in results.items():
        ax2.hist(pvals, bins=50, alpha=0.5, color=colors.get(label, "gray"),
                 label=label, density=True, edgecolor="white", linewidth=0.3)
    ax2.axhline(1.0, ls="--", color="black", lw=1, label="Uniform density")
    ax2.set_xlabel("p-value")
    ax2.set_ylabel("Density")
    ax2.set_title("Histogram")
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {outpath}")


def plot_root_split(results: list[dict], outpath: Path) -> None:
    """Bar chart of root split rates vs nominal alpha."""
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [r["label"] for r in results]
    rates = [r["split_rate"] for r in results]
    alphas = [r["alpha"] for r in results]

    x = np.arange(len(labels))
    bars = ax.bar(x, rates, color="#4C78A8", edgecolor="white", width=0.6)
    for i, (rate, alpha) in enumerate(zip(rates, alphas)):
        ax.hlines(alpha, i - 0.35, i + 0.35, colors="red", linestyles="--", linewidths=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Root split rate")
    ax.set_title("CIT root split rate under complete global null")

    # Add a legend entry for the red dashes
    from matplotlib.lines import Line2D
    ax.legend(
        [bars[0], Line2D([0], [0], color="red", ls="--", lw=1.5)],
        ["Empirical rate", "Nominal alpha"],
        loc="upper right", fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if args.quick:
        n_sims_ptest = 2000
        n_sims_root = 200
    else:
        n_sims_ptest = 10000
        n_sims_root = 1000

    summary_rows: list[dict] = []

    # --- Part 1: Filter-level p-value calibration ---
    print(f"[1/3] Filter-level p-value calibration ({n_sims_ptest} sims)...")
    ptest_results = run_ptest_calibration(
        n_sims=n_sims_ptest, n=200, p=1, B=199, seed=0,
    )

    # Empirical rejection rates at alpha=0.05
    for label, pvals in ptest_results.items():
        rej = (pvals <= 0.05).mean()
        summary_rows.append({
            "test": label, "mode": "fixed-B", "n": 200, "B": 199,
            "n_sims": n_sims_ptest, "nominal_alpha": 0.05,
            "empirical_rejection_rate": rej,
        })
        print(f"  {label} (fixed-B): reject rate = {rej:.4f} (nominal 0.05)")

    # --- Part 1b: Adaptive stopping calibration ---
    print(f"[1b/3] Adaptive stopping calibration ({n_sims_ptest} sims)...")
    pvals_adaptive = run_ptest_adaptive_calibration(
        n_sims=n_sims_ptest, n=200, B=199, seed=99999,
    )
    rej_adaptive = (pvals_adaptive <= 0.05).mean()
    summary_rows.append({
        "test": "ptest_mc", "mode": "adaptive", "n": 200, "B": 199,
        "n_sims": n_sims_ptest, "nominal_alpha": 0.05,
        "empirical_rejection_rate": rej_adaptive,
    })
    print(f"  ptest_mc (adaptive): reject rate = {rej_adaptive:.4f} (nominal 0.05)")

    # Combined ECDF plot
    all_pvals = {**ptest_results, "ptest_mc (adaptive)": pvals_adaptive}
    plot_ecdf(
        all_pvals,
        title=f"P-value calibration under complete global null (n=200, B=199, {n_sims_ptest} sims)",
        outpath=FIGURES_DIR / "calibration_pvalue_ecdf.png",
    )

    # --- Part 2: CIT root split rate ---
    print(f"[2/3] CIT root split rate ({n_sims_root} sims per condition)...")
    root_results = []
    conditions = [
        {"n": 200, "p": 10, "B": 199, "alpha": 0.05, "label": "n=200, p=10\nalpha=0.05"},
        {"n": 200, "p": 50, "B": 199, "alpha": 0.05, "label": "n=200, p=50\nalpha=0.05"},
        {"n": 200, "p": 100, "B": 199, "alpha": 0.05, "label": "n=200, p=100\nalpha=0.05"},
        {"n": 500, "p": 10, "B": 199, "alpha": 0.05, "label": "n=500, p=10\nalpha=0.05"},
        {"n": 500, "p": 100, "B": 199, "alpha": 0.05, "label": "n=500, p=100\nalpha=0.05"},
    ]
    for cond in conditions:
        rate = run_root_split_calibration(
            n_sims=n_sims_root, n=cond["n"], p=cond["p"],
            B=cond["B"], alpha=cond["alpha"], seed=0,
        )
        root_results.append({**cond, "split_rate": rate})
        summary_rows.append({
            "test": "CIT_root_split", "mode": f"n={cond['n']},p={cond['p']}",
            "n": cond["n"], "B": cond["B"], "n_sims": n_sims_root,
            "nominal_alpha": cond["alpha"],
            "empirical_rejection_rate": rate,
        })
        print(f"  n={cond['n']}, p={cond['p']}: split rate = {rate:.4f} (alpha={cond['alpha']})")

    plot_root_split(root_results, FIGURES_DIR / "calibration_root_split.png")

    # --- Part 3: Multiple B values for ptest_mc ---
    print(f"[3/3] B sensitivity ({n_sims_ptest} sims per B)...")
    for B in [49, 99, 199, 499, 999]:
        rng = np.random.default_rng(B)
        pvals_b = np.empty(n_sims_ptest, dtype=np.float64)
        for i in range(n_sims_ptest):
            x = rng.standard_normal(200).astype(np.float64)
            y = rng.integers(0, 2, size=200).astype(np.int64)
            pvals_b[i] = ptest_mc(
                x=x, y=y, n_classes=2,
                n_resamples=B, early_stopping=None,
                alpha=0.05, random_state=B + i,
            )
        rej = (pvals_b <= 0.05).mean()
        summary_rows.append({
            "test": "ptest_mc", "mode": f"fixed-B={B}", "n": 200, "B": B,
            "n_sims": n_sims_ptest, "nominal_alpha": 0.05,
            "empirical_rejection_rate": rej,
        })
        print(f"  B={B}: reject rate = {rej:.4f}")

    # --- Save summary ---
    df = pd.DataFrame(summary_rows)
    csv_path = TABLES_DIR / "calibration_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Wrote {csv_path}")
    print("\n=== SUMMARY ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
