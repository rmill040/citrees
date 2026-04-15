"""CIF vs R implementations under the paper benchmark contract.

2-panel figure (1x2): CIF vs r_cforest | CIF vs r_ctree
  - X-axis: feature budget k (5, 10, 25, 50, 100)
  - Y-axis: mean dataset-level balanced-accuracy delta (CIF - opponent)
  - CIF / opponent scores are first averaged within each
    `(dataset, downstream_model, k)` cell
  - The plotted delta at each `k` then averages over downstream models within
    dataset, matching the broad classification pairwise interpretation used in
    the manuscript
  - Error bars: bootstrap 95% CI of the mean dataset-level delta
  - Wilcoxon signed-rank p-value annotated at each k
  - n_datasets shown at each k

Usage:
    UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/fig_cif_vs_r.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from paper.scripts.analysis.benchmark_common import STANDARD_K, TASK_CONFIG, dataset_scores, load_real_task_frame, select_best_task_configs

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS = Path("paper/results")
FIGURES = RESULTS / "figures"

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
CIF_BLUE = "#2563EB"
R_ORANGE = "#F97316"
GREEN_POS = "#16A34A"
K_VALUES = list(STANDARD_K)

DISPLAY_NAMES = {
    "r_cforest": "R-cforest",
    "r_ctree": "R-ctree",
}


def _setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------
def _load_and_prepare() -> pd.DataFrame:
    """Load the broad classification benchmark surface for CIF-vs-R."""
    config = TASK_CONFIG["classification"]
    raw = load_real_task_frame(config["path"])
    standard_raw = raw[raw["k"].isin(K_VALUES)].copy()

    _, best_configs = select_best_task_configs(standard_raw, config["metric"])
    best = standard_raw.merge(best_configs[["method_base", "method_id"]], on=["method_base", "method_id"], how="inner")

    return dataset_scores(best, config["metric"])


def _compute_deltas(
    dataset_means: pd.DataFrame,
    opponent: str,
) -> dict[int, dict]:
    """Compute pairwise deltas (CIF - opponent) at each k with bootstrap CIs."""
    cif = dataset_means[dataset_means["method_base"] == "cif"]
    opp = dataset_means[dataset_means["method_base"] == opponent]

    results: dict[int, dict] = {}
    rng = np.random.default_rng(42)

    for k in K_VALUES:
        cif_k = cif[cif["k"] == k][["dataset", "downstream_model", "dataset_mean_score"]].rename(
            columns={"dataset_mean_score": "cif_score"}
        )
        opp_k = opp[opp["k"] == k][["dataset", "downstream_model", "dataset_mean_score"]].rename(
            columns={"dataset_mean_score": "opp_score"}
        )

        # Pair on the broad benchmark surface, then average downstream-model
        # deltas within dataset so the unit of comparison matches the paper's
        # dataset-level breadth summaries.
        merged = cif_k.merge(opp_k, on=["dataset", "downstream_model"], how="inner")
        if merged.empty:
            continue

        merged["delta"] = merged["cif_score"] - merged["opp_score"]
        by_dataset = merged.groupby("dataset", as_index=False)["delta"].mean()
        deltas = by_dataset["delta"].to_numpy()
        if len(deltas) < 2:
            continue

        mean_delta = deltas.mean()
        n_datasets = len(deltas)

        # Bootstrap 95% CI of the mean delta (1000 resamples)
        boot_means = np.empty(1000)
        for i in range(1000):
            sample = rng.choice(deltas, size=n_datasets, replace=True)
            boot_means[i] = sample.mean()
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        # Wilcoxon signed-rank test
        try:
            _, pval = stats.wilcoxon(deltas, alternative="two-sided")
        except ValueError:
            pval = 1.0

        results[k] = {
            "mean_delta": mean_delta,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "pval": pval,
            "n_datasets": n_datasets,
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _format_pval(p: float) -> str:
    if p < 0.001:
        return "$p < .001$"
    if p < 0.01:
        return f"$p = {p:.3f}$"
    if p < 0.05:
        return f"$p = {p:.2f}$"
    return f"$p = {p:.2f}$"


def _plot_panel(
    ax: plt.Axes,
    results: dict[int, dict],
    opponent_name: str,
) -> None:
    """Draw one panel of the delta plot."""
    ks = sorted(results.keys())
    means = [results[k]["mean_delta"] for k in ks]
    ci_los = [results[k]["ci_lo"] for k in ks]
    ci_his = [results[k]["ci_hi"] for k in ks]
    pvals = [results[k]["pval"] for k in ks]
    n_ds = [results[k]["n_datasets"] for k in ks]

    yerr_lo = [m - lo for m, lo in zip(means, ci_los)]
    yerr_hi = [hi - m for m, hi in zip(means, ci_his)]

    # Determine if all deltas are positive
    all_positive = all(m > 0 for m in means)
    point_color = GREEN_POS if all_positive else CIF_BLUE

    # Zero line
    ax.axhline(0, color="#9CA3AF", linewidth=0.8, linestyle="--", zorder=1)

    # Error bars + points
    ax.errorbar(
        ks,
        means,
        yerr=[yerr_lo, yerr_hi],
        fmt="o",
        color=point_color,
        markerfacecolor=point_color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        markersize=7,
        capsize=4,
        capthick=1.2,
        linewidth=1.2,
        ecolor=point_color,
        alpha=0.9,
        zorder=3,
    )

    # Connect points with a light line
    ax.plot(ks, means, color=point_color, linewidth=1.0, alpha=0.4, zorder=2)

    # Annotate p-values and n_datasets
    for idx, (k_val, m, p, n) in enumerate(zip(ks, means, pvals, n_ds)):
        ci_top = ci_his[idx]
        ci_bottom = ci_los[idx]

        # p-value above the upper CI cap
        ax.annotate(
            _format_pval(p),
            xy=(k_val, ci_top),
            xytext=(0, 8),
            textcoords="offset points",
            fontsize=6.5,
            ha="center",
            va="bottom",
            color="#374151",
        )
        # n_datasets below the lower CI cap
        ax.annotate(
            f"$n={n}$",
            xy=(k_val, ci_bottom),
            xytext=(0, -8),
            textcoords="offset points",
            fontsize=6.5,
            ha="center",
            va="top",
            color="#6B7280",
        )

    ax.set_xlabel("Feature budget $k$")
    ax.set_ylabel("$\\Delta$ balanced accuracy\n(CIF $-$ opponent)")
    ax.set_title(f"CIF vs {opponent_name}", fontsize=10, fontweight="bold")

    # Log-scale x-axis to spread out k=5,10 vs k=50,100
    ax.set_xscale("log")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.minorticks_off()

    # Set y-limits with headroom, then shade positive region
    y_lo = min(ci_los) - 0.03
    y_hi = max(ci_his) + 0.04
    ax.set_ylim(y_lo, y_hi)
    ax.axhspan(0, y_hi, color=GREEN_POS, alpha=0.03, zorder=0)


def main() -> None:
    _setup_style()
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    dataset_means = _load_and_prepare()

    print("Computing deltas...")
    opponents = [
        ("r_cforest", "R-cforest"),
        ("r_ctree", "R-ctree"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, (opp_base, opp_display) in zip(axes, opponents):
        results = _compute_deltas(dataset_means, opp_base)
        _plot_panel(ax, results, opp_display)

        # Print summary
        for k in sorted(results.keys()):
            r = results[k]
            print(
                f"  k={k:>3d}: delta={r['mean_delta']:+.4f} "
                f"CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
                f"p={r['pval']:.4f} n={r['n_datasets']}"
            )
        print()

    # Only first panel needs y-label
    axes[1].set_ylabel("")

    fig.tight_layout()
    out = FIGURES / "cif_vs_r.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
