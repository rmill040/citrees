"""Generate publication-quality ablation figures from CSV results.

Archived figure suite; not part of the current paper-facing rebuild path.

Reads ablation CSVs from paper/results/tables/ and produces figures
saved to paper/results/figures/ablation_*.png.

Usage:
    uv run python paper/scripts/backlog/analysis/fig_ablation_suite.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]  # citrees/
TABLE_DIR = ROOT / "paper" / "results" / "tables"
FIG_DIR = ROOT / "paper" / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

# Color palette — colorblind-friendly
CIF_DEFAULT = "#2563EB"  # blue-600
CIF_LIGHT = "#93C5FD"  # blue-300
CIF_DARK = "#1E40AF"  # blue-800


R_PRIMARY = "#F97316"  # orange-500
R_SECONDARY = "#FB923C"  # orange-400

# Categorical palette for dataset types
DATASET_PALETTE = {
    # classification
    "clf_standard_easy": "#2563EB",
    "clf_standard_hard": "#7C3AED",
    "clf_weak_signal": "#DC2626",
    "clf_nonlinear": "#059669",
    "clf_toeplitz": "#0891B2",
    "clf_confounder": "#D97706",
    "clf_bias": "#6D28D9",
    "clf_redundant": "#64748B",
    # regression
    "reg_friedman": "#2563EB",
    "reg_linear": "#7C3AED",
    "reg_highdim": "#DC2626",
    "reg_toeplitz": "#0891B2",
    "reg_weak_signal": "#D97706",
    "reg_confounder": "#059669",
}

DATASET_LABELS = {
    "clf_standard_easy": "Easy",
    "clf_standard_hard": "Hard (p=1000)",
    "clf_weak_signal": "Weak signal",
    "clf_nonlinear": "Nonlinear",
    "clf_toeplitz": "Toeplitz",
    "clf_confounder": "Confounder",
    "clf_bias": "Bias",
    "clf_redundant": "Redundant",
    "reg_friedman": "Friedman",
    "reg_linear": "Linear",
    "reg_highdim": "High-dim",
    "reg_toeplitz": "Toeplitz",
    "reg_weak_signal": "Weak signal",
    "reg_confounder": "Confounder",
}

# Markers for lines
DATASET_MARKERS = {
    "clf_standard_easy": "o",
    "clf_standard_hard": "s",
    "clf_weak_signal": "^",
    "clf_nonlinear": "D",
    "clf_toeplitz": "v",
    "clf_confounder": "P",
    "clf_bias": "X",
    "clf_redundant": "h",
    "reg_friedman": "o",
    "reg_linear": "s",
    "reg_highdim": "^",
    "reg_toeplitz": "v",
    "reg_weak_signal": "D",
    "reg_confounder": "P",
}


def _apply_style(ax: mpl.axes.Axes) -> None:
    """Apply publication style to an axes object."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.grid(axis="y", color="#D1D5DB", alpha=0.3, linewidth=0.5)
    ax.grid(axis="x", visible=False)
    ax.tick_params(labelsize=8, width=0.5)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)


def _save(fig: mpl.figure.Figure, name: str) -> None:
    """Save figure and close."""
    path = FIG_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> saved {path.relative_to(ROOT)}")


def _read_csv(name: str) -> pd.DataFrame | None:
    """Read a CSV file, returning None if it doesn't exist."""
    path = TABLE_DIR / name
    if not path.exists():
        print(f"  [SKIP] {name} not found")
        return None
    return pd.read_csv(path)


# ===================================================================
# Figure 1: Alpha vs. Precision@10 (proxy for depth control)
# ===================================================================
def fig_alpha_depth() -> None:
    print("Figure 1: ablation_alpha_depth.png")
    df = _read_csv("alpha_sweep.csv")
    if df is None:
        return

    # Focus on a curated set of dataset types that show the alpha story
    # We show both clf and reg to be comprehensive.
    # Use precision_at_10_mean as the y-axis since mean_depth is not available.
    focus_types = [
        "clf_standard_easy",
        "clf_standard_hard",
        "clf_weak_signal",
        "clf_toeplitz",
        "reg_toeplitz",
        "reg_weak_signal",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

    for ax, task_prefix, title in [
        (axes[0], "clf_", "Classification"),
        (axes[1], "reg_", "Regression"),
    ]:
        _apply_style(ax)
        task_types = [t for t in focus_types if t.startswith(task_prefix)]
        sub = df[df["dataset_type"].isin(task_types)].copy()

        for dtype in task_types:
            dd = sub[sub["dataset_type"] == dtype].sort_values("alpha_selector")
            if dd.empty:
                continue
            ax.plot(
                dd["alpha_selector"],
                dd["precision_at_10_mean"],
                marker=DATASET_MARKERS.get(dtype, "o"),
                color=DATASET_PALETTE.get(dtype, CIF_DEFAULT),
                label=DATASET_LABELS.get(dtype, dtype),
                markersize=5,
                linewidth=1.5,
            )
            # Error bands
            ax.fill_between(
                dd["alpha_selector"],
                dd["precision_at_10_mean"] - dd["precision_at_10_std"],
                dd["precision_at_10_mean"] + dd["precision_at_10_std"],
                alpha=0.1,
                color=DATASET_PALETTE.get(dtype, CIF_DEFAULT),
            )

        ax.set_xscale("log")
        ax.set_xlabel(r"$\alpha_{\mathrm{selector}}$")
        if ax is axes[0]:
            ax.set_ylabel("Precision@10")
        ax.set_title(title, fontsize=10, fontweight="medium", pad=6)
        ax.legend(fontsize=7, frameon=False, loc="best")

    fig.tight_layout(w_pad=2)
    _save(fig, "ablation_alpha_depth.png")


# ===================================================================
# Figure 2: n_estimators vs. Precision@10
# ===================================================================
def fig_n_estimators() -> None:
    print("Figure 2: ablation_n_estimators.png")
    df = _read_csv("n_estimators_sweep.csv")
    if df is None:
        return

    # Show both clf and reg side by side
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

    for ax, task_prefix, title in [
        (axes[0], "clf_", "Classification"),
        (axes[1], "reg_", "Regression"),
    ]:
        _apply_style(ax)
        sub = df[df["task"] == task_prefix.rstrip("_")].copy()
        dtypes = sub["dataset_type"].unique()

        for dtype in sorted(dtypes):
            dd = sub[sub["dataset_type"] == dtype].sort_values("n_estimators")
            if dd.empty:
                continue
            ax.plot(
                dd["n_estimators"],
                dd["precision_at_10_mean"],
                marker=DATASET_MARKERS.get(dtype, "o"),
                color=DATASET_PALETTE.get(dtype, CIF_DEFAULT),
                label=DATASET_LABELS.get(dtype, dtype),
                markersize=5,
                linewidth=1.5,
            )
            ax.fill_between(
                dd["n_estimators"],
                dd["precision_at_10_mean"] - dd["precision_at_10_std"],
                dd["precision_at_10_mean"] + dd["precision_at_10_std"],
                alpha=0.1,
                color=DATASET_PALETTE.get(dtype, CIF_DEFAULT),
            )

        ax.set_xlabel("Number of estimators")
        if ax is axes[0]:
            ax.set_ylabel("Precision@10")
        ax.set_title(title, fontsize=10, fontweight="medium", pad=6)
        ax.legend(fontsize=7, frameon=False, loc="best")

    fig.tight_layout(w_pad=2)
    _save(fig, "ablation_n_estimators.png")


# ===================================================================
# Figure 3: Scaling — runtime vs n_samples and n_features
# ===================================================================
def fig_scaling() -> None:
    """CIF-only scaling: runtime vs n and runtime vs p."""
    print("Figure 3: ablation_scaling.png")
    df = _read_csv("scaling_curves.csv")
    if df is None:
        return

    # Only show CIF — cross-method runtime comparisons are invalid
    # (different languages, libraries, hardware).
    cif = df[df["method"] == "cif"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # (a) Runtime vs n_samples (sweep == "n")
    ax = axes[0]
    _apply_style(ax)
    sub_n = cif[cif["sweep"] == "n"].sort_values("n_samples")
    if not sub_n.empty:
        ax.plot(
            sub_n["n_samples"],
            sub_n["elapsed_median"],
            marker="o",
            color=CIF_DEFAULT,
            label="CIF",
            markersize=5,
            linewidth=1.5,
        )
    ax.set_yscale("log")
    ax.set_xlabel("$n$ (samples)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("(a) Scaling with $n$", fontsize=10, fontweight="medium", pad=6)
    ax.legend(fontsize=8, frameon=False)

    # (b) Runtime vs n_features (sweep == "p")
    ax = axes[1]
    _apply_style(ax)
    sub_p = cif[cif["sweep"] == "p"].sort_values("n_features")
    if not sub_p.empty:
        ax.plot(
            sub_p["n_features"],
            sub_p["elapsed_median"],
            marker="o",
            color=CIF_DEFAULT,
            label="CIF",
            markersize=5,
            linewidth=1.5,
        )
    ax.set_yscale("log")
    ax.set_xlabel("$p$ (features)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("(b) Scaling with $p$", fontsize=10, fontweight="medium", pad=6)
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout(w_pad=2)
    _save(fig, "ablation_scaling.png")


# ===================================================================
# Figure 4: Bootstrap & feature subsampling
# ===================================================================
def fig_bootstrap() -> None:
    print("Figure 4: ablation_bootstrap.png")
    df = _read_csv("bootstrap_vs_subsampling.csv")
    if df is None:
        return

    # Parse the variant column: "boot_allfeats", "noboot_sqrt", etc.
    df["bootstrap"] = df["variant"].str.startswith("boot_")
    df["feat_subsample"] = df["variant"].str.split("_", n=1).str[1]

    # Focus on a few representative dataset types
    focus = ["clf_standard_easy", "clf_standard_hard", "clf_weak_signal", "clf_confounder"]
    sub = df[df["dataset_type"].isin(focus)].copy()

    # Grouped bar chart: for each dataset, bars for each variant
    feat_order = ["allfeats", "sqrt", "log2", "half"]
    feat_labels = ["All", r"$\sqrt{p}$", r"$\log_2 p$", r"$p/2$"]

    fig, axes = plt.subplots(1, len(focus), figsize=(7, 2.8), sharey=True)
    if len(focus) == 1:
        axes = [axes]

    bar_width = 0.35
    x = np.arange(len(feat_order))

    for ax, dtype in zip(axes, focus):
        _apply_style(ax)
        dd = sub[sub["dataset_type"] == dtype]

        # Bootstrap ON
        boot_vals = []
        boot_errs = []
        for fs in feat_order:
            row = dd[(dd["bootstrap"]) & (dd["feat_subsample"] == fs)]
            boot_vals.append(row["precision_at_10_mean"].values[0] if len(row) else 0)
            boot_errs.append(row["precision_at_10_std"].values[0] if len(row) else 0)

        # Bootstrap OFF
        noboot_vals = []
        noboot_errs = []
        for fs in feat_order:
            row = dd[(~dd["bootstrap"]) & (dd["feat_subsample"] == fs)]
            noboot_vals.append(row["precision_at_10_mean"].values[0] if len(row) else 0)
            noboot_errs.append(row["precision_at_10_std"].values[0] if len(row) else 0)

        ax.bar(
            x - bar_width / 2,
            boot_vals,
            bar_width,
            yerr=boot_errs,
            label="Bootstrap",
            color=CIF_DEFAULT,
            alpha=0.85,
            capsize=2,
            error_kw={"linewidth": 0.7},
        )
        ax.bar(
            x + bar_width / 2,
            noboot_vals,
            bar_width,
            yerr=noboot_errs,
            label="No bootstrap",
            color=CIF_LIGHT,
            alpha=0.85,
            capsize=2,
            error_kw={"linewidth": 0.7},
        )

        ax.set_xticks(x)
        ax.set_xticklabels(feat_labels, fontsize=7)
        ax.set_title(DATASET_LABELS.get(dtype, dtype), fontsize=9, fontweight="medium", pad=4)
        if ax is axes[0]:
            ax.set_ylabel("Precision@10", fontsize=9)
        if ax is axes[-1]:
            ax.legend(fontsize=7, frameon=False, loc="upper right")

    fig.tight_layout(w_pad=1.5)
    _save(fig, "ablation_bootstrap.png")


# ===================================================================
# Figure 5: Power analysis — rejection rate vs. signal strength
# ===================================================================
def fig_power() -> None:
    print("Figure 5: ablation_power.png")
    df = _read_csv("power_analysis.csv")
    if df is None:
        return

    # Focus on alpha=0.05, vary B and stopping
    fig, ax = plt.subplots(figsize=(3.5, 3))
    _apply_style(ax)

    alpha_focus = 0.05
    sub = df[df["alpha"] == alpha_focus].copy()

    # Build unique (B, stopping) combos, pick a few representative B values
    b_vals = sorted(sub["B"].unique())
    # Show B=49, 99, 499, 999 for adaptive & fixed_B
    b_show = [v for v in [49, 99, 499, 999] if v in b_vals]

    # Color gradient for B values
    blue_gradient = ["#93C5FD", "#60A5FA", "#2563EB", "#1E40AF"]
    orange_gradient = ["#FED7AA", "#FDBA74", "#F97316", "#EA580C"]

    for idx, b in enumerate(b_show):
        for stopping, ls, colors in [
            ("adaptive", "-", blue_gradient),
            ("fixed_B", "--", orange_gradient),
        ]:
            dd = sub[(sub["B"] == b) & (sub["stopping"] == stopping)].sort_values("class_sep")
            if dd.empty:
                continue
            color = colors[min(idx, len(colors) - 1)]
            lbl = f"B={b}, {stopping}" if stopping == "adaptive" else f"B={b}, full"
            ax.plot(
                dd["class_sep"],
                dd["rejection_rate"],
                color=color,
                linestyle=ls,
                linewidth=1.3,
                marker="o" if stopping == "adaptive" else "s",
                markersize=4,
                label=lbl,
            )

    # Nominal alpha line
    ax.axhline(y=alpha_focus, color="#EF4444", linestyle=":", linewidth=1.0, alpha=0.8, label=r"Nominal $\alpha$")

    ax.set_xlabel("Signal strength (class separation)")
    ax.set_ylabel("Rejection rate")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="center right")

    fig.tight_layout()
    _save(fig, "ablation_power.png")


# ===================================================================
# Figure 6: n_resamples (B) vs. Precision@10
# ===================================================================
def fig_nresamples() -> None:
    print("Figure 6: ablation_nresamples.png")
    df = _read_csv("resamples_and_honesty.csv")
    if df is None:
        return

    # Filter to B=* variants only (exclude honesty/no_honesty rows)
    b_rows = df[df["variant"].str.startswith("B=")].copy()
    b_rows["B"] = b_rows["variant"].str.replace("B=", "").astype(int)

    # Show both clf and reg
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

    for ax, task_prefix, title in [
        (axes[0], "clf_", "Classification"),
        (axes[1], "reg_", "Regression"),
    ]:
        _apply_style(ax)
        sub = b_rows[b_rows["task"] == task_prefix.rstrip("_")]
        dtypes = sorted(sub["dataset_type"].unique())

        for dtype in dtypes:
            dd = sub[sub["dataset_type"] == dtype].sort_values("B")
            if dd.empty:
                continue
            ax.plot(
                dd["B"],
                dd["precision_at_10_mean"],
                marker=DATASET_MARKERS.get(dtype, "o"),
                color=DATASET_PALETTE.get(dtype, CIF_DEFAULT),
                label=DATASET_LABELS.get(dtype, dtype),
                markersize=5,
                linewidth=1.5,
            )
            ax.fill_between(
                dd["B"],
                dd["precision_at_10_mean"] - dd["precision_at_10_std"],
                dd["precision_at_10_mean"] + dd["precision_at_10_std"],
                alpha=0.1,
                color=DATASET_PALETTE.get(dtype, CIF_DEFAULT),
            )

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks([49, 99, 199, 499, 999])
        ax.set_xticklabels(["49", "99", "199", "499", "999"], fontsize=7)
        ax.set_xlabel("$B$ (number of resamples)")
        if ax is axes[0]:
            ax.set_ylabel("Precision@10")
        ax.set_title(title, fontsize=10, fontweight="medium", pad=6)
        ax.legend(fontsize=7, frameon=False, loc="best")

    fig.tight_layout(w_pad=2)
    _save(fig, "ablation_nresamples.png")


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    print(f"Reading from: {TABLE_DIR.relative_to(ROOT)}")
    print(f"Saving to:    {FIG_DIR.relative_to(ROOT)}\n")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig_alpha_depth()
    fig_n_estimators()
    fig_scaling()
    fig_bootstrap()
    fig_power()
    fig_nresamples()

    print("\nDone.")


if __name__ == "__main__":
    main()
