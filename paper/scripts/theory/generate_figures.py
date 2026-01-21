#!/usr/bin/env python3
"""
Generate figures for feature muting power analysis.

Creates visualizations of:
1. Power curves (root vs gate) as a function of p
2. Gap region boundaries across sample sizes
3. Validation: empirical vs theoretical power

Usage:
    uv run python paper/scripts/theory/generate_figures.py
    uv run python paper/scripts/theory/generate_figures.py --output-dir paper/results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from theoretical_predictions import (
    RHO_GATE,
    critical_depth,
    find_gap_region,
    gate_power,
    root_power,
)

# Style settings
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "root": "#2E86AB",  # Blue
    "gate": "#A23B72",  # Magenta
    "gap": "#F18F01",  # Orange
    "theory": "#C73E1D",  # Red
    "empirical": "#3B1F2B",  # Dark
}


def plot_power_curves(
    n: int = 2000,
    alpha: float = 0.05,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Plot root and gate power as functions of gate probability p.

    Shows the power gap: region where gate power is high but root power is low.
    """
    p_values = np.linspace(0.001, 0.3, 200)

    root_powers = [root_power(p, n, alpha) for p in p_values]
    gate_powers = [gate_power(p, n, alpha) for p in p_values]

    # Find gap region
    gap = find_gap_region(n, alpha)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Power curves
    ax.plot(p_values, root_powers, color=COLORS["root"], linewidth=2.5, label="Root power")
    ax.plot(p_values, gate_powers, color=COLORS["gate"], linewidth=2.5, label="Gate power")

    # Gap region shading
    if gap.is_valid:
        ax.axvspan(gap.p_min, gap.p_max, alpha=0.2, color=COLORS["gap"], label=f"Gap region")
        ax.axvline(gap.p_min, color=COLORS["gap"], linestyle="--", alpha=0.7)
        ax.axvline(gap.p_max, color=COLORS["gap"], linestyle="--", alpha=0.7)

    # Reference lines
    ax.axhline(0.8, color="gray", linestyle=":", alpha=0.5, label="Power = 0.8")
    ax.axhline(0.2, color="gray", linestyle=":", alpha=0.5, label="Power = 0.2")

    ax.set_xlabel("Gate probability $p$", fontsize=12)
    ax.set_ylabel("Power", fontsize=12)
    ax.set_title(f"Feature Muting Power Analysis ($n={n:,}$, $\\alpha={alpha}$)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 1.05)

    # Add annotation for gap
    if gap.is_valid:
        mid_p = (gap.p_min + gap.p_max) / 2
        ax.annotate(
            f"Gap: [{gap.p_min:.3f}, {gap.p_max:.3f}]\nRatio: {gap.ratio:.1f}x",
            xy=(mid_p, 0.5),
            xytext=(mid_p + 0.05, 0.5),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"),
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
        )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_gap_region_by_n(
    n_values: list[int] | None = None,
    alpha: float = 0.05,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Plot gap region boundaries across different sample sizes.

    Shows how the gap region evolves with n.
    """
    if n_values is None:
        n_values = [100, 200, 500, 1000, 2000, 5000, 10000]

    p_mins = []
    p_maxs = []
    ratios = []
    valid_n = []

    for n in n_values:
        gap = find_gap_region(n, alpha)
        if gap.is_valid:
            valid_n.append(n)
            p_mins.append(gap.p_min)
            p_maxs.append(gap.p_max)
            ratios.append(gap.ratio)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Gap boundaries
    ax1 = axes[0]
    ax1.fill_between(valid_n, p_mins, p_maxs, alpha=0.3, color=COLORS["gap"], label="Gap region")
    ax1.plot(valid_n, p_mins, "o-", color=COLORS["gate"], linewidth=2, markersize=6, label="$p_{min}$ (gate constraint)")
    ax1.plot(valid_n, p_maxs, "s-", color=COLORS["root"], linewidth=2, markersize=6, label="$p_{max}$ (root constraint)")
    ax1.set_xlabel("Sample size $n$", fontsize=12)
    ax1.set_ylabel("Gate probability $p$", fontsize=12)
    ax1.set_title("Gap Region Boundaries", fontsize=14)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Right: Gap ratio
    ax2 = axes[1]
    ax2.plot(valid_n, ratios, "o-", color=COLORS["gap"], linewidth=2, markersize=8)
    ax2.set_xlabel("Sample size $n$", fontsize=12)
    ax2.set_ylabel("Gap ratio ($p_{max} / p_{min}$)", fontsize=12)
    ax2.set_title("Gap Strength vs Sample Size", fontsize=14)
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)

    # Add annotation
    ax2.annotate(
        "Larger n → wider gap\n(more room for\nlocal muting to help)",
        xy=(valid_n[-1], ratios[-1]),
        xytext=(valid_n[-1] / 3, ratios[-1] * 0.7),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray"),
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
    )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_depth_propagation(
    n: int = 2000,
    p: float = 0.05,
    alpha: float = 0.05,
    max_depth: int = 8,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Plot power degradation with tree depth.
    """
    depths = list(range(max_depth + 1))

    # At each depth, sample size halves
    n_at_depth = [n // (2**d) for d in depths]
    n_gate_at_depth = [int(and * p) for and in n_at_depth]

    # Power at each depth (testing at the node level, not root)
    # For root-style test at depth d with reduced sample
    power_node = []
    power_gate = []

    for d, and, ng in zip(depths, n_at_depth, n_gate_at_depth):
        if and >= 4:
            # Power of detecting the diluted signal at this node
            power_node.append(root_power(p, and, alpha))
        else:
            power_node.append(0.0)

        if ng >= 4:
            # Power of detecting the full signal in the gate
            from theoretical_predictions import power_correlation_test

            power_gate.append(power_correlation_test(RHO_GATE, ng, alpha))
        else:
            power_gate.append(0.0)

    # Critical depth
    d_crit = critical_depth(p, n, alpha)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(depths, power_node, "o-", color=COLORS["root"], linewidth=2, markersize=8, label="Root-style power at node")
    ax.plot(depths, power_gate, "s-", color=COLORS["gate"], linewidth=2, markersize=8, label="Gate power")

    # Reference lines
    ax.axhline(0.8, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(0.2, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(d_crit, color=COLORS["gap"], linestyle="--", alpha=0.7, label=f"Critical depth = {d_crit:.1f}")

    # Annotations for sample sizes
    for d, and, ng in zip(depths, n_at_depth, n_gate_at_depth):
        ax.annotate(
            f"n={and}\n$n_g$={ng}",
            xy=(d, -0.08),
            ha="center",
            fontsize=8,
            color="gray",
        )

    ax.set_xlabel("Tree depth", fontsize=12)
    ax.set_ylabel("Power", fontsize=12)
    ax.set_title(f"Power Degradation with Depth ($n={n:,}$, $p={p}$)", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(-0.5, max_depth + 0.5)
    ax.set_ylim(-0.15, 1.05)
    ax.set_xticks(depths)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_validation(
    summary_path: Path,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Plot empirical vs theoretical power comparison.

    Requires running muting_power_gap.py first to generate summary data.
    """
    df = pd.read_parquet(summary_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Root power comparison
    ax1 = axes[0]
    ax1.scatter(
        df["root_power_theory"],
        df["root_reject_rate"],
        c=[COLORS["gap"] if g else COLORS["root"] for g in df["in_gap_region"]],
        s=80,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.5,
    )
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect agreement")
    ax1.set_xlabel("Theoretical power", fontsize=12)
    ax1.set_ylabel("Empirical rejection rate", fontsize=12)
    ax1.set_title("Root Power Validation", fontsize=14)
    ax1.legend(loc="lower right")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")

    # Gate power comparison
    ax2 = axes[1]
    ax2.scatter(
        df["gate_power_theory"],
        df["gate_reject_rate"],
        c=[COLORS["gap"] if g else COLORS["gate"] for g in df["in_gap_region"]],
        s=80,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.5,
    )
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect agreement")
    ax2.set_xlabel("Theoretical power", fontsize=12)
    ax2.set_ylabel("Empirical rejection rate", fontsize=12)
    ax2.set_title("Gate Power Validation", fontsize=14)
    ax2.legend(loc="lower right")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect("equal")

    # Add legend for gap region
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=COLORS["root"], label="Standard config"),
        Patch(facecolor=COLORS["gap"], label="In gap region"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=10,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate feature muting power figures")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/results/figures"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2000,
        help="Sample size for power curve plot",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all figures
    print("Generating figures...")

    plot_power_curves(n=args.n, output_path=args.output_dir / "muting_power_curves.png")

    plot_gap_region_by_n(output_path=args.output_dir / "muting_gap_region.png")

    plot_depth_propagation(n=args.n, output_path=args.output_dir / "muting_depth_propagation.png")

    # Validation plot (requires simulation data)
    summary_path = Path("paper/results/theory/muting_power_gap_summary.parquet")
    if summary_path.exists():
        plot_validation(summary_path, output_path=args.output_dir / "muting_power_validation.png")
    else:
        print(f"Skipping validation plot: {summary_path} not found")
        print("  Run muting_power_gap.py first to generate simulation data")

    print("\nDone! Figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()
