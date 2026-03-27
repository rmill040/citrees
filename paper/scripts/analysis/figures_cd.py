"""Generate publication-quality Critical Difference diagrams.

Reads pre-computed ranking CSVs (method, avg_rank, std_rank, cd) and produces
clean, modern CD diagrams following the Demsar (2006) / Nemenyi layout.

Usage:
    uv run python paper/scripts/analysis/figures_cd.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Display-name mapping
# ---------------------------------------------------------------------------
DISPLAY_NAMES: dict[str, str] = {
    "cif": "CIF",
    "cit": "CIT",
    "rf": "RF",
    "et": "ExtraTrees",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "cat": "CatBoost",
    "rfe": "RFE",
    "boruta": "Boruta",
    "ptest_mc": "MC-ptest",
    "ptest_rdc": "RDC-ptest",
    "ptest_pc": "PC-ptest",
    "ptest_dc": "DC-ptest",
    "r_ctree": "R-ctree",
    "r_cforest": "R-cforest",
    "cpi": "CPI",
    "pi": "PI",
}

# ---------------------------------------------------------------------------
# Color mapping
# ---------------------------------------------------------------------------
COLOR_CIF = "#2563EB"       # Blue  -- our method (forest)
COLOR_CIT = "#60A5FA"       # Light blue -- our method (tree)
COLOR_R = "#F97316"         # Orange -- R baselines
COLOR_DEFAULT = "#374151"   # Dark gray -- everything else


def _color_for(method: str) -> str:
    if method == "cif":
        return COLOR_CIF
    if method == "cit":
        return COLOR_CIT
    if method in ("r_ctree", "r_cforest"):
        return COLOR_R
    return COLOR_DEFAULT


def _is_bold(method: str) -> bool:
    return method in ("cif", "cit")


# ---------------------------------------------------------------------------
# Nemenyi clique detection
# ---------------------------------------------------------------------------

def _find_cliques(avg_ranks: list[float], cd: float) -> list[tuple[float, float]]:
    """Return maximal non-significance cliques as (min_rank, max_rank) intervals.

    Two methods are *not* significantly different when |rank_i - rank_j| <= cd.
    We enumerate all maximal contiguous windows (sorted by rank) where the
    difference between the best and worst member is within cd.
    """
    sorted_ranks = sorted(avg_ranks)
    n = len(sorted_ranks)

    raw: list[tuple[float, float]] = []
    for i in range(n):
        j = i
        while j < n and sorted_ranks[j] - sorted_ranks[i] <= cd:
            j += 1
        j -= 1
        if j > i:
            raw.append((sorted_ranks[i], sorted_ranks[j]))

    # Remove subsumed intervals -- keep only maximal ones
    cliques: list[tuple[float, float]] = []
    for lo, hi in raw:
        if not any(
            elo <= lo and ehi >= hi and (elo, ehi) != (lo, hi)
            for elo, ehi in raw
        ):
            if (lo, hi) not in cliques:
                cliques.append((lo, hi))

    return cliques


# ---------------------------------------------------------------------------
# Core drawing
# ---------------------------------------------------------------------------

@dataclass
class CDDiagramSpec:
    """Specification for one CD diagram."""

    csv_path: Path
    output_path: Path
    title: str


def draw_cd_diagram(
    ranks_df: pd.DataFrame,
    cd: float,
    title: str,
    output_path: Path,
    fig_width: float = 7.0,
) -> None:
    """Draw a clean, modern Critical Difference diagram (Demsar-style).

    Layout:
        - Horizontal number line at the top (1 .. k).
        - Left-half methods (lower/better rank): short tick down from axis,
          horizontal line to the left margin, label at the margin.
        - Right-half methods: same, but to the right margin.
        - Connectors never cross because labels and ranks are both sorted.
        - Clique bars (non-significant groups) are drawn just below the axis.
        - A CD reference bar is drawn above the axis.
    """
    methods = ranks_df["method"].tolist()
    avg_ranks = ranks_df["avg_rank"].tolist()
    n_methods = len(methods)

    # ---- Split into left / right halves ----
    n_left = n_methods // 2
    n_right = n_methods - n_left
    n_rows = max(n_left, n_right)

    # ---- Layout geometry (inches) ----
    label_row_h = 0.26           # vertical space per label row
    axis_block_h = 0.55          # axis + CD bar + clique bars
    top_pad = 0.42
    bot_pad = 0.12
    label_block_h = n_rows * label_row_h

    fig_h = top_pad + axis_block_h + label_block_h + bot_pad
    fig = plt.figure(figsize=(fig_width, fig_h))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ---- x mapping: rank -> figure fraction ----
    margin_l = 0.22
    margin_r = 0.22

    def rx(rank: float) -> float:
        return margin_l + (rank - 1.0) / max(n_methods - 1.0, 1.0) * (1.0 - margin_l - margin_r)

    # ---- y positions (figure fraction, top-down) ----
    y_title = 1.0 - top_pad / fig_h * 0.40
    y_cd = 1.0 - top_pad / fig_h
    y_axis = y_cd - 0.06
    y_clique_top = y_axis - 0.035
    row_dy = label_row_h / fig_h

    # Compute lowest clique bar y so we know where labels can start
    cliques = _find_cliques(avg_ranks, cd)
    cliques.sort(key=lambda c: -(c[1] - c[0]))   # widest first
    clique_dy = 0.020
    clique_positions: list[tuple[float, float, float]] = []   # (xlo, xhi, y)
    for lo, hi in cliques:
        xlo, xhi = rx(lo), rx(hi)
        y_try = y_clique_top
        for plo, phi, py in clique_positions:
            if not (xhi < plo - 0.005 or xlo > phi + 0.005):
                y_try = min(y_try, py - clique_dy)
        clique_positions.append((xlo, xhi, y_try))

    y_lowest_clique = min((y for _, _, y in clique_positions), default=y_clique_top)
    y_labels_start = y_lowest_clique - 0.045

    # ---- Title ----
    ax.text(
        0.5, y_title, title,
        ha="center", va="center", fontsize=10, fontweight="bold",
        transform=ax.transAxes,
    )

    # ---- Rank axis ----
    ax.plot(
        [rx(1), rx(n_methods)], [y_axis, y_axis],
        color="black", linewidth=0.9, solid_capstyle="butt",
        transform=ax.transAxes, clip_on=False,
    )
    tick_step = 1 if n_methods <= 20 else 2
    for t in range(1, n_methods + 1, tick_step):
        xt = rx(t)
        ax.plot(
            [xt, xt], [y_axis - 0.007, y_axis + 0.007],
            color="black", linewidth=0.7,
            transform=ax.transAxes, clip_on=False,
        )
        ax.text(
            xt, y_axis + 0.018, str(t),
            ha="center", va="bottom", fontsize=6.5, color="#999999",
            transform=ax.transAxes,
        )

    # ---- CD bar ----
    cd_x0 = rx(1)
    cd_x1 = rx(1 + cd)
    ax.plot(
        [cd_x0, cd_x1], [y_cd, y_cd],
        color="#DC2626", linewidth=2.2, solid_capstyle="butt",
        transform=ax.transAxes, clip_on=False,
    )
    for cx in (cd_x0, cd_x1):
        ax.plot(
            [cx, cx], [y_cd - 0.007, y_cd + 0.007],
            color="#DC2626", linewidth=1.4,
            transform=ax.transAxes, clip_on=False,
        )
    ax.text(
        (cd_x0 + cd_x1) / 2, y_cd + 0.016,
        f"CD = {cd:.2f}",
        ha="center", va="bottom", fontsize=7.5,
        color="#DC2626", fontweight="bold",
        transform=ax.transAxes,
    )

    # ---- Clique bars ----
    for xlo, xhi, y_bar in clique_positions:
        ax.plot(
            [xlo, xhi], [y_bar, y_bar],
            color="#9CA3AF", linewidth=2.4, solid_capstyle="round",
            alpha=0.6, transform=ax.transAxes, clip_on=False,
        )

    # ---- Dots on axis ----
    for i in range(n_methods):
        color = _color_for(methods[i])
        ax.plot(
            rx(avg_ranks[i]), y_axis, "o",
            color=color, markersize=3.5, zorder=5,
            transform=ax.transAxes, clip_on=False,
        )

    # ---- Labels with elbow connectors ----
    #
    # Left side: label rows go top-to-bottom in rank order (best = top).
    #   Connector: short vertical tick down from axis dot, then horizontal
    #   line running LEFT to the label margin.
    #
    # Right side: same, but horizontal line runs RIGHT.
    #
    # Because both label rows and rank positions are monotonically ordered
    # (on each side), the vertical segments never cross.

    label_fs = 8.5
    line_alpha = 0.35
    line_lw = 0.65

    def _draw_side(indices: list[int], side: str) -> None:
        n = len(indices)
        if n == 0:
            return
        for row, mi in enumerate(indices):
            method = methods[mi]
            rank = avg_ranks[mi]
            disp = DISPLAY_NAMES.get(method, method)
            color = _color_for(method)
            bold = _is_bold(method)

            y_row = y_labels_start - row * row_dy
            x_rank = rx(rank)

            if side == "left":
                x_text = 0.005
                ha = "left"
                label = f"{disp}  ({rank:.1f})"
                x_hook = 0.012
            else:
                x_text = 0.995
                ha = "right"
                label = f"({rank:.1f})  {disp}"
                x_hook = 0.988

            # Vertical drop: axis -> label y
            ax.plot(
                [x_rank, x_rank], [y_axis, y_row],
                color=color, linewidth=line_lw, alpha=line_alpha,
                transform=ax.transAxes, clip_on=False,
            )
            # Horizontal run: rank x -> label margin
            ax.plot(
                [x_rank, x_hook], [y_row, y_row],
                color=color, linewidth=line_lw, alpha=line_alpha,
                transform=ax.transAxes, clip_on=False,
            )
            # Label
            ax.text(
                x_text, y_row, label,
                ha=ha, va="center", fontsize=label_fs,
                color=color, fontweight="bold" if bold else "normal",
                transform=ax.transAxes,
            )

    _draw_side(list(range(n_left)), "left")
    _draw_side(list(range(n_left, n_methods)), "right")

    # ---- Save ----
    fig.savefig(
        output_path, dpi=300, bbox_inches="tight",
        facecolor="white", edgecolor="none", pad_inches=0.10,
    )
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    tables_dir = Path("paper/results/tables")
    figures_dir = Path("paper/results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    specs: list[CDDiagramSpec] = [
        CDDiagramSpec(
            csv_path=tables_dir / "clf_ranking_balanced_accuracy.csv",
            output_path=figures_dir / "clf_cd_balanced_accuracy_clean.png",
            title="Classification: Balanced Accuracy (mean rank)",
        ),
        CDDiagramSpec(
            csv_path=tables_dir / "reg_ranking_r2.csv",
            output_path=figures_dir / "reg_cd_r2_clean.png",
            title=u"Regression: R\u00b2 (mean rank)",
        ),
        CDDiagramSpec(
            csv_path=tables_dir / "synthetic_ranking_precision_at_10.csv",
            output_path=figures_dir / "synthetic_cd_precision_at_10_clean.png",
            title="Synthetic: Precision@10 (mean rank)",
        ),
    ]

    for spec in specs:
        if not spec.csv_path.exists():
            print(f"SKIP (not found): {spec.csv_path}")
            continue

        df = pd.read_csv(spec.csv_path)
        if df.empty:
            print(f"SKIP (empty): {spec.csv_path}")
            continue

        cd = df["cd"].iloc[0]
        print(f"\n--- {spec.title} ---")
        print(f"  Methods: {len(df)},  CD: {cd:.3f}")
        for _, row in df.iterrows():
            display = DISPLAY_NAMES.get(row["method"], row["method"])
            print(f"    {display:15s}  rank={row['avg_rank']:.2f}")

        draw_cd_diagram(
            ranks_df=df,
            cd=cd,
            title=spec.title,
            output_path=spec.output_path,
        )


if __name__ == "__main__":
    main()
