"""Build compact paper-facing summary tables for the presentation package."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Final

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.scripts.analysis.benchmark_common import TABLES_DIR

BENCHMARK_AGG_PATH: Final[Path] = TABLES_DIR / "paper_benchmark_method_aggregate.csv"
BENCHMARK_MEMBERSHIP_PATH: Final[Path] = TABLES_DIR / "paper_benchmark_complete_case_membership.csv"
BENCHMARK_SPREAD_PATH: Final[Path] = TABLES_DIR / "paper_benchmark_spread.csv"
HETEROGENEITY_PATH: Final[Path] = TABLES_DIR / "paper_heterogeneity_method_summary.csv"
PAIRWISE_BREADTH_PATH: Final[Path] = TABLES_DIR / "paper_heterogeneity_cif_pairwise_breadth.csv"
MIRRORED_ABLATION_PATH: Final[Path] = TABLES_DIR / "paper_mirrored_knob_ablation_summary.csv"
THRESHOLD_ABLATION_PATH: Final[Path] = TABLES_DIR / "paper_threshold_ablation_summary.csv"

BENCHMARK_PAIR_BASELINES: Final[tuple[str, ...]] = ("r_ctree", "r_cforest", "cit")
PRACTICAL_VARIANTS: Final[dict[str, tuple[str, str]]] = {
    "cif_no_adaptive": ("mirrored", "cif_default"),
    "cif_no_bonferroni": ("mirrored", "cif_default"),
    "histogram_32": ("threshold", "histogram_256"),
    "exact_all": ("threshold", "histogram_256"),
}


def _complete_case_dataset_count(membership: pd.DataFrame, task: str, k: int) -> int:
    """Count datasets complete across all downstream models at one budget."""
    sub = membership[(membership["task"] == task) & (membership["k"] == k)].copy()
    grouped = sub.groupby("dataset", as_index=False)["is_complete_case"].all()
    return int(grouped["is_complete_case"].sum())


def _support_counts_by_k(membership: pd.DataFrame, task: str) -> tuple[int, int]:
    """Return dataset support at the smallest and largest standard budgets."""
    return _complete_case_dataset_count(membership, task, 5), _complete_case_dataset_count(membership, task, 100)


def build_benchmark_presentation_summary(
    aggregate: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    pairwise_breadth: pd.DataFrame,
    membership: pd.DataFrame,
    spread: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact CIF-centric benchmark summary for main-text use."""
    rows: list[dict[str, object]] = []

    for task in ("classification", "regression"):
        agg_task = aggregate[aggregate["task"] == task].sort_values("mean_rank")
        het_task = heterogeneity[(heterogeneity["task"] == task) & (heterogeneity["method_base"] == "cif")].iloc[0]
        cif_agg = agg_task[agg_task["method_base"] == "cif"].iloc[0]
        best_agg = agg_task.iloc[0]
        support_k5, support_k100 = _support_counts_by_k(membership, task)
        spread_task = spread[(spread["task"] == task) & (spread["comparison_scope"] == "standard_k")]
        range_k5 = float(spread_task[spread_task["k"] == 5]["mean_range"].iloc[0])
        range_k100 = float(spread_task[spread_task["k"] == 100]["mean_range"].iloc[0])

        row: dict[str, object] = {
            "task": task,
            "cif_rank_position": int(cif_agg["rank_position"]),
            "cif_mean_rank": float(cif_agg["mean_rank"]),
            "cif_mean_score": float(cif_agg["mean_score"]),
            "best_method_base": str(best_agg["method_base"]),
            "best_method_mean_rank": float(best_agg["mean_rank"]),
            "n_complete_case_datasets_k5": support_k5,
            "n_complete_case_datasets_k100": support_k100,
            "cif_top1_datasets": int(round(float(het_task["top1_share"]) * int(het_task["n_datasets"]))),
            "cif_top3_datasets": int(round(float(het_task["top3_share"]) * int(het_task["n_datasets"]))),
            "cif_top_half_datasets": int(round(float(het_task["top_half_share"]) * int(het_task["n_datasets"]))),
            "mean_cross_method_range_k5": range_k5,
            "mean_cross_method_range_k100": range_k100,
        }

        breadth_task = pairwise_breadth[pairwise_breadth["task"] == task]
        for baseline in BENCHMARK_PAIR_BASELINES:
            baseline_row = breadth_task[breadth_task["baseline"] == baseline].iloc[0]
            row[f"cif_positive_vs_{baseline}_datasets"] = int(baseline_row["wins"])
            row[f"cif_mean_delta_vs_{baseline}"] = float(baseline_row["mean_delta"])

        rows.append(row)

    return pd.DataFrame(rows)


def build_practical_controls_presentation_summary(
    mirrored: pd.DataFrame,
    threshold: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact cross-context table for the practical-control story."""
    rows: list[dict[str, object]] = []

    mirrored_rows = mirrored[
        mirrored["variant"].isin(
            {
                "cif_no_adaptive",
                "cif_no_scan",
                "cif_no_threshold_scan",
                "cif_no_bonferroni",
            }
        )
    ].copy()
    for _, row in mirrored_rows.iterrows():
        rows.append(
            {
                "family": "mirrored",
                "reference_variant": "cif_default",
                "task": row["task"],
                "dataset_group": row["dataset_group"],
                "variant": row["variant"],
                "runtime_ratio_vs_default": float(row["elapsed_seconds_ratio"]),
                "downstream_delta_vs_default": float(row["delta_downstream_score"]),
                "curve_recovery_delta_vs_default": float(row["delta_precision_over_standard_k"])
                if pd.notna(row["delta_precision_over_standard_k"])
                else pd.NA,
                "depth_delta_vs_default": float(row["delta_max_depth"]),
                "features_used_delta_vs_default": float(row["delta_features_used"]),
            }
        )

    threshold_rows = threshold[threshold["variant"].isin({"histogram_32", "exact_all"})].copy()
    for _, row in threshold_rows.iterrows():
        rows.append(
            {
                "family": "threshold",
                "reference_variant": "histogram_256",
                "task": row["task"],
                "dataset_group": row["dataset_group"],
                "variant": row["variant"],
                "runtime_ratio_vs_default": float(row["elapsed_seconds_ratio_vs_default"]),
                "downstream_delta_vs_default": float(row["delta_real_downstream_vs_default"]),
                "curve_recovery_delta_vs_default": float(row["delta_precision_over_standard_k_vs_default"])
                if pd.notna(row["delta_precision_over_standard_k_vs_default"])
                else pd.NA,
                "depth_delta_vs_default": float(row["delta_depth_vs_default"]),
                "features_used_delta_vs_default": float(row["delta_features_used_vs_default"]),
            }
        )

    order = {"classification": 0, "regression": 1, "real": 0, "synthetic": 1}
    out = pd.DataFrame(rows)
    return out.sort_values(
        by=["family", "variant", "task", "dataset_group"],
        key=lambda col: col.map(order).fillna(col),
    ).reset_index(drop=True)


def main() -> None:
    """Build and save compact presentation-summary tables."""
    parser = argparse.ArgumentParser(description="Build compact presentation summary tables")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TABLES_DIR,
        help="Directory for presentation-summary outputs",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_summary = build_benchmark_presentation_summary(
        pd.read_csv(BENCHMARK_AGG_PATH),
        pd.read_csv(HETEROGENEITY_PATH),
        pd.read_csv(PAIRWISE_BREADTH_PATH),
        pd.read_csv(BENCHMARK_MEMBERSHIP_PATH),
        pd.read_csv(BENCHMARK_SPREAD_PATH),
    )
    benchmark_summary.to_csv(output_dir / "paper_presentation_benchmark_summary.csv", index=False)

    practical_summary = build_practical_controls_presentation_summary(
        pd.read_csv(MIRRORED_ABLATION_PATH),
        pd.read_csv(THRESHOLD_ABLATION_PATH),
    )
    practical_summary.to_csv(output_dir / "paper_presentation_practical_controls_summary.csv", index=False)


if __name__ == "__main__":
    main()
