"""Build summary tables for the CIT-only runtime ablation.

The EC2 run writes one raw CSV per dataset shard. This script combines those
shards into a single traceable raw table, recomputes paired runtime ratios
against `cit_default`, and writes compact tables for paper-facing analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.scripts.analysis.benchmark_common import TABLES_DIR  # noqa: E402

DEFAULT_VARIANT: Final[str] = "cit_default"
CIT_VARIANTS: Final[tuple[str, ...]] = (
    "cit_default",
    "cit_no_adaptive",
    "cit_no_feature_scan",
    "cit_no_threshold_scan",
    "cit_no_feature_mute",
    "cit_exact_thresholds",
    "cit_no_bonferroni",
)
TASK_LABELS: Final[dict[str, str]] = {
    "clf": "classification",
    "reg": "regression",
}
PAIR_KEYS: Final[list[str]] = ["task", "dataset_source", "dataset_type", "seed"]
RAW_OUTPUT: Final[str] = "cit_runtime_ablation_raw.csv"
DATASET_OUTPUT: Final[str] = "cit_runtime_ablation_dataset_summary.csv"
PAPER_OUTPUT: Final[str] = "paper_cit_runtime_ablation_summary.csv"


def _quantile(percentile: float):
    """Return a named quantile aggregation function."""

    def quantile(series: pd.Series) -> float:
        return float(series.quantile(percentile))

    quantile.__name__ = f"p{int(percentile * 100)}"
    return quantile


def load_raw_shards(input_dir: Path) -> pd.DataFrame:
    """Load and combine all CIT runtime raw shard outputs."""
    raw_files = sorted(input_dir.rglob("*_raw.csv"))
    if not raw_files:
        msg = f"No raw runtime-ablation CSVs found under {input_dir}"
        raise FileNotFoundError(msg)

    frames: list[pd.DataFrame] = []
    for path in raw_files:
        frame = pd.read_csv(path)
        frame["source_file"] = str(path.relative_to(input_dir))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def add_paired_runtime_ratios(raw: pd.DataFrame) -> pd.DataFrame:
    """Add elapsed-time ratios against the same dataset/seed `cit_default` row."""
    df = raw.copy()
    defaults = (
        df[df["variant"] == DEFAULT_VARIANT][PAIR_KEYS + ["elapsed_seconds"]]
        .rename(columns={"elapsed_seconds": "default_elapsed_seconds"})
        .drop_duplicates(PAIR_KEYS)
    )
    df = df.merge(defaults, on=PAIR_KEYS, how="left", validate="many_to_one")
    df["elapsed_ratio_vs_default"] = df["elapsed_seconds"] / df["default_elapsed_seconds"]
    df["elapsed_speedup_vs_default"] = df["default_elapsed_seconds"] / df["elapsed_seconds"]
    return df


def validate_raw(raw: pd.DataFrame) -> None:
    """Fail fast on common ingestion mistakes."""
    missing_columns = {
        "task",
        "dataset_source",
        "dataset_type",
        "seed",
        "variant",
        "method_family",
        "fit_success",
        "elapsed_seconds",
    } - set(raw.columns)
    if missing_columns:
        msg = f"Raw table is missing required columns: {sorted(missing_columns)}"
        raise ValueError(msg)

    method_families = set(raw["method_family"].dropna().unique())
    if method_families != {"cit"}:
        msg = f"Expected only CIT rows, found method_family={sorted(method_families)}"
        raise ValueError(msg)

    variants = set(raw["variant"].dropna().unique())
    unexpected = variants - set(CIT_VARIANTS)
    missing = set(CIT_VARIANTS) - variants
    if unexpected or missing:
        msg = f"Unexpected CIT variant inventory. unexpected={sorted(unexpected)} missing={sorted(missing)}"
        raise ValueError(msg)

    failed = raw[~raw["fit_success"].astype(bool)]
    if not failed.empty:
        msg = f"Found {len(failed)} failed fit rows in the runtime ablation"
        raise ValueError(msg)

    per_dataset_counts = raw.groupby("dataset_type")["variant"].size()
    if per_dataset_counts.nunique() != 1:
        msg = "Dataset shards have inconsistent row counts"
        raise ValueError(msg)

    expected_per_dataset = raw["seed"].nunique() * len(CIT_VARIANTS)
    if int(per_dataset_counts.iloc[0]) != expected_per_dataset:
        msg = f"Expected {expected_per_dataset} rows per dataset, found {int(per_dataset_counts.iloc[0])}"
        raise ValueError(msg)


def build_dataset_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize the ablation within each dataset and variant."""
    grouped = raw.groupby(
        ["task", "dataset_source", "dataset_type", "variant"], as_index=False, dropna=False
    )
    summary = grouped.agg(
        n_runs=("elapsed_seconds", "size"),
        n_success=("fit_success", "sum"),
        mean_elapsed_seconds=("elapsed_seconds", "mean"),
        median_elapsed_seconds=("elapsed_seconds", "median"),
        p25_elapsed_seconds=("elapsed_seconds", _quantile(0.25)),
        p75_elapsed_seconds=("elapsed_seconds", _quantile(0.75)),
        mean_elapsed_ratio_vs_default=("elapsed_ratio_vs_default", "mean"),
        median_elapsed_ratio_vs_default=("elapsed_ratio_vs_default", "median"),
        mean_elapsed_speedup_vs_default=("elapsed_speedup_vs_default", "mean"),
        median_elapsed_speedup_vs_default=("elapsed_speedup_vs_default", "median"),
        mean_depth=("mean_depth", "mean"),
        mean_max_depth=("max_depth", "mean"),
        mean_features_used=("mean_features_used_per_tree", "mean"),
        mean_importance_nonzero_count=("importance_nonzero_count", "mean"),
        mean_precision_at_10=("precision_at_10", "mean"),
        mean_recall_at_10=("recall_at_10", "mean"),
        mean_f1_at_10=("f1_at_10", "mean"),
        changes_statistical_rule=("changes_statistical_rule", "max"),
    )
    return summary.sort_values(["task", "dataset_source", "dataset_type", "variant"]).reset_index(
        drop=True
    )


def build_paper_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate dataset/seed-paired runtime evidence by task, source, and variant."""
    grouped = raw.groupby(["task", "dataset_source", "variant"], as_index=False, dropna=False)
    summary = grouped.agg(
        n_dataset_types=("dataset_type", "nunique"),
        n_runs=("elapsed_seconds", "size"),
        n_success=("fit_success", "sum"),
        runtime_ratio_vs_default_mean=("elapsed_ratio_vs_default", "mean"),
        runtime_ratio_vs_default_median=("elapsed_ratio_vs_default", "median"),
        runtime_ratio_vs_default_p25=("elapsed_ratio_vs_default", _quantile(0.25)),
        runtime_ratio_vs_default_p75=("elapsed_ratio_vs_default", _quantile(0.75)),
        runtime_speedup_vs_default_mean=("elapsed_speedup_vs_default", "mean"),
        runtime_speedup_vs_default_median=("elapsed_speedup_vs_default", "median"),
        n_pairs_faster_5pct=(
            "elapsed_ratio_vs_default",
            lambda series: int((series <= 0.95).sum()),
        ),
        n_pairs_near_default=(
            "elapsed_ratio_vs_default",
            lambda series: int(((series > 0.95) & (series < 1.05)).sum()),
        ),
        n_pairs_slower_5pct=(
            "elapsed_ratio_vs_default",
            lambda series: int((series >= 1.05).sum()),
        ),
        mean_depth=("mean_depth", "mean"),
        mean_max_depth=("max_depth", "mean"),
        mean_features_used=("mean_features_used_per_tree", "mean"),
        mean_importance_nonzero_count=("importance_nonzero_count", "mean"),
        mean_precision_at_10=("precision_at_10", "mean"),
        mean_recall_at_10=("recall_at_10", "mean"),
        mean_f1_at_10=("f1_at_10", "mean"),
        changes_statistical_rule=("changes_statistical_rule", "max"),
    )

    defaults = summary[summary["variant"] == DEFAULT_VARIANT][
        [
            "task",
            "dataset_source",
            "mean_depth",
            "mean_features_used",
            "mean_importance_nonzero_count",
            "mean_precision_at_10",
            "mean_recall_at_10",
            "mean_f1_at_10",
        ]
    ].rename(
        columns={
            "mean_depth": "default_mean_depth",
            "mean_features_used": "default_mean_features_used",
            "mean_importance_nonzero_count": "default_mean_importance_nonzero_count",
            "mean_precision_at_10": "default_mean_precision_at_10",
            "mean_recall_at_10": "default_mean_recall_at_10",
            "mean_f1_at_10": "default_mean_f1_at_10",
        }
    )
    summary = summary.merge(
        defaults, on=["task", "dataset_source"], how="left", validate="many_to_one"
    )
    summary["runtime_ratio_scope"] = "paired_seed_dataset_vs_cit_default"
    summary["delta_mean_depth"] = summary["mean_depth"] - summary["default_mean_depth"]
    summary["delta_mean_features_used"] = (
        summary["mean_features_used"] - summary["default_mean_features_used"]
    )
    summary["delta_mean_importance_nonzero_count"] = (
        summary["mean_importance_nonzero_count"] - summary["default_mean_importance_nonzero_count"]
    )
    summary["delta_mean_precision_at_10"] = (
        summary["mean_precision_at_10"] - summary["default_mean_precision_at_10"]
    )
    summary["delta_mean_recall_at_10"] = (
        summary["mean_recall_at_10"] - summary["default_mean_recall_at_10"]
    )
    summary["delta_mean_f1_at_10"] = summary["mean_f1_at_10"] - summary["default_mean_f1_at_10"]
    summary["task"] = summary["task"].map(TASK_LABELS)

    variant_order = {variant: idx for idx, variant in enumerate(CIT_VARIANTS)}
    summary["_variant_order"] = summary["variant"].map(variant_order)
    summary = (
        summary.sort_values(["task", "dataset_source", "_variant_order"])
        .drop(columns="_variant_order")
        .reset_index(drop=True)
    )

    column_order = [
        "task",
        "dataset_source",
        "variant",
        "runtime_ratio_scope",
        "n_dataset_types",
        "n_runs",
        "n_success",
        "runtime_ratio_vs_default_median",
        "runtime_ratio_vs_default_p25",
        "runtime_ratio_vs_default_p75",
        "runtime_ratio_vs_default_mean",
        "runtime_speedup_vs_default_median",
        "runtime_speedup_vs_default_mean",
        "n_pairs_faster_5pct",
        "n_pairs_near_default",
        "n_pairs_slower_5pct",
        "mean_depth",
        "mean_max_depth",
        "mean_features_used",
        "mean_importance_nonzero_count",
        "mean_precision_at_10",
        "mean_recall_at_10",
        "mean_f1_at_10",
        "changes_statistical_rule",
        "delta_mean_depth",
        "delta_mean_features_used",
        "delta_mean_importance_nonzero_count",
        "delta_mean_precision_at_10",
        "delta_mean_recall_at_10",
        "delta_mean_f1_at_10",
    ]
    return summary[column_order]


def main() -> None:
    """Build and save combined CIT runtime-ablation tables."""
    parser = argparse.ArgumentParser(description="Build CIT runtime-ablation summary tables")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "scratch" / "cit_runtime_ablation_s3",
        help="Directory containing downloaded EC2 shard outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TABLES_DIR,
        help="Directory for combined and summary outputs",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_shards(args.input_dir.resolve())
    validate_raw(raw)
    raw = add_paired_runtime_ratios(raw)
    raw.to_csv(output_dir / RAW_OUTPUT, index=False)

    dataset_summary = build_dataset_summary(raw)
    dataset_summary.to_csv(output_dir / DATASET_OUTPUT, index=False)

    paper_summary = build_paper_summary(raw)
    paper_summary.to_csv(output_dir / PAPER_OUTPUT, index=False)

    print(f"Wrote {output_dir / RAW_OUTPUT} ({len(raw)} rows)")
    print(f"Wrote {output_dir / DATASET_OUTPUT} ({len(dataset_summary)} rows)")
    print(f"Wrote {output_dir / PAPER_OUTPUT} ({len(paper_summary)} rows)")


if __name__ == "__main__":
    main()
