"""Build paper-facing high-p endpoint tables.

This script turns the existing stage-2 endpoint evaluations into a clean,
paper-facing analysis layer for datasets whose endpoint exceeds the standard
headline budget (`k > 100`).

Outputs:
  - paper/results/tables/paper_high_p_endpoint_inventory.csv
  - paper/results/tables/paper_high_p_endpoint_method_presence.csv
  - paper/results/tables/paper_high_p_delta_vs_endpoint_cells.csv
  - paper/results/tables/paper_high_p_delta_vs_endpoint_method.csv
  - paper/results/tables/paper_high_p_delta_vs_endpoint_overall.csv
  - paper/results/tables/paper_high_p_endpoint_aggregate.csv
  - paper/results/tables/paper_high_p_cif_endpoint_summary.csv
  - paper/results/tables/paper_high_p_cif_endpoint_examples.csv
  - paper/results/tables/paper_high_p_cif_best_observed_k_summary.csv
  - paper/results/tables/paper_high_p_cif_best_observed_k_examples.csv
  - paper/results/tables/paper_high_p_endpoint_pairwise.csv
  - paper/results/tables/paper_high_p_endpoint_spread.csv

Usage:
  UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_high_p_saturation_tables.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Final

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.scripts.analysis.benchmark_common import (
    STANDARD_K,
    TABLES_DIR,
    TASK_CONFIG,
    dataset_scores,
    load_real_task_frame,
    select_best_task_configs,
)

HIGH_P_MIN_ENDPOINT: Final[int] = max(STANDARD_K)
SCORE_TOL: Final[float] = 1e-12


def build_task_scores(task: str, path: Path, metric: str) -> pd.DataFrame:
    """Load one task and keep the task-wide best config for each method family."""
    raw = load_real_task_frame(path)
    standard_raw = raw[raw["k"].isin(STANDARD_K)].copy()
    _, best_configs = select_best_task_configs(standard_raw, metric)
    best_full = raw.merge(best_configs[["method_base", "method_id"]], on=["method_base", "method_id"], how="inner")
    scores = dataset_scores(best_full, metric)
    endpoint = scores.groupby("dataset", as_index=False)["k"].max().rename(columns={"k": "endpoint_k"})
    scores = scores.merge(endpoint, on="dataset", how="inner")
    scores.insert(0, "task", task)
    return scores


def filter_high_p(scores: pd.DataFrame) -> pd.DataFrame:
    """Keep datasets whose endpoint budget exceeds the standard headline range."""
    return scores[scores["endpoint_k"] > HIGH_P_MIN_ENDPOINT].copy()


def endpoint_rows(scores: pd.DataFrame) -> pd.DataFrame:
    """Keep only endpoint rows (`k = endpoint_k`)."""
    return scores[scores["k"] == scores["endpoint_k"]].copy()


def k100_rows(scores: pd.DataFrame) -> pd.DataFrame:
    """Keep the `k=100` rows for high-p datasets."""
    return scores[scores["k"] == HIGH_P_MIN_ENDPOINT].copy()


def complete_case_endpoint(scores: pd.DataFrame) -> pd.DataFrame:
    """Keep endpoint dataset/downstream cells where every method family is present."""
    n_methods = scores["method_base"].nunique()
    ep = endpoint_rows(scores)
    counts = ep.groupby(["dataset", "downstream_model"], as_index=False)["method_base"].nunique()
    complete = counts[counts["method_base"] == n_methods][["dataset", "downstream_model"]]
    return ep.merge(complete, on=["dataset", "downstream_model"], how="inner")


def complete_case_k100(scores: pd.DataFrame) -> pd.DataFrame:
    """Keep `k=100` dataset/downstream cells where every method family is present."""
    n_methods = scores["method_base"].nunique()
    base = k100_rows(scores)
    counts = base.groupby(["dataset", "downstream_model"], as_index=False)["method_base"].nunique()
    complete = counts[counts["method_base"] == n_methods][["dataset", "downstream_model"]]
    return base.merge(complete, on=["dataset", "downstream_model"], how="inner")


def build_endpoint_inventory(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize endpoint completeness for each high-p dataset."""
    ep = endpoint_rows(scores)
    expected_methods = int(scores["method_base"].nunique())
    expected_downstreams = int(scores["downstream_model"].nunique())

    inventory = (
        ep.groupby(["task", "dataset"], as_index=False)
        .agg(
            endpoint_k=("endpoint_k", "first"),
            n_methods=("method_base", "nunique"),
            n_downstreams=("downstream_model", "nunique"),
            n_method_downstream_cells=("dataset_mean_score", "size"),
        )
        .sort_values(["task", "endpoint_k", "dataset"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    inventory["expected_methods"] = expected_methods
    inventory["expected_downstreams"] = expected_downstreams
    inventory["is_complete_endpoint_surface"] = (
        (inventory["n_methods"] == expected_methods)
        & (inventory["n_downstreams"] == expected_downstreams)
        & (inventory["n_method_downstream_cells"] == expected_methods * expected_downstreams)
    )
    return inventory


def build_endpoint_method_presence(scores: pd.DataFrame) -> pd.DataFrame:
    """Record method-level endpoint presence for each high-p dataset/downstream cell."""
    expected_methods = sorted(scores["method_base"].unique().tolist())
    base_cells = (
        scores[["task", "dataset", "downstream_model", "endpoint_k"]]
        .drop_duplicates()
        .sort_values(["task", "dataset", "downstream_model"])
        .reset_index(drop=True)
    )
    methods = pd.DataFrame({"method_base": expected_methods})
    grid = base_cells.merge(methods, how="cross")
    ep = endpoint_rows(scores)[["task", "dataset", "downstream_model", "method_base", "method_id"]].copy()
    ep["has_endpoint_row"] = True
    merged = grid.merge(ep, on=["task", "dataset", "downstream_model", "method_base"], how="left")
    merged["has_endpoint_row"] = merged["method_id"].notna()
    merged["missing_endpoint_row"] = ~merged["has_endpoint_row"]
    return merged.sort_values(["task", "dataset", "downstream_model", "method_base"]).reset_index(drop=True)


def build_delta_vs_endpoint_cells(scores: pd.DataFrame) -> pd.DataFrame:
    """Expose the full high-p cell-level surface relative to each method's endpoint."""
    ep = endpoint_rows(scores)[
        ["task", "dataset", "downstream_model", "method_base", "method_id", "dataset_mean_score"]
    ].rename(columns={"dataset_mean_score": "endpoint_score"})
    merged = scores.merge(ep, on=["task", "dataset", "downstream_model", "method_base", "method_id"], how="inner")
    best_observed = merged.groupby(["task", "dataset", "downstream_model", "method_base", "method_id"])[
        "dataset_mean_score"
    ].transform("max")
    merged["score_minus_endpoint"] = merged["dataset_mean_score"] - merged["endpoint_score"]
    merged["is_endpoint"] = merged["k"] == merged["endpoint_k"]
    merged["is_standard_k"] = merged["k"].isin(STANDARD_K)
    merged["is_best_over_observed_k"] = merged["dataset_mean_score"] >= best_observed - SCORE_TOL
    merged = merged.sort_values(
        ["task", "dataset", "downstream_model", "method_base", "k"]
    ).reset_index(drop=True)
    return merged[
        [
            "task",
            "dataset",
            "downstream_model",
            "method_base",
            "method_id",
            "k",
            "endpoint_k",
            "dataset_mean_score",
            "endpoint_score",
            "score_minus_endpoint",
            "is_standard_k",
            "is_endpoint",
            "is_best_over_observed_k",
        ]
    ]


def build_delta_vs_endpoint_method(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize how each standard budget compares to endpoint on high-p datasets."""
    ep = endpoint_rows(scores)[["task", "dataset", "downstream_model", "method_base", "method_id", "dataset_mean_score"]]
    ep = ep.rename(columns={"dataset_mean_score": "endpoint_score"})
    merged = scores.merge(ep, on=["task", "dataset", "downstream_model", "method_base", "method_id"], how="inner")
    merged["score_minus_endpoint"] = merged["dataset_mean_score"] - merged["endpoint_score"]
    merged = merged[merged["k"].isin(STANDARD_K)].copy()

    summary = (
        merged.groupby(["task", "k", "method_base", "method_id"], as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            n_cells=("score_minus_endpoint", "size"),
            mean_score_minus_endpoint=("score_minus_endpoint", "mean"),
            median_score_minus_endpoint=("score_minus_endpoint", "median"),
        )
        .sort_values(["task", "k", "method_base"])
        .reset_index(drop=True)
    )
    summary["support_type"] = "dataset_downstream_mean_over_high_p_with_endpoint"
    return summary


def build_delta_vs_endpoint_overall(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize average standard-budget performance relative to endpoint across all methods."""
    ep = endpoint_rows(scores)[["task", "dataset", "downstream_model", "method_base", "method_id", "dataset_mean_score"]]
    ep = ep.rename(columns={"dataset_mean_score": "endpoint_score"})
    merged = scores.merge(ep, on=["task", "dataset", "downstream_model", "method_base", "method_id"], how="inner")
    merged["score_minus_endpoint"] = merged["dataset_mean_score"] - merged["endpoint_score"]
    merged = merged[merged["k"].isin(STANDARD_K)].copy()

    summary = (
        merged.groupby(["task", "k"], as_index=False)
        .agg(
            n_methods=("method_base", "nunique"),
            n_datasets=("dataset", "nunique"),
            n_cells=("score_minus_endpoint", "size"),
            mean_score_minus_endpoint=("score_minus_endpoint", "mean"),
            median_score_minus_endpoint=("score_minus_endpoint", "median"),
        )
        .sort_values(["task", "k"])
        .reset_index(drop=True)
    )
    summary["support_type"] = "dataset_downstream_mean_over_high_p_with_endpoint_all_methods"
    return summary


def build_endpoint_aggregate(scores: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Aggregate complete-case endpoint ranks over high-p datasets."""
    complete = complete_case_endpoint(scores).copy()
    complete["rank"] = complete.groupby(["task", "dataset", "downstream_model"])["dataset_mean_score"].rank(
        ascending=False,
        method="average",
    )
    by_dataset = (
        complete.groupby(["task", "dataset", "method_base", "method_id"], as_index=False)
        .agg(
            n_cells=("rank", "size"),
            mean_rank=("rank", "mean"),
            mean_score=("dataset_mean_score", "mean"),
        )
    )
    aggregate = (
        by_dataset.groupby(["task", "method_base", "method_id"], as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            mean_dataset_cells=("n_cells", "mean"),
            mean_rank=("mean_rank", "mean"),
            median_rank=("mean_rank", "median"),
            mean_score=("mean_score", "mean"),
        )
        .sort_values(["task", "mean_rank", "method_base"])
        .reset_index(drop=True)
    )
    aggregate.insert(1, "metric", metric)
    aggregate["support_type"] = "dataset_mean_over_high_p_endpoint_complete_case"
    aggregate["rank_position"] = aggregate.groupby("task")["mean_rank"].rank(ascending=True, method="average")
    return aggregate


def build_cif_endpoint_summary(scores: pd.DataFrame, focus_method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize CIF score and rank movement from `k=100` to endpoint."""
    k100 = k100_rows(scores)
    ep = endpoint_rows(scores)

    focus_100 = k100[k100["method_base"] == focus_method][
        ["task", "dataset", "downstream_model", "dataset_mean_score", "endpoint_k"]
    ].rename(columns={"dataset_mean_score": "score_k100"})
    focus_ep = ep[ep["method_base"] == focus_method][
        ["task", "dataset", "downstream_model", "dataset_mean_score", "endpoint_k"]
    ].rename(columns={"dataset_mean_score": "score_endpoint"})
    examples = focus_100.merge(focus_ep, on=["task", "dataset", "downstream_model", "endpoint_k"], how="inner")
    examples["score_endpoint_minus_k100"] = examples["score_endpoint"] - examples["score_k100"]

    complete_100 = complete_case_k100(scores).copy()
    complete_ep = complete_case_endpoint(scores).copy()
    complete_100["rank_k100"] = complete_100.groupby(["task", "dataset", "downstream_model"])["dataset_mean_score"].rank(
        ascending=False,
        method="average",
    )
    complete_ep["rank_endpoint"] = complete_ep.groupby(["task", "dataset", "downstream_model"])["dataset_mean_score"].rank(
        ascending=False,
        method="average",
    )

    rank_100 = complete_100[complete_100["method_base"] == focus_method][
        ["task", "dataset", "downstream_model", "rank_k100"]
    ]
    rank_ep = complete_ep[complete_ep["method_base"] == focus_method][
        ["task", "dataset", "downstream_model", "rank_endpoint"]
    ]
    rank_examples = rank_100.merge(rank_ep, on=["task", "dataset", "downstream_model"], how="inner")
    examples = examples.merge(rank_examples, on=["task", "dataset", "downstream_model"], how="left")
    examples["rank_change_endpoint_minus_k100"] = examples["rank_endpoint"] - examples["rank_k100"]

    focus_all = scores[scores["method_base"] == focus_method].copy()
    best_observed = focus_all.groupby(["task", "dataset", "downstream_model"])["dataset_mean_score"].transform("max")
    focus_all["is_best_over_observed_k"] = focus_all["dataset_mean_score"] >= best_observed - SCORE_TOL
    endpoint_best = focus_all[focus_all["k"] == focus_all["endpoint_k"]][
        ["task", "dataset", "downstream_model", "is_best_over_observed_k"]
    ]
    examples = examples.merge(endpoint_best, on=["task", "dataset", "downstream_model"], how="left")

    summary = (
        examples.groupby(["task", "downstream_model"], as_index=False)
        .agg(
            n_cells=("dataset", "size"),
            n_datasets=("dataset", "nunique"),
            mean_score_endpoint_minus_k100=("score_endpoint_minus_k100", "mean"),
            median_score_endpoint_minus_k100=("score_endpoint_minus_k100", "median"),
            mean_rank_change_endpoint_minus_k100=("rank_change_endpoint_minus_k100", "mean"),
            median_rank_change_endpoint_minus_k100=("rank_change_endpoint_minus_k100", "median"),
            score_improved_cells=("score_endpoint_minus_k100", lambda s: int((s > SCORE_TOL).sum())),
            score_degraded_cells=("score_endpoint_minus_k100", lambda s: int((s < -SCORE_TOL).sum())),
            rank_improved_cells=("rank_change_endpoint_minus_k100", lambda s: int((s < -SCORE_TOL).sum())),
            rank_worsened_cells=("rank_change_endpoint_minus_k100", lambda s: int((s > SCORE_TOL).sum())),
            endpoint_best_cells=("is_best_over_observed_k", "sum"),
        )
        .sort_values(["task", "downstream_model"])
        .reset_index(drop=True)
    )
    summary["endpoint_best_share"] = summary["endpoint_best_cells"] / summary["n_cells"]

    overall = (
        examples.groupby("task", as_index=False)
        .agg(
            n_cells=("dataset", "size"),
            n_datasets=("dataset", "nunique"),
            mean_score_endpoint_minus_k100=("score_endpoint_minus_k100", "mean"),
            median_score_endpoint_minus_k100=("score_endpoint_minus_k100", "median"),
            mean_rank_change_endpoint_minus_k100=("rank_change_endpoint_minus_k100", "mean"),
            median_rank_change_endpoint_minus_k100=("rank_change_endpoint_minus_k100", "median"),
            score_improved_cells=("score_endpoint_minus_k100", lambda s: int((s > SCORE_TOL).sum())),
            score_degraded_cells=("score_endpoint_minus_k100", lambda s: int((s < -SCORE_TOL).sum())),
            rank_improved_cells=("rank_change_endpoint_minus_k100", lambda s: int((s < -SCORE_TOL).sum())),
            rank_worsened_cells=("rank_change_endpoint_minus_k100", lambda s: int((s > SCORE_TOL).sum())),
            endpoint_best_cells=("is_best_over_observed_k", "sum"),
        )
        .sort_values("task")
        .reset_index(drop=True)
    )
    overall.insert(1, "downstream_model", "all")
    overall["endpoint_best_share"] = overall["endpoint_best_cells"] / overall["n_cells"]

    combined = pd.concat([overall, summary], ignore_index=True)
    return combined, examples.sort_values(
        ["task", "score_endpoint_minus_k100", "dataset", "downstream_model"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def build_cif_best_observed_k_summary(scores: pd.DataFrame, focus_method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize where the first best observed CIF budget occurs on the high-p surface."""
    focus = scores[scores["method_base"] == focus_method].copy()
    best_observed = focus.groupby(["task", "dataset", "downstream_model"])["dataset_mean_score"].transform("max")
    best_rows = focus[focus["dataset_mean_score"] >= best_observed - SCORE_TOL].copy()
    best_rows = best_rows.sort_values(["task", "dataset", "downstream_model", "k"])
    first_best = best_rows.groupby(["task", "dataset", "downstream_model"], as_index=False).first()

    first_best = first_best.rename(columns={"k": "first_best_k", "dataset_mean_score": "first_best_score"})
    first_best["best_k_fraction_of_endpoint"] = first_best["first_best_k"] / first_best["endpoint_k"]
    first_best["first_best_under_100"] = first_best["first_best_k"] < HIGH_P_MIN_ENDPOINT
    first_best["first_best_at_100"] = first_best["first_best_k"] == HIGH_P_MIN_ENDPOINT
    first_best["first_best_intermediate"] = (
        (first_best["first_best_k"] > HIGH_P_MIN_ENDPOINT) & (first_best["first_best_k"] < first_best["endpoint_k"])
    )
    first_best["first_best_at_endpoint"] = first_best["first_best_k"] == first_best["endpoint_k"]
    first_best["best_k_bucket"] = np.select(
        [
            first_best["first_best_under_100"],
            first_best["first_best_at_100"],
            first_best["first_best_intermediate"],
            first_best["first_best_at_endpoint"],
        ],
        [
            "under_100",
            "k100",
            "between_100_and_endpoint",
            "endpoint",
        ],
        default="other",
    )

    summary = (
        first_best.groupby(["task", "downstream_model"], as_index=False)
        .agg(
            n_cells=("dataset", "size"),
            n_datasets=("dataset", "nunique"),
            mean_first_best_k=("first_best_k", "mean"),
            median_first_best_k=("first_best_k", "median"),
            mean_best_k_fraction_of_endpoint=("best_k_fraction_of_endpoint", "mean"),
            median_best_k_fraction_of_endpoint=("best_k_fraction_of_endpoint", "median"),
            under_100_cells=("first_best_under_100", "sum"),
            k100_cells=("first_best_at_100", "sum"),
            intermediate_cells=("first_best_intermediate", "sum"),
            endpoint_cells=("first_best_at_endpoint", "sum"),
        )
        .sort_values(["task", "downstream_model"])
        .reset_index(drop=True)
    )

    overall = (
        first_best.groupby("task", as_index=False)
        .agg(
            n_cells=("dataset", "size"),
            n_datasets=("dataset", "nunique"),
            mean_first_best_k=("first_best_k", "mean"),
            median_first_best_k=("first_best_k", "median"),
            mean_best_k_fraction_of_endpoint=("best_k_fraction_of_endpoint", "mean"),
            median_best_k_fraction_of_endpoint=("best_k_fraction_of_endpoint", "median"),
            under_100_cells=("first_best_under_100", "sum"),
            k100_cells=("first_best_at_100", "sum"),
            intermediate_cells=("first_best_intermediate", "sum"),
            endpoint_cells=("first_best_at_endpoint", "sum"),
        )
        .sort_values("task")
        .reset_index(drop=True)
    )
    overall.insert(1, "downstream_model", "all")

    combined = pd.concat([overall, summary], ignore_index=True)
    for col in ["under_100", "k100", "intermediate", "endpoint"]:
        combined[f"{col}_share"] = combined[f"{col}_cells"] / combined["n_cells"]

    examples = first_best.sort_values(
        ["task", "dataset", "downstream_model", "first_best_k"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return combined, examples


def build_endpoint_pairwise(scores: pd.DataFrame, focus_method: str) -> pd.DataFrame:
    """Summarize high-p endpoint CIF-versus-baseline deltas."""
    ep = endpoint_rows(scores)
    focus = ep[ep["method_base"] == focus_method][["task", "dataset", "downstream_model", "dataset_mean_score"]]
    focus = focus.rename(columns={"dataset_mean_score": "focus_score"})
    others = ep[ep["method_base"] != focus_method][
        ["task", "dataset", "downstream_model", "method_base", "dataset_mean_score"]
    ].rename(columns={"dataset_mean_score": "baseline_score"})
    merged = others.merge(focus, on=["task", "dataset", "downstream_model"], how="inner")
    merged["delta"] = merged["focus_score"] - merged["baseline_score"]

    by_dataset = (
        merged.groupby(["task", "method_base", "dataset"], as_index=False)
        .agg(
            n_cells=("delta", "size"),
            mean_delta=("delta", "mean"),
        )
    )
    summary = (
        by_dataset.groupby(["task", "method_base"], as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            mean_dataset_cells=("n_cells", "mean"),
            mean_delta=("mean_delta", "mean"),
            median_delta=("mean_delta", "median"),
            wins=("mean_delta", lambda s: int((s > SCORE_TOL).sum())),
            losses=("mean_delta", lambda s: int((s < -SCORE_TOL).sum())),
            ties=("mean_delta", lambda s: int((s.abs() <= SCORE_TOL).sum())),
        )
        .sort_values(["task", "mean_delta", "method_base"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    summary = summary.rename(columns={"method_base": "baseline"})
    summary.insert(1, "focus_method", focus_method)
    summary["support_type"] = "dataset_mean_over_high_p_endpoint_pairwise_available"
    return summary


def build_endpoint_spread(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize how much method spread compresses from `k=100` to endpoint."""
    frames: list[pd.DataFrame] = []

    for label, frame in [("k100", complete_case_k100(scores)), ("endpoint", complete_case_endpoint(scores))]:
        spread = (
            frame.groupby(["task", "dataset", "downstream_model"], as_index=False)["dataset_mean_score"]
            .agg(["std", "min", "max"])
            .reset_index()
        )
        spread["range"] = spread["max"] - spread["min"]
        summary = (
            spread.groupby("task", as_index=False)
            .agg(
                n_dataset_downstream_cells=("range", "size"),
                mean_std=("std", "mean"),
                median_std=("std", "median"),
                mean_range=("range", "mean"),
                median_range=("range", "median"),
            )
            .sort_values("task")
            .reset_index(drop=True)
        )
        summary.insert(1, "comparison_point", label)
        frames.append(summary)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    """Build and save the paper-facing high-p endpoint tables."""
    parser = argparse.ArgumentParser(description="Build high-p endpoint summary tables")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TABLES_DIR,
        help="Directory for summary outputs",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory_frames: list[pd.DataFrame] = []
    endpoint_presence_frames: list[pd.DataFrame] = []
    delta_cell_frames: list[pd.DataFrame] = []
    delta_method_frames: list[pd.DataFrame] = []
    delta_overall_frames: list[pd.DataFrame] = []
    endpoint_aggregate_frames: list[pd.DataFrame] = []
    cif_summary_frames: list[pd.DataFrame] = []
    cif_example_frames: list[pd.DataFrame] = []
    cif_best_k_summary_frames: list[pd.DataFrame] = []
    cif_best_k_example_frames: list[pd.DataFrame] = []
    endpoint_pairwise_frames: list[pd.DataFrame] = []
    endpoint_spread_frames: list[pd.DataFrame] = []

    for task, config in TASK_CONFIG.items():
        scores = filter_high_p(build_task_scores(task, config["path"], config["metric"]))
        if scores.empty:
            continue

        inventory_frames.append(build_endpoint_inventory(scores))
        endpoint_presence_frames.append(build_endpoint_method_presence(scores))
        delta_cell_frames.append(build_delta_vs_endpoint_cells(scores))
        delta_method_frames.append(build_delta_vs_endpoint_method(scores))
        delta_overall_frames.append(build_delta_vs_endpoint_overall(scores))
        endpoint_aggregate_frames.append(build_endpoint_aggregate(scores, config["metric"]))
        cif_summary, cif_examples = build_cif_endpoint_summary(scores, config["focus_method"])
        cif_summary_frames.append(cif_summary)
        cif_example_frames.append(cif_examples)
        cif_best_k_summary, cif_best_k_examples = build_cif_best_observed_k_summary(scores, config["focus_method"])
        cif_best_k_summary_frames.append(cif_best_k_summary)
        cif_best_k_example_frames.append(cif_best_k_examples)
        endpoint_pairwise_frames.append(build_endpoint_pairwise(scores, config["focus_method"]))
        endpoint_spread_frames.append(build_endpoint_spread(scores))

    outputs = {
        "paper_high_p_endpoint_inventory.csv": pd.concat(inventory_frames, ignore_index=True),
        "paper_high_p_endpoint_method_presence.csv": pd.concat(endpoint_presence_frames, ignore_index=True),
        "paper_high_p_delta_vs_endpoint_cells.csv": pd.concat(delta_cell_frames, ignore_index=True),
        "paper_high_p_delta_vs_endpoint_method.csv": pd.concat(delta_method_frames, ignore_index=True),
        "paper_high_p_delta_vs_endpoint_overall.csv": pd.concat(delta_overall_frames, ignore_index=True),
        "paper_high_p_endpoint_aggregate.csv": pd.concat(endpoint_aggregate_frames, ignore_index=True),
        "paper_high_p_cif_endpoint_summary.csv": pd.concat(cif_summary_frames, ignore_index=True),
        "paper_high_p_cif_endpoint_examples.csv": pd.concat(cif_example_frames, ignore_index=True),
        "paper_high_p_cif_best_observed_k_summary.csv": pd.concat(cif_best_k_summary_frames, ignore_index=True),
        "paper_high_p_cif_best_observed_k_examples.csv": pd.concat(cif_best_k_example_frames, ignore_index=True),
        "paper_high_p_endpoint_pairwise.csv": pd.concat(endpoint_pairwise_frames, ignore_index=True),
        "paper_high_p_endpoint_spread.csv": pd.concat(endpoint_spread_frames, ignore_index=True),
    }

    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)


if __name__ == "__main__":
    main()
