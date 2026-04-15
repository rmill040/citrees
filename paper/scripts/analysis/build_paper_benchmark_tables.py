"""Build benchmark tables for the paper rewrite.

This script creates a single summary surface that:

1. uses one explicit config-selection rule per task,
2. mirrors classification and regression,
3. keeps support accounting explicit, and
4. separates stratified from aggregate reporting.

Config contract:
  - one best global config per method family within task
  - selected over real datasets, standard k values, and all downstream models

Outputs:
  - paper/results/tables/paper_benchmark_best_configs.csv
  - paper/results/tables/paper_benchmark_selected_config_details.csv
  - paper/results/tables/paper_benchmark_observed_k_values.csv
  - paper/results/tables/paper_benchmark_complete_case_membership.csv
  - paper/results/tables/paper_benchmark_fixed_panel_membership.csv
  - paper/results/tables/paper_benchmark_fixed_panel_aggregate.csv
  - paper/results/tables/paper_benchmark_spread.csv
  - paper/results/tables/paper_benchmark_stratified.csv
  - paper/results/tables/paper_benchmark_extended_stratified.csv
  - paper/results/tables/paper_benchmark_method_aggregate.csv
  - paper/results/tables/paper_benchmark_pairwise_stratified.csv
  - paper/results/tables/paper_benchmark_pairwise_aggregate.csv

Usage:
  UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_paper_benchmark_tables.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper.scripts.analysis.benchmark_common import (
    STANDARD_K,
    TABLES_DIR,
    TASK_CONFIG,
    complete_case_scores,
    dataset_scores,
    load_real_task_frame,
    rank_complete_case_scores,
    select_best_task_configs,
    task_global_config_scores,
)
from paper.scripts.analysis.config_resolution import resolve_method_config_details
from paper.scripts.pipeline.methods import get_method_config_count


def _build_stratified_summary(
    ranked: pd.DataFrame,
    task: str,
    metric: str,
    support_type: str,
) -> pd.DataFrame:
    """Summarize complete-case scores by downstream and k."""
    summary = (
        ranked.groupby(["downstream_model", "k", "method_base", "method_id"], as_index=False)
        .agg(
            n_complete_datasets=("dataset", "nunique"),
            mean_score=("dataset_mean_score", "mean"),
            median_score=("dataset_mean_score", "median"),
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
        )
    )
    summary.insert(0, "task", task)
    summary.insert(1, "metric", metric)
    summary["support_type"] = support_type
    summary["rank_position"] = summary.groupby(["task", "downstream_model", "k"])["mean_rank"].rank(
        ascending=True,
        method="average",
    )
    return summary.sort_values(["task", "downstream_model", "k", "rank_position", "method_base"]).reset_index(
        drop=True
    )


def _build_method_aggregate(
    ranked: pd.DataFrame,
    task: str,
    metric: str,
) -> pd.DataFrame:
    """Aggregate complete-case ranks over all downstreams and standard k values."""
    by_dataset = (
        ranked.groupby(["dataset", "method_base", "method_id"], as_index=False)
        .agg(
            n_cells=("rank", "size"),
            mean_rank=("rank", "mean"),
            mean_score=("dataset_mean_score", "mean"),
        )
    )

    aggregate = (
        by_dataset.groupby(["method_base", "method_id"], as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            mean_dataset_cells=("n_cells", "mean"),
            mean_rank=("mean_rank", "mean"),
            median_rank=("mean_rank", "median"),
            mean_score=("mean_score", "mean"),
        )
    )
    aggregate.insert(0, "task", task)
    aggregate.insert(1, "metric", metric)
    aggregate["support_type"] = "dataset_mean_over_all_complete_case_downstream_k"
    aggregate["rank_position"] = aggregate.groupby("task")["mean_rank"].rank(
        ascending=True,
        method="average",
    )
    return aggregate.sort_values(["task", "rank_position", "method_base"]).reset_index(drop=True)


def _build_complete_case_membership(scores: pd.DataFrame, task: str) -> pd.DataFrame:
    """Record dataset membership for the standard-k all-method complete-case surface."""
    expected_methods = int(scores["method_base"].nunique())
    membership = (
        scores.groupby(["downstream_model", "k", "dataset"], as_index=False)["method_base"]
        .nunique()
        .rename(columns={"method_base": "n_methods_present"})
        .sort_values(["downstream_model", "k", "dataset"])
        .reset_index(drop=True)
    )
    membership.insert(0, "task", task)
    membership["expected_methods"] = expected_methods
    membership["is_complete_case"] = membership["n_methods_present"] == expected_methods
    return membership


def _build_fixed_panel_membership(scores: pd.DataFrame, task: str) -> pd.DataFrame:
    """Record which datasets stay complete across every downstream x k cell."""
    membership = _build_complete_case_membership(scores, task)
    required_complete_cells = int(scores["downstream_model"].nunique() * scores["k"].nunique())
    dataset_membership = (
        membership.groupby("dataset", as_index=False)
        .agg(
            n_observed_cells=("is_complete_case", "size"),
            n_complete_case_cells=("is_complete_case", "sum"),
        )
        .sort_values("dataset")
        .reset_index(drop=True)
    )
    dataset_membership.insert(0, "task", task)
    dataset_membership["required_complete_cells"] = required_complete_cells
    dataset_membership["n_missing_complete_cells"] = (
        required_complete_cells - dataset_membership["n_complete_case_cells"]
    )
    dataset_membership["is_fixed_panel"] = (
        dataset_membership["n_complete_case_cells"] == dataset_membership["required_complete_cells"]
    )
    return dataset_membership


def _build_benchmark_spread(scores: pd.DataFrame, task: str) -> pd.DataFrame:
    """Summarize cross-method score spread on the complete-case standard-k surface."""
    spread = (
        scores.groupby(["dataset", "downstream_model", "k"], as_index=False)["dataset_mean_score"]
        .agg(["std", "min", "max"])
        .reset_index()
    )
    spread["range"] = spread["max"] - spread["min"]

    by_k = (
        spread.groupby("k", as_index=False)
        .agg(
            n_dataset_downstream_cells=("range", "size"),
            mean_std=("std", "mean"),
            median_std=("std", "median"),
            mean_range=("range", "mean"),
            median_range=("range", "median"),
        )
        .sort_values("k")
        .reset_index(drop=True)
    )
    by_k.insert(0, "task", task)
    by_k.insert(1, "comparison_scope", "standard_k")

    pooled = pd.DataFrame(
        [
            {
                "task": task,
                "comparison_scope": "all_standard_k",
                "k": pd.NA,
                "n_dataset_downstream_cells": int(len(spread)),
                "mean_std": float(spread["std"].mean()),
                "median_std": float(spread["std"].median()),
                "mean_range": float(spread["range"].mean()),
                "median_range": float(spread["range"].median()),
            }
        ]
    )
    return pd.concat([by_k, pooled], ignore_index=True)


def _build_pairwise_stratified(
    scores: pd.DataFrame,
    task: str,
    focus_method: str,
) -> pd.DataFrame:
    """Summarize CIF-vs-baseline deltas for each downstream and k."""
    rows: list[dict[str, object]] = []

    for (downstream_model, k), cell in scores.groupby(["downstream_model", "k"]):
        pivot = cell.pivot(index="dataset", columns="method_base", values="dataset_mean_score")
        if focus_method not in pivot.columns:
            continue

        for baseline in sorted(col for col in pivot.columns if col != focus_method):
            common = pivot[[focus_method, baseline]].dropna()
            if common.empty:
                continue

            delta = common[focus_method] - common[baseline]
            rows.append(
                {
                    "task": task,
                    "downstream_model": downstream_model,
                    "k": int(k),
                    "focus_method": focus_method,
                    "baseline": baseline,
                    "support_type": "pairwise_available_standard_k",
                    "n_datasets": int(len(common)),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(delta.median()),
                    "wins": int((delta > 0).sum()),
                    "losses": int((delta < 0).sum()),
                    "ties": int((delta == 0).sum()),
                    "win_share": float((delta > 0).mean()),
                    "nonloss_share": float((delta >= 0).mean()),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["task", "downstream_model", "k", "baseline"]
    ).reset_index(drop=True)


def _build_pairwise_aggregate(
    scores: pd.DataFrame,
    task: str,
    focus_method: str,
) -> pd.DataFrame:
    """Aggregate CIF-vs-baseline deltas over all downstreams and supported k values."""
    rows: list[dict[str, object]] = []

    for baseline in sorted(m for m in scores["method_base"].unique() if m != focus_method):
        focus = scores[scores["method_base"] == focus_method][["dataset", "downstream_model", "k", "dataset_mean_score"]]
        focus = focus.rename(columns={"dataset_mean_score": "focus_score"})
        base = scores[scores["method_base"] == baseline][["dataset", "downstream_model", "k", "dataset_mean_score"]]
        base = base.rename(columns={"dataset_mean_score": "baseline_score"})

        merged = focus.merge(base, on=["dataset", "downstream_model", "k"], how="inner")
        if merged.empty:
            continue

        merged["delta"] = merged["focus_score"] - merged["baseline_score"]

        by_dataset = (
            merged.groupby("dataset", as_index=False)
            .agg(
                n_cells=("delta", "size"),
                mean_delta=("delta", "mean"),
                win_share=("delta", lambda x: float((x > 0).mean())),
                nonloss_share=("delta", lambda x: float((x >= 0).mean())),
            )
        )

        rows.append(
            {
                "task": task,
                "focus_method": focus_method,
                "baseline": baseline,
                "support_type": "dataset_mean_over_all_supported_downstream_standard_k",
                "n_datasets": int(by_dataset["dataset"].nunique()),
                "mean_dataset_cells": float(by_dataset["n_cells"].mean()),
                "mean_delta": float(by_dataset["mean_delta"].mean()),
                "median_delta": float(by_dataset["mean_delta"].median()),
                "wins": int((by_dataset["mean_delta"] > 0).sum()),
                "losses": int((by_dataset["mean_delta"] < 0).sum()),
                "ties": int((by_dataset["mean_delta"] == 0).sum()),
                "win_share": float((by_dataset["mean_delta"] > 0).mean()),
                "nonloss_share": float((by_dataset["mean_delta"] >= 0).mean()),
                "mean_cell_win_share": float(by_dataset["win_share"].mean()),
                "mean_cell_nonloss_share": float(by_dataset["nonloss_share"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(["task", "baseline"]).reset_index(drop=True)


def _build_config_selection_audit(df: pd.DataFrame, task: str, metric: str) -> pd.DataFrame:
    """Summarize tuning flexibility for the selected config in each method family."""
    perf = task_global_config_scores(df, metric)
    counts = get_method_config_count(sorted(perf["method_base"].unique()), task)
    rows: list[dict[str, object]] = []

    for method_base, family in perf.groupby("method_base"):
        family = family.sort_values(["task_global_mean_metric", "method_id"], ascending=[False, True]).reset_index(drop=True)
        best = family.iloc[0]
        runner_up = family.iloc[1] if len(family) > 1 else None
        family_mean = float(family["task_global_mean_metric"].mean())
        rows.append(
            {
                "task": task,
                "metric": metric,
                "method_base": method_base,
                "method_id": str(best["method_id"]),
                "candidate_config_count": int(counts[method_base]),
                "observed_config_count": int(len(family)),
                "runner_up_method_id": None if runner_up is None else str(runner_up["method_id"]),
                "runner_up_task_global_mean_metric": (
                    None if runner_up is None else float(runner_up["task_global_mean_metric"])
                ),
                "selected_minus_runner_up": (
                    None
                    if runner_up is None
                    else float(best["task_global_mean_metric"] - runner_up["task_global_mean_metric"])
                ),
                "family_mean_task_global_metric": family_mean,
                "selected_minus_family_mean_task_global_metric": float(best["task_global_mean_metric"] - family_mean),
                "family_task_global_metric_std": float(family["task_global_mean_metric"].std(ddof=0)),
            }
        )

    return pd.DataFrame(rows).sort_values(["task", "method_base"]).reset_index(drop=True)


def _build_fixed_panel_aggregate(
    scores: pd.DataFrame,
    task: str,
    metric: str,
    headline_aggregate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate ranks on a dataset panel that stays complete across all downstream x k cells."""
    fixed_panel_membership = _build_fixed_panel_membership(scores, task)
    fixed_datasets = fixed_panel_membership[fixed_panel_membership["is_fixed_panel"]]["dataset"]
    fixed_scores = scores[scores["dataset"].isin(fixed_datasets)].copy()
    fixed_ranked = rank_complete_case_scores(fixed_scores)
    fixed_aggregate = _build_method_aggregate(fixed_ranked, task, metric)
    fixed_aggregate["support_type"] = "dataset_mean_over_all_standard_downstream_k_fixed_panel"

    headline_cols = [
        "method_base",
        "n_datasets",
        "mean_dataset_cells",
        "mean_rank",
        "mean_score",
        "rank_position",
    ]
    merged = fixed_aggregate.merge(
        headline_aggregate[headline_cols].rename(
            columns={
                "n_datasets": "headline_n_datasets",
                "mean_dataset_cells": "headline_mean_dataset_cells",
                "mean_rank": "headline_mean_rank",
                "mean_score": "headline_mean_score",
                "rank_position": "headline_rank_position",
            }
        ),
        on="method_base",
        how="left",
    )
    merged["delta_mean_rank_vs_headline"] = merged["mean_rank"] - merged["headline_mean_rank"]
    merged["delta_mean_score_vs_headline"] = merged["mean_score"] - merged["headline_mean_score"]
    return fixed_panel_membership, merged


def main() -> None:
    """Build and save the paper-facing benchmark summary tables."""
    parser = argparse.ArgumentParser(description="Build contract-compliant benchmark summary tables")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TABLES_DIR,
        help="Directory for summary outputs",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    best_config_frames: list[pd.DataFrame] = []
    observed_k_frames: list[pd.DataFrame] = []
    membership_frames: list[pd.DataFrame] = []
    fixed_panel_membership_frames: list[pd.DataFrame] = []
    fixed_panel_aggregate_frames: list[pd.DataFrame] = []
    selection_audit_frames: list[pd.DataFrame] = []
    spread_frames: list[pd.DataFrame] = []
    stratified_frames: list[pd.DataFrame] = []
    extended_stratified_frames: list[pd.DataFrame] = []
    method_aggregate_frames: list[pd.DataFrame] = []
    pairwise_stratified_frames: list[pd.DataFrame] = []
    pairwise_aggregate_frames: list[pd.DataFrame] = []

    for task, config in TASK_CONFIG.items():
        raw = load_real_task_frame(config["path"])
        standard_raw = raw[raw["k"].isin(STANDARD_K)].copy()
        metric = config["metric"]
        focus_method = config["focus_method"]

        _, best_configs = select_best_task_configs(standard_raw, metric)
        best_configs.insert(0, "task", task)
        best_configs.insert(1, "metric", metric)
        best_config_frames.append(best_configs)
        selection_audit_frames.append(_build_config_selection_audit(standard_raw, task, metric))

        # Apply the same task-wide best configs to the full observed real-data
        # k surface so the endpoint / dataset-specific summaries use the same
        # config contract as the standard benchmark layer.
        best_full = raw.merge(best_configs[["method_base", "method_id"]], on=["method_base", "method_id"], how="inner")
        full_scores = dataset_scores(best_full, metric)
        standard_scores = full_scores[full_scores["k"].isin(STANDARD_K)].copy()
        membership_frames.append(_build_complete_case_membership(standard_scores, task))

        observed_k = (
            full_scores.groupby("k", as_index=False)["dataset"]
            .nunique()
            .rename(columns={"dataset": "n_datasets"})
            .sort_values("k")
        )
        observed_k.insert(0, "task", task)
        observed_k["is_standard_k"] = observed_k["k"].isin(STANDARD_K)
        observed_k_frames.append(observed_k)

        standard_complete = complete_case_scores(standard_scores)
        standard_ranked = rank_complete_case_scores(standard_complete)
        spread_frames.append(_build_benchmark_spread(standard_complete, task))

        full_complete = complete_case_scores(full_scores)
        full_ranked = rank_complete_case_scores(full_complete)

        stratified_frames.append(
            _build_stratified_summary(
                standard_ranked,
                task,
                metric,
                support_type="all_method_complete_case_standard_k",
            )
        )
        extended_stratified_frames.append(
            _build_stratified_summary(
                full_ranked,
                task,
                metric,
                support_type="all_method_complete_case_all_observed_k",
            )
        )
        headline_aggregate = _build_method_aggregate(standard_ranked, task, metric)
        method_aggregate_frames.append(headline_aggregate)
        fixed_panel_membership, fixed_panel_aggregate = _build_fixed_panel_aggregate(
            standard_scores,
            task,
            metric,
            headline_aggregate,
        )
        fixed_panel_membership_frames.append(fixed_panel_membership)
        fixed_panel_aggregate_frames.append(fixed_panel_aggregate)
        pairwise_stratified_frames.append(_build_pairwise_stratified(standard_scores, task, focus_method))
        pairwise_aggregate_frames.append(_build_pairwise_aggregate(standard_scores, task, focus_method))

    best_configs = pd.concat(best_config_frames, ignore_index=True)
    selection_audit = pd.concat(selection_audit_frames, ignore_index=True)
    selected_config_details = resolve_method_config_details(best_configs).merge(
        selection_audit,
        on=["task", "metric", "method_base", "method_id"],
        how="left",
    )
    outputs = {
        "paper_benchmark_best_configs.csv": best_configs,
        "paper_benchmark_selected_config_details.csv": selected_config_details,
        "paper_benchmark_observed_k_values.csv": pd.concat(observed_k_frames, ignore_index=True),
        "paper_benchmark_complete_case_membership.csv": pd.concat(membership_frames, ignore_index=True),
        "paper_benchmark_fixed_panel_membership.csv": pd.concat(fixed_panel_membership_frames, ignore_index=True),
        "paper_benchmark_fixed_panel_aggregate.csv": pd.concat(fixed_panel_aggregate_frames, ignore_index=True),
        "paper_benchmark_spread.csv": pd.concat(spread_frames, ignore_index=True),
        "paper_benchmark_stratified.csv": pd.concat(stratified_frames, ignore_index=True),
        "paper_benchmark_extended_stratified.csv": pd.concat(extended_stratified_frames, ignore_index=True),
        "paper_benchmark_method_aggregate.csv": pd.concat(method_aggregate_frames, ignore_index=True),
        "paper_benchmark_pairwise_stratified.csv": pd.concat(pairwise_stratified_frames, ignore_index=True),
        "paper_benchmark_pairwise_aggregate.csv": pd.concat(pairwise_aggregate_frames, ignore_index=True),
    }

    for filename, frame in outputs.items():
        out_path = output_dir / filename
        frame.to_csv(out_path, index=False)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
