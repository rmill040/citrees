"""Build fixed-panel paired uncertainty summaries for CIF.

This script adds a compact support-matched comparison layer on top of the
paper-facing benchmark contract:

1. one selected config per method family within task,
2. real datasets only,
3. standard k values only, and
4. datasets that remain complete across every downstream x k cell.

Output:
  - paper/results/tables/paper_benchmark_fixed_panel_pairwise_ci.csv

Usage:
  UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_fixed_panel_uncertainty_table.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Final

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.scripts.analysis.benchmark_common import STANDARD_K, TABLES_DIR, TASK_CONFIG, load_real_task_frame


BEST_CONFIGS_PATH: Final[Path] = TABLES_DIR / "paper_benchmark_best_configs.csv"
FIXED_PANEL_MEMBERSHIP_PATH: Final[Path] = TABLES_DIR / "paper_benchmark_fixed_panel_membership.csv"
OUTPUT_PATH: Final[Path] = TABLES_DIR / "paper_benchmark_fixed_panel_pairwise_ci.csv"
BOOTSTRAP_SEED: Final[int] = 1718
DEFAULT_N_BOOT: Final[int] = 20_000


def _bootstrap_mean_ci(values: np.ndarray, *, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a paired delta vector."""
    n = len(values)
    draws = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        draws[idx] = float(sample.mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def _load_fixed_panel_datasets(task: str) -> list[str]:
    membership = pd.read_csv(FIXED_PANEL_MEMBERSHIP_PATH)
    fixed = membership[(membership["task"] == task) & (membership["is_fixed_panel"])]
    return sorted(fixed["dataset"].tolist())


def _selected_method_ids(task: str) -> dict[str, str]:
    best = pd.read_csv(BEST_CONFIGS_PATH)
    sub = best[best["task"] == task][["method_base", "method_id"]].copy()
    return dict(zip(sub["method_base"], sub["method_id"], strict=False))


def _dataset_mean_scores(task: str) -> pd.DataFrame:
    task_cfg = TASK_CONFIG[task]
    selected = _selected_method_ids(task)
    fixed_panel_datasets = _load_fixed_panel_datasets(task)

    frame = load_real_task_frame(task_cfg["path"])
    frame = frame[
        frame["dataset"].isin(fixed_panel_datasets)
        & frame["k"].isin(STANDARD_K)
        & frame["method_id"].isin(selected.values())
    ].copy()

    metric = task_cfg["metric"]
    cell_scores = (
        frame.groupby(["dataset", "downstream_model", "k", "method_base", "method_id"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "dataset_mean_score"})
    )
    dataset_scores = (
        cell_scores.groupby(["dataset", "method_base", "method_id"], as_index=False)["dataset_mean_score"]
        .mean()
        .sort_values(["dataset", "method_base"])
        .reset_index(drop=True)
    )
    return dataset_scores


def build_fixed_panel_pairwise_ci(*, n_boot: int = DEFAULT_N_BOOT) -> pd.DataFrame:
    """Summarize paired CIF-vs-baseline deltas on the fixed panel."""
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for task, task_cfg in TASK_CONFIG.items():
        dataset_scores = _dataset_mean_scores(task)
        focus_method = task_cfg["focus_method"]
        wide = dataset_scores.pivot(index="dataset", columns="method_base", values="dataset_mean_score")
        if focus_method not in wide.columns:
            continue

        for baseline in sorted(col for col in wide.columns if col != focus_method):
            paired = wide[[focus_method, baseline]].dropna()
            if paired.empty:
                continue

            delta = paired[focus_method] - paired[baseline]
            wins = int((delta > 0).sum())
            losses = int((delta < 0).sum())
            ties = int((delta == 0).sum())
            non_ties = wins + losses
            ci_lower, ci_upper = _bootstrap_mean_ci(delta.to_numpy(), n_boot=n_boot, rng=rng)
            sign_pvalue = (
                float(binomtest(wins, non_ties, 0.5, alternative="greater").pvalue) if non_ties else np.nan
            )

            rows.append(
                {
                    "task": task,
                    "focus_method": focus_method,
                    "baseline": baseline,
                    "n_datasets": int(len(paired)),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(delta.median()),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "win_share": float(wins / len(paired)),
                    "nonloss_share": float((wins + ties) / len(paired)),
                    "sign_test_pvalue": sign_pvalue,
                    "bootstrap_replicates": int(n_boot),
                }
            )

    return pd.DataFrame(rows).sort_values(["task", "focus_method", "baseline"]).reset_index(drop=True)


def main() -> None:
    """Build and save the fixed-panel uncertainty summary."""
    parser = argparse.ArgumentParser(description="Build fixed-panel paired uncertainty summary for CIF")
    parser.add_argument(
        "--n-boot",
        type=int,
        default=DEFAULT_N_BOOT,
        help="Number of bootstrap replicates for percentile CIs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output CSV path",
    )
    args = parser.parse_args()

    out = build_fixed_panel_pairwise_ci(n_boot=args.n_boot)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)


if __name__ == "__main__":
    main()
