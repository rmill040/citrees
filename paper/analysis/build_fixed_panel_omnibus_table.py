"""Build omnibus test summaries for the 14-dataset paper benchmark.

This script evaluates the already-selected paper-facing benchmark surface on the
shared 14-dataset benchmark datasets. For each task it:

1. filters to datasets that are complete across every standard downstream x k
   cell,
2. averages each method over those standard downstream x k cells within
   dataset, and
3. runs a Friedman omnibus test across methods on that shared dataset matrix.

Outputs:
  - paper/results/tables/paper_benchmark_fixed_panel_omnibus.csv

Usage:
  uv run python paper/analysis/build_fixed_panel_omnibus_table.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import pandas as pd
from scipy import stats

TABLES_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "results" / "tables"
FIXED_PANEL_MEMBERSHIP_PATH: Final[Path] = TABLES_DIR / "paper_benchmark_fixed_panel_membership.csv"
DATASET_SURFACE_PATH: Final[Path] = TABLES_DIR / "paper_heterogeneity_dataset_method_surface.csv"
OUTPUT_PATH: Final[Path] = TABLES_DIR / "paper_benchmark_fixed_panel_omnibus.csv"


def _kendalls_w(chi2_friedman: float, n_datasets: int, k_methods: int) -> float:
    """Return Kendall's W effect size for a Friedman test."""
    return chi2_friedman / (n_datasets * (k_methods - 1))


def build_fixed_panel_omnibus() -> pd.DataFrame:
    """Compute Friedman omnibus summaries on the 14-dataset benchmark surface."""
    membership = pd.read_csv(FIXED_PANEL_MEMBERSHIP_PATH)
    surface = pd.read_csv(DATASET_SURFACE_PATH)
    rows: list[dict[str, object]] = []

    for task in sorted(surface["task"].unique()):
        fixed_datasets = membership[(membership["task"] == task) & (membership["is_fixed_panel"])][
            "dataset"
        ]
        task_surface = surface[
            (surface["task"] == task) & (surface["dataset"].isin(fixed_datasets))
        ].copy()
        if task_surface.empty:
            continue

        matrix = task_surface.pivot(
            index="dataset", columns="method_base", values="mean_score"
        ).sort_index(axis=1)
        values = [matrix[col].to_numpy() for col in matrix.columns]
        chi2, p_value = stats.friedmanchisquare(*values)
        n_datasets = int(len(matrix))
        n_methods = int(len(matrix.columns))

        rows.append(
            {
                "task": task,
                "n_datasets": n_datasets,
                "n_methods": n_methods,
                "chi_square": float(chi2),
                "p_value": float(p_value),
                "kendalls_w": float(_kendalls_w(chi2, n_datasets, n_methods)),
                "support_type": "dataset_mean_over_all_standard_downstream_k_fixed_panel",
                "source_table": DATASET_SURFACE_PATH.name,
            }
        )

    return pd.DataFrame(rows).sort_values("task").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    out = build_fixed_panel_omnibus()
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
