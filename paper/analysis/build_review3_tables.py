#!/usr/bin/env python3
"""Tables for the review-round-3 control experiments (EC2 run of 2026-09-17).

Inputs live under ``../data/review3/results/<box>/`` (tarballs from
``repairs/h2h-rerun/source-e78d84a.../review3/``, archived in
``reference/v3/ablations/review3/``). Outputs, all under ``paper/results/tables``:

- ``paper_stageb_fixed_budget.csv``: Stage-B-only size and power of the
  per-threshold Bonferroni rule at a fixed budget of 999 permutations per
  candidate (``bonferroni_fixed``), by task, n, bins, stopping, level, effect.
  The adaptive column is the ``minimum`` rule in disguise (the sequential paths
  raise any budget to the floor), so the exhaustive column is the fixed-B arm.
- ``paper_stageb_scaling_3arm.csv``: one-run timing of bonferroni (scaled
  budget), bonferroni_fixed, and maxt on the scaling grid, with tree depth.
- ``paper_behavior_scanning_control.csv``: matched-behavior agreement with
  feature and threshold scanning on (positive control for the tie mechanism).
- ``paper_performance_equal_work.csv``: the 88-shard exhaustive performance
  campaign rerun with the permutation budget held at 999 per predictor in
  citrees (``scale_resamples_with_tests=False``), median and quartiles by cell.
- ``paper_nhanes_controls.csv``, ``paper_cif_all_recovery.csv``,
  ``paper_stageb_power_replicates.csv``: written when their boxes are in.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT.parent / "data" / "review3" / "results"
TABLES = ROOT / "paper" / "results" / "tables"


def _cat(pattern: str, reader=pd.read_csv) -> pd.DataFrame:
    files = sorted(glob.glob(str(RES / pattern)))
    return pd.concat([reader(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def fixed_budget() -> pd.DataFrame:
    d = _cat("r3-stageb-fixed-*/stageb_only.csv")
    if d.empty:
        return d
    d["budget_note"] = d["stopping"].map(
        {
            "exhaustive": "fixed 999 per candidate",
            "adaptive": "raised to the floor ceil(K/alpha) by the sequential path",
        }
    )
    return d.sort_values(["task", "n", "k", "stopping", "nominal_alpha", "effect"])


def scaling_3arm() -> pd.DataFrame:
    return _cat("r3-scaling-3arm/synthetic_scaling.csv")


def behavior_scanning() -> pd.DataFrame:
    d = _cat("r3-behavior-scan-shard*/behavior_fold_summary.parquet", pd.read_parquet)
    if d.empty:
        return d
    cols = ["split_decision_agreement", "root_agreement_given_both_split", "prediction_agreement"]
    g = d.groupby(["task", "model_family"], as_index=False)[cols].agg(["mean", "count"])
    g.columns = ["_".join(c).rstrip("_") for c in g.columns]
    g["feature_scanning"] = True
    return g


def equal_work() -> pd.DataFrame:
    d = _cat("r3-perf-equalwork-b*/shard-*/performance_raw.parquet", pd.read_parquet)
    if d.empty:
        return d
    g = d.groupby(["task", "model_family", "method", "axis", "axis_value"], as_index=False)[
        "elapsed_seconds"
    ].agg(median="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75), n="size")
    g["citrees_budget"] = "999 permutations per predictor (scale_resamples_with_tests=False)"
    return g


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, fn in [
        ("paper_stageb_fixed_budget.csv", fixed_budget),
        ("paper_stageb_scaling_3arm.csv", scaling_3arm),
        ("paper_behavior_scanning_control.csv", behavior_scanning),
        ("paper_performance_equal_work.csv", equal_work),
    ]:
        df = fn()
        if df.empty:
            print(f"{name}: no inputs yet")
            continue
        df.to_csv(TABLES / name, index=False)
        print(f"{name}: {len(df)} rows")


if __name__ == "__main__":
    main()
