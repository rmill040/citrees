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

import numpy as np
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


def power_replicates() -> pd.DataFrame:
    """Size-matched power with standard errors from the high-replicate (seed 3) and
    off-centre (85th-percentile step) Stage-B-only runs, pooled per cell with the
    seed 0-2 centred runs where the design is identical (nominal grid and step)."""
    import sys

    sys.path.insert(0, str(ROOT))
    from paper.analysis.build_stageb_only_pooled import CELL, clopper_pearson, size_matched

    frames = []
    for pattern, label in (
        ("r3-power-reps-*/stageb_only.csv", "centred_highrep"),
        ("r3-power-q85-*/stageb_only.csv", "offcentre_q85"),
    ):
        d = _cat(pattern)
        if d.empty:
            continue
        d["design"] = label
        frames.append(d)
    # Cells completed by boxes that were terminated before writing their CSV
    # (n = 500 high-replicate runs, stopped inside the K^2/alpha worst-case block),
    # parsed from their stdout: one line per finished cell.
    import re

    for f in sorted(glob.glob(str(RES.parent / "partial" / "r3-power-reps-*.stdout"))):
        rows = []
        with open(f) as handle:
            lines = handle.readlines()
        for line in lines:
            m = re.match(
                r"stageb (\w+) n=(\d+) k=(\d+) (\w+) (\w+) alpha=([\d.]+) delta=([\d.]+): ([\d.]+)",
                line.strip(),
            )
            if not m:
                continue
            task, n_, k_, test, stopping, alpha, delta, rate = m.groups()
            rows.append(
                dict(
                    study="stageb_only",
                    task=task,
                    n=int(n_),
                    k=int(k_),
                    threshold_test=test,
                    stopping=stopping,
                    nominal_alpha=float(alpha),
                    effect=float(delta),
                    replicates=1500 if float(delta) == 0 else 1000,
                    split_rate=float(rate),
                    step_quantile=0.5,
                    design="centred_highrep",
                    source="partial_stdout",
                )
            )
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw["splits"] = (raw["split_rate"] * raw["replicates"]).round().astype(int)
    out = []
    for design, sub in raw.groupby("design"):
        g = sub.groupby(CELL + ["nominal_alpha", "effect"], as_index=False).agg(
            splits=("splits", "sum"), replicates=("replicates", "sum")
        )
        g["rate"] = g["splits"] / g["replicates"]
        g["se"] = (g["rate"] * (1 - g["rate"]) / g["replicates"]) ** 0.5
        ci = [
            clopper_pearson(int(x), int(m))
            for x, m in zip(g["splits"], g["replicates"], strict=True)
        ]
        g["ci_low"], g["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
        m = size_matched(g)
        # attach the alternative-cell SE at the matched level (nearest nominal level by realized size)
        m["design"] = design
        g["design"] = design
        out.append((g, m))
    pooled = pd.concat([g for g, _ in out], ignore_index=True)
    matched = pd.concat([m for _, m in out], ignore_index=True)

    # SE of the size-matched power difference: combine the two arms' alternative SEs at the
    # nominal level whose realized size is closest to the match size
    def arm_se(row):
        sub = pooled[
            (pooled.design == row.design)
            & (pooled.task == row.task)
            & (pooled.n == row.n)
            & (pooled.k == row.k)
            & (pooled.threshold_test == row.threshold_test)
            & (pooled.stopping == row.stopping)
        ]
        null = sub[sub.effect == 0].set_index("nominal_alpha")["rate"]
        lvl = (null - row.match_size).abs().idxmin()
        alt = sub[(sub.effect == row.effect) & (sub.nominal_alpha == lvl)]
        return float(alt["se"].iloc[0]) if len(alt) else float("nan")

    matched["power_se"] = matched.apply(arm_se, axis=1)
    # Parametric bootstrap of the whole size-matching procedure: resample binomial
    # counts for every null and alternative cell, redo the interpolation, and take
    # the SD of the maxt-minus-bonferroni size-matched difference per cell.
    rng = np.random.default_rng(0)
    boot = {}
    for design, sub in pooled.groupby("design"):
        diffs = {}
        for _ in range(200):
            b = sub.copy()
            b["splits"] = rng.binomial(b["replicates"].astype(int), b["rate"].clip(0, 1))
            b["rate"] = b["splits"] / b["replicates"]
            mb = size_matched(b)
            pv = mb.pivot_table(
                index=["task", "n", "k", "stopping", "effect"],
                columns="threshold_test",
                values="power_at_match",
            )
            if "maxt" in pv and "bonferroni" in pv:
                for idx, val in (pv["maxt"] - pv["bonferroni"]).items():
                    diffs.setdefault(idx, []).append(val)
        boot[design] = {k: float(np.std(v)) for k, v in diffs.items()}
    matched["delta_se_bootstrap"] = [
        boot.get(r.design, {}).get((r.task, r.n, r.k, r.stopping, r.effect), float("nan"))
        for r in matched.itertuples()
    ]
    pooled.to_csv(TABLES / "paper_stageb_power_replicates_pooled.csv", index=False)
    return matched


def nhanes_controls() -> pd.DataFrame:
    """Mean rank of every column under the five forests, single draw and redraw, plus
    the per-fold contrasts on the shuffled survey weight."""
    rows = []
    for box, draw in (
        ("r3-nhanes-controls", "single_draw"),
        ("r3-nhanes-controls-redraw", "redraw_per_repeat"),
    ):
        f = RES / box / "fold_ranks.parquet"
        if not f.exists():
            continue
        r = pd.read_parquet(f)
        m = r.pivot_table(
            index="feature", columns="method", values="rank", aggfunc="mean"
        ).reset_index()
        m.insert(0, "draw", draw)
        m["kind"] = "mean_rank"
        rows.append(m)
        prim = r[r["feature"] == "shuffled_mec_weight"].pivot_table(
            index="fold", columns="method", values="rank"
        )
        for a, b in (
            ("rf", "cif"),
            ("rf_oobperm", "cif"),
            ("rf", "rf_oobperm"),
            ("cif_noscreen", "cif"),
            ("cforest", "cif"),
        ):
            d = prim[a] - prim[b]
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "draw": draw,
                            "feature": "shuffled_mec_weight",
                            "kind": f"contrast_{a}_minus_{b}",
                            "mean_diff": d.mean(),
                            "sd_diff": d.std(),
                            "share_positive": (d > 0).mean(),
                            "folds": len(d),
                        }
                    ]
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, fn in [
        ("paper_stageb_fixed_budget.csv", fixed_budget),
        ("paper_stageb_scaling_3arm.csv", scaling_3arm),
        ("paper_behavior_scanning_control.csv", behavior_scanning),
        ("paper_performance_equal_work.csv", equal_work),
        ("paper_stageb_power_replicates.csv", power_replicates),
        ("paper_nhanes_controls.csv", nhanes_controls),
    ]:
        df = fn()
        if df.empty:
            print(f"{name}: no inputs yet")
            continue
        df.to_csv(TABLES / name, index=False)
        print(f"{name}: {len(df)} rows")


if __name__ == "__main__":
    main()
