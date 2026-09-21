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
- ``paper_behavior_scanning_control.csv``: matched-behavior agreement with
  feature and threshold scanning on (positive control for the tie mechanism).
- ``paper_performance_equal_work.csv``: the 88-shard exhaustive performance
  campaign rerun with the permutation budget held at 999 per predictor in
  citrees (``scale_resamples_with_tests=False``), median and quartiles by cell.
- ``paper_nhanes_controls.csv``, ``paper_stageb_power_replicates.csv``.

Review-round-4 measurements (EC2, 2026-09-18) live under ``../data/review4/``:

- ``paper_performance_equal_work.csv`` also carries the rerun with the citrees
  budget one above partykit's (the 999-permutation run never split; see the
  ``run`` and ``fitted_depth`` columns).
- ``paper_performance_decomposition.csv``: the recommended forest at the
  reference condition under variants isolating its cost (serial, parallel,
  threshold test off, adjustment off, max-type), with depth and node counts.
- ``paper_noadjust_calibration.csv``, ``paper_noadjust_stageb.csv``:
  complete-node false-split rate and Stage-B-only size and power of the
  per-threshold rule without the Bonferroni adjustment over candidates.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT.parent / "data" / "review3" / "results"
RES4 = ROOT.parent / "data" / "review4"
TABLES = ROOT / "paper" / "results" / "tables"


def _cat(pattern: str, reader=pd.read_csv, root: Path = RES) -> pd.DataFrame:
    files = sorted(glob.glob(str(root / pattern)))
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
    """Both equal-work runs. ``fitted_depth`` is the tree's maximum depth counting
    the root as 1 (``fit_result_size`` for trees; the number of trees for forests),
    so 1 means the root never split. In the 2026-09-17 run (999 permutations) the
    citrees trees never split at 50 or 200 predictors: 1/(999+1) is not strictly
    below 0.05/50; that run also timed the first fit of each kernel on a host,
    which included its compilation. The 2026-09-18 run uses the budget one above
    partykit's and times the second identical fit; the r4-perf-equalwork run in
    between (budget fixed, protocol not) is superseded and not read."""
    frames = []
    for pattern, root, run, budget in (
        (
            "r3-perf-equalwork-b*/shard-*/performance_raw.parquet",
            RES,
            "2026-09-17",
            "999 permutations per predictor (never splits at p >= 50)",
        ),
        (
            "r4b-perf-equalwork-b*/shard-*/performance_raw.parquet",
            RES4,
            "2026-09-18 second-fit protocol",
            "partykit budget + 1 per predictor (1,000 at the reference)",
        ),
        (
            "r4b-perf-partykit-b*/shard-*/performance_raw.parquet",
            RES4,
            "2026-09-18 second-fit protocol",
            "partykit at its own budget",
        ),
    ):
        d = _cat(pattern, pd.read_parquet, root)
        if d.empty:
            continue
        d["run"] = run
        d["citrees_budget"] = budget
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    g = d.groupby(
        ["run", "task", "model_family", "method", "axis", "axis_value", "citrees_budget"],
        as_index=False,
    ).agg(
        median=("elapsed_seconds", "median"),
        q1=("elapsed_seconds", lambda x: x.quantile(0.25)),
        q3=("elapsed_seconds", lambda x: x.quantile(0.75)),
        n=("elapsed_seconds", "size"),
        fitted_depth=("fit_result_size", "median"),
    )
    return g


def perf_decomposition() -> pd.DataFrame:
    d = _cat("r4-perf-decomp/performance_decomposition.csv", root=RES4)
    if d.empty:
        return d
    return d.groupby(
        ["task", "variant", "kind", "n_jobs"], as_index=False, sort=False, dropna=False
    ).agg(
        seconds_median=("seconds", "median"),
        seconds_min=("seconds", "min"),
        seconds_max=("seconds", "max"),
        depth_mean=("depth_mean", "mean"),
        internal_nodes_mean=("internal_nodes_mean", "mean"),
        repeats=("seconds", "size"),
    )


def noadjust_calibration() -> pd.DataFrame:
    return _cat("r4-noadjust-null/null_calibration.csv", root=RES4)


def noadjust_stageb() -> pd.DataFrame:
    return _cat("r4-noadjust-stageb/stageb_only.csv", root=RES4)


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


def weak_signal() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weak-signal cardinality designs and support sizes."""
    d = _cat("r3-weak-designs-*/designs__*.parquet", pd.read_parquet)
    sup = _cat("r3-weak-support-*/support__*.parquet", pd.read_parquet)
    designs = pd.DataFrame()
    if not d.empty:
        fits = d.drop_duplicates(["task", "signal", "seed", "fold_idx", "method"])
        share_cols = [c for c in fits.columns if "_share_at_" in c] + ["support_size"]
        g = fits.groupby(["task", "signal", "method"])[share_cols].agg(["mean", "std", "count"])
        g.columns = ["_".join(c) for c in g.columns]
        for c in share_cols:
            g[f"{c}_se"] = g[f"{c}_std"] / np.sqrt(g[f"{c}_count"])
        metric = {"classification": "balanced_accuracy", "regression": "r2"}
        score = []
        for task, sub in d.groupby("task"):
            m = sub.groupby(["task", "signal", "method", "k"])[metric[task]].mean().unstack("k")
            m.columns = [f"downstream_at_{int(k)}" for k in m.columns]
            score.append(m)
        designs = g.join(pd.concat(score)).reset_index()
    support = pd.DataFrame()
    if not sup.empty:
        support = sup.groupby(["task", "dataset", "method"], as_index=False).agg(
            support_size=("support_size", "mean"),
            informative_in_support=("informative_in_support", "mean"),
            n_informative=("n_informative", "first"),
            n_features=("n_features", "first"),
            fits=("support_size", "size"),
        )
    return designs, support


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, fn in [
        ("paper_stageb_fixed_budget.csv", fixed_budget),
        ("paper_behavior_scanning_control.csv", behavior_scanning),
        ("paper_performance_equal_work.csv", equal_work),
        ("paper_stageb_power_replicates.csv", power_replicates),
        ("paper_nhanes_controls.csv", nhanes_controls),
        ("paper_performance_decomposition.csv", perf_decomposition),
        ("paper_noadjust_calibration.csv", noadjust_calibration),
        ("paper_noadjust_stageb.csv", noadjust_stageb),
    ]:
        df = fn()
        if df.empty:
            print(f"{name}: no inputs yet")
            continue
        df.to_csv(TABLES / name, index=False)
        print(f"{name}: {len(df)} rows")
    designs, support = weak_signal()
    for name, df in (
        ("paper_weak_signal_cardinality.csv", designs),
        ("paper_support_size.csv", support),
    ):
        if df.empty:
            print(f"{name}: no inputs yet")
            continue
        df.to_csv(TABLES / name, index=False)
        print(f"{name}: {len(df)} rows")


if __name__ == "__main__":
    main()
