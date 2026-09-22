#!/usr/bin/env python3
"""Cluster-bootstrap intervals for the size-matched Stage B power difference.

Reads the size-matched power tables of the Stage-B-only study and its two
replicate runs, forms the max-type minus Bonferroni difference for every
(task, n, k, stopping, effect) combination, and resamples the (task, n, k)
cells with replacement, so the interval treats the cell as the unit and keeps
the effects and stopping modes of a cell together.

Usage:
    uv run python paper/analysis/build_stageb_power_intervals.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TABLES = Path(__file__).resolve().parents[1] / "results" / "tables"
CELL = ["task", "n", "k"]
KEY = CELL + ["stopping", "effect"]
N_BOOT = 20_000
SEED = 0


def deltas(d: pd.DataFrame) -> pd.DataFrame:
    w = d.pivot_table(index=KEY, columns="threshold_test", values="power_at_match").dropna()
    w["delta"] = w["maxt"] - w["bonferroni"]
    return w.reset_index().merge(d[KEY + ["match_size"]].drop_duplicates(KEY), on=KEY)


def cluster_interval(w: pd.DataFrame) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    groups = [g["delta"].to_numpy() for _, g in w.groupby(CELL)]
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        boot[b] = np.concatenate([groups[i] for i in pick]).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(w["delta"].mean()), float(lo), float(hi)


def report(label: str, w: pd.DataFrame) -> None:
    mean, lo, hi = cluster_interval(w)
    n_cells = w.groupby(CELL).ngroups
    print(
        f"{label}: {n_cells} cells, {len(w)} combinations, mean {mean:+.4f}, "
        f"95% cell-bootstrap [{lo:+.4f}, {hi:+.4f}], range "
        f"{w['delta'].min():+.3f} to {w['delta'].max():+.3f}, "
        f"lower {(w['delta'] < 0).sum()}, higher {(w['delta'] > 0).sum()}"
    )


def main() -> None:
    w = deltas(pd.read_csv(TABLES / "paper_stageb_only_size_matched.csv"))
    report("Stage-B-only, all cells", w)
    report("Stage-B-only, matched at 0.05", w[w["match_size"] >= 0.05])
    report("Stage-B-only, matched below 0.05", w[w["match_size"] < 0.05])
    rep = pd.read_csv(TABLES / "paper_stageb_power_replicates.csv")
    for design, d in rep.groupby("design"):
        report(f"replicates, {design}", deltas(d))


if __name__ == "__main__":
    main()
