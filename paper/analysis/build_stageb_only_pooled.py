#!/usr/bin/env python3
"""Pool the Stage-B-only shards and compute size-matched power.

Reads every ``paper/jss/results/stageb-only-seed*-*/stageb_only.csv``, pools
seeds by summing splits and replicates per (task, n, k, threshold_test,
stopping, nominal_alpha, effect), attaches Clopper-Pearson intervals, and for
each (task, n, k, threshold_test, stopping, effect) interpolates power against
realized size at a target size of 0.05. Where a test's realized size never
reaches 0.05 within the nominal grid, the largest realized size shared by both
tests in that cell is used instead and recorded in ``match_size``.

Usage:
    uv run python paper/analysis/build_stageb_only_pooled.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables" / "paper_stageb_only_pooled.csv"
TARGET = 0.05
CELL = ["task", "n", "k", "threshold_test", "stopping"]


def clopper_pearson(x: int, n: int, level: float = 0.95) -> tuple[float, float]:
    lo = 0.0 if x == 0 else float(beta.ppf((1 - level) / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(1 - (1 - level) / 2, x + 1, n - x))
    return lo, hi


def pool() -> pd.DataFrame:
    files = sorted(
        glob.glob(str(ROOT / "jss" / "results" / "stageb-only-seed*-*" / "stageb_only.csv"))
    )
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["seed_dir"] = Path(f).parent.name
        d["splits"] = (d["split_rate"] * d["replicates"]).round().astype(int)
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    g = raw.groupby(CELL + ["nominal_alpha", "effect"], as_index=False).agg(
        splits=("splits", "sum"), replicates=("replicates", "sum"), n_shards=("seed_dir", "nunique")
    )
    g["rate"] = g["splits"] / g["replicates"]
    ci = np.array(
        [clopper_pearson(int(x), int(m)) for x, m in zip(g["splits"], g["replicates"], strict=True)]
    )
    g["ci_low"], g["ci_high"] = ci[:, 0], ci[:, 1]
    return g, len(files)


def size_matched(g: pd.DataFrame) -> pd.DataFrame:
    """Interpolate power against realized size per cell and effect."""
    rows = []
    for key, cell in g.groupby(CELL[:3]):  # task, n, k: match the two tests within a cell
        sizes = {}
        for test, sub in cell.groupby("threshold_test"):
            for stopping, ss in sub.groupby("stopping"):
                sz = ss[ss["effect"] == 0].sort_values("nominal_alpha")
                sizes[(test, stopping)] = sz.set_index("nominal_alpha")["rate"]
        for stopping in cell["stopping"].unique():
            tests = [t for t in cell["threshold_test"].unique() if (t, stopping) in sizes]
            maxima = [sizes[(t, stopping)].max() for t in tests]
            match = TARGET if min(maxima) >= TARGET else float(min(maxima))
            for test in tests:
                sz = sizes[(test, stopping)]
                sub = cell[(cell["threshold_test"] == test) & (cell["stopping"] == stopping)]
                for effect, pe in sub[sub["effect"] > 0].groupby("effect"):
                    pw = pe.set_index("nominal_alpha")["rate"].reindex(sz.index)
                    order = np.argsort(sz.values)
                    x, y = sz.values[order], pw.values[order]
                    ok = ~np.isnan(y)
                    x, y = x[ok], y[ok]
                    if len(x) == 0:
                        continue
                    if match <= x.min():
                        p = float(y[0])
                        note = "below smallest realized size; smallest-level power reported"
                    elif match >= x.max():
                        p = float(y[-1])
                        note = "at largest realized size" if match < TARGET else ""
                    else:
                        p = float(np.interp(match, x, y))
                        note = "" if match == TARGET else "matched at largest common realized size"
                    rows.append(
                        dict(
                            task=key[0],
                            n=key[1],
                            k=key[2],
                            threshold_test=test,
                            stopping=stopping,
                            effect=effect,
                            match_size=match,
                            power_at_match=p,
                            max_realized_size=float(sz.max()),
                            note=note,
                        )
                    )
    return pd.DataFrame(rows)


def main() -> None:
    g, nfiles = pool()
    g["study"] = "stageb_only_pooled"
    m = size_matched(g)
    g.to_csv(OUT, index=False)
    m.to_csv(OUT.with_name("paper_stageb_only_size_matched.csv"), index=False)
    print(f"{nfiles} shards -> {len(g)} pooled rows, {len(m)} size-matched rows")
    sz = g[(g["effect"] == 0) & (g["nominal_alpha"] == 0.05)]
    print(sz.groupby(["task", "threshold_test", "stopping"])["rate"].agg(["min", "max"]).round(3))
    piv = m.pivot_table(
        index=["task", "stopping", "effect"],
        columns="threshold_test",
        values="power_at_match",
        aggfunc="mean",
    )
    piv["delta_maxt_minus_bonf"] = piv["maxt"] - piv["bonferroni"]
    print(piv.round(3))
    print(m["note"].value_counts())


if __name__ == "__main__":
    main()
