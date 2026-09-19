"""Head-to-head forest timing protocol used for the JSS performance tables.

Times 100-tree classification forests, fit-only, in a fresh child process per
fit, on one 32-core host: the recommended ``citrees`` configuration, the same
without the Bonferroni adjustment over candidate thresholds
(``citrees_splitgate_nobonf``), the same with the max-type Stage B test
(``citrees_maxt``), ``partykit::cforest`` at its defaults on 1 and 32 cores
(``partykit_sqrt_1``, ``partykit_sqrt_32``), and a scikit-learn random forest
(``sklearn_rf``). Trees have depth at most 3, minimum split 20, minimum leaf 7,
alpha 0.05. Cells: synthetic Gaussian predictors with a weak or strong planted
signal at n in {500, 2000, 8000, 32000} x p in {20, 100}, plus p = 500 at
n in {500, 2000}, and the ten real benchmark datasets with at least 500
observations (parquet files under ``/root/paper-data/classification/real``).
Two repeats per cell, one on letter, isolet, and gisette; a fit is censored at
``FIT_TIMEOUT`` seconds (default 1200). Since 2026-09-19 each citrees fit is
timed as the second of two identical fits, so compilation is excluded.

Usage on a host with the pinned image (``/out`` writable, ``/app`` the repo):

    python -m paper.jss.replication.head_to_head_timing <shard-index> <num-shards>

Environment: ``REAL`` (comma list of real datasets), ``NS`` (sample sizes),
``CONFIGS`` (configurations), ``REAL_ONLY``, ``FIT_TIMEOUT``, ``FULL_POOL_WARMUP``.
Output: ``/out/results_<shard>.json``, one record per fit with ``fit_s``,
``n``, ``p``, ``classes``, and the cell keys; the 2026-09-13 run is summarized
in ``paper/results/tables/paper_h2h_rerun_summary.csv``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPS = 2
MAX_DEPTH, MIN_SPLIT, MIN_LEAF, TREES, ALPHA = 3, 20, 7, 100, 0.05
DEFAULT_REAL = "letter,gamma,pendigits,isolet,musk,gisette,page-blocks,spam,madelon,vowel-context"
DEFAULT_CONFIGS = (
    "citrees,citrees_splitgate_nobonf,citrees_maxt,partykit_sqrt_1,partykit_sqrt_32,sklearn_rf"
)
DEFAULT_NS = "500,2000,8000,32000"

CHILD = r"""
import os, sys, time, json, numpy as np, pandas as pd
kind, name, n, p, signal, config = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
MAX_DEPTH, MIN_SPLIT, MIN_LEAF, TREES, ALPHA = 3, 20, 7, 100, 0.05
if kind == "synthetic":
    rng = np.random.default_rng(1718 + n + p)
    X = rng.standard_normal((n, p)); active = min(5, p)
    coef = np.linspace(1.0, 0.25, active) if signal == "weak" else np.ones(active)
    noise = 1.0 if signal == "weak" else 0.5
    latent = X[:, :active] @ coef + noise * rng.standard_normal(n)
    y = (latent >= np.median(latent)).astype(np.int64)
else:
    d = pd.read_parquet(f"/root/paper-data/classification/real/clf_{name}.parquet")
    y = pd.factorize(d["y"])[0].astype(np.int64); X = d.drop(columns=["y"]).to_numpy(dtype=np.float64)
    n, p = X.shape
wi = np.random.default_rng(0).permutation(n)[:150]  # warm-up rows: random, so both classes are present
t0 = time.time()
if config.startswith("citrees"):
    from citrees import ConditionalInferenceForestClassifier as F, ConditionalInferenceTreeClassifier as T
    rec = dict(selector="mc", alpha_selector=ALPHA, adjust_alpha_selector=True, n_resamples_selector="minimum",
               early_stopping_selector="adaptive", early_stopping_confidence_selector=0.95, n_resamples_splitter="minimum",
               adjust_alpha_splitter=True, early_stopping_splitter="adaptive", early_stopping_confidence_splitter=0.95,
               feature_muting=True, feature_scanning=True, threshold_scanning=True, threshold_method="histogram",
               max_thresholds=256, max_depth=MAX_DEPTH, min_samples_split=MIN_SPLIT, min_samples_leaf=MIN_LEAF, verbose=0)
    extra = {}
    if config == "citrees_nosplitgate": rec.update(n_resamples_splitter=None, early_stopping_splitter=None, adjust_alpha_splitter=False)
    if config == "citrees_splitgate_nobonf": rec.update(adjust_alpha_splitter=False)
    if config == "citrees_allfeatures": extra = {"max_features": None}
    if config == "citrees_maxt": rec.update(threshold_test="maxt")
    warm_trees = 64 if os.environ.get("FULL_POOL_WARMUP") else 3
    F(**rec, n_estimators=warm_trees, n_jobs=-1, random_state=1).fit(X[wi][:, :min(p, 8)], y[wi])  # warm-up: JIT + worker pool
    # one identical full fit first: the small warm-up does not reach kernels selected by size, so the
    # first full fit on a host can include their compilation (found 2026-09-18 in the reference study)
    F(**rec, n_estimators=TREES, n_jobs=-1, random_state=0, **extra).fit(X, y)
    t = time.time(); F(**rec, n_estimators=TREES, n_jobs=-1, random_state=0, **extra).fit(X, y); el = time.time() - t
elif config.startswith("partykit"):
    cores = int(config.rsplit("_", 1)[1])
    from paper.benchmark.pipeline.r_methods import r_cforest_fit
    kw = dict(task="classification", teststat="quadratic", testtype=("Bonferroni", "MonteCarlo"), mincriterion=1 - ALPHA,
              nresample=999, mtry="sqrt", maxdepth=MAX_DEPTH, minsplit=MIN_SPLIT, minbucket=MIN_LEAF, replace=True, fraction=1.0, cores=cores)
    r_cforest_fit(X[wi][:, :min(p, 8)], y[wi], ntree=3, random_state=1, **kw)  # warm-up: R session
    t = time.time(); r_cforest_fit(X, y, ntree=TREES, random_state=0, **kw); el = time.time() - t
else:
    from sklearn.ensemble import RandomForestClassifier as RF
    t = time.time(); RF(n_estimators=TREES, max_depth=MAX_DEPTH, min_samples_split=MIN_SPLIT, min_samples_leaf=MIN_LEAF, n_jobs=-1, random_state=0).fit(X, y); el = time.time() - t
print(json.dumps({"fit_s": round(el, 2), "n": int(n), "p": int(p), "classes": int(len(np.unique(y)))}))
"""


def build_cells(
    real: str | None = None, ns: str | None = None, real_only: bool = False
) -> list[tuple]:
    """The ordered list of (kind, name, n, p, signal) cells."""
    real_names = (real or os.environ.get("REAL", DEFAULT_REAL)).split(",")
    sizes = tuple(int(x) for x in (ns or os.environ.get("NS", DEFAULT_NS)).split(","))
    skip_syn = real_only or bool(os.environ.get("REAL_ONLY"))
    cells: list[tuple] = []
    for n in () if skip_syn else sizes:
        for p in (20, 100):
            for sig in ("weak", "strong"):
                cells.append(("synthetic", "syn", n, p, sig))
    for n in () if skip_syn else (500, 2000):
        for sig in ("weak", "strong"):
            cells.append(("synthetic", "syn", n, 500, sig))
    for name in real_names:
        cells.append(("real", name, 0, 0, "real"))
    return cells


def configs() -> list[str]:
    return os.environ.get("CONFIGS", DEFAULT_CONFIGS).split(",")


def main() -> None:
    idx, num = int(sys.argv[1]), int(sys.argv[2])
    Path("/out/child.py").write_text(CHILD)
    jobs = [(c, cfg) for c in build_cells() for cfg in configs()]
    mine = jobs[idx::num]
    print(f"box {idx}/{num}: {len(mine)} fits x {REPS} reps", flush=True)
    res = []
    for (kind, name, n, p, sig), cfg in mine:
        reps = 1 if (kind == "real" and name in ("gisette", "isolet", "letter")) else REPS
        for r in range(reps):
            try:
                o = subprocess.run(
                    [sys.executable, "/out/child.py", kind, name, str(n), str(p), sig, cfg],
                    capture_output=True,
                    text=True,
                    cwd="/app",
                    timeout=int(os.environ.get("FIT_TIMEOUT", "1200")),
                )
                d = json.loads(o.stdout.strip().splitlines()[-1])
            except subprocess.TimeoutExpired:
                d = {"fit_s": None, "timeout": True}
            except Exception:
                d = {"fit_s": None, "err": (o.stderr[-300:] if "o" in dir() else "?")}
            d.update(dict(kind=kind, dataset=name, n_req=n, p_req=p, signal=sig, config=cfg, rep=r))
            res.append(d)
            print(
                f"{kind:9s} {name:12s} n={d.get('n', n):>6} p={d.get('p', p):>5} {sig:6s} {cfg:17s} rep{r}: {d.get('fit_s')}{' TIMEOUT' if d.get('timeout') else ''} {d.get('err', '')[:100]}",
                flush=True,
            )
            Path(f"/out/results_{idx}.json").write_text(json.dumps(res, indent=1))
    print("H2H_DONE", flush=True)


if __name__ == "__main__":
    main()
