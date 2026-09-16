#!/usr/bin/env python3
"""Summarize the mechanism-matched CIF check (CIF ranked by permutation importance).

Inputs: the per-dataset parquets written by
``paper/benchmark/experiments/cif_permutation_importance_check.py`` (default
location ``../data/cif-perm-check/results/*/``) and the benchmark evaluation
surfaces ``paper/results/{clf,reg}_evaluation.parquet``.

Outputs ``paper/results/tables/paper_cif_permutation_importance_check.csv`` with
the directed pairwise aggregate (as in ``build_benchmark_package_tables``) for
cif_perm vs cforest, cif_perm vs cif, and cif vs cforest on the same datasets,
plus a refit-validation column: the top-10 agreement between the refit's split
ranking and the stored Stage 1 ranking.

Usage:
    uv run python paper/analysis/build_cif_permutation_importance_check.py [--results-dir DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "paper" / "results" / "tables"
STANDARD_K = (5, 10, 25, 50, 100)
SELECTED = {
    "classification": {"cif": "cif__4ee6725d43ec8e5a", "r_cforest": "r_cforest__37be32340835a93b"},
    "regression": {"cif": "cif__40426895fc45caef", "r_cforest": "r_cforest__5c2653de657569d8"},
}
METRIC = {"classification": "balanced_accuracy", "regression": "r2"}


def load_check(results_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(results_dir / "*" / "*.parquet")))
    if not files:
        raise SystemExit(f"no parquets under {results_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def refit_agreement(check: pd.DataFrame, rankings_dir: Path) -> pd.DataFrame:
    rows = []
    for (task, ds, seed, fold), g in check.groupby(["task", "dataset", "seed", "fold_idx"]):
        f = rankings_dir / task / ds / f"{SELECTED[task]['cif']}_seed{seed}.parquet"
        if not f.exists():
            continue
        st = pd.read_parquet(f)
        st = st[st["fold_idx"] == fold]
        if st.empty:
            continue
        stored = np.array(st["feature_ranking"].iloc[0])[:10]
        split = np.array(json.loads(g["split_top10"].iloc[0]))
        rows.append(
            dict(
                task=task,
                dataset=ds,
                seed=seed,
                fold_idx=fold,
                top10_agreement=float(np.mean(stored == split)),
                top10_jaccard=len(set(stored) & set(split)) / len(set(stored) | set(split)),
            )
        )
    return pd.DataFrame(rows)


def dataset_scores(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = df[df["k"].isin(STANDARD_K)]
    return (
        df.groupby(["method_base", "dataset", "downstream_model", "k"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "dataset_mean_score"})
    )


def pairwise(scores: pd.DataFrame, focus: str, baseline: str, task: str) -> dict[str, object]:
    a = scores[scores["method_base"] == focus][
        ["dataset", "downstream_model", "k", "dataset_mean_score"]
    ].rename(columns={"dataset_mean_score": "f"})
    b = scores[scores["method_base"] == baseline][
        ["dataset", "downstream_model", "k", "dataset_mean_score"]
    ].rename(columns={"dataset_mean_score": "b"})
    m = a.merge(b, on=["dataset", "downstream_model", "k"], how="inner")
    m["delta"] = m["f"] - m["b"]
    by = m.groupby("dataset", as_index=False).agg(
        mean_delta=("delta", "mean"), n_cells=("delta", "size")
    )
    return dict(
        task=task,
        focus_method=focus,
        baseline=baseline,
        n_datasets=int(by["dataset"].nunique()),
        mean_delta=float(by["mean_delta"].mean()),
        median_delta=float(by["mean_delta"].median()),
        wins=int((by["mean_delta"] > 0).sum()),
        losses=int((by["mean_delta"] < 0).sum()),
        ties=int((by["mean_delta"] == 0).sum()),
        datasets=";".join(sorted(by["dataset"])),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir", type=Path, default=ROOT.parent / "data" / "cif-perm-check" / "results"
    )
    ap.add_argument("--rankings-dir", type=Path, default=ROOT.parent / "data" / "rankings")
    ap.add_argument(
        "--output", type=Path, default=TABLES / "paper_cif_permutation_importance_check.csv"
    )
    args = ap.parse_args()
    check = load_check(args.results_dir)
    agree = refit_agreement(check, args.rankings_dir)
    rows = []
    for task in ("classification", "regression"):
        sub = check[check["task"] == task]
        if sub.empty:
            continue
        metric = METRIC[task]
        ev = pd.read_parquet(
            ROOT
            / "paper"
            / "results"
            / f"{'clf' if task == 'classification' else 'reg'}_evaluation.parquet"
        )
        ev = ev[
            (ev["dataset_type"] == "real")
            & ev["method_id"].isin(SELECTED[task].values())
            & ev["dataset"].isin(sub["dataset"].unique())
        ]
        scores = pd.concat(
            [dataset_scores(sub, metric), dataset_scores(ev, metric)], ignore_index=True
        )
        ag = agree[agree["task"] == task]
        for focus, base in (("cif_perm", "r_cforest"), ("cif_perm", "cif"), ("cif", "r_cforest")):
            r = pairwise(scores, focus, base, task)
            r["metric"] = metric
            r["refit_top10_agreement_mean"] = (
                float(ag["top10_agreement"].mean()) if not ag.empty else np.nan
            )
            r["refit_top10_jaccard_mean"] = (
                float(ag["top10_jaccard"].mean()) if not ag.empty else np.nan
            )
            r["refit_folds_compared"] = int(len(ag))
            rows.append(r)
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.drop(columns=["datasets"]).round(4).to_string(index=False))
    print("datasets:", sorted(check["dataset"].unique()))
    if not agree.empty:
        print(
            agree.groupby(["task", "dataset"])[["top10_agreement", "top10_jaccard"]]
            .mean()
            .round(3)
            .to_string()
        )


if __name__ == "__main__":
    main()
