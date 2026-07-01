#!/usr/bin/env python3
"""Max-T multi-selector comparison: Bonferroni vs no-correction vs max-T.

Archived exploratory study; not part of the current paper-facing rebuild path.

Tests three strategies for multiple testing correction in CIF:
  1. selector='mc', adjust_alpha=True  (Bonferroni)
  2. selector='mc', adjust_alpha=False (no correction)
  3. selector=['mc', 'rdc']            (max-T via Westfall-Young)

Runs on all 8 synthetic CLF datasets to compare ranking quality.

Usage:
    uv run python -m paper.scripts.archive.experiments.study_multi_selector_correction
"""

from __future__ import annotations

import sys

import pandas as pd

from paper.scripts.experiments._common import (
    CLF_ALL,
    N_SEEDS,
    RANDOM_STATE,
    aggregate_seeds,
    build_cif,
    fit_and_evaluate,
    format_line,
    save_results,
    warmup_jit,
)

EXPERIMENT_NAME = "max_t_selector"

VARIANTS: list[tuple[str, dict]] = [
    ("mc_bonferroni", dict(selector="mc", adjust_alpha_selector=True)),
    ("mc_no_correction", dict(selector="mc", adjust_alpha_selector=False)),
    ("rdc_bonferroni", dict(selector="rdc", adjust_alpha_selector=True)),
    ("rdc_no_correction", dict(selector="rdc", adjust_alpha_selector=False)),
    ("maxt_mc_rdc", dict(selector=["mc", "rdc"])),
]


def run() -> pd.DataFrame:
    """Run the max-T multi-selector comparison experiment."""
    rows: list[dict] = []

    for ds_fn in CLF_ALL:
        X_base, y_base, info_base, dtype = ds_fn(RANDOM_STATE)
        is_conf = "confounder" in dtype
        n_base = X_base.shape[1] - 20 if is_conf else None
        print(f"\n  {dtype} (n={X_base.shape[0]}, p={X_base.shape[1]})")

        for vname, overrides in VARIANTS:
            seed_results: list[dict] = []
            for s in range(N_SEEDS):
                seed = RANDOM_STATE + s
                X, y, info, _ = ds_fn(seed)
                model = build_cif("clf", seed, **overrides)
                result = fit_and_evaluate(X, y, info, model, seed, "clf", is_conf, n_base)
                seed_results.append(result)

            base_row = {
                "experiment": EXPERIMENT_NAME,
                "task": "clf",
                "dataset_type": dtype,
                "variant": vname,
                "n_features": X_base.shape[1],
                "n_samples": X_base.shape[0],
            }
            agg = aggregate_seeds(seed_results, base_row)
            rows.append(agg)
            print(format_line(vname, agg, is_conf))

    return pd.DataFrame(rows)


def main() -> None:
    """Entry point."""
    sys.stdout.reconfigure(line_buffering=True)
    print(f"=== {EXPERIMENT_NAME} ===")
    warmup_jit()
    df = run()
    path = save_results(df, EXPERIMENT_NAME)
    print(f"\nSaved: {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
