#!/usr/bin/env python3
"""Noise robustness: at what noise-feature ratio does CIF break vs RF/ET?

Archived exploratory study; not part of the current paper-facing rebuild path.

Progressively adds noise features (0 to 1000) to three base datasets.
CIF maintains precision longer than RF/ET due to hypothesis testing.

Usage:
    uv run python -m paper.scripts.backlog.experiments.study_noise_feature_robustness
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from paper.scripts.experiments._common import (
    N_SEEDS,
    RANDOM_STATE,
    aggregate_seeds,
    build_baseline,
    build_cif,
    fit_and_evaluate,
    save_results,
    warmup_jit,
)

EXPERIMENT_NAME = "noise_robustness"


def run() -> pd.DataFrame:
    """Run the noise injection experiment."""
    noise_counts = [0, 10, 50, 100, 500, 1000]

    base_configs = [
        {"name": "easy", "n_features": 50, "n_informative": 10, "class_sep": 1.5, "n_samples": 1000},
        {"name": "moderate", "n_features": 50, "n_informative": 10, "class_sep": 0.5, "n_samples": 1000},
        {"name": "many_info", "n_features": 50, "n_informative": 20, "class_sep": 1.0, "n_samples": 1000},
    ]

    rows: list[dict] = []
    for base_cfg in base_configs:
        bname = base_cfg["name"]
        print(f"\n=== Base: {bname} (p={base_cfg['n_features']}, k={base_cfg['n_informative']}) ===")
        for variant in ["cif", "rf", "et"]:
            print(f"\n--- {variant.upper()} ---")
            for p_noise in noise_counts:
                seed_results: list[dict] = []
                for s in range(N_SEEDS):
                    seed = RANDOM_STATE + s
                    X, y = make_classification(
                        n_samples=base_cfg["n_samples"],
                        n_features=base_cfg["n_features"],
                        n_informative=base_cfg["n_informative"],
                        n_redundant=0,
                        class_sep=base_cfg["class_sep"],
                        flip_y=0.0,
                        random_state=seed,
                        shuffle=False,
                    )
                    true_info = list(range(base_cfg["n_informative"]))
                    if p_noise > 0:
                        rng = np.random.RandomState(seed + 100)
                        noise = rng.randn(base_cfg["n_samples"], p_noise)
                        X = np.hstack([X, noise])

                    if variant == "cif":
                        model = build_cif("clf", seed)
                    else:
                        model = build_baseline(variant, "clf", seed)

                    result = fit_and_evaluate(X, y, true_info, model, seed, "clf")
                    seed_results.append(result)

                base_row = {
                    "experiment": EXPERIMENT_NAME,
                    "task": "clf",
                    "variant": variant,
                    "dataset_type": bname,
                    "p_noise": p_noise,
                    "p_total": base_cfg["n_features"] + p_noise,
                    "n_samples": base_cfg["n_samples"],
                    "n_informative": base_cfg["n_informative"],
                }
                agg = aggregate_seeds(seed_results, base_row)
                rows.append(agg)
                print(f"  p_noise={p_noise:4d} (total={base_cfg['n_features']+p_noise:4d}): "
                      f"P@10={agg.get('precision_at_10_mean', 0):.3f} "
                      f"F1={agg.get('f1_at_10_mean', 0):.3f} "
                      f"ds={agg.get('lr_ba_mean', 0):.3f}")

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
