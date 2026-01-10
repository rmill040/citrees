"""Timing experiments for citrees hyperparameter configurations.

This script benchmarks different citrees configurations to understand
the performance impact of various hyperparameters:
- n_resamples: standard ~1k vs auto
- early_stopping: enabled vs disabled
- feature_muting: enabled vs disabled
- feature_scanning / threshold_scanning
- threshold_method: exact vs histogram

Results are saved to paper/results/timing_results.parquet
"""
import time
from itertools import product
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
)

N = 500
P = 500
NOISE = 2
RANDOM_STATE = 1718
N_REPEATS = 10


def tree_splits(tree: Any, splits: List[Any]) -> None:
    """Recursively get feature splits from conditional inference model.

    Parameters
    ----------
    tree : Any
        Fitted tree model.

    splits : List[Any]
        Index of features used for splitting.
    """
    if tree.get("value", None) is None:
        splits.append(tree["feature"])
        tree_splits(tree["left_child"], splits)
        tree_splits(tree["right_child"], splits)


def main() -> None:
    """Run timing experiments across hyperparameter configurations."""
    results = []

    for i in range(1, N_REPEATS + 1):
        print(f"=== Repeat {i}/{N_REPEATS} ===")
        X, y = make_friedman1(
            n_samples=N + 100,
            n_features=P,
            noise=NOISE,
            random_state=RANDOM_STATE + i,
        )
        y_binary = (y >= np.median(y)).astype(int)

        ######################
        # CLASSIFIER - TREES #
        ######################

        adjust_alpha_selector = [True, False]
        adjust_alpha_splitter = [True, False]
        n_resamples_selector = ["auto", 1_000]
        n_resamples_splitter = ["auto", 1_000]
        early_stopping_selector = [True, False]
        early_stopping_splitter = [True, False]
        threshold_method = ["histogram", "exact"]
        feature_muting = [True, False]
        feature_scanning = [True, False]
        threshold_scanning = [True, False]

        # Create all combinations
        configs = list(
            product(*[
                adjust_alpha_selector,
                adjust_alpha_splitter,
                n_resamples_selector,
                n_resamples_splitter,
                early_stopping_selector,
                early_stopping_splitter,
                threshold_method,
                feature_muting,
                feature_scanning,
                threshold_scanning,
            ])
        )
        n_configs = len(configs)
        for config in configs:
            hps = {
                "adjust_alpha_selector": config[0],
                "adjust_alpha_splitter": config[1],
                "n_resamples_selector": config[2],
                "n_resamples_splitter": config[3],
                "early_stopping_selector": config[4],
                "early_stopping_splitter": config[5],
                "threshold_method": config[6],
                "feature_muting": config[7],
                "feature_scanning": config[8],
                "threshold_scanning": config[9],
                "max_thresholds": 128 if config[6] == "histogram" else None,
                "random_state": RANDOM_STATE + i,
            }

            clf = ConditionalInferenceTreeClassifier(**hps)
                                        
            # Time
            tic = time.time()
            clf.fit(X[:N], y_binary[:N])
            toc = time.time()

            # Get splits
            splits = []
            tree_splits(clf.tree_, splits)

            # Save results
            results.append({
                "model": "cit",
                "repeat": i,
                "time": toc - tic,
                "accuracy": np.mean(clf.predict(X[N:]) == y_binary[N:]),
                "n_splits": len(splits),
                "correct_features": sum(1 for s in splits if s < 5),  # Friedman1 has 5 informative
                **hps,
            })

        # Baseline: Decision Tree
        clf = DecisionTreeClassifier(random_state=RANDOM_STATE + i)
        tic = time.time()
        clf.fit(X[:N], y_binary[:N])
        toc = time.time()
        results.append({
            "model": "dt",
            "repeat": i,
            "time": toc - tic,
            "accuracy": np.mean(clf.predict(X[N:]) == y_binary[N:]),
            "n_splits": clf.tree_.node_count,
            "correct_features": None,
        })

        # Baseline: Random Forest
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE + i, n_jobs=-1)
        tic = time.time()
        clf.fit(X[:N], y_binary[:N])
        toc = time.time()
        results.append({
            "model": "rf",
            "repeat": i,
            "time": toc - tic,
            "accuracy": np.mean(clf.predict(X[N:]) == y_binary[N:]),
            "n_splits": None,
            "correct_features": None,
        })

    # Save results
    df = pd.DataFrame(results)
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "timing_results.parquet"
    df.to_parquet(output_path)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n=== TIMING SUMMARY ===")
    summary = df.groupby("model")["time"].agg(["mean", "std", "min", "max"])
    print(summary.round(3))

    print("\n=== ACCURACY SUMMARY ===")
    summary = df.groupby("model")["accuracy"].agg(["mean", "std"])
    print(summary.round(3))


if __name__ == "__main__":
    main()
