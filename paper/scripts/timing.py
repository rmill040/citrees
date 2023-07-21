import time
from itertools import product
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

"""
Compare:
    * n_permutations -- standard ~1k vs auto
    * early stopping vs not
    * feature muting vs not
    * feature scanning/threshold scanning
    * threshold_method
"""


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
    """ADD HERE."""

    results = []
    for i in range(1, 11):
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
                "params": clf.get_params(),
                "time": toc - tic,
                "accuracy": np.mean(clf.predict(X[N:]) == y_binary[N:]),
                "splits": splits,
            })

        # # Decision tree classifier
        # clf = DecisionTreeClassifier()

        # tic = time.time()
        # clf.fit(X[:N], y_binary[:N])
        # toc = time.time()
        # results.append({
        #     "model": "dt",
        #     "time": toc - tic,
        #     "accuracy": np.mean(clf.predict(X[N:]) == y_binary[N:])
        # })
        # import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
