"""Synthetic data experiments with controlled informative features.

This script evaluates feature selection methods on synthetic datasets with:
- Varying total features (p = 50, 100, 500, 1000)
- Varying informative features (k = 5, 10, 20)
- Varying sample sizes (n = 200, 500, 1000)
- Varying noise levels
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from citrees import ConditionalInferenceForestClassifier, ConditionalInferenceTreeClassifier


@dataclass
class SyntheticConfig:
    """Configuration for synthetic experiment."""

    n_samples: int
    n_features: int
    n_informative: int
    n_redundant: int = 0
    n_clusters_per_class: int = 2
    class_sep: float = 1.0
    random_state: int = 42


# Feature selection methods
def get_feature_rankers() -> dict[str, Any]:
    """Return feature ranking methods to compare."""
    return {
        # Conditional Inference Trees
        "citree": ConditionalInferenceTreeClassifier(
            selector="mc",
            n_resamples_selector="auto",
            alpha_selector=0.05,
            verbose=0,
            random_state=42,
        ),
        "ciforest": ConditionalInferenceForestClassifier(
            n_estimators=100,
            selector="mc",
            n_resamples_selector=None,
            alpha_selector=0.05,
            n_jobs=-1,
            verbose=0,
            random_state=42,
        ),
        # Standard tree-based methods
        "rf": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "et": ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "dt": DecisionTreeClassifier(random_state=42),
        # Gradient boosting methods
        "xgb": XGBClassifier(n_estimators=100, n_jobs=-1, random_state=42, verbosity=0),
        "lgbm": LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=42, verbose=-1),
    }


def rank_features(method_name: str, model: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Get feature rankings from a model."""
    model.fit(X, y)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        # Fallback for models without feature_importances_
        importances = np.zeros(X.shape[1])

    # Return indices sorted by importance (highest first)
    return np.argsort(importances)[::-1]


def evaluate_ranking(
    ranking: np.ndarray, true_informative: np.ndarray, k_values: list[int]
) -> dict[str, float]:
    """Evaluate how well the ranking recovers true informative features."""
    results = {}

    for k in k_values:
        top_k = set(ranking[:k])
        true_set = set(true_informative)

        # Precision@k: what fraction of top-k are truly informative
        precision = len(top_k & true_set) / k if k > 0 else 0

        # Recall@k: what fraction of true informative are in top-k
        recall = len(top_k & true_set) / len(true_set) if len(true_set) > 0 else 0

        results[f"precision@{k}"] = precision
        results[f"recall@{k}"] = recall

    return results


def downstream_accuracy(
    X: np.ndarray, y: np.ndarray, ranking: np.ndarray, n_features_to_use: int
) -> dict[str, float]:
    """Evaluate downstream classification with selected features."""
    selected_features = ranking[:n_features_to_use]
    X_selected = X[:, selected_features]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for train_idx, test_idx in cv.split(X_selected, y):
        X_train, y_train = X_selected[train_idx], y[train_idx]
        X_test, y_test = X_selected[test_idx], y[test_idx]

        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
        pipe.fit(X_train, y_train)
        accuracies.append(pipe.score(X_test, y_test))

    return {"accuracy_mean": np.mean(accuracies), "accuracy_std": np.std(accuracies)}


def run_single_experiment(config: SyntheticConfig) -> dict[str, Any]:
    """Run a single synthetic experiment."""
    # Generate data
    X, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_informative=config.n_informative,
        n_redundant=config.n_redundant,
        n_clusters_per_class=config.n_clusters_per_class,
        class_sep=config.class_sep,
        random_state=config.random_state,
        shuffle=False,  # Keep informative features at the start
    )

    # True informative features are indices 0 to n_informative-1
    true_informative = np.arange(config.n_informative)

    # Now shuffle to hide the true features
    perm = np.random.RandomState(config.random_state).permutation(config.n_features)
    X = X[:, perm]
    true_informative = np.where(np.isin(perm, true_informative))[0]

    results = {
        "n_samples": config.n_samples,
        "n_features": config.n_features,
        "n_informative": config.n_informative,
        "class_sep": config.class_sep,
        "random_state": config.random_state,
    }

    # Evaluate each feature selection method
    rankers = get_feature_rankers()
    k_values = [5, 10, 20, config.n_informative]

    for method_name, model in rankers.items():
        try:
            ranking = rank_features(method_name, model, X, y)

            # Ranking quality metrics
            ranking_metrics = evaluate_ranking(ranking, true_informative, k_values)
            for k, v in ranking_metrics.items():
                results[f"{method_name}_{k}"] = v

            # Downstream accuracy with true number of informative features
            downstream = downstream_accuracy(X, y, ranking, config.n_informative)
            results[f"{method_name}_downstream_acc_mean"] = downstream["accuracy_mean"]
            results[f"{method_name}_downstream_acc_std"] = downstream["accuracy_std"]

        except Exception as e:
            logger.error(f"Method {method_name} failed: {e}")
            for k in k_values:
                results[f"{method_name}_precision@{k}"] = np.nan
                results[f"{method_name}_recall@{k}"] = np.nan
            results[f"{method_name}_downstream_acc_mean"] = np.nan
            results[f"{method_name}_downstream_acc_std"] = np.nan

    return results


def main():
    """Run all synthetic experiments."""
    # Experiment grid
    configs = []

    # Varying dimensions
    for n_features in [50, 100, 500, 1000]:
        for n_informative in [5, 10, 20]:
            if n_informative < n_features:
                for n_samples in [200, 500, 1000]:
                    for class_sep in [0.5, 1.0, 2.0]:  # Easy to hard
                        for seed in range(5):  # 5 repetitions
                            configs.append(
                                SyntheticConfig(
                                    n_samples=n_samples,
                                    n_features=n_features,
                                    n_informative=n_informative,
                                    class_sep=class_sep,
                                    random_state=42 + seed,
                                )
                            )

    logger.info(f"Running {len(configs)} synthetic experiments")

    # Run experiments in parallel
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_single_experiment)(config) for config in configs
    )

    # Save results
    df = pd.DataFrame(results)
    output_path = Path(__file__).parent.parent / "results" / "synthetic_experiments.parquet"
    output_path.parent.mkdir(exist_ok=True)
    df.to_parquet(output_path)
    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n=== SUMMARY ===")
    precision_cols = [c for c in df.columns if c.endswith("_precision@10")]
    if precision_cols:
        print(df.groupby(["n_features", "n_informative"])[precision_cols].mean().round(3))


if __name__ == "__main__":
    main()
