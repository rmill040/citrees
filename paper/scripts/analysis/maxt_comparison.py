"""Block 12: Max-T vs Bonferroni comparison.

Tests the principled alternative to Bonferroni correction:
  1. selector='mc', adjust_alpha=True  (Bonferroni — current default)
  2. selector='mc', adjust_alpha=False (no correction — ablation finding)
  3. selector=['mc', 'rdc'] (max-T via Westfall-Young — principled fix)

Runs on 8 synthetic CLF datasets, 5 seeds, CIF with 100 estimators.

Usage:
    uv run python paper/scripts/analysis/maxt_comparison.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_friedman1

from citrees import ConditionalInferenceForestClassifier

RANDOM_STATE = 1718
N_SEEDS = 5
N_ESTIMATORS = 100
K = 10
RESULTS_DIR = Path("paper/results/tables")


def precision_at_k(ranking, info, k):
    return len(set(ranking[:k]) & set(info)) / k


def recall_at_k(ranking, info, k):
    return len(set(ranking[:k]) & set(info)) / len(info) if info else 0.0


def f1_at_k(ranking, info, k):
    p, r = precision_at_k(ranking, info, k), recall_at_k(ranking, info, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _downstream_clf(X, y, ranking, k, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    sel = list(ranking[:k])
    if not sel:
        return {"lr_ba": 0.0}
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    scores = cross_val_score(lr, X[:, sel], y, cv=5, scoring="balanced_accuracy")
    return {"lr_ba": float(scores.mean())}


# --- Dataset factories (same as comprehensive_ablation.py) ---

def clf_standard_easy(seed):
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=10,
                               n_redundant=0, n_clusters_per_class=1, class_sep=1.5, random_state=seed)
    return X, y, list(range(10)), "clf_standard_easy"

def clf_standard_hard(seed):
    X, y = make_classification(n_samples=200, n_features=1000, n_informative=5,
                               n_redundant=0, n_clusters_per_class=1, class_sep=1.0, random_state=seed)
    return X, y, list(range(5)), "clf_standard_hard"

def clf_weak_signal(seed):
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=10,
                               n_redundant=0, n_clusters_per_class=1, class_sep=0.3, random_state=seed)
    return X, y, list(range(10)), "clf_weak_signal"

def clf_nonlinear(seed):
    X, y_cont = make_friedman1(n_samples=1000, n_features=100, noise=1.0, random_state=seed)
    y = (y_cont > np.median(y_cont)).astype(int)
    return X, y, list(range(5)), "clf_nonlinear"

def clf_toeplitz(seed):
    rng = np.random.default_rng(seed)
    n, p, k = 1000, 100, 10
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = 0.9 ** abs(i - j)
    X = rng.multivariate_normal(np.zeros(p), cov, size=n)
    beta = np.zeros(p); beta[:k] = rng.standard_normal(k)
    y = (X @ beta > 0).astype(int)
    return X, y, list(range(k)), "clf_toeplitz"

def clf_confounder(seed):
    X, y = make_classification(n_samples=1000, n_features=50, n_informative=10,
                               n_redundant=0, n_clusters_per_class=1, class_sep=1.0, random_state=seed)
    rng = np.random.default_rng(seed)
    confounders = np.zeros((1000, 20))
    for i in range(20):
        confounders[:, i] = y * 0.5 + rng.standard_normal(1000) * 0.5
    X = np.hstack([X, confounders])
    return X, y, list(range(10)), "clf_confounder"

def clf_bias(seed):
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=10,
                               n_redundant=0, n_clusters_per_class=1, class_sep=1.0, random_state=seed)
    rng = np.random.default_rng(seed + 1000)
    for i in range(5):
        X[:, 90 + i] = rng.choice(100, size=1000).astype(float)
    return X, y, list(range(10)), "clf_bias"

def clf_redundant(seed):
    rng = np.random.default_rng(seed)
    n, p, k = 1000, 100, 10
    X = rng.standard_normal((n, p))
    beta = np.zeros(p); beta[:k] = rng.standard_normal(k)
    y = (X @ beta > 0).astype(int)
    for i in range(k, k + 10):
        X[:, i] = X[:, i % k] + rng.standard_normal(n) * 0.1
    return X, y, list(range(k)), "clf_redundant"


ALL_DATASETS = [clf_standard_easy, clf_standard_hard, clf_weak_signal, clf_nonlinear,
                clf_toeplitz, clf_confounder, clf_bias, clf_redundant]

# --- CIF configs to compare ---
CONFIGS = [
    ("bonferroni_mc", dict(
        selector="mc", adjust_alpha_selector=True, alpha_selector=0.05,
        early_stopping_selector="adaptive", n_resamples_selector="minimum",
    )),
    ("no_bonf_mc_a05", dict(
        selector="mc", adjust_alpha_selector=False, alpha_selector=0.05,
        early_stopping_selector="adaptive", n_resamples_selector="minimum",
    )),
    ("no_bonf_mc_a10", dict(
        selector="mc", adjust_alpha_selector=False, alpha_selector=0.10,
        early_stopping_selector="adaptive", n_resamples_selector="minimum",
    )),
    ("maxt_mc_rdc", dict(
        selector=["mc", "rdc"], adjust_alpha_selector=True, alpha_selector=0.05,
        early_stopping_selector="adaptive", n_resamples_selector="minimum",
    )),
    ("maxt_mc_rdc_a10", dict(
        selector=["mc", "rdc"], adjust_alpha_selector=True, alpha_selector=0.10,
        early_stopping_selector="adaptive", n_resamples_selector="minimum",
    )),
]


def main():
    print("=" * 70)
    print("BLOCK 12: MAX-T vs BONFERRONI COMPARISON")
    print("=" * 70)
    print(f"  seeds={N_SEEDS}, estimators={N_ESTIMATORS}")
    print(f"  Configs: {[c[0] for c in CONFIGS]}")
    print()

    # Warmup JIT
    print("Warming up Numba JIT...")
    X_w, y_w, _, _ = clf_standard_easy(0)
    m = ConditionalInferenceForestClassifier(n_estimators=2, n_resamples_selector="minimum",
                                             random_state=0)
    m.fit(X_w[:50], y_w[:50])
    print("JIT warmup complete.\n")

    rows = []

    for ds_fn in ALL_DATASETS:
        X0, y0, info0, dtype = ds_fn(RANDOM_STATE)
        print(f"  {dtype} (n={X0.shape[0]}, p={X0.shape[1]})")

        for vname, overrides in CONFIGS:
            seed_results = []
            for s in range(N_SEEDS):
                seed = RANDOM_STATE + s
                X, y, info, _ = ds_fn(seed)

                params = dict(
                    n_estimators=N_ESTIMATORS,
                    splitter="gini",
                    alpha_splitter=0.05,
                    n_resamples_splitter="minimum",
                    early_stopping_splitter="adaptive",
                    threshold_method="histogram",
                    max_thresholds=256,
                    n_jobs=-1,
                    random_state=seed,
                    **overrides,
                )

                t0 = time.perf_counter()
                model = ConditionalInferenceForestClassifier(**params)
                model.fit(X, y)
                elapsed = time.perf_counter() - t0

                importances = model.feature_importances_
                ranking = list(np.argsort(importances)[::-1])

                result = {
                    "elapsed_seconds": elapsed,
                    "precision_at_10": precision_at_k(ranking, info, K),
                    "recall_at_10": recall_at_k(ranking, info, K),
                    "f1_at_10": f1_at_k(ranking, info, K),
                    "n_features_used": int((importances > 0).sum()),
                }
                # Tree depth
                depths = [t.max_depth for t in model.estimators_
                          if hasattr(t, "max_depth") and t.max_depth is not None]
                if depths:
                    result["mean_depth"] = float(np.mean(depths))

                result.update(_downstream_clf(X, y, ranking, K, seed))
                seed_results.append(result)

            # Aggregate
            agg = {"block": "maxt_comparison", "dataset_type": dtype, "variant": vname,
                   "n_features": X0.shape[1], "n_samples": X0.shape[0]}
            for key in seed_results[0]:
                vals = [r[key] for r in seed_results if key in r]
                if vals and isinstance(vals[0], (int, float)):
                    agg[f"{key}_mean"] = float(np.mean(vals))
                    agg[f"{key}_std"] = float(np.std(vals))
            rows.append(agg)

            p10 = agg.get("precision_at_10_mean", 0)
            ds = agg.get("lr_ba_mean", 0)
            depth = agg.get("mean_depth_mean", 0)
            feats = agg.get("n_features_used_mean", 0)
            t = agg.get("elapsed_seconds_mean", 0)
            print(f"    {vname:20s}: P@10={p10:.3f} ds={ds:.3f} "
                  f"depth={depth:.1f} feats={feats:.1f} t={t:.1f}s")

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "ablation_block12_maxt.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows → {out}")


if __name__ == "__main__":
    main()
