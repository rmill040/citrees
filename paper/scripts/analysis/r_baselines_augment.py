"""R-only augmentation: fill in r_ctree/r_cforest baselines for Block 10 & 11.

Runs r_ctree (Bonferroni + MonteCarlo) and r_cforest (Bonferroni + MonteCarlo)
on the same datasets used in comprehensive_ablation.py blocks 10 and 11.

Usage:
    uv run python paper/scripts/analysis/r_baselines_augment.py --blocks 10
    uv run python paper/scripts/analysis/r_baselines_augment.py --blocks 11
    uv run python paper/scripts/analysis/r_baselines_augment.py --blocks 10,11
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
    make_friedman1,
    make_regression,
)
from sklearn.preprocessing import StandardScaler

# Ensure R_HOME is set before importing rpy2
for _rp in ("/usr/lib64/R", "/usr/lib/R"):
    if Path(_rp).exists() and not os.environ.get("R_HOME"):
        os.environ["R_HOME"] = _rp
        break

from paper.scripts.pipeline.r_methods import r_cforest_ranking, r_ctree_ranking

RANDOM_STATE = 1718
N_SEEDS = 5
N_ESTIMATORS = 100
K = 10
RESULTS_DIR = Path("paper/results/tables")


# ---------------------------------------------------------------------------
# Metrics (same as comprehensive_ablation.py)
# ---------------------------------------------------------------------------

def precision_at_k(ranking: list[int], info: list[int], k: int) -> float:
    top_k = set(ranking[:k])
    return len(top_k & set(info)) / k


def recall_at_k(ranking: list[int], info: list[int], k: int) -> float:
    top_k = set(ranking[:k])
    return len(top_k & set(info)) / len(info) if info else 0.0


def f1_at_k(ranking: list[int], info: list[int], k: int) -> float:
    p = precision_at_k(ranking, info, k)
    r = recall_at_k(ranking, info, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _compute_spread(ranking: list[int], n_features: int, k: int = 10) -> float:
    top_k = ranking[:k]
    if len(top_k) < 2 or n_features <= 1:
        return 0.0
    return (max(top_k) - min(top_k)) / (n_features - 1)


def _compute_confounder_rate(ranking: list[int], info: list[int], n_base: int, k: int) -> float:
    top_k = ranking[:k]
    confounders = [i for i in range(n_base, n_base + 20)]
    return len(set(top_k) & set(confounders)) / k


def _downstream_clf(X, y, ranking, k, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    sel = list(ranking[:k])
    if not sel:
        return {"lr_ba": 0.0}
    X_sel = X[:, sel]
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    scores = cross_val_score(lr, X_sel, y, cv=5, scoring="balanced_accuracy")
    return {"lr_ba": float(scores.mean())}


def _downstream_reg(X, y, ranking, k, seed):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    sel = list(ranking[:k])
    if not sel:
        return {"ridge_r2": 0.0}
    X_sel = X[:, sel]
    ridge = Ridge(alpha=1.0, random_state=seed)
    scores = cross_val_score(ridge, X_sel, y, cv=5, scoring="r2")
    return {"ridge_r2": float(scores.mean())}


# ---------------------------------------------------------------------------
# Synthetic dataset factories (identical to comprehensive_ablation.py)
# ---------------------------------------------------------------------------

def clf_standard_easy(seed):
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=10, n_redundant=0,
                               n_clusters_per_class=1, class_sep=1.5, random_state=seed)
    return X, y, list(range(10)), "clf_standard_easy"


def clf_standard_hard(seed):
    X, y = make_classification(n_samples=200, n_features=1000, n_informative=5, n_redundant=0,
                               n_clusters_per_class=1, class_sep=1.0, random_state=seed)
    return X, y, list(range(5)), "clf_standard_hard"


def clf_weak_signal(seed):
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=10, n_redundant=0,
                               n_clusters_per_class=1, class_sep=0.3, random_state=seed)
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
    beta = np.zeros(p)
    beta[:k] = rng.standard_normal(k)
    y = (X @ beta > 0).astype(int)
    return X, y, list(range(k)), "clf_toeplitz"


def clf_confounder(seed):
    X, y = make_classification(n_samples=1000, n_features=50, n_informative=10, n_redundant=0,
                               n_clusters_per_class=1, class_sep=1.0, random_state=seed)
    rng = np.random.default_rng(seed)
    confounders = np.zeros((1000, 20))
    for i in range(20):
        confounders[:, i] = y * 0.5 + rng.standard_normal(1000) * 0.5
    X = np.hstack([X, confounders])
    return X, y, list(range(10)), "clf_confounder"


def clf_bias(seed):
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=10, n_redundant=0,
                               n_clusters_per_class=1, class_sep=1.0, random_state=seed)
    rng = np.random.default_rng(seed + 1000)
    n_high_card = 5
    for i in range(n_high_card):
        X[:, 90 + i] = rng.choice(100, size=1000).astype(float)
    return X, y, list(range(10)), "clf_bias"


def clf_redundant(seed):
    rng = np.random.default_rng(seed)
    n, p, k = 1000, 100, 10
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:k] = rng.standard_normal(k)
    y = (X @ beta > 0).astype(int)
    for i in range(k, k + 10):
        X[:, i] = X[:, i % k] + rng.standard_normal(n) * 0.1
    return X, y, list(range(k)), "clf_redundant"


def reg_friedman(seed):
    X, y = make_friedman1(n_samples=1000, n_features=100, noise=1.0, random_state=seed)
    return X, y, list(range(5)), "reg_friedman"


def reg_linear(seed):
    X, y = make_regression(n_samples=1000, n_features=100, n_informative=10,
                           noise=10.0, random_state=seed)
    return X, y, list(range(10)), "reg_linear"


def reg_toeplitz(seed):
    rng = np.random.default_rng(seed)
    n, p, k = 1000, 100, 10
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = 0.9 ** abs(i - j)
    X = rng.multivariate_normal(np.zeros(p), cov, size=n)
    beta = np.zeros(p)
    beta[:k] = rng.standard_normal(k)
    y = X @ beta + rng.standard_normal(n) * 0.5
    return X, y, list(range(k)), "reg_toeplitz"


def reg_weak_signal(seed):
    X, y = make_regression(n_samples=1000, n_features=100, n_informative=10,
                           noise=50.0, random_state=seed)
    return X, y, list(range(10)), "reg_weak_signal"


CLF_SYNTHETIC = [clf_standard_easy, clf_standard_hard, clf_weak_signal, clf_nonlinear,
                 clf_toeplitz, clf_confounder, clf_bias, clf_redundant]
REG_SYNTHETIC = [reg_friedman, reg_linear, reg_toeplitz, reg_weak_signal]


# ---------------------------------------------------------------------------
# Real dataset loaders (same as comprehensive_ablation.py)
# ---------------------------------------------------------------------------

def _load_real_clf(name):
    loaders = {"iris": load_iris, "wine": load_wine,
               "breast_cancer": load_breast_cancer, "digits": load_digits}
    if name in loaders:
        data = loaders[name]()
        X, y = data.data, data.target
    elif name.startswith("openml_"):
        from sklearn.datasets import fetch_openml
        ds_name = name.replace("openml_", "")
        data = fetch_openml(name=ds_name, as_frame=False, parser="auto")
        X, y = data.data, data.target
        if y.dtype.kind in ("U", "S", "O"):
            from sklearn.preprocessing import LabelEncoder
            y = LabelEncoder().fit_transform(y)
        y = y.astype(np.int64)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    X = StandardScaler().fit_transform(X)
    return X, y, f"real_{name}"


def _load_real_reg(name):
    loaders = {"diabetes": load_diabetes}
    if name in loaders:
        data = loaders[name]()
        X, y = data.data, data.target
    elif name == "california":
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        X, y = data.data, data.target
    else:
        raise ValueError(f"Unknown dataset: {name}")
    X = StandardScaler().fit_transform(X)
    return X, y, f"real_{name}"


REAL_CLF_NAMES = ["iris", "wine", "breast_cancer", "digits",
                  "openml_madelon", "openml_waveform", "openml_optdigits"]
REAL_REG_NAMES = ["diabetes", "california"]


# ---------------------------------------------------------------------------
# R methods to run
# ---------------------------------------------------------------------------

R_METHODS = [
    ("r_ctree_bonf", r_ctree_ranking, dict(testtype="Bonferroni")),
    ("r_ctree_mc", r_ctree_ranking, dict(testtype="MonteCarlo", nresample=199)),
    ("r_cforest_bonf", r_cforest_ranking, dict(testtype="Bonferroni", ntree=N_ESTIMATORS)),
    ("r_cforest_mc", r_cforest_ranking, dict(testtype="MonteCarlo", nresample=199, ntree=N_ESTIMATORS)),
]


def _run_r_on_dataset(X, y, info, dtype, task, is_conf=False, n_base=None):
    """Run all 4 R methods on one dataset, return list of row dicts."""
    r_task = "classification" if task == "clf" else "regression"
    rows = []

    for r_method, r_func, r_params in R_METHODS:
        seed_results = []
        for s in range(N_SEEDS):
            seed = RANDOM_STATE + s
            # Regenerate from factory would be ideal but we pass the same X,y
            # The key thing is the R method sees the same data
            t0 = time.perf_counter()
            try:
                ranking = list(r_func(X, y, task=r_task, **r_params))
            except Exception as e:
                print(f"    ERROR {r_method} seed {s}: {e}")
                continue
            elapsed = time.perf_counter() - t0

            result = {"elapsed_seconds": elapsed,
                      "spread_at_10": _compute_spread(ranking, X.shape[1], K)}
            if info is not None:
                result["precision_at_10"] = precision_at_k(ranking, info, K)
                result["recall_at_10"] = recall_at_k(ranking, info, K)
                result["f1_at_10"] = f1_at_k(ranking, info, K)
            if task == "clf":
                result.update(_downstream_clf(X, y, ranking, K, seed))
            else:
                result.update(_downstream_reg(X, y, ranking, K, seed))
            if is_conf and n_base is not None:
                result["confounder_rate_at_5"] = _compute_confounder_rate(ranking, info, n_base, 5)
                result["confounder_rate_at_10"] = _compute_confounder_rate(ranking, info, n_base, 10)
            seed_results.append(result)

        if not seed_results:
            continue

        # Aggregate across seeds
        agg = {"block": "r_baselines", "task": task, "dataset_type": dtype,
               "variant": r_method, "n_features": X.shape[1], "n_samples": X.shape[0]}
        for key in seed_results[0]:
            vals = [r[key] for r in seed_results if key in r]
            if vals and isinstance(vals[0], (int, float)):
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_std"] = float(np.std(vals))
        rows.append(agg)

        p10 = agg.get("precision_at_10_mean", None)
        ds = agg.get("lr_ba_mean", agg.get("ridge_r2_mean", None))
        t = agg.get("elapsed_seconds_mean", 0)
        line = f"  {r_method:22s}:"
        if p10 is not None:
            line += f" P@10={p10:.3f}"
        if ds is not None:
            line += f" ds={ds:.3f}"
        line += f" t={t:.1f}s"
        print(line)

    return rows


def run_block10_r(save=True):
    """R baselines for Block 10 datasets (synthetic)."""
    print("\n" + "=" * 80)
    print("R BASELINES FOR BLOCK 10 (SYNTHETIC)")
    print("=" * 80)

    rows = []

    for task, datasets in [("clf", CLF_SYNTHETIC), ("reg", REG_SYNTHETIC)]:
        print(f"\n--- {task.upper()} ---")
        for ds_fn in datasets:
            # Use seed 0 for dataset generation (R methods are deterministic given data)
            # but we run across seeds for cross-validation stability
            X, y, info, dtype = ds_fn(RANDOM_STATE)
            is_conf = "confounder" in dtype
            n_base = X.shape[1] - 20 if is_conf else None
            print(f"\n  {dtype} (n={X.shape[0]}, p={X.shape[1]})")

            # Run each seed with its own data generation
            r_task = "classification" if task == "clf" else "regression"
            for r_method, r_func, r_params in R_METHODS:
                seed_results = []
                for s in range(N_SEEDS):
                    seed = RANDOM_STATE + s
                    X_s, y_s, info_s, _ = ds_fn(seed)
                    t0 = time.perf_counter()
                    try:
                        ranking = list(r_func(X_s, y_s, task=r_task, **r_params))
                    except Exception as e:
                        print(f"    ERROR {r_method} seed {s}: {e}")
                        continue
                    elapsed = time.perf_counter() - t0

                    result = {"elapsed_seconds": elapsed,
                              "spread_at_10": _compute_spread(ranking, X_s.shape[1], K)}
                    if info_s is not None:
                        result["precision_at_10"] = precision_at_k(ranking, info_s, K)
                        result["recall_at_10"] = recall_at_k(ranking, info_s, K)
                        result["f1_at_10"] = f1_at_k(ranking, info_s, K)
                    if task == "clf":
                        result.update(_downstream_clf(X_s, y_s, ranking, K, seed))
                    else:
                        result.update(_downstream_reg(X_s, y_s, ranking, K, seed))
                    if is_conf and n_base is not None:
                        result["confounder_rate_at_5"] = _compute_confounder_rate(ranking, info_s, n_base, 5)
                        result["confounder_rate_at_10"] = _compute_confounder_rate(ranking, info_s, n_base, 10)
                    seed_results.append(result)

                if not seed_results:
                    continue

                agg = {"block": "strictness_continuum", "task": task, "dataset_type": dtype,
                       "variant": r_method, "n_features": X.shape[1], "n_samples": X.shape[0]}
                for key in seed_results[0]:
                    vals = [r[key] for r in seed_results if key in r]
                    if vals and isinstance(vals[0], (int, float)):
                        agg[f"{key}_mean"] = float(np.mean(vals))
                        agg[f"{key}_std"] = float(np.std(vals))
                rows.append(agg)

                p10 = agg.get("precision_at_10_mean", None)
                ds = agg.get("lr_ba_mean", agg.get("ridge_r2_mean", None))
                t = agg.get("elapsed_seconds_mean", 0)
                line = f"  {r_method:22s}:"
                if p10 is not None:
                    line += f" P@10={p10:.3f}"
                if ds is not None:
                    line += f" ds={ds:.3f}"
                line += f" t={t:.1f}s"
                print(line)

    df = pd.DataFrame(rows)
    if save:
        out = RESULTS_DIR / "ablation_block10_r_baselines.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved {len(df)} rows → {out}")
    return df


def run_block11_r(save=True):
    """R baselines for Block 11 datasets (real)."""
    print("\n" + "=" * 80)
    print("R BASELINES FOR BLOCK 11 (REAL DATASETS)")
    print("=" * 80)

    rows = []

    print("\n--- REAL CLF ---")
    for ds_name in REAL_CLF_NAMES:
        try:
            X, y, dtype = _load_real_clf(ds_name)
        except Exception as e:
            print(f"\n  SKIP {ds_name}: {e}")
            continue
        print(f"\n  {dtype} (n={X.shape[0]}, p={X.shape[1]})")
        rows.extend(_run_r_on_dataset(X, y, None, dtype, "clf"))

    print("\n--- REAL REG ---")
    for ds_name in REAL_REG_NAMES:
        try:
            X, y, dtype = _load_real_reg(ds_name)
        except Exception as e:
            print(f"\n  SKIP {ds_name}: {e}")
            continue
        print(f"\n  {dtype} (n={X.shape[0]}, p={X.shape[1]})")
        rows.extend(_run_r_on_dataset(X, y, None, dtype, "reg"))

    df = pd.DataFrame(rows)
    if save:
        out = RESULTS_DIR / "ablation_block11_r_baselines.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved {len(df)} rows → {out}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run R baselines for ablation blocks")
    parser.add_argument("--blocks", type=str, default="10,11",
                        help="Comma-separated block numbers (10, 11, or 10,11)")
    args = parser.parse_args()

    blocks = [int(b.strip()) for b in args.blocks.split(",")]

    for block in blocks:
        if block == 10:
            run_block10_r()
        elif block == 11:
            run_block11_r()
        else:
            print(f"Unknown block: {block}")
