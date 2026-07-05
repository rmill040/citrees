"""Small statistical helpers used by paper analysis tests and builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def bootstrap_ci(
    scores: np.ndarray, n_bootstrap: int = 2000, ci: float = 0.95, random_state: int = 42
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(random_state)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))
    alpha = 1 - ci
    return (
        float(np.percentile(bootstrap_means, alpha / 2 * 100)),
        float(np.percentile(bootstrap_means, (1 - alpha / 2) * 100)),
    )


def _aligned_pair(data: pd.DataFrame, col1: str, col2: str) -> tuple[np.ndarray, np.ndarray]:
    """Return complete-case paired values for two columns."""
    subset = data[[col1, col2]].dropna()
    return subset[col1].values, subset[col2].values


def pairwise_wilcoxon_holm(data: pd.DataFrame, methods: list[str], metric: str) -> pd.DataFrame:
    """Run paired Wilcoxon signed-rank tests with Holm correction."""
    pvalues = []
    pairs = []

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i >= j:
                continue
            col1 = f"{m1}_{metric}"
            col2 = f"{m2}_{metric}"
            if col1 not in data.columns or col2 not in data.columns:
                continue
            v1, v2 = _aligned_pair(data, col1, col2)
            if len(v1) < 10:
                continue
            stat, pval = wilcoxon(v1, v2, alternative="two-sided")
            pvalues.append(pval)
            pairs.append((m1, m2, stat, len(v1)))

    if not pvalues:
        return pd.DataFrame()

    _, corrected_pvals, _, _ = multipletests(pvalues, method="holm")

    rows = []
    for (m1, m2, stat, n_pairs), pval, corrected in zip(
        pairs, pvalues, corrected_pvals, strict=False
    ):
        rows.append(
            {
                "method1": m1,
                "method2": m2,
                "statistic": stat,
                "p_value": pval,
                "p_value_corrected": corrected,
                "significant": corrected < 0.05,
                "n_pairs": n_pairs,
            }
        )

    return pd.DataFrame(rows)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Return Cohen's d standardized mean difference."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def interpret_cohens_d(d: float) -> str:
    """Return a conventional text label for a Cohen's d value."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


def compute_noise_selection_rate(
    feature_ranking: list[int], noise_indices: list[int], k: int
) -> float:
    """Return the fraction of top-k ranked features that are known noise."""
    if k == 0 or len(noise_indices) == 0:
        return 0.0
    top_k = set(feature_ranking[:k])
    noise_set = set(noise_indices)
    return len(top_k & noise_set) / k


def friedman_test(
    data: pd.DataFrame, methods: list[str], metric: str
) -> tuple[float, float, int, int]:
    """Run a Friedman test across methods using complete-case rows."""
    method_cols = [(m, f"{m}_{metric}") for m in methods if f"{m}_{metric}" in data.columns]
    if len(method_cols) < 3:
        return np.nan, np.nan, 0, len(method_cols)

    cols = [c for _, c in method_cols]
    aligned = data[cols].dropna()
    n_datasets = len(aligned)
    if n_datasets < 2:
        return np.nan, np.nan, n_datasets, len(method_cols)

    values = [aligned[c].values for c in cols]
    stat, pvalue = stats.friedmanchisquare(*values)
    return stat, pvalue, n_datasets, len(method_cols)
