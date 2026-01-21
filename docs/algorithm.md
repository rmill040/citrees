# The Conditional Inference Algorithm

## Overview

citrees implements **Conditional Inference Trees** (CIT), which differ
fundamentally from traditional CART-style trees. The key innovation is
separating **variable selection** from **split point selection** using
statistical hypothesis testing.

## The Problem with Traditional Trees

Traditional decision trees (CART, ID3, C4.5) use a greedy algorithm:

```
For each node:
    For each feature X_j:
        For each possible split point c:
            Compute weighted impurity reduction:
            Δ = Gini(parent) - (|L|/n)·Gini(left) - (|R|/n)·Gini(right)
    Select (feature, split) that maximizes Δ
```

**The problem**: This approach is **biased toward features with more split
points**. A feature with 1000 unique values has 999 chances to find a "good"
split by chance, while a binary feature has only 1 chance. This leads to:

1. **Variable selection bias**: High-cardinality features appear more important
2. **Spurious splits**: Trees can find splits on noise features by chance
3. **Overfitting**: The tree structure adapts to noise, not signal

## The Conditional Inference Solution

Conditional Inference Trees address this by using **permutation tests**:

```
For each node:
    Step 1: VARIABLE SELECTION (test association with target)
        For each feature X_j:
            H0: X_j is independent of Y
            Compute p-value using permutation test
        Select feature with lowest p-value
        IF p-value >= alpha: STOP (create leaf)

    Step 2: SPLIT POINT SELECTION (given the selected feature)
        For each possible split point c:
            H0: Split at c provides no improvement
            Compute p-value using permutation test
        Select split with lowest p-value
```

### Why This Works

1. **Reduced high-cardinality bias**: Stage A selects variables via
   permutation-test p-values rather than by searching over many thresholds for
   the largest impurity improvement. This avoids the pure multiple-comparisons
   mechanism that favors features with many candidate split points (at a fixed
   node; see the scope note in `docs/permutation-tests.md`).

2. **Principled stopping**: The tree stops growing when no feature has a
   statistically significant association with the target. No need for pruning.

3. **Interpretable importance**: Feature importance is based on how often a
   feature passes the statistical test and how much impurity it reduces.

---

## Implementation Details

### Feature Selection (Selector)

citrees supports multiple association measures:

| Selector | For            | Measures                                                                            |
| -------- | -------------- | ----------------------------------------------------------------------------------- |
| `mc`     | Classification | Multiple Correlation (η) - how much variance in X is explained by class membership |
| `mi`     | Classification | Mutual Information - non-linear dependence                                          |
| `rdc`    | Both           | Randomized Dependence Coefficient - O(n log n) non-linear dependence                |
| `pc`     | Regression     | Pearson Correlation                                                                 |
| `dc`     | Regression     | Distance Correlation - captures non-linear relationships                            |

**Multiple Correlation (mc)** - Default for classification:

```python
# Implemented in _selector.py
η = sqrt(SSB / SST)  # Correlation ratio (monotone in η² = SSB/SST)

# SST = Σ(x_i - x̄)²  (total sum of squares)
# SSB = Σ n_k(x̄_k - x̄)²  (between-class sum of squares)
```

### Permutation Testing

The p-value is computed via Monte Carlo permutation test:

```python
# Pseudocode - see _selector.py for implementation
def _ptest(func, x, y, n_resamples, alpha, early_stopping):
    θ = |func(x, y)|  # Observed statistic

    θ_perm = []
    for i in range(n_resamples):
        y_shuffled = permute(y)
        θ_perm[i] = func(x, y_shuffled)  # Statistic under null

    # +1 corrected Monte Carlo permutation p-value (Phipson & Smyth, 2010)
    p_value = (1 + sum(|θ_perm| >= θ)) / (1 + n_resamples)
    return p_value
```

**Key parameters**:

- `n_resamples_selector`: Number of permutations (`"auto"`, `"minimum"`, `"maximum"`, `int`, or `None`)
- `alpha_selector`: Significance threshold (default: 0.05)
- `early_stopping_selector`: `"adaptive"`, `"simple"`, or `None` (fixed-$B$)
- `early_stopping_confidence_selector`: Posterior-confidence threshold γ for `"adaptive"` (default: 0.95)

### Bonferroni Correction

When testing multiple features, the alpha is adjusted:

```python
# Bonferroni correction controls family-wise error rate
alpha_adjusted = alpha / n_features
```

This controls the **family-wise error rate** - the probability of at least one
false positive.

### Feature Muting

Features that clearly fail the test are "muted" (removed from consideration):

```python
# Feature muting accelerates training by removing uninformative features
if feature_muting and pval >= max(alpha, 1 - alpha):
    mute_feature(feature)  # Remove from available features
```

For $\alpha \le 1/2$, this is an *upper-tail* rule: `pval >= 1 - alpha` (e.g., $\alpha=0.05 \Rightarrow p\ge 0.95$).
It is a speed-only heuristic: it avoids repeatedly testing features that look extremely unpromising.

Muting is **subtree-local**: muted features are removed only for descendants of the current node (siblings are
isolated), which avoids traversal-order dependence:

```
Root available: {0,1,2}
├─ Left child sees {0,1,2} → may drop feature 1 in its subtree → descendants see {0,2}
└─ Right child sees {0,1,2} (unaffected by left)
```

---

## Honest Estimation

citrees supports **honest estimation**, a sample-splitting technique for
reducing adaptive bias in leaf estimation (conditional unbiasedness under sample
splitting assumptions). For full details, see the dedicated documentation:
**[Honest Estimation](honest-estimation.md)**.

---

## Summary: What Each Feature Adds

| Feature                   | Scientific Contribution                          | When to Use                   |
| ------------------------- | ------------------------------------------------ | ----------------------------- |
| **Permutation tests**     | Reduced selection bias, test-based stopping      | Always (core algorithm)       |
| **Bonferroni correction** | Controls family-wise error rate                  | When interpretability matters |
| **Feature muting**        | Accelerates training                             | Large feature spaces          |
| **Honesty**               | Reduced adaptive bias in leaf estimation         | Causal/estimation contexts    |

## References

1. Hothorn, T., Hornik, K., & Zeileis, A. (2006). Unbiased Recursive
   Partitioning: A Conditional Inference Framework. _JCGS_, 15(3), 651-674.

2. Strobl, C., Boulesteix, A. L., Zeileis, A., & Hothorn, T. (2007). Bias in
   Random Forest Variable Importance Measures. _BMC Bioinformatics_, 8(1), 25.

3. Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous
   Treatment Effects using Random Forests. _JASA_, 113(523), 1228-1242.

4. Strobl, C., Boulesteix, A. L., Kneib, T., Augustin, T., & Zeileis, A. (2008).
   Conditional Variable Importance for Random Forests. _BMC Bioinformatics_,
   9(1), 307.
