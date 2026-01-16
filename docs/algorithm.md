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
            Compute impurity reduction: Gini(parent) - Gini(left) - Gini(right)
    Select (feature, split) that maximizes impurity reduction
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

1. **Unbiased variable selection**: The p-value is computed under the null
   hypothesis. A feature with 1000 values doesn't have more "chances" - the
   permutation test accounts for this.

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
| `mc`     | Classification | Multiple Correlation (η²) - how much variance in X is explained by class membership |
| `mi`     | Classification | Mutual Information - non-linear dependence                                          |
| `rdc`    | Both           | Randomized Dependence Coefficient - O(n log n) non-linear dependence                |
| `pc`     | Regression     | Pearson Correlation                                                                 |
| `dc`     | Regression     | Distance Correlation - captures non-linear relationships                            |

**Multiple Correlation (mc)** - Default for classification:

```python
# Implemented in _selector.py
η² = SSB / SST  # Between-class variance / Total variance

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

    p_value = mean(|θ_perm| >= θ)  # Proportion as extreme as observed
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

This accelerates training by not re-testing clearly uninformative features within a subtree: muted features are removed
only for descendants of the current node (siblings are isolated).

---

## What is Honesty?

**Honest estimation** is a technique from the causal inference literature (Wager
& Athey, 2018) that produces **unbiased leaf predictions** by using different
data for tree structure and leaf values.

### The Problem with Adaptive Estimation

In a standard tree, the same data is used to:

1. Choose which features to split on
2. Choose where to split
3. Estimate the prediction in each leaf

This creates **overfitting bias** - the leaf predictions are optimistically
biased because the tree structure was chosen to make them look good on the
training data.

### The Honest Solution

```python
# Split data into two parts
X_split, X_est = train_test_split(X, test_size=honesty_fraction)

# Build tree structure using splitting sample
tree = build_tree(X_split, y_split)

# Re-estimate leaf values using estimation sample
for leaf in tree.leaves:
    samples_in_leaf = X_est[X_est falls into leaf]
    leaf.value = mean(y_est[samples_in_leaf])
```

### citrees Implementation

```python
# Honest estimation splits data for structure and prediction
if self.honesty:
    X_split, X_est, y_split, y_est = train_test_split(
        X, y, test_size=self.honesty_fraction, random_state=self._random_state
    )
    # Build tree structure using splitting sample
    self.tree_ = self._build_tree(X_split, y_split, depth=1)
    # Re-estimate leaf values using estimation sample
    self._reestimate_leaf_values(X_est, y_est)
```

### When to Use Honesty

| Use Case           | Honesty? | Why                                     |
| ------------------ | -------- | --------------------------------------- |
| Pure prediction    | Optional | May hurt accuracy (less training data)  |
| Causal inference   | **Yes**  | Required for valid confidence intervals |
| Feature importance | Optional | Reduces selection bias                  |
| Small datasets     | No       | Can't afford to split data              |
| Large datasets     | Yes      | Benefits outweigh data loss             |

### Current Implementation Issues

1. **Forest honesty**: Each tree in a forest uses its own honest split. This is
   correct but different from GRF which uses out-of-bag samples.
2. **No stratification**: The honest split is unstratified (for both classifiers
   and regressors) to preserve the independence assumptions used in the theory.

2. **No variance estimation**: True honest forests (like GRF) can provide valid
   confidence intervals. Our implementation doesn't compute these yet.

---

## Summary: What Each Feature Adds

| Feature                   | Scientific Contribution                          | When to Use                   |
| ------------------------- | ------------------------------------------------ | ----------------------------- |
| **Permutation tests**     | Unbiased variable selection, principled stopping | Always (core algorithm)       |
| **Bonferroni correction** | Controls family-wise error rate                  | When interpretability matters |
| **Feature muting**        | Accelerates training                             | Large feature spaces          |
| **Honesty**               | Unbiased leaf predictions                        | Causal inference              |

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
