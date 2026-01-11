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

- `n_resamples_selector`: Number of permutations (default: "auto" ≈ 100-500)
- `alpha_selector`: Significance threshold (default: 0.05)
- `early_stopping_selector`: Stop testing once significance is achieved

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

This accelerates training by not re-testing clearly uninformative features.

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
        X, y, test_size=self.honesty_fraction, stratify=y
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

2. **No variance estimation**: True honest forests (like GRF) can provide valid
   confidence intervals. Our implementation doesn't compute these yet.

---

## What is Conformal Prediction?

**Conformal Prediction** provides **distribution-free prediction intervals**
with guaranteed coverage. Unlike parametric methods, it makes no assumptions
about the data distribution.

### The Guarantee

For any exchangeable dataset and any significance level α:

```
P(Y_new ∈ prediction_set(X_new)) ≥ 1 - α
```

This holds for ANY distribution, ANY model, finite samples.

### How It Works (Split Conformal)

```python
# 1. Split data
X_train, X_cal = train_test_split(X)

# 2. Train model on training data
model.fit(X_train, y_train)

# 3. Compute nonconformity scores on calibration data
scores = |y_cal - model.predict(X_cal)|  # For regression

# 4. Find quantile
q = quantile(scores, (1-α)(1 + 1/n_cal))  # Finite-sample correction

# 5. Prediction interval for new point
y_pred = model.predict(X_new)
interval = [y_pred - q, y_pred + q]
```

### citrees Implementation

**ConformalClassifier** (Adaptive Prediction Sets):

```python
# From _conformal.py - produces prediction SETS for classification
# A set of classes that contains the true class with probability 1-α

prediction_sets = clf.predict_set(X_test)
# Returns: [{class1, class2}, {class1}, {class1, class3}, ...]
```

**ConformalRegressor** (Split Conformal):

```python
# From _conformal.py - produces prediction INTERVALS for regression
lower, upper = reg.predict_interval(X_test)
# Guarantee: P(lower <= y_true <= upper) >= 1 - α
```

**CQR** (Conformalized Quantile Regression):

```python
# From _conformal.py - adaptive intervals that vary with local uncertainty
# Uses the spread of tree predictions as a quantile estimate
lower, upper = cqr.predict_interval(X_test)
# Intervals are WIDER where the model is uncertain
```

### Why Conformal + Citrees?

1. **Principled uncertainty**: Citrees' statistical tests already embody a
   frequentist worldview. Conformal prediction extends this to predictions.

2. **Distribution-free**: Works without assumptions about residual distribution.

3. **Finite-sample valid**: Coverage guarantee holds for any sample size.

---

## What is SHAP?

**SHAP (SHapley Additive exPlanations)** provides theoretically grounded feature
attributions based on game theory.

### The Problem with MDI

citrees' built-in `feature_importances_` uses Mean Decrease Impurity (MDI):

```python
importance[j] = Σ (impurity_decrease at nodes using feature j)
```

**Problems with MDI**:

1. Biased toward high-cardinality features (in CART)
2. Doesn't account for feature correlations
3. Not additive: importances don't sum to prediction

### SHAP Values

For a prediction f(x), SHAP decomposes it:

```
f(x) = base_value + φ₁ + φ₂ + ... + φₚ

Where:
- base_value = E[f(X)]  (expected prediction)
- φⱼ = contribution of feature j to this specific prediction
```

**Properties**:

- **Local accuracy**: φ values sum to (f(x) - base_value)
- **Consistency**: If feature j contributes more in model A than B, φⱼ is larger
- **Missingness**: φⱼ = 0 if feature j doesn't affect prediction

### citrees Implementation

```python
# See _importance.py for full implementation
class SHAPExplainer:
    def __init__(self, model, background_data):
        # Uses shap's model-agnostic Explainer
        # NOT TreeExplainer (our tree structure isn't compatible)
        self.explainer_ = shap.Explainer(model.predict_proba, background_data)
```

**Current limitation**: We use the model-agnostic SHAP explainer, not TreeSHAP.
TreeSHAP is O(TLD²) for tree models, while model-agnostic is slower. A native
TreeSHAP implementation would be much faster.

---

## Conditional Permutation Importance (CPI)

CPI addresses a limitation of standard permutation importance with correlated
features.

### The Problem

Standard permutation importance:

```python
for feature j:
    X_permuted = X.copy()
    X_permuted[:, j] = permute(X[:, j])
    importance[j] = score(X) - score(X_permuted)
```

**Problem**: When features are correlated, permuting one creates an unrealistic
data distribution (extrapolation). This inflates the importance of correlated
features.

### CPI Solution

Permute feature j **conditionally** on correlated features:

```python
# Pseudocode - see _importance.py for full implementation
def conditional_permutation_importance(model, X, y):
    for feature j:
        correlated = find_correlated_features(X, j)
        strata = create_strata(X[:, correlated])  # Bin correlated features

        for stratum in strata:
            # Permute only within each stratum
            X_permuted[stratum, j] = permute(X[stratum, j])

        importance[j] = score(X) - score(X_permuted)
```

This maintains the correlation structure while testing the marginal effect of
feature j.

---

## Summary: What Each Feature Adds

| Feature                   | Scientific Contribution                          | When to Use                            |
| ------------------------- | ------------------------------------------------ | -------------------------------------- |
| **Permutation tests**     | Unbiased variable selection, principled stopping | Always (core algorithm)                |
| **Bonferroni correction** | Controls family-wise error rate                  | When interpretability matters          |
| **Feature muting**        | Accelerates training                             | Large feature spaces                   |
| **Honesty**               | Unbiased leaf predictions                        | Causal inference, confidence intervals |
| **Conformal prediction**  | Distribution-free coverage guarantees            | Uncertainty quantification             |
| **SHAP**                  | Theoretically grounded feature attributions      | Local explanations                     |
| **CPI**                   | Handles correlated features                      | When features are correlated           |

## References

1. Hothorn, T., Hornik, K., & Zeileis, A. (2006). Unbiased Recursive
   Partitioning: A Conditional Inference Framework. _JCGS_, 15(3), 651-674.

2. Strobl, C., Boulesteix, A. L., Zeileis, A., & Hothorn, T. (2007). Bias in
   Random Forest Variable Importance Measures. _BMC Bioinformatics_, 8(1), 25.

3. Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous
   Treatment Effects using Random Forests. _JASA_, 113(523), 1228-1242.

4. Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a
   Random World. Springer.

5. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting
   Model Predictions. _NeurIPS_.

6. Strobl, C., Boulesteix, A. L., Kneib, T., Augustin, T., & Zeileis, A. (2008).
   Conditional Variable Importance for Random Forests. _BMC Bioinformatics_,
   9(1), 307.
