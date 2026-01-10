# SHAP Integration

SHAP (SHapley Additive exPlanations) provides theoretically grounded feature attributions based on cooperative game theory. citrees supports TreeSHAP, an efficient algorithm for computing exact SHAP values in polynomial time for tree-based models.

## Overview

| Aspect | Description |
|--------|-------------|
| Purpose | Feature attribution / explanation |
| Foundation | Shapley values from game theory |
| Complexity | $O(TLD^2)$ for trees (vs $O(2^M)$ exact) |
| Key reference | [Lundberg & Lee (2017)](https://arxiv.org/abs/1705.07874) |

---

## The Problem: Feature Importance

### Limitations of Standard Importance

Standard tree feature importance (Mean Decrease Impurity, MDI) has known issues:

| Problem | Description |
|---------|-------------|
| **Bias toward high-cardinality** | Features with more unique values get inflated importance |
| **Correlation blindness** | Cannot distinguish correlated features |
| **No direction** | Only magnitude, not positive/negative effect |
| **Global only** | Same importance for all predictions |

### SHAP Advantages

| Advantage | Description |
|-----------|-------------|
| **Local explanations** | Different importance for each prediction |
| **Directional** | Shows positive/negative contribution |
| **Theoretically grounded** | Based on Shapley values |
| **Consistent** | If feature contributes more, importance increases |

---

## Shapley Values: The Theory

### Game Theory Foundation

Shapley values come from **cooperative game theory**. Given:
- Players: Features $\{1, 2, \ldots, M\}$
- Value function: $v(S) = $ prediction using features in $S$

The Shapley value for feature $j$ is:

$$\phi_j = \sum_{S \subseteq \{1,\ldots,M\} \setminus \{j\}} \frac{|S|!(M-|S|-1)!}{M!} [v(S \cup \{j\}) - v(S)]$$

This is the **average marginal contribution** of feature $j$ across all possible coalitions.

### Properties (Axioms)

| Property | Description |
|----------|-------------|
| **Efficiency** | $\sum_j \phi_j = f(x) - \mathbb{E}[f(X)]$ |
| **Symmetry** | Equal features get equal attribution |
| **Linearity** | Additive across models |
| **Null player** | Unused features get zero attribution |

### Interpretation

For a prediction $f(x)$:

$$f(x) = \phi_0 + \sum_{j=1}^{M} \phi_j(x)$$

where:
- $\phi_0 = \mathbb{E}[f(X)]$ (base value / expected prediction)
- $\phi_j(x)$ = contribution of feature $j$ for this instance

---

## TreeSHAP Algorithm

### The Challenge

Exact Shapley values require summing over $2^M$ subsets - exponential in the number of features.

**TreeSHAP** (Lundberg et al., 2019) exploits tree structure to compute exact SHAP values in polynomial time.

### Key Insight

For decision trees, the value function $v(S)$ can be computed efficiently by:
1. Following paths where features in $S$ determine splits
2. Averaging over branches where features not in $S$ would split

### Algorithm

```
Algorithm: TreeSHAP (Simplified)
Input: Tree T, instance x, background data X_bg

1. Initialize:
   φ[j] ← 0 for all features j
   base_value ← average prediction over X_bg

2. Recursive tree traversal:
   function RECURSE(node, path_features, path_weight):
       If node is leaf:
           For each feature j in path_features:
               # Contribution proportional to path weight
               φ[j] += path_weight × contribution(j, path)
           Return

       j ← split feature at node
       threshold ← split threshold at node

       If j in path_features:
           # Feature value known, follow single path
           If x[j] ≤ threshold:
               RECURSE(left_child, path_features, path_weight)
           Else:
               RECURSE(right_child, path_features, path_weight)
       Else:
           # Feature unknown, follow both paths weighted by data
           w_left ← fraction of X_bg going left
           w_right ← fraction of X_bg going right
           RECURSE(left_child, path_features ∪ {j}, path_weight × w_left)
           RECURSE(right_child, path_features ∪ {j}, path_weight × w_right)

3. Return φ, base_value
```

### Complexity

| Method | Time Complexity | Space |
|--------|-----------------|-------|
| Exact Shapley | $O(2^M)$ | $O(M)$ |
| KernelSHAP | $O(TL \cdot 2^M)$ | $O(M)$ |
| TreeSHAP | $O(TLD^2)$ | $O(D)$ |
| FastTreeSHAP v1 | ~$O(TLD^2)$ | $O(D)$ |
| FastTreeSHAP v2 | ~$O(TLD^{1.5})$ | $O(D^2)$ |

Where:
- $T$ = number of trees
- $L$ = maximum leaves per tree
- $D$ = maximum depth
- $M$ = number of features

---

## SHAP Value Interpretation

### Local Explanations

For a single prediction:

```
Prediction: 0.73 (probability of class 1)
Base value: 0.52

Feature contributions:
  age:      +0.15  (being older increases probability)
  income:   +0.08  (higher income increases probability)
  location: -0.02  (this location slightly decreases probability)

Sum: 0.52 + 0.15 + 0.08 - 0.02 = 0.73 ✓
```

### Global Feature Importance

Aggregate SHAP values across instances:

$$I_j = \frac{1}{n} \sum_{i=1}^{n} |\phi_j(x_i)|$$

This gives **global feature importance** that:
- Accounts for both positive and negative effects
- Is consistent with local explanations
- Reduces bias from MDI importance

### Visualization Types

| Plot | Purpose |
|------|---------|
| **Summary plot** | Distribution of SHAP values per feature |
| **Bar plot** | Mean absolute SHAP value per feature |
| **Waterfall** | Single prediction breakdown |
| **Force plot** | Interactive single prediction |
| **Dependence** | SHAP value vs feature value |
| **Interaction** | Pairwise feature interactions |

---

## Implementation in citrees

### Basic Usage

```python
from citrees import ConditionalInferenceForestClassifier
import shap

# Train model
forest = ConditionalInferenceForestClassifier(n_estimators=100)
forest.fit(X_train, y_train)

# Create TreeExplainer
explainer = shap.TreeExplainer(forest)

# Compute SHAP values
shap_values = explainer.shap_values(X_test)

# For binary classification: shap_values is (n_samples, n_features)
# For multiclass: shap_values is list of (n_samples, n_features) per class
```

### Visualization

```python
import shap
import matplotlib.pyplot as plt

# Summary plot (beeswarm)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Bar plot (global importance)
shap.summary_plot(shap_values, X_test, plot_type="bar",
                  feature_names=feature_names)

# Single prediction waterfall
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test[0],
    feature_names=feature_names
))

# Dependence plot (feature effect)
shap.dependence_plot("age", shap_values, X_test,
                     feature_names=feature_names)
```

### Native citrees Method

```python
# Built-in SHAP support
forest = ConditionalInferenceForestClassifier(n_estimators=100)
forest.fit(X_train, y_train)

# Get SHAP values directly
shap_values = forest.shap_values(X_test)

# Get SHAP-based feature importance
shap_importance = forest.shap_feature_importances(X_test)
```

---

## Comparison: SHAP vs MDI Importance

### Mean Decrease Impurity (MDI)

Standard tree feature importance:

$$MDI_j = \sum_{nodes\ where\ j\ splits} \frac{n_{node}}{n} \Delta_{impurity}$$

### Comparison

| Aspect | MDI | SHAP |
|--------|-----|------|
| **Scope** | Global only | Local + Global |
| **Direction** | Magnitude only | Signed contribution |
| **Computation** | Fast | Slower |
| **Consistency** | Can be inconsistent | Theoretically consistent |
| **Correlation handling** | Poor | Better |
| **High cardinality bias** | Yes | Reduced |

### Example: Correlated Features

```
True model: y = x₁ + noise
x₁ and x₂ are highly correlated (r = 0.95)

MDI Importance:
  x₁: 0.55
  x₂: 0.45  ← Inflated due to correlation

SHAP Importance:
  x₁: 0.82
  x₂: 0.18  ← More accurate attribution
```

---

## SHAP for Conditional Inference Trees

### Interaction with Permutation Tests

citrees uses permutation tests for feature selection, which provides statistical significance. SHAP provides complementary information:

| Method | What It Measures |
|--------|------------------|
| **Permutation p-value** | Is feature associated with target? |
| **SHAP importance** | How much does feature contribute to predictions? |

### Recommended Workflow

```python
from citrees import ConditionalInferenceForestClassifier
import shap

# 1. Train with statistical feature selection
forest = ConditionalInferenceForestClassifier(
    alpha_selector=0.05,  # Only significant features used
    n_estimators=100,
)
forest.fit(X_train, y_train)

# 2. Get statistically grounded importance
stat_importance = forest.feature_importances_  # MDI from selected features

# 3. Get SHAP-based importance for interpretation
explainer = shap.TreeExplainer(forest)
shap_values = explainer.shap_values(X_test)
shap_importance = np.abs(shap_values).mean(axis=0)

# 4. Compare
for i, name in enumerate(feature_names):
    print(f"{name}: stat={stat_importance[i]:.3f}, shap={shap_importance[i]:.3f}")
```

### Benefits of Combining

| Benefit | Description |
|---------|-------------|
| **Statistical validity** | Permutation tests control false positives |
| **Interpretability** | SHAP provides local explanations |
| **Robustness** | Two independent measures of importance |

---

## FastTreeSHAP

### Performance Improvements

[FastTreeSHAP](https://engineering.linkedin.com/blog/2022/fasttreeshap--accelerating-shap-value-computation-for-trees) (LinkedIn, 2021) provides faster computation:

| Version | Speedup | Memory |
|---------|---------|--------|
| TreeSHAP (original) | 1× | Baseline |
| FastTreeSHAP v1 | 1.5× | Same |
| FastTreeSHAP v2 | 2.5× | Higher |

### Usage

```python
import fasttreeshap

# Faster SHAP computation
explainer = fasttreeshap.TreeExplainer(
    forest,
    algorithm='v2',  # or 'v1', 'auto'
)
shap_values = explainer.shap_values(X_test)
```

---

## Practical Considerations

### Background Data

TreeSHAP needs background data to compute expected values:

```python
# Option 1: Use training data (default)
explainer = shap.TreeExplainer(forest, X_train)

# Option 2: Use representative sample
background = shap.sample(X_train, 100)
explainer = shap.TreeExplainer(forest, background)

# Option 3: Interventional vs Tree-path dependent
explainer = shap.TreeExplainer(
    forest,
    data=X_train,
    feature_perturbation='interventional',  # or 'tree_path_dependent'
)
```

### Interventional vs Path-Dependent

| Method | Behavior | Use Case |
|--------|----------|----------|
| **Interventional** | Breaks feature correlations | Causal interpretation |
| **Tree-path dependent** | Respects correlations | Predictive interpretation |

### Large Datasets

For large datasets, use sampling:

```python
# Sample for background
background = shap.sample(X_train, 1000)
explainer = shap.TreeExplainer(forest, background)

# Sample for explanation
sample_idx = np.random.choice(len(X_test), 500, replace=False)
shap_values = explainer.shap_values(X_test[sample_idx])
```

---

## SHAP Interaction Values

### Pairwise Interactions

Beyond main effects, TreeSHAP can compute interaction effects:

$$f(x) = \phi_0 + \sum_j \phi_j + \sum_{j < k} \phi_{jk}$$

where $\phi_{jk}$ is the interaction between features $j$ and $k$.

### Computing Interactions

```python
# Interaction values (slower, O(TLD²M²))
interaction_values = explainer.shap_interaction_values(X_test[:100])

# Shape: (n_samples, n_features, n_features)
# interaction_values[i, j, k] = interaction between j and k for sample i

# Visualize
shap.summary_plot(interaction_values[:, :, 0], X_test[:100])  # Interactions with feature 0
```

---

## Configuration Examples

### Quick Explanation

```python
import shap

# Fast global importance
explainer = shap.TreeExplainer(forest)
shap_values = explainer.shap_values(X_train[:1000])  # Sample

# Bar plot
shap.summary_plot(shap_values, X_train[:1000], plot_type="bar")
```

### Detailed Analysis

```python
import shap

# Full analysis
explainer = shap.TreeExplainer(forest, X_train)
shap_values = explainer.shap_values(X_test)

# Multiple visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Summary (beeswarm)
plt.sca(axes[0, 0])
shap.summary_plot(shap_values, X_test, show=False)

# 2. Bar (global importance)
plt.sca(axes[0, 1])
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)

# 3. Dependence (top feature)
plt.sca(axes[1, 0])
top_feature = np.abs(shap_values).mean(0).argmax()
shap.dependence_plot(top_feature, shap_values, X_test, ax=axes[1, 0], show=False)

# 4. Waterfall (single prediction)
plt.sca(axes[1, 1])
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test[0],
), show=False)

plt.tight_layout()
plt.savefig("shap_analysis.png", dpi=300)
```

### With Interaction Analysis

```python
# Full interaction analysis (expensive)
explainer = shap.TreeExplainer(forest)

# Main effects
shap_values = explainer.shap_values(X_test)

# Interaction effects (for small sample)
interaction_values = explainer.shap_interaction_values(X_test[:50])

# Main effect + interaction for feature pair
feature_i, feature_j = 0, 1
main_i = interaction_values[:, feature_i, feature_i]
main_j = interaction_values[:, feature_j, feature_j]
interaction_ij = interaction_values[:, feature_i, feature_j]

print(f"Main effect {feature_names[feature_i]}: {np.abs(main_i).mean():.3f}")
print(f"Main effect {feature_names[feature_j]}: {np.abs(main_j).mean():.3f}")
print(f"Interaction: {np.abs(interaction_ij).mean():.3f}")
```

---

## References

1. **Original SHAP Paper**: [Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.](https://arxiv.org/abs/1705.07874)

2. **TreeSHAP**: [Lundberg, S. M., Erion, G., Chen, H., et al. (2019). Explainable AI for Trees: From Local Explanations to Global Understanding.](https://arxiv.org/abs/1905.04610)

3. **FastTreeSHAP**: [Yang, J. (2021). Fast TreeSHAP: Accelerating SHAP Value Computation for Trees. NeurIPS XAI4Debugging Workshop.](https://xai4debugging.github.io/files/papers/fast_treeshap_accelerating_sha.pdf)

4. **SHAP Documentation**: [shap.readthedocs.io](https://shap.readthedocs.io/)

5. **Interpretable ML Book**: [Molnar, C. (2024). Interpretable Machine Learning. Chapter on SHAP.](https://christophm.github.io/interpretable-ml-book/shap.html)

6. **treeshap R Package**: [ModelOriented/treeshap](https://github.com/ModelOriented/treeshap)

7. **SHAP for Feature Selection**: [Marcílio, W. E., & Eler, D. M. (2024). From Explanations to Feature Selection: A Comprehensive Analysis of SHAP Values.](https://link.springer.com/article/10.1186/s40537-024-00905-w)
