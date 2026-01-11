# Honest Estimation

Honest estimation is a sample-splitting technique that provides unbiased
predictions and valid inference from decision trees. It separates the data used
for tree construction from the data used for leaf estimation, eliminating the
overfitting bias inherent in standard trees.

## Overview

| Aspect        | Description                                                                                |
| ------------- | ------------------------------------------------------------------------------------------ |
| Purpose       | Unbiased leaf predictions, valid confidence intervals                                      |
| Method        | Sample splitting (cross-fitting)                                                           |
| Trade-off     | Reduced effective sample size for higher statistical validity                              |
| Key reference | [Wager & Athey (2018)](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1319839) |

---

## The Problem with Standard Trees

### Adaptive Bias

Standard decision trees use the **same data** for both:

1. **Structure selection** - Choosing split points
2. **Leaf estimation** - Computing predictions

This creates **adaptive bias**: the tree structure is optimized to fit the
training data, making leaf estimates overly optimistic.

```
Standard Tree Bias
──────────────────

Training Data ─────┬──────────────────────────────────────┐
                   │                                       │
                   ▼                                       ▼
            ┌─────────────┐                        ┌─────────────┐
            │   Choose    │                        │   Estimate  │
            │   Splits    │                        │   Leaves    │
            └─────────────┘                        └─────────────┘
                   │                                       │
                   │              SAME DATA                │
                   │         ═══════════════════           │
                   │         Predictions biased            │
                   │         toward training data          │
                   ▼                                       ▼
            ┌─────────────────────────────────────────────────┐
            │              BIASED TREE                         │
            │  - Overfit predictions                           │
            │  - Invalid confidence intervals                  │
            │  - Inflated feature importance                   │
            └─────────────────────────────────────────────────┘
```

### Consequences

| Issue             | Description                                  |
| ----------------- | -------------------------------------------- |
| Overfitting       | Leaf predictions fit noise in training data  |
| Invalid inference | Confidence intervals have incorrect coverage |
| Biased importance | Features appear more important than they are |

---

## Honest Trees: The Solution

### Core Idea

**Honest trees** use different subsamples for structure and estimation:

```
Honest Tree Architecture
────────────────────────

Training Data ─────────────────────────────────────────────
                          │
                    ┌─────┴─────┐
                    │   SPLIT   │
                    └─────┬─────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
   ┌─────────────┐                 ┌─────────────┐
   │  Splitting  │                 │ Estimation  │
   │  Subsample  │                 │  Subsample  │
   │     Jˢ      │                 │     Jᵉ      │
   └──────┬──────┘                 └──────┬──────┘
          │                               │
          ▼                               ▼
   ┌─────────────┐                 ┌─────────────┐
   │   Build     │                 │  Populate   │
   │   Tree      │                 │   Leaves    │
   │  Structure  │                 │             │
   └──────┬──────┘                 └──────┬──────┘
          │                               │
          └───────────────┬───────────────┘
                          │
                          ▼
            ┌─────────────────────────────┐
            │        HONEST TREE          │
            │  - Unbiased predictions     │
            │  - Valid confidence intervals│
            │  - Correct coverage         │
            └─────────────────────────────┘
```

### Mathematical Definition

Given training data $\{(X_i, Y_i)\}_{i=1}^n$, an honest tree:

1. **Splits** the data into disjoint subsets:
   - $\mathcal{J}^s$ (splitting sample) for tree structure
   - $\mathcal{J}^e$ (estimation sample) for leaf predictions

2. **Builds** the tree using only $\mathcal{J}^s$

3. **Estimates** leaf values using only $\mathcal{J}^e$:
   $$\hat{\mu}(x) = \frac{1}{|L(x) \cap \mathcal{J}^e|} \sum_{i \in L(x) \cap \mathcal{J}^e} Y_i$$

where $L(x)$ is the leaf containing point $x$.

---

## Algorithm

### Honest Tree Construction

```
Algorithm: Honest Decision Tree
Input: Data (X, y), honesty_fraction η ∈ (0, 1)

1. Sample Splitting:
   n ← |y|
   n_split ← ⌊η × n⌋
   n_est ← n - n_split

   # Random partition
   indices ← RandomPermutation(1, ..., n)
   J_split ← indices[1:n_split]
   J_est ← indices[n_split+1:n]

2. Tree Construction (using J_split only):
   tree ← BuildTree(X[J_split], y[J_split])
   # Standard recursive partitioning
   # Only splitting sample influences structure

3. Leaf Estimation (using J_est only):
   For each leaf L in tree:
       # Find estimation samples in this leaf
       est_indices ← {i ∈ J_est : X[i] falls in L}

       If |est_indices| > 0:
           L.prediction ← mean(y[est_indices])  # Regression
           L.prediction ← mode(y[est_indices])  # Classification
       Else:
           L.prediction ← tree_default  # Fallback

4. Return tree
```

### Honest Forest Construction

For forests, honesty is applied to each tree independently:

```
Algorithm: Honest Random Forest
Input: Data (X, y), n_estimators T, honesty_fraction η

1. For t = 1 to T:
   # Bootstrap or subsample
   sample_t ← BootstrapSample(X, y)

   # Split into honest subsamples
   J_split_t, J_est_t ← SplitSample(sample_t, η)

   # Build honest tree
   tree_t ← HonestTree(J_split_t, J_est_t)

2. Forest prediction:
   ŷ(x) = (1/T) Σₜ tree_t.predict(x)

3. Return forest
```

---

## Properties of Honest Trees

### Unbiasedness

For honest trees with sufficient data in each leaf:

$$\mathbb{E}[\hat{\mu}(x)] = \mu(x) + O(1/n_{leaf})$$

The bias vanishes as the number of estimation samples per leaf increases.

### Asymptotic Normality

Under regularity conditions, honest tree predictions are asymptotically normal:

$$\sqrt{n}(\hat{\mu}(x) - \mu(x)) \xrightarrow{d} N(0, \sigma^2(x))$$

This enables valid confidence intervals and hypothesis tests.

### Consistency

Honest forests are consistent: as $n \to \infty$,

$$\hat{\mu}(x) \xrightarrow{p} \mu(x)$$

provided the trees grow slowly relative to $n$.

---

## Trade-offs

### Benefits

| Benefit              | Description                                |
| -------------------- | ------------------------------------------ |
| Unbiased predictions | No overfitting to training structure       |
| Valid inference      | Confidence intervals have correct coverage |
| Causal inference     | Enables estimation of treatment effects    |

### Costs

| Cost                | Description                                     |
| ------------------- | ----------------------------------------------- |
| Reduced sample size | Each tree sees only half the data for each task |
| Empty leaves        | Some leaves may have no estimation samples      |
| Variance increase   | Predictions may be more variable                |

### When to Use Honesty

| Use Case                       | Recommendation                       |
| ------------------------------ | ------------------------------------ |
| Prediction only                | Honesty optional (may hurt accuracy) |
| Inference/confidence intervals | Honesty required                     |
| Causal effect estimation       | Honesty required                     |
| Feature importance             | Honesty recommended                  |
| Small datasets                 | Consider without honesty             |
| Large datasets                 | Honesty recommended                  |

---

## Implementation in citrees

### Basic Usage

```python
from citrees import (
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceForestClassifier,
)

# Honest single tree
tree = ConditionalInferenceTreeClassifier(
    honesty=True,
    honesty_fraction=0.5,  # 50% for splitting, 50% for estimation
)
tree.fit(X_train, y_train)

# Honest forest
forest = ConditionalInferenceForestClassifier(
    n_estimators=100,
    honesty=True,
    honesty_fraction=0.5,
)
forest.fit(X_train, y_train)
```

### Parameters

| Parameter          | Type  | Default | Description                                  |
| ------------------ | ----- | ------- | -------------------------------------------- |
| `honesty`          | bool  | False   | Enable honest estimation                     |
| `honesty_fraction` | float | 0.5     | Fraction for splitting (rest for estimation) |

### Choosing `honesty_fraction`

| Value | Splitting | Estimation | Use Case                        |
| ----- | --------- | ---------- | ------------------------------- |
| 0.3   | 30%       | 70%        | Prioritize estimation precision |
| 0.5   | 50%       | 50%        | Balanced (default)              |
| 0.7   | 70%       | 30%        | Prioritize tree structure       |

---

## Honest Estimation for Causal Inference

### Heterogeneous Treatment Effects

Honest trees are essential for **Causal Forests** (Wager & Athey, 2018), which
estimate heterogeneous treatment effects:

$$\tau(x) = \mathbb{E}[Y(1) - Y(0) | X = x]$$

where:

- $Y(1)$ = potential outcome under treatment
- $Y(0)$ = potential outcome under control
- $\tau(x)$ = conditional average treatment effect (CATE)

### Why Honesty Matters for Causality

```
Without Honesty:
───────────────
Tree finds splits that maximize Y_treated - Y_control
→ Splits are optimized for observed difference
→ CATE estimates are biased upward
→ Invalid confidence intervals

With Honesty:
─────────────
Splitting sample finds heterogeneous regions
Estimation sample provides unbiased CATE
→ Valid inference on treatment effects
→ Correct coverage for CIs
```

### Connection to GRF

The [Generalized Random Forests (GRF)](https://grf-labs.github.io/grf/) package
implements honest causal forests in R. citrees provides similar capabilities in
Python.

---

## Handling Empty Leaves

### The Problem

With sample splitting, some leaves may have no estimation samples:

```
Splitting Sample              Estimation Sample
────────────────              ─────────────────

     [Root]                        [Root]
      / \                           / \
     /   \                         /   \
  [L1]   [L2]                   [L1]   [L2]
   / \     |                     |       |
  /   \    |                     |       |
[L3] [L4] [L5]                 [L3]   [L5]
 ▲     ▲                              ▲
 │     │                              │
 5     3    samples                   0     samples
                                      ↑
                                  EMPTY LEAF!
```

### Solutions

| Strategy              | Description                      | Implementation     |
| --------------------- | -------------------------------- | ------------------ |
| **Parent fallback**   | Use parent node prediction       | Default in citrees |
| **Sibling borrowing** | Use sibling leaf prediction      | Alternative        |
| **Minimum leaf size** | Require min samples in structure | Preventive         |

```python
# Prevent empty leaves with minimum leaf size
tree = ConditionalInferenceTreeClassifier(
    honesty=True,
    min_samples_leaf=5,  # Ensures structure has enough samples
)
```

---

## Comparison with Other Methods

### Honest Trees vs. Pruning

| Aspect     | Honest Trees       | Pruning             |
| ---------- | ------------------ | ------------------- |
| Goal       | Unbiased inference | Prevent overfitting |
| Method     | Sample splitting   | Complexity penalty  |
| Inference  | Valid CIs          | Invalid CIs         |
| Prediction | May be worse       | Optimized           |

### Honest Trees vs. Cross-Validation

| Aspect      | Honest Trees | Cross-Validation |
| ----------- | ------------ | ---------------- |
| Data usage  | Single split | K-fold splits    |
| Inference   | Valid        | Invalid          |
| Computation | 1× training  | K× training      |
| Structure   | Fixed        | Averaged         |

---

## Recent Research

### Accuracy Limits (2024-2025)

Recent work by
[Cattaneo, Klusowski, and Yu (2025)](https://mdcattaneo.github.io/papers/Cattaneo-Klusowski-Yu_2025_CausalTrees.pdf)
establishes theoretical limitations:

- Honest causal trees cannot achieve polynomial convergence rates under basic
  conditions
- Performance depends strongly on regularity of the conditional expectation
  function
- Adaptive methods may exhibit arbitrarily slow convergence

### Practical Guidance

A
[2023 practical guide](https://academic.oup.com/aje/advance-article/doi/10.1093/aje/kwad043/7056288)
in the American Journal of Epidemiology provides recommendations:

1. Use honest forests for CATE estimation
2. Report confidence intervals from the forest
3. Validate with refutation tests
4. Consider ensemble methods over single trees

---

## Configuration Examples

### For Prediction (Honesty Optional)

```python
# Standard tree - may have better prediction
tree = ConditionalInferenceTreeClassifier(
    honesty=False,  # Use all data for everything
)

# Honest tree - unbiased but may have higher variance
tree = ConditionalInferenceTreeClassifier(
    honesty=True,
    honesty_fraction=0.5,
)
```

### For Inference (Honesty Required)

```python
# Valid confidence intervals require honesty
forest = ConditionalInferenceForestClassifier(
    n_estimators=500,  # More trees for stable inference
    honesty=True,
    honesty_fraction=0.5,
    min_samples_leaf=5,  # Prevent empty leaves
)
```

### For Causal Inference

```python
# Heterogeneous treatment effect estimation
forest = ConditionalInferenceForestRegressor(
    n_estimators=1000,
    honesty=True,
    honesty_fraction=0.5,
    max_depth=None,  # Grow deep trees
    min_samples_leaf=5,
)
```

---

## References

1. **Foundational Paper**:
   [Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. JASA, 113(523), 1228-1242.](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1319839)

2. **Generalized Random Forests**:
   [Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized Random Forests. Annals of Statistics, 47(2), 1148-1178.](https://arxiv.org/abs/1610.01271)

3. **GRF Package**: [grf-labs.github.io/grf/](https://grf-labs.github.io/grf/)

4. **Practical Guide**:
   [Haggerty, C. J., et al. (2023). Practical Guide to Honest Causal Forests for Identifying Heterogeneous Treatment Effects. American Journal of Epidemiology.](https://academic.oup.com/aje/advance-article/doi/10.1093/aje/kwad043/7056288)

5. **Theoretical Limits**:
   [Cattaneo, M. D., Klusowski, J. M., & Yu, Z. (2025). The Honest Truth About Causal Trees: Accuracy Limits.](https://mdcattaneo.github.io/papers/Cattaneo-Klusowski-Yu_2025_CausalTrees.pdf)

6. **Applied Review**:
   [Brand, J. E., et al. (2024). How Do Applied Researchers Use the Causal Forest? A Methodological Review.](https://arxiv.org/abs/2404.13356)
