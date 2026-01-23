# Honest Estimation

Honest estimation is a sample-splitting technique that **reduces adaptive bias**
in leaf estimation by separating the data used for tree construction from the
data used for estimating leaf values.

**Scope note.** In citrees, `honesty=True` performs a single random split of the
training set: one part is used to learn the tree structure, and the other part
is used to re-estimate leaf values. This can yield **conditional unbiasedness of
leaf means/proportions** under standard i.i.d. assumptions, but it does **not by
itself** provide confidence intervals, hypothesis tests, or causal effect
estimation.

## Overview

| Aspect        | Description                                                                                |
| ------------- | ------------------------------------------------------------------------------------------ |
| Purpose       | Reduce adaptive bias in leaf estimation                                                    |
| Method        | Sample splitting (single split in citrees)                                                 |
| Trade-off     | Fewer samples for structure and estimation (often higher variance)                         |
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
   n_est ← ⌊η × n⌋       # honesty_fraction determines estimation sample size
   n_split ← n - n_est   # remainder used for tree structure

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

### Conditional unbiasedness of leaf values (what we can safely claim)

Let the training set be split at random into two disjoint subsets:

- a **splitting** sample $S$ used to learn the tree structure (the partition),
  and
- an **estimation** sample $E$ used to estimate leaf values.

Let $\Pi$ denote the (random) partition learned from $S$, and let $L\in\Pi$ be
any leaf. For regression, define the honest leaf mean using only
estimation-sample points that fall into the leaf:

$$
\widehat{\mu}(L) := \frac{1}{|E(L)|}\sum_{i\in E(L)} Y_i,\qquad E(L):=\{i\in E : X_i\in L\},
$$

whenever $|E(L)|\ge 1$.

Under standard i.i.d. sampling and an **independent random split** $(S,E)$,
$\widehat{\mu}(L)$ is unbiased conditional on the learned partition (on leaves
that receive estimation samples):

$$
\mathbb{E}\!\left[\widehat{\mu}(L)\mid \Pi,\ |E(L)|\ge 1\right]=\mu(L):=\mathbb{E}[Y\mid X\in L].
$$

For classification, the same statement holds with $\widehat{\mu}(L)$ replaced by
the empirical class proportions in the leaf.

**Important caveat (empty leaves).** If a leaf receives **no** estimation
samples ($|E(L)|=0$), the honest estimator is undefined. In citrees, such leaves
keep their original value from the splitting sample, so the
conditional-unbiasedness statement above does not apply to those leaves.

### What we do not claim here

- We do not claim that `honesty=True` alone yields confidence intervals or
  hypothesis tests in citrees.
- Asymptotic normality/inference results exist in the literature for certain
  _honest random forests_ under additional conditions; see Wager & Athey (2018)
  for context.

---

## Trade-offs

### Benefits

| Benefit               | Description                                                          |
| --------------------- | -------------------------------------------------------------------- |
| Reduced adaptive bias | Leaf values use data not used to choose splits (on non-empty leaves) |

### Costs

| Cost                | Description                                     |
| ------------------- | ----------------------------------------------- |
| Reduced sample size | Each tree sees only half the data for each task |
| Empty leaves        | Some leaves may have no estimation samples      |
| Variance increase   | Predictions may be more variable                |

### When to Use Honesty

| Use Case                       | Recommendation                                          |
| ------------------------------ | ------------------------------------------------------- |
| Prediction only                | Honesty optional (may hurt accuracy)                    |
| Reduce adaptive bias in leaves | Consider honesty (especially with forests and larger n) |
| Small datasets                 | Consider without honesty                                |
| Large datasets                 | Honesty recommended                                     |

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
    honesty_fraction=0.5,  # 50% for estimation, 50% for splitting
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
| `honesty_fraction` | float | 0.5     | Fraction for estimation (rest for splitting) |

### Choosing `honesty_fraction`

| Value | Splitting | Estimation | Use Case                        |
| ----- | --------- | ---------- | ------------------------------- |
| 0.3   | 70%       | 30%        | Prioritize tree structure       |
| 0.5   | 50%       | 50%        | Balanced (default)              |
| 0.7   | 30%       | 70%        | Prioritize estimation precision |

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
| **Leaf fallback**     | Keep splitting-sample leaf value | Default in citrees |
| **Parent fallback**   | Use parent node prediction       | Not implemented    |
| **Sibling borrowing** | Use sibling leaf prediction      | Not implemented    |
| **Minimum leaf size** | Require min samples in structure | Preventive         |

```python
# Prevent empty leaves with minimum leaf size
tree = ConditionalInferenceTreeClassifier(
    honesty=True,
    min_samples_leaf=5,  # Ensures structure has enough samples
)
```

---

## Configuration Examples

### For Prediction (Honesty Optional)

```python
# Standard tree - may have better prediction
tree = ConditionalInferenceTreeClassifier(
    honesty=False,  # Use all data for everything
)

# Honest tree - reduced adaptive bias but may have higher variance
tree = ConditionalInferenceTreeClassifier(
    honesty=True,
    honesty_fraction=0.5,
)
```

---

## References

1. [Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. JASA.](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1319839)
2. [Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized Random Forests. Annals of Statistics.](https://arxiv.org/abs/1610.01271)
