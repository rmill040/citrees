# Split Criteria (Splitters)

Splitters evaluate the quality of a potential split point. Once a feature is
selected, citrees tests various thresholds and chooses the split that best
separates the target variable.

## Overview

| Splitter  | Task           | Measures            | Range               |
| --------- | -------------- | ------------------- | ------------------- |
| `gini`    | Classification | Impurity            | [0, 0.5] for binary |
| `entropy` | Classification | Information gain    | [0, log₂K]          |
| `mse`     | Regression     | Mean squared error  | [0, ∞)              |
| `mae`     | Regression     | Mean absolute error | [0, ∞)              |

---

## Gini Impurity (gini)

**Default for classification.** Measures the probability of incorrect
classification if a random sample was randomly labeled according to the class
distribution.

### Mathematical Definition

For a node with class distribution $p_1, p_2, \ldots, p_K$:

$$Gini = 1 - \sum_{k=1}^{K} p_k^2 = \sum_{k=1}^{K} p_k(1 - p_k)$$

For a split:

$$Gini_{split} = \frac{n_L}{n} Gini_L + \frac{n_R}{n} Gini_R$$

$$\Delta Gini = Gini_{parent} - Gini_{split}$$

### Properties

| Property      | Value                               |
| ------------- | ----------------------------------- |
| Minimum       | 0 (pure node, single class)         |
| Maximum       | 0.5 (binary), $(K-1)/K$ (K classes) |
| Perfect split | $\Delta Gini = Gini_{parent}$       |

### Algorithm

```
Algorithm: Gini Impurity Split
Input: Feature x ∈ ℝⁿ, labels y ∈ {1,...,K}ⁿ, threshold c

1. Partition data:
   L = {i : xᵢ ≤ c}
   R = {i : xᵢ > c}

2. Compute class distributions:
   For k = 1 to K:
       p_L[k] = |{i ∈ L : yᵢ = k}| / |L|
       p_R[k] = |{i ∈ R : yᵢ = k}| / |R|

3. Compute Gini for each child:
   Gini_L = 1 - Σₖ p_L[k]²
   Gini_R = 1 - Σₖ p_R[k]²

4. Compute weighted split Gini:
   Gini_split = (|L|/n) · Gini_L + (|R|/n) · Gini_R

5. Return Gini_split (lower is better)
```

### Implementation

```python
@njit(cache=True, fastmath=True, nogil=True)
def gini(y_left, y_right, n_classes):
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    if n_left == 0 or n_right == 0:
        return 1.0  # Invalid split

    # Count classes in each partition
    counts_left = np.zeros(n_classes)
    counts_right = np.zeros(n_classes)

    for i in range(n_left):
        counts_left[y_left[i]] += 1
    for i in range(n_right):
        counts_right[y_right[i]] += 1

    # Compute Gini
    gini_left = 1.0 - np.sum((counts_left / n_left) ** 2)
    gini_right = 1.0 - np.sum((counts_right / n_right) ** 2)

    # Weighted average
    return (n_left * gini_left + n_right * gini_right) / n_total
```

### Comparison with Entropy

| Aspect             | Gini                     | Entropy              |
| ------------------ | ------------------------ | -------------------- |
| Computation        | Faster (no log)          | Slower               |
| Behavior           | Favors larger partitions | More balanced splits |
| Typical difference | Minimal                  | Minimal              |

In practice, Gini and entropy produce similar trees. Gini is preferred for
computational efficiency.

---

## Entropy / Information Gain (entropy)

Measures the expected information content of the class distribution. Based on
Shannon's information theory.

### Mathematical Definition

Entropy of a node:

$$H = -\sum_{k=1}^{K} p_k \log_2(p_k)$$

Information gain from a split:

$$IG = H_{parent} - \left[\frac{n_L}{n} H_L + \frac{n_R}{n} H_R\right]$$

### Properties

| Property         | Value                             |
| ---------------- | --------------------------------- |
| Minimum entropy  | 0 (pure node)                     |
| Maximum entropy  | $\log_2 K$ (uniform distribution) |
| Information gain | Higher = better split             |

### Algorithm

```
Algorithm: Entropy Split
Input: Feature x ∈ ℝⁿ, labels y ∈ {1,...,K}ⁿ, threshold c

1. Partition data:
   L = {i : xᵢ ≤ c}
   R = {i : xᵢ > c}

2. Compute class probabilities:
   For k = 1 to K:
       p_L[k] = |{i ∈ L : yᵢ = k}| / |L|
       p_R[k] = |{i ∈ R : yᵢ = k}| / |R|

3. Compute entropy for each child:
   H_L = -Σₖ p_L[k] · log₂(p_L[k])  (0·log(0) = 0 by convention)
   H_R = -Σₖ p_R[k] · log₂(p_R[k])

4. Compute weighted split entropy:
   H_split = (|L|/n) · H_L + (|R|/n) · H_R

5. Return H_split (lower is better)
```

### Interpretation

| H value    | Interpretation             |
| ---------- | -------------------------- |
| 0          | Pure node (all same class) |
| 1.0        | Binary, 50/50 split        |
| $\log_2 K$ | K classes, uniform         |

---

## Mean Squared Error (mse)

**Default for regression.** Measures the average squared deviation from the mean
prediction in each child node.

### Mathematical Definition

For a split into left (L) and right (R) children:

$$MSE_{split} = \frac{n_L}{n} \cdot \frac{1}{n_L}\sum_{i \in L}(y_i - \bar{y}_L)^2 + \frac{n_R}{n} \cdot \frac{1}{n_R}\sum_{i \in R}(y_i - \bar{y}_R)^2$$

This simplifies to the weighted variance:

$$MSE_{split} = \frac{1}{n}\left[\sum_{i \in L}(y_i - \bar{y}_L)^2 + \sum_{i \in R}(y_i - \bar{y}_R)^2\right]$$

### Properties

| Property      | Value                       |
| ------------- | --------------------------- |
| Minimum       | 0 (perfect prediction)      |
| Maximum       | $\sigma^2$ (no improvement) |
| Optimal split | Minimizes total variance    |

### Algorithm

```
Algorithm: MSE Split
Input: Feature x ∈ ℝⁿ, target y ∈ ℝⁿ, threshold c

1. Partition data:
   L = {i : xᵢ ≤ c}
   R = {i : xᵢ > c}

2. Compute means:
   ȳ_L = (1/|L|) Σᵢ∈L yᵢ
   ȳ_R = (1/|R|) Σᵢ∈R yᵢ

3. Compute MSE for each child:
   MSE_L = (1/|L|) Σᵢ∈L (yᵢ - ȳ_L)²
   MSE_R = (1/|R|) Σᵢ∈R (yᵢ - ȳ_R)²

4. Compute weighted MSE:
   MSE_split = (|L|/n) · MSE_L + (|R|/n) · MSE_R

5. Return MSE_split (lower is better)
```

### Efficient Computation

Using the variance formula $Var(Y) = E[Y^2] - E[Y]^2$:

```python
@njit(cache=True, fastmath=True, nogil=True)
def mse_split(y_left, y_right):
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    if n_left == 0 or n_right == 0:
        return np.inf

    # Using variance formula: Var = E[Y²] - E[Y]²
    mean_left = np.mean(y_left)
    mean_right = np.mean(y_right)

    mse_left = np.mean((y_left - mean_left) ** 2)
    mse_right = np.mean((y_right - mean_right) ** 2)

    return (n_left * mse_left + n_right * mse_right) / n_total
```

---

## Mean Absolute Error (mae)

More robust to outliers than MSE. Uses absolute deviations instead of squared
deviations.

### Mathematical Definition

$$MAE_{split} = \frac{n_L}{n} \cdot \frac{1}{n_L}\sum_{i \in L}|y_i - \text{median}(y_L)| + \frac{n_R}{n} \cdot \frac{1}{n_R}\sum_{i \in R}|y_i - \text{median}(y_R)|$$

Note: MAE is minimized by the **median**, not the mean.

### Properties

| Property           | Value                            |
| ------------------ | -------------------------------- |
| Robustness         | More robust to outliers than MSE |
| Gradient           | Non-smooth at 0                  |
| Optimal prediction | Median                           |

### Algorithm

```
Algorithm: MAE Split
Input: Feature x ∈ ℝⁿ, target y ∈ ℝⁿ, threshold c

1. Partition data:
   L = {i : xᵢ ≤ c}
   R = {i : xᵢ > c}

2. Compute medians:
   m_L = median(y_L)
   m_R = median(y_R)

3. Compute MAE for each child:
   MAE_L = (1/|L|) Σᵢ∈L |yᵢ - m_L|
   MAE_R = (1/|R|) Σᵢ∈R |yᵢ - m_R|

4. Compute weighted MAE:
   MAE_split = (|L|/n) · MAE_L + (|R|/n) · MAE_R

5. Return MAE_split (lower is better)
```

### Comparison: MSE vs MAE

| Aspect              | MSE               | MAE                   |
| ------------------- | ----------------- | --------------------- |
| Outlier sensitivity | High              | Low                   |
| Gradient            | Smooth            | Non-smooth            |
| Central tendency    | Mean              | Median                |
| Computation         | O(n)              | O(n log n) for median |
| Common use          | Most applications | Robust regression     |

---

## Permutation Tests for Splitters

In citrees, the selected split is validated using a permutation test:

### Null Hypothesis

$$H_0: \text{The split provides no improvement in prediction}$$

### Test Statistic

The test statistic is the impurity reduction:

$$\Delta = \text{Impurity}_{parent} - \text{Impurity}_{split}$$

### Permutation Procedure

```
Algorithm: Splitter Permutation Test
Input: x ∈ ℝⁿ, y, threshold c, n_resamples

1. Compute observed statistic:
   Δ_obs = Impurity(y) - Impurity_split(x, y, c)

2. Generate null distribution:
   For b = 1 to n_resamples:
       y_perm = shuffle(y)
       Δ_perm[b] = Impurity(y_perm) - Impurity_split(x, y_perm, c)

3. Compute p-value:
   p = (1 + Σ_b 𝟙[Δ_perm[b] ≥ Δ_obs]) / (1 + n_resamples)

4. Return p
```

The `+1` in numerator and denominator is a finite-sample correction ensuring
valid p-values.

---

## Threshold Methods

citrees supports multiple methods for generating candidate split thresholds:

| Method       | Description       | Use Case                |
| ------------ | ----------------- | ----------------------- |
| `exact`      | All unique values | Small datasets, precise |
| `random`     | Random subset     | Large datasets          |
| `percentile` | Quantile-based    | Robust to outliers      |
| `histogram`  | Equal-width bins  | Very large datasets     |

### Algorithm: Threshold Generation

```
Algorithm: Generate Thresholds
Input: x ∈ ℝⁿ, method, max_thresholds

Case method:
    "exact":
        thresholds = unique(x)[:-1]  # All but last

    "random":
        unique_vals = unique(x)
        k = min(max_thresholds, len(unique_vals))
        thresholds = random_sample(unique_vals, k)

    "percentile":
        percentiles = linspace(0, 100, max_thresholds + 2)[1:-1]
        thresholds = percentile(x, percentiles)

    "histogram":
        bins = linspace(min(x), max(x), max_thresholds + 1)
        thresholds = bins[1:-1]

Return thresholds
```

---

## References

1. **Gini Impurity**: Breiman, L., Friedman, J., Stone, C.J., & Olshen, R.A.
   (1984). Classification and Regression Trees. CRC Press.

2. **Information Gain**: Quinlan, J.R. (1986). Induction of Decision Trees.
   Machine Learning.

3. **MSE/MAE**: Friedman, J.H. (2001). Greedy Function Approximation: A Gradient
   Boosting Machine. Annals of Statistics.

4. **Permutation Tests for Splits**: Hothorn, T., Hornik, K., & Zeileis, A.
   (2006). Unbiased Recursive Partitioning. JCGS.
