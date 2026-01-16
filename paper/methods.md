# citrees: Detailed Methods and Algorithms

This document provides comprehensive algorithmic details for the citrees library, intended to serve as the foundation for a methods section in a research paper. It complements `theory.md` which focuses on theoretical guarantees and proofs.

## Table of Contents

1. [Algorithm Overview](#1-algorithm-overview)
2. [Tree Building Algorithm](#2-tree-building-algorithm)
3. [Feature Selection (Stage A)](#3-feature-selection-stage-a)
4. [Split Selection (Stage B)](#4-split-selection-stage-b)
5. [Permutation Testing](#5-permutation-testing)
6. [Sequential Permutation Testing](#6-sequential-permutation-testing)
7. [Multiple Testing Correction](#7-multiple-testing-correction)
8. [Multi-Selector Mode](#8-multi-selector-mode)
9. [Computational Heuristics](#9-computational-heuristics)
10. [Forest Ensemble](#10-forest-ensemble)
11. [Honest Estimation](#11-honest-estimation)
12. [Implementation Details](#12-implementation-details)

---

## 1. Algorithm Overview

citrees implements conditional inference trees and forests that use permutation-based hypothesis testing for variable and split selection. The key innovation over CART-style trees is replacing greedy optimization with statistical testing, which provides:

1. **Unbiased variable selection**: Features are selected based on statistical significance rather than impurity optimization, eliminating selection bias toward high-cardinality features.

2. **Built-in stopping criterion**: The tree stops growing when no feature shows statistically significant association with the response, providing principled regularization.

3. **Interpretable p-values**: Each split decision is accompanied by a p-value quantifying the evidence against the null hypothesis of no association.

### High-Level Algorithm

```
Algorithm 1: Conditional Inference Tree
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Training data (X, y), significance levels α_sel, α_split
Output: Decision tree T

function BuildTree(X, y, depth):
    if StoppingCriteria(X, y, depth):
        return LeafNode(value = AggregateResponse(y))

    # Stage A: Feature Selection
    F ← SelectCandidateFeatures(X)
    (j*, p*_sel) ← TestFeatureAssociation(X, y, F)

    if p*_sel ≥ α_sel / |F|:  # Bonferroni-adjusted threshold
        return LeafNode(value = AggregateResponse(y))

    # Stage B: Split Selection
    C ← GenerateThresholdCandidates(X[:, j*])
    (c*, p*_split) ← TestSplitQuality(X[:, j*], y, C)

    if p*_split ≥ α_split / |C|:  # Bonferroni-adjusted threshold
        return LeafNode(value = AggregateResponse(y))

    # Check impurity decrease constraint
    if ImpurityDecrease(y, X[:, j*], c*) < min_impurity_decrease:
        return LeafNode(value = AggregateResponse(y))

    # Recursive split
    (X_L, y_L), (X_R, y_R) ← SplitData(X, y, j*, c*)
    left_child ← BuildTree(X_L, y_L, depth + 1)
    right_child ← BuildTree(X_R, y_R, depth + 1)

    return InternalNode(feature=j*, threshold=c*, left=left_child, right=right_child)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 2. Tree Building Algorithm

### 2.1 Stopping Criteria

The tree building process terminates at a node when any of the following conditions are met:

1. **Sample size**: $n_t < \text{min\_samples\_split}$ (default: 2)
2. **Maximum depth**: $\text{depth} > \text{max\_depth}$
3. **Pure node**: All labels $y_t$ are identical
4. **No significant features**: Stage A fails to reject any null hypothesis
5. **No significant splits**: Stage B fails to reject any null hypothesis
6. **Insufficient impurity decrease**: $\Delta I < \text{min\_impurity\_decrease}$
7. **Minimum leaf size violated**: $n_{\text{left}} < \text{min\_samples\_leaf}$ or $n_{\text{right}} < \text{min\_samples\_leaf}$

### 2.2 Leaf Value Computation

**Classification**: The leaf value is a probability vector over classes:
$$
\hat{p}_k(L) = \frac{1}{|L|} \sum_{i \in L} \mathbf{1}\{y_i = k\}, \quad k = 1, \ldots, K
$$

**Regression**: The leaf value is the mean of responses:
$$
\hat{\mu}(L) = \frac{1}{|L|} \sum_{i \in L} y_i
$$

---

## 3. Feature Selection (Stage A)

### 3.1 Overview

At each node $t$ with samples $(X_t, y_t)$ where $n_t = |I_t|$, we test for association between each candidate feature $X_{t,j}$ and the response $y_t$.

### 3.2 Null Hypothesis

For each feature $j$ in the candidate set $F_t$:
$$
H^{\text{sel}}_{t,j}: X_{t,j} \perp y_t
$$
i.e., feature $j$ is independent of the response at node $t$.

### 3.3 Test Statistics (Selectors)

citrees implements the following association measures:

#### 3.3.1 Multiple Correlation (mc) - Classification

The multiple correlation coefficient measures the strength of linear association between a feature and class membership:

$$
\text{mc}(x, y) = \sqrt{\frac{\text{SSB}}{\text{SST}}}
$$

where:
- **SST** (Total Sum of Squares): $\text{SST} = \sum_{i=1}^n (x_i - \bar{x})^2$
- **SSB** (Between-class Sum of Squares): $\text{SSB} = \sum_{k=1}^K n_k (\bar{x}_k - \bar{x})^2$
- $\bar{x}$ is the overall mean
- $\bar{x}_k$ is the mean of feature values for class $k$
- $n_k$ is the number of samples in class $k$

**Properties**:
- Range: $[0, 1]$
- $\text{mc} = 0$ when class means are identical (no linear association)
- $\text{mc} = 1$ when classes perfectly separate on the feature

**Implementation** (`citrees/_selector.py:356-420`):
```python
@njit(cache=True, nogil=True, fastmath=True)
def mc(x, y, n_classes, random_state=None):
    mu = x.mean()
    sst = np.sum((x - mu) ** 2)
    ssb = 0.0
    for j in range(n_classes):
        x_j = x[y == j]
        if len(x_j) > 0:
            mu_j = x_j.mean()
            ssb += len(x_j) * (mu_j - mu) ** 2
    return np.sqrt(ssb / sst) if sst > 0 else 0.0
```

#### 3.3.2 Mutual Information (mi) - Classification

Mutual information quantifies the information shared between the feature and class labels:

$$
I(X; Y) = \sum_{k=1}^K \int p(x, y=k) \log \frac{p(x, y=k)}{p(x) p(y=k)} dx
$$

citrees uses the scikit-learn implementation based on k-nearest neighbor estimation (Kraskov et al., 2004).

**Properties**:
- Range: $[0, \infty)$ (unbounded)
- Non-negative: $I(X; Y) \geq 0$
- $I(X; Y) = 0$ if and only if $X \perp Y$

**Important**: Because MI is unbounded, it cannot be combined with other selectors in multi-selector mode.

#### 3.3.3 Pearson Correlation (pc) - Regression

The absolute Pearson correlation coefficient:

$$
\text{pc}(x, y) = \left| \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}} \right|
$$

**Properties**:
- Range: $[0, 1]$ (after absolute value)
- Measures linear association only
- $\text{pc} = 0$ does not imply independence (only uncorrelatedness)

**Implementation** (`citrees/_selector.py:518-558`):
```python
@njit(cache=True, nogil=True, fastmath=True)
def _correlation(x, y):
    n = len(x)
    sx, sy, sx2, sy2, sxy = 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        sx += x[i]; sy += y[i]
        sx2 += x[i]*x[i]; sy2 += y[i]*y[i]
        sxy += x[i]*y[i]
    cov = n * sxy - sx * sy
    denom = np.sqrt((n * sx2 - sx*sx) * (n * sy2 - sy*sy))
    return cov / denom if denom > 0 else 0.0
```

#### 3.3.4 Distance Correlation (dc) - Regression

Distance correlation (Székely et al., 2007) detects both linear and nonlinear dependencies:

$$
\text{dCor}(X, Y) = \frac{\text{dCov}(X, Y)}{\sqrt{\text{dVar}(X) \cdot \text{dVar}(Y)}}
$$

where distance covariance is defined via doubly-centered distance matrices.

**Properties**:
- Range: $[0, 1]$
- $\text{dCor}(X, Y) = 0$ if and only if $X \perp Y$ (under mild conditions)
- Detects nonlinear dependencies
- Complexity: $O(n^2)$

**Implementation**: citrees wraps the `dcor` library.

#### 3.3.5 Randomized Dependence Coefficient (rdc) - Both

The RDC (Lopez-Paz et al., 2013) is a computationally efficient nonlinear dependence measure:

$$
\text{rdc}(x, y) = \max_{j,k} |\text{corr}(\phi_j(F_x(x)), \phi_k(F_y(y)))|
$$

where:
- $F_x, F_y$ are empirical CDFs (rank transform)
- $\phi_j$ are random nonlinear features: $\phi(u) = [\cos(w^\top u), \sin(w^\top u)]$
- $w \sim \mathcal{N}(0, s^2 I)$ with bandwidth $s = 1/6$

**Algorithm**:
```
Algorithm 2: Randomized Dependence Coefficient
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Vectors x, y ∈ ℝⁿ, number of projections k=10, bandwidth s=1/6
Output: RDC score ∈ [0, 1]

1. Apply empirical CDF transform:
   u_i ← rank(x_i) / n,  v_i ← rank(y_i) / n

2. Augment with bias term:
   U ← [u, 1] ∈ ℝⁿˣ²,  V ← [v, 1] ∈ ℝⁿˣ²

3. Generate random projections:
   W_x, W_y ~ N(0, s²) ∈ ℝ²ˣᵏ

4. Create nonlinear features:
   Φ_x ← [cos(U W_x), sin(U W_x)] ∈ ℝⁿˣ²ᵏ
   Φ_y ← [cos(V W_y), sin(V W_y)] ∈ ℝⁿˣ²ᵏ

5. Return max |corr(Φ_x[:, j], Φ_y[:, k])| over all j, k
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Properties**:
- Range: $[0, 1]$
- Complexity: $O(n \log n)$ due to sorting for rank transform
- Detects nonlinear dependencies
- Consistent: converges to copula-based dependence measure

**Implementation** (`citrees/_selector.py:613-731`):
```python
@njit(cache=True, nogil=True, fastmath=True)
def _rdc(x, y, k, s, seed):
    X_feat = _rdc_features(x, k, s, seed)
    Y_feat = _rdc_features(y, k, s, seed + 1000)
    return _rdc_cancor(X_feat, Y_feat)
```

### 3.4 Feature Selection Procedure

```
Algorithm 3: Feature Selection at Node t
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Data (X_t, y_t), candidate features F_t, α_sel, B permutations, adjust_alpha
Output: Best feature j*, p-value p*, rejection decision

1. If |F_t| > max_features:
   F_t ← RandomSubset(F_t, max_features)

2. If feature_scanning and early_stopping enabled:
   F_t ← SortByAssociationScore(F_t, X_t, y_t)  # Most promising first

3. Set effective threshold:
   α_eff ← α_sel/|F_t| if adjust_alpha else α_sel

4. Initialize: p* ← ∞, j* ← F_t[0]

5. For each feature j in F_t:
   a. Compute association score: θ_0 ← |T(X_t[:,j], y_t)|
   b. Compute p-value via permutation test:
      p_j ← PermutationTest(X_t[:,j], y_t, B, α_eff)

   c. If p_j < p*:
      p* ← p_j, j* ← j

   d. Early stopping check (if enabled):
      If p* < α_eff: break  # first significant new best

   e. Feature muting (if enabled):
      If p_j ≥ max(α_eff, 1-α_eff):
         Remove j from available features globally

6. Return (j*, p*, p* < α_eff)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 4. Split Selection (Stage B)

### 4.1 Overview

Given the selected feature $j^*$, we find the best threshold $c$ to partition the data into left and right children.

### 4.2 Null Hypothesis

For each threshold $c$ in the candidate set $C_{t,j^*}$:
$$
H^{\text{split}}_{t,j^*,c}: y_t \text{ is exchangeable w.r.t. the partition } (I^L_t, I^R_t)
$$
where $I^L_t = \{i : X_{i,j^*} \leq c\}$ and $I^R_t = I_t \setminus I^L_t$.

### 4.3 Test Statistics (Splitters)

#### 4.3.1 Gini Index - Classification

$$
\text{Gini}(y) = 1 - \sum_{k=1}^K \hat{p}_k^2
$$

where $\hat{p}_k = \frac{1}{n} \sum_{i=1}^n \mathbf{1}\{y_i = k\}$.

**Split statistic** (lower is better):
$$
T^{\text{split}}(c) = \text{Gini}(y_L) + \text{Gini}(y_R)
$$

**Properties**:
- Range: $[0, 1 - 1/K]$
- $\text{Gini} = 0$ for pure nodes
- Maximum when all classes are equally represented

**Implementation** (`citrees/_splitter.py:329-349`):
```python
@njit(cache=True, fastmath=True, nogil=True)
def gini(y):
    n = len(y)
    p = np.bincount(y) / n
    return 1 - np.sum(p * p)
```

#### 4.3.2 Entropy - Classification

$$
\text{Entropy}(y) = -\sum_{k=1}^K \hat{p}_k \log_2(\hat{p}_k)
$$

**Properties**:
- Range: $[0, \log_2 K]$
- $\text{Entropy} = 0$ for pure nodes
- Maximum for uniform distribution

**Implementation** (`citrees/_splitter.py:352-373`):
```python
@njit(cache=True, fastmath=True, nogil=True)
def entropy(y):
    n = len(y)
    p = np.bincount(y) / n
    p = p[p != 0]  # Avoid log(0)
    return -np.sum(np.log2(p) * p)
```

#### 4.3.3 Mean Squared Error (MSE) - Regression

$$
\text{MSE}(y) = \frac{1}{n} \sum_{i=1}^n (y_i - \bar{y})^2
$$

This is the empirical variance of the response.

**Implementation** (`citrees/_splitter.py:508-532`):
```python
@njit(cache=True, fastmath=True, nogil=True)
def mse(y):
    dev = y - y.mean()
    return np.mean(dev * dev)
```

#### 4.3.4 Mean Absolute Error (MAE) - Regression

$$
\text{MAE}(y) = \frac{1}{n} \sum_{i=1}^n |y_i - \bar{y}|
$$

More robust to outliers than MSE.

**Implementation** (`citrees/_splitter.py:535-558`):
```python
@njit(cache=True, fastmath=True, nogil=True)
def mae(y):
    dev = np.abs(y - y.mean())
    return np.mean(dev)
```

### 4.4 Threshold Generation Methods

citrees supports four methods for generating candidate thresholds:

#### 4.4.1 Exact (default)

Uses all midpoints between consecutive unique values:
$$
C = \left\{ \frac{x_{(i)} + x_{(i+1)}}{2} : i = 1, \ldots, m-1 \right\}
$$
where $x_{(1)} < x_{(2)} < \cdots < x_{(m)}$ are the unique sorted values.

**Implementation** (`citrees/_threshold_method.py:8-37`):
```python
@njit(cache=True, fastmath=True, nogil=True)
def exact(x, max_thresholds=None, random_state=None):
    values = np.unique(x)
    midpoints = (values[:-1] + values[1:]) / 2
    return midpoints
```

#### 4.4.2 Random

Random subsample of midpoints:
$$
C = \text{RandomSample}(\text{Midpoints}, k)
$$

#### 4.4.3 Percentile

Equally spaced percentiles of midpoints:
$$
C = \{Q_p(\text{Midpoints}) : p \in \text{linspace}(0, 100, k)\}
$$

#### 4.4.4 Histogram

Histogram bin edges of midpoints.

### 4.5 Split Selection Procedure

```
Algorithm 4: Split Selection at Node t
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Feature x = X_t[:,j*], response y_t, α_split, B permutations, adjust_alpha
Output: Best threshold c*, p-value p*, rejection decision

1. Generate threshold candidates:
   C ← ThresholdMethod(x)
   If |C| > max_thresholds:
      C ← Subsample(C, max_thresholds)

2. If threshold_scanning and early_stopping enabled:
   C ← SortByImpurityReduction(C, x, y_t)  # Best splits first

3. Set effective threshold:
   α_eff ← α_split/|C| if adjust_alpha else α_split

4. Initialize: p* ← ∞, c* ← C[0]

5. For each threshold c in C:
   a. Compute split impurity:
      θ_0 ← Impurity(y[x ≤ c]) + Impurity(y[x > c])

   b. Compute p-value via permutation test (LEFT-TAIL):
      p_c ← PermutationTestSplit(x, y_t, c, B, α_eff)

   c. If p_c < p*:
      p* ← p_c, c* ← c

   d. Early stopping (if enabled):
      If p* < α_eff: break  # first significant new best

5. Check deterministic split constraints:
   If min_samples_leaf violated by c*: set reject=False

6. Return (c*, p*, p* < α_eff)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Important**: Split selection uses a **left-tail** test (smaller impurity = better split), while feature selection uses a **right-tail** test (larger association = stronger signal).

---

## 5. Permutation Testing

### 5.1 Core Principle

Under the null hypothesis $H_0: X \perp Y$, the joint distribution of $(X, Y)$ is invariant under permutations of $Y$. Therefore:
$$
T(X, Y) \stackrel{d}{=} T(X, \pi(Y)) \quad \text{for all permutations } \pi
$$

### 5.2 Monte Carlo Permutation Test Algorithm

```
Algorithm 5: Monte Carlo Permutation Test (Fixed-B)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Data (x, y), test statistic T, number of permutations B
Output: P-value p

1. Compute observed statistic: θ_0 ← |T(x, y)|

2. Initialize count: k ← 0

3. For b = 1, ..., B:
   a. Generate random permutation π_b
   b. Compute permuted statistic: θ_b ← |T(x, π_b(y))|
   c. If θ_b ≥ θ_0: k ← k + 1  # (right-tail for selectors)
      # Or: If θ_b ≤ θ_0: k ← k + 1  (left-tail for splitters)

4. Return p-value with +1 correction:
   p ← (k + 1) / (B + 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 5.3 Phipson-Smyth +1 Correction

The p-value formula $p = (k+1)/(B+1)$ instead of $p = k/B$ ensures:

1. **Non-zero p-values**: $p \geq 1/(B+1) > 0$, critical for multiple testing correction
2. **Conservative estimate**: $\mathbb{E}[p | H_0] = p^* + (1-p^*)/(B+1) \geq p^*$
3. **Valid Type I error**: $\mathbb{P}(p \leq \alpha) \leq \alpha$ for all $\alpha$

**Reference**: Phipson & Smyth (2010). "Permutation P-values Should Never Be Zero." SAGMB 9(1):39.

### 5.4 Implementation Details

**Right-tail test** (feature selection - larger = more extreme):
```python
# citrees/_selector.py:78-89
theta = np.abs(func(x, y, func_arg))
theta_p = np.empty(n_resamples)
for i in range(n_resamples):
    np.random.shuffle(y_)
    theta_p[i] = func(x, y_, func_arg)
return (1 + np.sum(np.abs(theta_p) >= theta)) / (1 + n_resamples)
```

**Left-tail test** (split selection - smaller = more extreme):
```python
# citrees/_splitter.py:82-87
idx = x <= threshold
theta = func(y[idx]) + func(y[~idx])
for i in range(n_resamples):
    np.random.shuffle(y_)
    theta_p[i] = func(y_[idx]) + func(y_[~idx])
return (1 + np.sum(theta_p <= theta)) / (1 + n_resamples)
```

---

## 6. Sequential Permutation Testing

citrees implements sequential Monte Carlo stopping rules to reduce computational cost in permutation testing.

**Important inferential note.** When `early_stopping_*=None`, citrees reports fixed-$B$ Monte Carlo permutation p-values
with the Phipson–Smyth (+1) correction, and standard permutation-test guarantees apply. When
`early_stopping_* ∈ {"adaptive","simple"}`, the algorithm may stop at a data-dependent time and returns the +1 Monte
Carlo estimate evaluated at that stopping time; this number should not be treated as a classical fixed-$B$ p-value.
For publication-grade p-value claims, use fixed-$B$ mode (`early_stopping_*=None`) and report $B$ explicitly.

### 6.1 Motivation

In a fixed-$B$ permutation test:
- Under $H_0$: Need many permutations to establish non-significance
- Under $H_1$: Few permutations often suffice to establish significance
- Problem: Fixed-$B$ is wasteful when the result is "obvious"

### 6.2 Simple Sequential (Baseline)

Stops early under two conditions:

1. **Significance**: Current p-value $< \alpha$ (after minimum resamples)
2. **Futility**: Best possible p-value $\geq \alpha$ (cannot reject)

```
Algorithm 6: Simple Sequential Permutation Test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Data (x, y), statistic T, max permutations B, threshold α
Output: P-value p

1. Compute θ_0 ← |T(x, y)|
2. min_resamples ← ⌈1/α⌉
3. k ← 0  # extreme count

4. For b = 1, ..., B:
   a. Permute and compute: θ_b ← |T(x, π_b(y))|
   b. If θ_b ≥ θ_0: k ← k + 1
   c. p_current ← (k + 1) / (b + 1)

   d. If b ≥ min_resamples:
      # Early significance
      If p_current < α: return p_current

      # Futility: best possible p-value
      p_best ← (k + 1) / (B + 1)
      If p_best ≥ α and k ≥ 3: return p_current

5. Return (k + 1) / (B + 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Warning**: This method inflates Type I error to ~9% because it "peeks" at running p-values without proper adjustment.

### 6.3 Adaptive Sequential (Bayesian)

Uses Bayesian posterior to make stopping decisions:

**Model**: After $n$ permutations with $k$ exceedances:
$$
k \mid p \sim \text{Binomial}(n, p)
$$
$$
p \sim \text{Beta}(1, 1) \quad \text{(uniform prior)}
$$
$$
p \mid k, n \sim \text{Beta}(1 + k, 1 + n - k)
$$

**Stopping rule**: Stop when confident about significance/non-significance:
$$
P(p < \alpha \mid k, n) \geq \gamma \quad \text{or} \quad P(p \geq \alpha \mid k, n) \geq \gamma
$$
where $\gamma = 0.95$ (default confidence).

```
Algorithm 7: Adaptive Sequential Permutation Test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Data (x, y), statistic T, max permutations B, threshold α, confidence γ
Output: P-value p

1. Compute θ_0 ← |T(x, y)|
2. min_resamples ← ⌈1/α⌉
3. k ← 0  # extreme count

4. For b = 1, ..., B:
   a. Permute and compute: θ_b ← |T(x, π_b(y))|
   b. If θ_b ≥ θ_0: k ← k + 1

   c. If b ≥ min_resamples:
      # Compute Beta CDF: P(p < α | k, b)
      prob_sig ← BetaCDF(α; 1+k, 1+b-k)

      # Confident significant
      If prob_sig ≥ γ: return (k + 1) / (b + 1)

      # Confident non-significant
      If (1 - prob_sig) ≥ γ: return (k + 1) / (b + 1)

5. Return (k + 1) / (B + 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 6.4 Beta CDF Computation

The Beta CDF $I_\alpha(a, b) = P(X \leq \alpha)$ for $X \sim \text{Beta}(a, b)$ is computed using Lentz's continued fraction algorithm:

```python
# citrees/_sequential.py:17-68
@njit(cache=True, fastmath=True)
def _beta_cdf(x, a, b):
    if x <= 0: return 0.0
    if x >= 1: return 1.0

    # Use symmetry for numerical stability
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _beta_cdf(1 - x, b, a)

    # Continued fraction expansion
    log_prefix = a * log(x) + b * log(1-x) - log(a)
    log_prefix += lgamma(a+b) - lgamma(a) - lgamma(b)

    # Lentz's algorithm for continued fraction
    # ... (see implementation)

    return exp(log_prefix) * result
```

### 6.5 Empirical Performance

The tradeoff is best understood empirically via null/signal simulations; see `paper/scripts/theory/` for reproducible
calibration scripts.

Example summary (illustrative; regenerate from scripts for the paper):

| Method | Type I Error | Avg Perms (null) | Power | Speedup |
|--------|-------------|------------------|-------|---------|
| Fixed-B (1000) | 5.6% | 1000 | 97.0% | 1x |
| Simple | **9.1%** | 135 | 97.8% | 7x |
| Adaptive (γ=0.95) | **5.5%** | 48 | 96.4% | **21x** |

Adaptive stopping can reduce computation substantially on clearly non-significant tests while keeping the *rejection
rate* close to nominal in many regimes; however, fixed-$B$ mode remains the clean option for classical p-value claims.

---

## 7. Multiple Testing Correction

### 7.1 Bonferroni Correction

At each node, citrees performs $m$ hypothesis tests (one per candidate feature or threshold). To control family-wise error rate (FWER), Bonferroni correction is applied:

$$
\text{Reject } H_j \text{ if } p_j < \frac{\alpha}{m}
$$

**Properties**:
- FWER $\leq \alpha$ under any dependence structure
- Conservative when tests are positively dependent
- No independence assumptions required

### 7.2 Dynamic Adjustment of Permutation Count

When Bonferroni is enabled, the effective threshold becomes $\alpha/m$. To maintain resolution for smaller thresholds, citrees scales the number of permutations:

**NResamples modes**:

1. **minimum**: $B = \lceil 1/(\alpha/m) \rceil = \lceil m/\alpha \rceil$
   - Minimum required for rejection possibility

2. **maximum**: $B = \lceil 1/(4(\alpha/m)^2) \rceil$
   - High precision, diminishing returns

3. **auto** (default): $B = \max(\lceil 1/(\alpha/m) \rceil, z_\alpha^2 (1-\alpha/m)/(\alpha/m))$
   - Balances precision and computation
   - Uses normal approximation for Monte Carlo error

**Implementation** (`citrees/_tree.py:670-703`):
```python
def _bonferroni_correction(self, *, adjust: str, n_tests: int):
    if n_tests > 1:
        _alpha = alpha / n_tests
        if n_resamples == NResamples.MINIMUM:
            _n_resamples = ceil(1 / _alpha)
        elif n_resamples == NResamples.MAXIMUM:
            _n_resamples = ceil(1 / (4 * _alpha * _alpha))
        else:  # AUTO
            z = norm.ppf(1 - _alpha)
            upper_limit = ceil(z * z * (1 - _alpha) / _alpha)
            _n_resamples = max(ceil(1/_alpha), upper_limit)
```

### 7.3 Scaling Examples

| Features | α | Effective α | Min B | Auto B |
|----------|---|-------------|-------|--------|
| 10 | 0.05 | 0.005 | 199 | 1,321 |
| 50 | 0.05 | 0.001 | 999 | 9,540 |
| 100 | 0.05 | 0.0005 | 1,999 | 21,645 |
| 1,000 | 0.05 | 0.00005 | 19,999 | 302,719 |

---

## 8. Multi-Selector Mode

### 8.1 Overview

citrees supports combining multiple selectors (e.g., `selector=['mc', 'rdc']`). This is useful when:
- Linear and nonlinear associations may both be present
- Different selectors have complementary strengths
- Increased sensitivity is desired

### 8.2 Max-T Method (Westfall & Young, 1993)

To maintain valid Type I error when using multiple selectors, citrees uses the **max-T method**:

**Composite statistic**:
$$
T^{\text{max}}(x, y) = \max_{s \in \mathcal{S}} |T_s(x, y)|
$$

**Key insight**: Compute the maximum **inside** each permutation, not just on the observed data.

```
Algorithm 8: Max-T Multi-Selector Permutation Test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Data (x, y), selectors {T_1, ..., T_S}, permutations B, threshold α
Output: P-value p

1. Compute observed max statistic:
   θ_0 ← max_{s} |T_s(x, y)|

2. k ← 0

3. For b = 1, ..., B:
   a. Permute: y_b ← π_b(y)
   b. Compute max statistic under permutation:
      θ_b ← max_{s} |T_s(x, y_b)|  # MAX INSIDE PERMUTATION
   c. If θ_b ≥ θ_0: k ← k + 1

4. Return p ← (k + 1) / (B + 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 8.3 Validity

**Theorem**: The max-T p-value is valid (super-uniform under $H_0$).

**Proof sketch**: The composite statistic $T^{\text{max}}$ is a measurable function of the data. Under $H_0$, exchangeability of $Y$ implies:
$$
(T^{\text{max}}_0, T^{\text{max}}_1, \ldots, T^{\text{max}}_B) \text{ is exchangeable}
$$
where $T^{\text{max}}_b = \max_s |T_s(x, \pi_b(y))|$. The rank argument from Theorem 1 then applies directly.

### 8.4 Scale Compatibility

**Requirement**: All selectors in multi-selector mode must be on a comparable scale.

| Selector | Scale | Can Combine? |
|----------|-------|--------------|
| mc | [0, 1] | Yes |
| rdc | [0, 1] | Yes |
| pc | [0, 1] (after abs) | Yes |
| dc | [0, 1] | Yes |
| mi | [0, ∞) | **No** |

Mutual information (mi) cannot be combined because it's unbounded, which would bias the maximum toward mi.

### 8.5 Empirical Validation

10,000 simulations under global null (α = 0.05):

| Mode | Rejection Rate | 95% CI |
|------|---------------|--------|
| Single (mc) | 5.3% | [4.9%, 5.7%] |
| Multi (mc + rdc) | 5.9% | [5.3%, 6.6%] |

Both are consistent with nominal α = 0.05.

---

## 9. Computational Heuristics

citrees implements several heuristics to improve computational efficiency without substantially affecting Type I error or power.

### 9.1 Feature Muting

**Idea**: Remove features that show no association with the response from future consideration.

**Criterion**: A feature $j$ is muted if:
$$
p_j \geq \max(\alpha, 1 - \alpha)
$$

This removes features that are clearly non-significant while keeping marginal cases.

**Effect on theory**: Feature muting adaptively changes the hypothesis family across nodes, making global FWER statements more complex. Use `feature_muting=False` for clean theoretical guarantees.

### 9.2 Feature Scanning

**Idea**: When early stopping is enabled, test the most promising features first.

**Procedure**:
1. Compute association scores for all candidate features (no permutation)
2. Sort features by decreasing score
3. Test in sorted order

This increases the probability of early stopping when a significant feature exists.

### 9.3 Threshold Scanning

**Idea**: Similar to feature scanning, but for thresholds.

**Procedure**:
1. Compute split impurity for all candidate thresholds (no permutation)
2. Sort thresholds by increasing impurity
3. Test in sorted order

### 9.4 Parallel Permutation Tests

For large permutation counts without early stopping, citrees uses Numba's parallel loops:

```python
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_mc_parallel(x, y, n_classes, n_resamples, random_state):
    theta_p = np.empty(n_resamples)
    for i in prange(n_resamples):  # Parallel loop
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)
        theta_p[i] = mc(x, y_perm, n_classes)
    return (1 + np.sum(np.abs(theta_p) >= theta)) / (1 + n_resamples)
```

Threshold: `_PARALLEL_THRESHOLD = 200` permutations.

---

## 10. Forest Ensemble

### 10.1 Overview

`ConditionalInferenceForestClassifier` and `ConditionalInferenceForestRegressor` build ensembles of conditional inference trees using bagging.

### 10.2 Bootstrap Methods

#### 10.2.1 Classic Bootstrap

Standard bootstrap with replacement:
$$
\mathcal{D}^{(b)} = \{(X_{\pi_i}, Y_{\pi_i})\}_{i=1}^n, \quad \pi_i \stackrel{\text{iid}}{\sim} \text{Uniform}\{1, \ldots, n\}
$$

#### 10.2.2 Bayesian Bootstrap (default)

Weights samples according to a Dirichlet prior:
$$
w \sim \text{Dirichlet}(1, \ldots, 1) = \frac{1}{n-1} \text{Exp}(1)^{\otimes n} \text{ (normalized)}
$$

The Bayesian bootstrap samples indices with probabilities proportional to $w$.

**Implementation** (`citrees/_utils.py:118-138`):
```python
@njit(cache=True, fastmath=True, nogil=True)
def bayesian_bootstrap_proba(*, n: int, random_state: int):
    np.random.seed(random_state)
    p = np.random.exponential(scale=1.0, size=n)
    return p / p.sum()
```

### 10.3 Sampling Methods (Classification)

#### 10.3.1 Stratified (default)

Maintains class proportions in each bootstrap sample:
- Sample separately within each class
- Combine to form final sample

#### 10.3.2 Balanced

Forces equal class sizes:
- Sample $n_{\min} = \min_k n_k$ from each class
- Useful for imbalanced datasets

### 10.4 Aggregation

**Classification**: Average predicted probabilities
$$
\hat{p}(y = k \mid x) = \frac{1}{T} \sum_{t=1}^T \hat{p}_t(y = k \mid x)
$$

**Regression**: Average predictions
$$
\hat{y}(x) = \frac{1}{T} \sum_{t=1}^T \hat{y}_t(x)
$$

### 10.5 Feature Importance

Mean Decrease in Impurity (MDI) aggregated across trees:
$$
\text{Importance}_j = \sum_{t=1}^T \sum_{\text{node } v \text{ splits on } j} \Delta I_v
$$

Normalized to sum to 1.

---

## 11. Honest Estimation

### 11.1 Overview

Honest estimation (Wager & Athey, 2018) uses sample splitting to decouple tree structure learning from leaf value estimation, reducing adaptive bias.

### 11.2 Procedure

```
Algorithm 9: Honest Tree Building
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Data (X, y), honesty_fraction η (default 0.5)
Output: Honest tree T

1. Split data into:
   - Splitting sample: (X_S, y_S) with n_S = ⌊(1-η)n⌋ samples
   - Estimation sample: (X_E, y_E) with n_E = ⌈ηn⌉ samples

   Note: Use random split (not stratified) for theoretical guarantees

2. Build tree structure using splitting sample:
   T ← BuildTree(X_S, y_S)

3. Re-estimate leaf values using estimation sample:
   For each leaf L in T:
      E_L ← {i ∈ E : X_i routes to L}
      If |E_L| > 0:
         L.value ← AggregateResponse(y_{E_L})
      # Else: keep original value from splitting sample

4. Return T
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 11.3 Theoretical Guarantee

**Proposition** (Unbiased honest estimation): Under honest estimation with independent sample split:
$$
\mathbb{E}[\hat{\mu}(L) \mid \Pi] = \mu(L) := \mathbb{E}[Y \mid X \in L]
$$
for any leaf $L$ with $|E_L| \geq 1$, where $\Pi$ is the learned partition.

**Proof**: See `theory.md` Proposition 4.

### 11.4 Implementation Details

**Sample splitting** (`citrees/_tree.py:1138-1148`):
```python
if self.honesty:
    X_split, X_est, y_split, y_est = train_test_split(
        X, y,
        test_size=self.honesty_fraction,
        random_state=self._random_state,
        # stratify=None for theoretical guarantees
    )
    self.tree_ = self._build_tree(X_split, y_split, depth=1)
    self._reestimate_leaf_values(X_est, y_est)
```

**Path-based leaf identification** (robust to serialization):
```python
def _get_leaf_path(self, x, tree=None, path=()):
    if tree is None:
        tree = self.tree_
    if "value" in tree:
        return path
    if x[tree["feature"]] <= tree["threshold"]:
        return self._get_leaf_path(x, tree["left_child"], path + ("L",))
    else:
        return self._get_leaf_path(x, tree["right_child"], path + ("R",))
```

---

## 12. Evaluation Metrics

This section defines the metrics used to evaluate feature selection methods,
particularly for demonstrating citrees' unbiased selection property.

### 12.1 Precision@k, Recall@k, F1@k

For synthetic datasets with known ground truth (informative feature indices):

$$\text{Precision}@k = \frac{|\text{top}_k \cap \text{informative}|}{k}$$

$$\text{Recall}@k = \frac{|\text{top}_k \cap \text{informative}|}{|\text{informative}|}$$

$$\text{F1}@k = 2 \cdot \frac{\text{Precision}@k \cdot \text{Recall}@k}{\text{Precision}@k + \text{Recall}@k}$$

**Redundant features (informative+redundant).**  
For datasets with redundant features (linear combinations of informative ones), we also report metrics using the union
of informative and redundant indices as ground truth:

$$\text{Precision}^{\mathrm{IR}}@k = \frac{|\text{top}_k \cap (\text{informative} \cup \text{redundant})|}{k}$$

$$\text{Recall}^{\mathrm{IR}}@k = \frac{|\text{top}_k \cap (\text{informative} \cup \text{redundant})|}{|\text{informative} \cup \text{redundant}|}$$

This avoids penalizing methods that select redundant-but-correct proxies of the signal.

### 12.2 Noise Selection Rate (False Positive Rate)

The noise selection rate measures how often a feature selection method
incorrectly ranks noise features in the top-k:

$$\text{NSR}@k = \frac{|\text{top}_k \cap \text{noise}|}{k}$$

Where:
- `top_k` = set of top k features by ranking (from feature selection method)
- `noise` = set of indices of known noise features (from synthetic ground truth)
- Range: [0, 1], lower is better
- 0.0 = no noise selected (perfect)
- 1.0 = all selected features are noise (worst case)

**Significance for Selection Bias:**

This metric is critical for evaluating selection bias. Traditional methods
like CART and Random Forest are known to favor high-cardinality features
(features with many unique values), leading to elevated NSR even when those
features are pure noise.

Conditional inference methods (citrees) use permutation-based hypothesis
testing which is invariant to feature cardinality, and should therefore
maintain NSR near the nominal α level (e.g., ~0.05 for α=0.05).

**Confounder selection rate (correlated noise).**  
For confounder datasets (noise features correlated with informative features), we report:

$$\text{ConfounderRate}@k = \frac{|\text{top}_k \cap \text{confounders}|}{k}$$

This measures how often a method is misled by correlated-but-noncausal features.

**Expected Results:**

| Method Type | Expected NSR@10 | Reason |
|-------------|-----------------|--------|
| citrees (CIT, CIF) | ~0.05 | Permutation test controls Type I error |
| RF, XGBoost, etc. | 0.25-0.40 | Biased toward high-cardinality features |

### 12.3 Nogueira Stability Index

Measures consistency of feature selection across repeated runs (e.g., CV folds, random seeds):

$$\text{Stability} = 1 - \frac{\frac{1}{p} \sum_{j=1}^{p} \hat{f}_j (1 - \hat{f}_j)}{\frac{\bar{k}}{p} (1 - \frac{\bar{k}}{p})}$$

Where:
- $p$ = total number of features
- $\hat{f}_j$ = selection frequency of feature $j$ across $M$ runs
- $\bar{k}$ = average number of selected features

Range: [-1, 1], higher is better. 1.0 indicates perfect consistency.

### 12.4 Pairwise Statistical Comparisons

After establishing overall ranking differences via Friedman test, we perform
pairwise comparisons using the **Wilcoxon signed-rank test** with **Holm-Bonferroni
correction** for family-wise error rate (FWER) control.

**Wilcoxon signed-rank test** is a non-parametric paired test that compares
matched samples without assuming normality. For each pair of methods $(i, j)$:
- $H_0$: Methods $i$ and $j$ have equal performance distributions
- $H_1$: Methods $i$ and $j$ differ in performance

**Holm-Bonferroni correction** controls FWER by a step-down procedure:
1. Sort $m$ p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. For $i = 1, \ldots, m$: reject $H_{(i)}$ if $p_{(i)} \leq \alpha/(m-i+1)$
3. Stop at first non-rejection

This is more powerful than Bonferroni while maintaining FWER $\leq \alpha$.

### 12.5 Cohen's d Effect Size

To quantify practical significance beyond statistical significance, we report
Cohen's $d$ effect size for each pairwise comparison:

$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}$$

where $s_{\text{pooled}} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$

**Interpretation:**
| $|d|$ Range | Effect Size |
|-------------|-------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| ≥ 0.8 | Large |

### 12.6 Statistical Analysis Pipeline

All statistical analyses follow a unified pipeline applied to each dataset type
(synthetic, classification, regression):

1. **Friedman omnibus test**: Tests whether at least one method differs
2. **Pairwise Wilcoxon + Holm**: Identifies which specific pairs differ
3. **Cohen's d**: Quantifies effect magnitude for each significant pair
4. **Bootstrap CIs**: 95% confidence intervals via 2000 bootstrap resamples
5. **Critical difference diagrams**: Visualizes method rankings with CD bars

This ensures consistent, reproducible statistical comparisons across all experiments.

### 12.7 Bootstrap Confidence Intervals

To quantify uncertainty in performance estimates, we compute bootstrap
confidence intervals using the percentile method:

1. **Resample**: Draw $B = 2000$ bootstrap samples with replacement
2. **Compute**: Calculate mean for each bootstrap sample
3. **Percentile**: Take [2.5th, 97.5th] percentiles for 95% CI

$$\text{CI}_{95\%} = [q_{0.025}, q_{0.975}]$$

This non-parametric approach makes no distributional assumptions and
provides valid intervals for any sample size $\geq 5$.

**Output format**: Results are reported as `mean [CI_lo, CI_hi]`, e.g.,
`0.847 [0.823, 0.871]`.

---

## 13. Implementation Details

### 13.1 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Core | NumPy | Array operations |
| JIT compilation | Numba | Performance-critical functions |
| Validation | Pydantic v2 | Parameter validation |
| API | scikit-learn | BaseEstimator, ClassifierMixin, RegressorMixin |
| Parallelism | joblib | Forest training |
| Distance correlation | dcor | Specialized dCor implementation |

### 13.2 Registry Pattern

Selectors and splitters are registered via decorators for extensibility:

```python
from citrees._registry import ClassifierSelectors

@ClassifierSelectors.register("mc")
@njit(cache=True, nogil=True, fastmath=True)
def mc(x, y, n_classes, random_state=None):
    ...
```

Available registries:
- `ClassifierSelectors` / `ClassifierSelectorTests`
- `RegressorSelectors` / `RegressorSelectorTests`
- `ClassifierSplitters` / `ClassifierSplitterTests`
- `RegressorSplitters` / `RegressorSplitterTests`
- `ThresholdMethods`

### 13.3 Type System

All types are centralized in `citrees/_types.py`:

```python
# Numeric constraints
ProbabilityFloat = Annotated[float, Field(gt=0.0, le=1.0)]
PositiveInt = Annotated[int, Field(gt=0)]
ConfidenceFloat = Annotated[float, Field(gt=0.5, lt=1.0)]

# String enums
class EarlyStopping(StrEnum):
    ADAPTIVE = "adaptive"
    SIMPLE = "simple"

class NResamples(StrEnum):
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    AUTO = "auto"
```

### 13.4 Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Single permutation | O(n log n) | Dominated by shuffle |
| Feature selection (1 feature) | O(B · n) | B permutations |
| Feature selection (m features) | O(m · B · n) | With early stopping: much less |
| Split selection (k thresholds) | O(k · B · n) | With early stopping: much less |
| Tree building | O(d · m · B · n) | d = depth, typical d << n |
| Forest building (T trees) | O(T · d · m · B · n) | Embarrassingly parallel |

**Memory**: O(n · p) for data storage, O(B) for permutation statistics.

### 13.5 Numerical Stability

- **Log-space computation**: Beta CDF uses log-gamma and log-exp
- **Continued fraction**: Lentz's algorithm with underflow protection
- **Division by zero**: Guarded with explicit checks and fallback values

---

## References

1. Hothorn, T., Hornik, K., & Zeileis, A. (2006). Unbiased recursive partitioning: A conditional inference framework. *Journal of Computational and Graphical Statistics*, 15(3), 651-674.

2. Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be zero: calculating exact P-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39.

3. Westfall, P. H., & Young, S. S. (1993). *Resampling-based multiple testing: Examples and methods for p-value adjustment*. John Wiley & Sons.

4. Lopez-Paz, D., Hennig, P., & Schölkopf, B. (2013). The randomized dependence coefficient. *Advances in Neural Information Processing Systems*, 26.

5. Székely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007). Measuring and testing dependence by correlation of distances. *The Annals of Statistics*, 35(6), 2769-2794.

6. Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523), 1228-1242.

7. Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. *Physical Review E*, 69(6), 066138.

8. Strobl, C., Boulesteix, A. L., Zeileis, A., & Hothorn, T. (2007). Bias in random forest variable importance measures: Illustrations, sources and a solution. *BMC Bioinformatics*, 8(1), 25.
