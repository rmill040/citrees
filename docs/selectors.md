# Feature Selectors

Feature selectors in citrees measure the association between a feature and the
target variable. The selector with the strongest association (lowest p-value) is
chosen for splitting at each node.

## Overview

| Selector | Task           | Captures   | Complexity | Scale  |
| -------- | -------------- | ---------- | ---------- | ------ |
| `mc`     | Classification | Linear     | O(n)       | [0, 1] |
| `mi`     | Classification | Non-linear | O(n log n) | [0, ∞) |
| `rdc`    | Both           | Non-linear | O(n log n) | [0, 1] |
| `pc`     | Regression     | Linear     | O(n)       | [0, 1] |
| `dc`     | Regression     | Non-linear | O(n²)      | [0, 1] |

---

## Multiple Correlation (mc)

**Default for classification.** Measures how much variance in the feature is
explained by class membership using ANOVA.

### Mathematical Definition

For a feature $X$ and categorical target $Y$ with $K$ classes:

$$\eta = \sqrt{\frac{SS_B}{SS_T}} = \sqrt{\frac{\sum_{k=1}^{K} n_k (\bar{x}_k - \bar{x})^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}}$$

Where:

- $SS_B$ = Between-class sum of squares
- $SS_T$ = Total sum of squares
- $n_k$ = Number of samples in class $k$
- $\bar{x}_k$ = Mean of $X$ in class $k$
- $\bar{x}$ = Overall mean of $X$

### Properties

- **Range**: [0, 1]
- **Interpretation**: $\eta = 0$ means no linear separation; $\eta = 1$ means
  perfect separation
- **Notes**: Some references define the statistic as $\eta^2 = SS_B/SS_T$
  (correlation ratio squared). citrees uses $\eta=\sqrt{SS_B/SS_T}$; this is a
  monotone transformation, so fixed-$B$ permutation p-values and feature
  rankings are unchanged.
- **Equivalent to**: ANOVA $F$-statistic (monotonically related)
- **Limitation**: Only captures linear class separation

### Algorithm

```
Algorithm: Multiple Correlation
Input: Feature x ∈ ℝⁿ, class labels y ∈ {1,...,K}ⁿ

1. Compute overall mean: x̄ = (1/n) Σᵢ xᵢ
2. Compute total sum of squares: SS_T = Σᵢ (xᵢ - x̄)²
3. For each class k:
   a. Compute class mean: x̄ₖ = (1/nₖ) Σᵢ:yᵢ=k xᵢ
   b. Compute class size: nₖ = |{i : yᵢ = k}|
4. Compute between-class SS: SS_B = Σₖ nₖ(x̄ₖ - x̄)²
5. Return η = √(SS_B / SS_T)
```

### Implementation

```python
# From _selector.py - JIT-compiled for performance
@njit(cache=True, fastmath=True, nogil=True)
def mc(x, y, n_classes, random_state=None):
    x_mean = np.mean(x)
    ss_total = np.sum((x - x_mean) ** 2)

    if ss_total == 0:
        return 0.0

    ss_between = 0.0
    for k in range(n_classes):
        mask = y == k
        n_k = np.sum(mask)
        if n_k > 0:
            x_k_mean = np.mean(x[mask])
            ss_between += n_k * (x_k_mean - x_mean) ** 2

    return np.sqrt(ss_between / ss_total)
```

---

## Mutual Information (mi)

Measures non-linear statistical dependence using information theory. In
`citrees`, this selector delegates to
`sklearn.feature_selection.mutual_info_classif` for a one-column feature matrix.
It uses scikit-learn's default estimator settings unless the implementation is
changed upstream: dense features are treated as continuous, the target is
discrete, and `n_neighbors=3`.

### Mathematical Definition

$$MI(X; Y) = H(Y) - H(Y|X) = \sum_{y \in Y} \sum_{x \in X} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

Where:

- $H(Y)$ = Entropy of $Y$
- $H(Y|X)$ = Conditional entropy of $Y$ given $X$

For continuous dense features with a discrete target, the delegated
scikit-learn estimator uses nearest-neighbor entropy estimation for mixed
continuous-discrete data. `citrees` passes `random_state` through to
scikit-learn so the small tie-breaking noise added to continuous variables is
reproducible.

### Properties

- **Range**: [0, ∞)
- **Units**: Nats
- **Zero iff independent**: True mutual information is zero iff independent;
  finite-sample estimates should be interpreted as estimates.
- **Negative estimates**: scikit-learn clips negative estimates to zero.
- **Limitation**: Unbounded scale makes it incompatible with other selectors

### Algorithm

```
Algorithm: Mutual Information (scikit-learn delegated)
Input: Feature x ∈ ℝⁿ, class labels y ∈ {1,...,K}ⁿ, random_state

1. Reshape x to an n × 1 feature matrix if needed.
2. Call sklearn.feature_selection.mutual_info_classif(
      x, y, random_state=random_state
   )
   with scikit-learn's default discrete_features="auto" and n_neighbors=3.
3. Return the first estimated MI value.
```

### Implementation

```python
# From _selector.py
def mi(x: np.ndarray, y: np.ndarray, n_classes: int, random_state: int) -> float:
    if x.ndim == 1:
        x = x[:, None]

    return mutual_info_classif(x, y, random_state=random_state)[0]
```

### Usage Note

Because MI is unbounded, it **cannot** be combined with other selectors in
multi-selector mode:

```python
# Valid
clf = ConditionalInferenceTreeClassifier(selector="mi")

# Invalid - will raise error
clf = ConditionalInferenceTreeClassifier(selector=["mc", "mi"])  # ❌
```

---

## Randomized Dependence Coefficient (rdc)

**Recommended for non-linear relationships.** RDC detects arbitrary dependence
with O(n log n) complexity.

### Mathematical Definition

RDC is defined as the largest canonical correlation between random non-linear
projections of the copula-transformed variables. **citrees uses a fast
approximation** that replaces full CCA with a max pairwise-correlation step.

$$RDC(X, Y) = \sup_{f,g \in \mathcal{F}} \text{Corr}(f(\Phi(X)), g(\Phi(Y)))$$

Where:

- $\Phi$ = Copula transform (rank-based, makes marginals uniform)
- $\mathcal{F}$ = Random Fourier features

### Algorithm

```
Algorithm: Randomized Dependence Coefficient
Input: x ∈ ℝⁿ, y ∈ ℝⁿ, projections k=20, bandwidth s=1/6

1. Copula Transform (rank-based):
   x_copula[i] = rank(x[i]) / n
   y_copula[i] = rank(y[i]) / n

2. Random Non-linear Projections:
   Sample ω_x, ω_y ~ N(0, s²I_k)

   Φ_x = [sin(x_copula · ω_x), cos(x_copula · ω_x)]  ∈ ℝⁿˣ²ᵏ
   Φ_y = [sin(y_copula · ω_y), cos(y_copula · ω_y)]  ∈ ℝⁿˣ²ᵏ

3. Approximate Canonical Correlation:
   Compute max|Corr(Φ_x[:,j], Φ_y[:,k])| across all column pairs
   (O(k²) approximation to true CCA)

4. Return: max pairwise correlation
```

### Properties (Interpretation)

RDC is a **randomized** dependence score designed to capture nonlinear
association efficiently.

- **Defined for non-constant inputs**: If either input is constant, dependence
  measures are ill-posed; citrees returns 0 in such cases.
- **Symmetric**: RDC(X, Y) = RDC(Y, X).
- **Bounded**: 0 ≤ RDC(X, Y) ≤ 1 (as a correlation magnitude).
- **Monotone-transform robustness**: The rank (copula) transform makes the score
  insensitive to strictly monotone transforms of each marginal (up to
  finite-sample ranking ties).
- **Dependence detection (heuristic at finite n)**: In finite samples with a
  finite number of random features and the approximation used in citrees, RDC
  should be read as a practical dependence score, not as an exact “0 iff
  independent” guarantee.

### Complexity

| Operation                  | Complexity               |
| -------------------------- | ------------------------ |
| Copula transform (sorting) | O(n log n)               |
| Random projections         | O(nk)                    |
| CCA                        | O(nk²)                   |
| **Total**                  | **O(n log n)** for k ≪ n |

### Why RDC for citrees?

| Comparison           | RDC              | Distance Correlation |
| -------------------- | ---------------- | -------------------- |
| Non-linear detection | ✓                | ✓                    |
| Time complexity      | O(n log n)       | O(n²)                |
| Space complexity     | O(nk)            | O(n²)                |
| Large n scaling      | Typically faster | Typically slower     |

---

## Pearson Correlation (pc)

**Default for regression.** Measures linear association between feature and
continuous target.

### Mathematical Definition

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

For feature selection, we use the absolute value: $|r|$

### Properties

- **Range**: [-1, 1], using |r| gives [0, 1]
- **Interpretation**: 0 = no linear relationship, 1 = perfect linear
  relationship
- **Limitation**: Only detects linear relationships
- **Complexity**: O(n)

### Algorithm

```
Algorithm: Pearson Correlation
Input: Feature x ∈ ℝⁿ, target y ∈ ℝⁿ

1. Compute means: x̄ = mean(x), ȳ = mean(y)
2. Compute centered values: x' = x - x̄, y' = y - ȳ
3. Compute covariance: cov = Σᵢ x'ᵢy'ᵢ
4. Compute variances: var_x = Σᵢ x'ᵢ², var_y = Σᵢ y'ᵢ²
5. Return |cov / √(var_x · var_y)|
```

---

## Distance Correlation (dc)

Measures arbitrary (linear and non-linear) dependence between variables. Zero if
and only if independent.

### Mathematical Definition

Distance correlation is based on distance covariance:

$$dCov^2(X, Y) = \frac{1}{n^2} \sum_{i,j} A_{ij} B_{ij}$$

Where:

- $A_{ij} = a_{ij} - \bar{a}_{i\cdot} - \bar{a}_{\cdot j} + \bar{a}_{\cdot\cdot}$
  (doubly-centered distance matrix for X)
- $B_{ij}$ = analogous for Y
- $a_{ij} = \|x_i - x_j\|$

Distance correlation:

$$dCor(X, Y) = \frac{dCov(X, Y)}{\sqrt{dVar(X) \cdot dVar(Y)}}$$

### Properties

- **Range**: [0, 1]
- **Zero iff independent**: dCor(X, Y) = 0 ⟺ X ⊥ Y (for finite first moments)
- **Detects all dependence**: Captures both linear and non-linear relationships
- **Limitation**: O(n²) complexity limits scalability

### Algorithm

```
Algorithm: Distance Correlation
Input: x ∈ ℝⁿ, y ∈ ℝⁿ

1. Compute distance matrices:
   a[i,j] = |x[i] - x[j]|
   b[i,j] = |y[i] - y[j]|

2. Double-center the matrices:
   A[i,j] = a[i,j] - row_mean[i] - col_mean[j] + grand_mean
   B[i,j] = b[i,j] - row_mean[i] - col_mean[j] + grand_mean

3. Compute distance covariance:
   dCov² = (1/n²) Σᵢⱼ A[i,j] · B[i,j]

4. Compute distance variances:
   dVar_X = (1/n²) Σᵢⱼ A[i,j]²
   dVar_Y = (1/n²) Σᵢⱼ B[i,j]²

5. Return dCor = dCov / √(dVar_X · dVar_Y)
```

### Complexity Warning

| n (samples) | Time   | Memory  |
| ----------- | ------ | ------- |
| 1,000       | ~0.5s  | ~8 MB   |
| 10,000      | ~50s   | ~800 MB |
| 100,000     | ~5000s | ~80 GB  |

**Recommendation**: Use `rdc` instead of `dc` for datasets with more than ~5,000
samples.

---

## Multi-Selector Mode

citrees supports combining multiple selectors using the **max-T method**
(Westfall & Young, 1993). The permutation test computes max(selector_scores)
inside each permutation. In fixed-$B$ mode at a fixed node, this yields a
standard exchangeability-based (super-uniform) permutation p-value for the
composite max statistic.

Each selector in the list must be unique - duplicates are not allowed.

### Usage

```python
# Classification: combine mc and rdc (both [0,1] scale)
clf = ConditionalInferenceTreeClassifier(selector=['mc', 'rdc'])

# Regression: combine all three
reg = ConditionalInferenceTreeRegressor(selector=['pc', 'dc', 'rdc'])
```

### Algorithm

```
Algorithm: Max-T Permutation Test (per feature)
Input: feature x ∈ ℝⁿ, labels y, selectors S = {s₁, ..., sₘ}, permutations B
Output: p-value p

1. Observed max statistic:
   T0 ← max_{s ∈ S} |s(x, y)|

2. Count exceedances:
   L ← 0
   For b = 1..B:
       y_perm ← Permute(y)
       Tb ← max_{s ∈ S} |s(x, y_perm)|
       If Tb ≥ T0: L ← L + 1

3. +1 corrected p-value:
   p ← (L + 1) / (B + 1)

4. Return p

Stage A then computes p_j for each feature j using this max-T test and selects
the feature with the smallest p-value.
```

### Compatibility

| Combination           | Valid | Reason           |
| --------------------- | ----- | ---------------- |
| `['mc', 'rdc']`       | ✓     | Both [0,1] scale |
| `['pc', 'dc', 'rdc']` | ✓     | All [0,1] scale  |
| `['mc', 'mi']`        | ✗     | MI is unbounded  |

---

## References

1. **Multiple Correlation**: Fisher, R.A. (1925). Statistical Methods for
   Research Workers.

2. **Mutual Information**: citrees delegates to scikit-learn's
   [`mutual_info_classif`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html),
   which cites nearest-neighbor entropy estimators for continuous and mixed
   continuous-discrete variables.

3. **RDC**: Lopez-Paz, D., Hennig, P., & Schölkopf, B. (2013).
   [The Randomized Dependence Coefficient](https://arxiv.org/abs/1304.7717).
   NeurIPS.

4. **Distance Correlation**: Székely, G.J., Rizzo, M.L., & Bakirov, N.K. (2007).
   [Measuring and Testing Dependence by Correlation of Distances](https://projecteuclid.org/journals/annals-of-statistics/volume-35/issue-6/Measuring-and-testing-dependence-by-correlation-of-distances/10.1214/009053607000000505.full).
   Annals of Statistics.
