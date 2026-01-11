# Feature Selectors

Feature selectors in citrees measure the association between a feature and the target variable. The selector with the strongest association (lowest p-value) is chosen for splitting at each node.

## Overview

| Selector | Task | Captures | Complexity | Scale |
|----------|------|----------|------------|-------|
| `mc` | Classification | Linear | O(n) | [0, 1] |
| `mi` | Classification | Non-linear | O(n log n) | [0, ∞) |
| `rdc` | Both | Non-linear | O(n log n) | [0, 1] |
| `pc` | Regression | Linear | O(n) | [0, 1] |
| `dc` | Regression | Non-linear | O(n²) | [0, 1] |

---

## Multiple Correlation (mc)

**Default for classification.** Measures how much variance in the feature is explained by class membership using ANOVA.

### Mathematical Definition

For a feature $X$ and categorical target $Y$ with $K$ classes:

$$\eta^2 = \frac{SS_B}{SS_T} = \frac{\sum_{k=1}^{K} n_k (\bar{x}_k - \bar{x})^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$

Where:
- $SS_B$ = Between-class sum of squares
- $SS_T$ = Total sum of squares
- $n_k$ = Number of samples in class $k$
- $\bar{x}_k$ = Mean of $X$ in class $k$
- $\bar{x}$ = Overall mean of $X$

### Properties

- **Range**: [0, 1]
- **Interpretation**: $\eta^2 = 0$ means no linear separation; $\eta^2 = 1$ means perfect separation
- **Equivalent to**: ANOVA F-statistic (monotonically related)
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
5. Return η² = SS_B / SS_T
```

### Implementation

```python
# From _selector.py - JIT-compiled for performance
@njit(cache=True, fastmath=True, nogil=True)
def multiple_correlation(x, y, n_classes, random_state=None):
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

    return ss_between / ss_total
```

---

## Mutual Information (mi)

Measures non-linear statistical dependence using information theory. Based on k-nearest neighbors estimation.

### Mathematical Definition

$$MI(X; Y) = H(Y) - H(Y|X) = \sum_{y \in Y} \sum_{x \in X} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

Where:
- $H(Y)$ = Entropy of $Y$
- $H(Y|X)$ = Conditional entropy of $Y$ given $X$

For continuous features, MI is estimated using the [KSG estimator](https://arxiv.org/abs/cond-mat/0305641):

$$\hat{MI}(X; Y) = \psi(k) - \langle \psi(n_x + 1) + \psi(n_y + 1) \rangle + \psi(N)$$

### Properties

- **Range**: [0, ∞)
- **Units**: Nats (natural log) or bits (log₂)
- **Zero iff independent**: $MI(X;Y) = 0 \Leftrightarrow X \perp Y$
- **Limitation**: Unbounded scale makes it incompatible with other selectors

### Algorithm

```
Algorithm: Mutual Information (KSG Estimator)
Input: Feature x ∈ ℝⁿ, class labels y ∈ {1,...,K}ⁿ, neighbors k

1. For each point i:
   a. Find k-th nearest neighbor in joint space (x, y)
   b. Let ε[i] = distance to k-th neighbor
   c. Count n_x[i] = points within ε[i] in x-space
   d. Count n_y[i] = points within ε[i] in y-space

2. Compute MI estimate:
   MI = ψ(k) - (1/n)Σᵢ[ψ(n_x[i]+1) + ψ(n_y[i]+1)] + ψ(n)

   where ψ is the digamma function
```

### Usage Note

Because MI is unbounded, it **cannot** be combined with other selectors in multi-selector mode:

```python
# Valid
clf = ConditionalInferenceTreeClassifier(selector="mi")

# Invalid - will raise error
clf = ConditionalInferenceTreeClassifier(selector=["mc", "mi"])  # ❌
```

---

## Randomized Dependence Coefficient (rdc)

**Recommended for non-linear relationships.** RDC detects arbitrary dependence with O(n log n) complexity.

### Mathematical Definition

RDC is defined as the largest canonical correlation between random non-linear projections of the copula-transformed variables:

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

3. Canonical Correlation Analysis:
   Find u, v that maximize Corr(Φ_x · u, Φ_y · v)

4. Return: max canonical correlation
```

### Properties (Rényi's Axioms)

RDC satisfies all of Rényi's axioms for dependence measures:

1. **Defined for any pair**: Works for non-constant X, Y
2. **Symmetric**: RDC(X, Y) = RDC(Y, X)
3. **Bounded**: 0 ≤ RDC(X, Y) ≤ 1
4. **Zero iff independent**: RDC(X, Y) = 0 ⟺ X ⊥ Y
5. **Invariant to monotonic transforms**: RDC(f(X), g(Y)) = RDC(X, Y)
6. **Maximum for functional relationships**: RDC(X, f(X)) = 1

### Complexity

| Operation | Complexity |
|-----------|------------|
| Copula transform (sorting) | O(n log n) |
| Random projections | O(nk) |
| CCA | O(nk²) |
| **Total** | **O(n log n)** for k ≪ n |

### Why RDC for citrees?

| Comparison | RDC | Distance Correlation |
|------------|-----|---------------------|
| Non-linear detection | ✓ | ✓ |
| Time complexity | O(n log n) | O(n²) |
| Space complexity | O(nk) | O(n²) |
| 10,000 samples | ~0.1s | ~50s |

---

## Pearson Correlation (pc)

**Default for regression.** Measures linear association between feature and continuous target.

### Mathematical Definition

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

For feature selection, we use the absolute value: $|r|$

### Properties

- **Range**: [-1, 1], using |r| gives [0, 1]
- **Interpretation**: 0 = no linear relationship, 1 = perfect linear relationship
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

Measures arbitrary (linear and non-linear) dependence between variables. Zero if and only if independent.

### Mathematical Definition

Distance correlation is based on distance covariance:

$$dCov^2(X, Y) = \frac{1}{n^2} \sum_{i,j} A_{ij} B_{ij}$$

Where:
- $A_{ij} = a_{ij} - \bar{a}_{i\cdot} - \bar{a}_{\cdot j} + \bar{a}_{\cdot\cdot}$ (doubly-centered distance matrix for X)
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

| n (samples) | Time | Memory |
|-------------|------|--------|
| 1,000 | ~0.5s | ~8 MB |
| 10,000 | ~50s | ~800 MB |
| 100,000 | ~5000s | ~80 GB |

**Recommendation**: Use `rdc` instead of `dc` for datasets with more than ~5,000 samples.

---

## Multi-Selector Mode

citrees supports combining multiple selectors. The feature with the maximum score across all selectors is selected, and the permutation test uses that selector.

### Usage

```python
# Classification: combine mc and rdc (both [0,1] scale)
clf = ConditionalInferenceTreeClassifier(selector=['mc', 'rdc'])

# Regression: combine all three
reg = ConditionalInferenceTreeRegressor(selector=['pc', 'dc', 'rdc'])
```

### Algorithm

```
Algorithm: Multi-Selector Feature Selection
Input: X ∈ ℝⁿˣᵖ, y, selectors S = {s₁, ..., sₘ}

For each feature j:
    scores[j] = max_{s ∈ S} s(X[:,j], y)
    best_selector[j] = argmax_{s ∈ S} s(X[:,j], y)

j* = argmax_j scores[j]
p_value = PermutationTest(X[:,j*], y, selector=best_selector[j*])
```

### Compatibility

| Combination | Valid | Reason |
|-------------|-------|--------|
| `['mc', 'rdc']` | ✓ | Both [0,1] scale |
| `['pc', 'dc', 'rdc']` | ✓ | All [0,1] scale |
| `['mc', 'mi']` | ✗ | MI is unbounded |

---

## References

1. **Multiple Correlation**: Fisher, R.A. (1925). Statistical Methods for Research Workers.

2. **Mutual Information**: Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). [Estimating mutual information](https://arxiv.org/abs/cond-mat/0305641). Physical Review E.

3. **RDC**: Lopez-Paz, D., Hennig, P., & Schölkopf, B. (2013). [The Randomized Dependence Coefficient](https://arxiv.org/abs/1304.7717). NeurIPS.

4. **Distance Correlation**: Székely, G.J., Rizzo, M.L., & Bakirov, N.K. (2007). [Measuring and Testing Dependence by Correlation of Distances](https://projecteuclid.org/journals/annals-of-statistics/volume-35/issue-6/Measuring-and-testing-dependence-by-correlation-of-distances/10.1214/009053607000000505.full). Annals of Statistics.
