# Algorithm Analysis & Potential Improvements

## Current Implementation Analysis

### Feature Selectors

citrees currently implements these association measures:

| Selector | Type | Complexity | Non-linear? | Notes |
|----------|------|------------|-------------|-------|
| `mc` (multiple correlation) | Classification | O(n) | No | Only captures linear class separation |
| `mi` (mutual information) | Classification | O(n log n) | Yes | sklearn implementation, k-NN based |
| `pc` (Pearson correlation) | Regression | O(n) | No | Only linear relationships |
| `dc` (distance correlation) | Regression | **O(n²)** | Yes | Captures all dependence but SLOW |

### The Distance Correlation Problem

Distance correlation (`dc`) from the `dcor` library has **O(n²)** time AND space complexity:

```python
# From _selector.py line 414-442
@RegressorSelectors.register("dc")
def distance_correlation(x, y, standardize, random_state):
    return _d_correlation(x, y)  # Uses dcor library - O(n²)
```

For a dataset with 10,000 samples:
- O(n²) = 100,000,000 operations per feature per permutation
- With 100 permutations × 100 features = 1 trillion operations

**This is why citrees can be slow on large datasets.**

---

## The Randomized Dependence Coefficient (RDC)

### What is RDC?

RDC (Lopez-Paz et al., 2013) is a non-linear dependence measure with several key advantages:

**Paper**: [The Randomized Dependence Coefficient](https://arxiv.org/abs/1304.7717)

### Properties (Rényi's Axioms)

A good dependence measure ρ*(X, Y) should satisfy:

1. **Defined for any pair**: Works for any non-constant X, Y
2. **Symmetric**: ρ*(X, Y) = ρ*(Y, X)
3. **Bounded**: 0 ≤ ρ*(X, Y) ≤ 1
4. **Zero iff independent**: ρ*(X, Y) = 0 ⟺ X ⊥ Y
5. **Invariant to monotonic transforms**: ρ*(f(X), g(Y)) = ρ*(X, Y) for bijective f, g
6. **Max for functional relationships**: ρ*(X, Y) = 1 if Y = f(X)

**RDC satisfies ALL of these. Pearson correlation only satisfies 1-4.**

### Algorithm

```python
def rdc(x, y, k=20, s=1/6):
    """
    Randomized Dependence Coefficient

    1. Apply copula transform (rank-based, makes marginals uniform)
    2. Apply random non-linear projections
    3. Compute canonical correlation
    """
    n = len(x)

    # Step 1: Copula transform - O(n log n) for sorting
    x_copula = rank(x) / n  # Uniform marginals
    y_copula = rank(y) / n

    # Step 2: Random non-linear projections - O(nk)
    # Sample random frequencies from N(0, s)
    wx = np.random.normal(0, s, (1, k))
    wy = np.random.normal(0, s, (1, k))

    # Apply sin/cos projections (like random Fourier features)
    x_proj = np.hstack([
        np.sin(x_copula @ wx),
        np.cos(x_copula @ wx)
    ])
    y_proj = np.hstack([
        np.sin(y_copula @ wy),
        np.cos(y_copula @ wy)
    ])

    # Step 3: Canonical correlation - O(nk²)
    # Find maximum correlation between linear combinations
    cca = CCA(n_components=1)
    cca.fit(x_proj, y_proj)
    x_c, y_c = cca.transform(x_proj, y_proj)

    return np.corrcoef(x_c.T, y_c.T)[0, 1]
```

### Complexity Comparison

| Measure | Time Complexity | Space Complexity | Non-linear |
|---------|----------------|------------------|------------|
| Pearson | O(n) | O(1) | No |
| Distance Correlation | **O(n²)** | **O(n²)** | Yes |
| Mutual Information | O(n log n) | O(n) | Yes |
| **RDC** | **O(n log n)** | **O(nk)** | **Yes** |

Where k is the number of random projections (typically 20-50).

### Why RDC is Better for citrees

1. **Speed**: 100-1000x faster than distance correlation on large datasets
2. **Non-linear**: Captures non-linear relationships (unlike mc/pc)
3. **Invariant**: Copula transform makes it invariant to marginal distributions
4. **Simple**: Can be implemented with numba for additional speedup

---

## Implementation Proposal

### New Selector: `rdc`

```python
@ClassifierSelectors.register("rdc")
@RegressorSelectors.register("rdc")
@njit(cache=True, nogil=True, fastmath=True)
def randomized_dependence_coefficient(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 20,          # Number of random projections
    s: float = 1/6,       # Bandwidth parameter
    random_state: int = 42
) -> float:
    """
    Compute RDC between x and y.

    For classification, y should be one-hot encoded or we compute
    RDC between x and each class indicator, then take max.
    """
    n = len(x)

    # Copula transform (rank-based)
    x_ranks = np.argsort(np.argsort(x)) / n
    y_ranks = np.argsort(np.argsort(y)) / n

    # Random projections
    np.random.seed(random_state)
    wx = np.random.randn(k) * s
    wy = np.random.randn(k) * s

    # Non-linear features via sin/cos
    x_features = np.empty((n, 2*k))
    y_features = np.empty((n, 2*k))

    for i in range(n):
        for j in range(k):
            x_features[i, j] = np.sin(x_ranks[i] * wx[j])
            x_features[i, k+j] = np.cos(x_ranks[i] * wx[j])
            y_features[i, j] = np.sin(y_ranks[i] * wy[j])
            y_features[i, k+j] = np.cos(y_ranks[i] * wy[j])

    # Simplified CCA: correlation of first principal components
    # (Full CCA would be more accurate but slower)
    return _max_correlation(x_features, y_features)
```

### Expected Impact

| Dataset Size | dc Time | rdc Time | Speedup |
|--------------|---------|----------|---------|
| 1,000 | 0.5s | 0.01s | 50x |
| 10,000 | 50s | 0.1s | 500x |
| 100,000 | 5000s | 1s | 5000x |

---

## Other Algorithmic Improvements

### 1. Fast Distance Correlation (Univariate Only)

For univariate x and y, there exists an O(n log n) algorithm:

```python
# Huo & Szekely (2016) - "Fast Computing for Distance Covariance"
# Only works for 1D x and 1D y
def fast_dcov(x, y):
    n = len(x)
    ix = np.argsort(x)
    iy = np.argsort(y)
    # ... involves sorted partial sums
    return dcov  # O(n log n)
```

**Problem**: citrees already has univariate features, but the dcor library doesn't use this optimization.

### 2. Permutation Test Improvements

Current implementation runs full permutation tests:

```python
# Current: O(n_resamples * n * association_cost)
for _ in range(n_resamples):
    y_perm = permute(y)
    stat = association(x, y_perm)  # O(n) to O(n²)
```

**Improvements**:

1. **Asymptotic p-values**: For large n, use asymptotic distribution instead of permutation
2. **Sequential testing**: Stop early if p-value is clearly significant or not
3. **Parallel permutations**: Already implemented for some selectors

### 3. Feature Screening

Before running permutation tests, quickly eliminate clearly uninformative features:

```python
def screen_features(X, y, threshold=0.1):
    """Fast screening using approximate tests."""
    # Use RDC or simple correlation as a quick filter
    quick_scores = [rdc(X[:, j], y) for j in range(X.shape[1])]
    # Only run expensive permutation tests on promising features
    candidates = np.where(quick_scores > threshold)[0]
    return candidates
```

### 4. Tree-Specific Optimizations

**Current**: Permutation test at every split candidate
**Better**: Use impurity-based selection, only test the BEST split

```python
# Current approach
for feature in features:
    pval = permutation_test(x, y)  # Expensive!

# Better approach
for feature in features:
    impurity = gini(x, y)  # Fast
best_feature = argmin(impurity)
pval = permutation_test(X[:, best_feature], y)  # Only test once
```

This would be a significant algorithmic change but could improve speed 10-100x.

---

## Summary of Proposed Changes

| Change | Impact | Effort | Priority |
|--------|--------|--------|----------|
| Add RDC selector | 100-1000x faster for non-linear | Medium | HIGH |
| Fast univariate dcor | 10-100x faster for dc | Low | Medium |
| Asymptotic p-values | Faster for large n | Medium | Medium |
| Feature screening | Faster for many features | Low | Medium |
| Single-test-per-split | 10-100x faster overall | High | Research |

## References

1. Lopez-Paz, D., Hennig, P., & Schölkopf, B. (2013). [The Randomized Dependence Coefficient](https://arxiv.org/abs/1304.7717). *NeurIPS*.

2. Székely, G. J., & Rizzo, M. L. (2014). Partial distance correlation with methods for dissimilarities. *Annals of Statistics*.

3. Huo, X., & Székely, G. J. (2016). [Fast Computing for Distance Covariance](https://www.researchgate.net/publication/328575900_A_fast_algorithm_for_computing_distance_correlation). *Technometrics*.

4. There's also a Python implementation of RDC available: [GitHub - garydoranjr/rdc](https://github.com/garydoranjr/rdc)
