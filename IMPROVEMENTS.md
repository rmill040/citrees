# Critical Gaps: citrees vs Original ctree

This document analyzes the fundamental differences between the current citrees implementation and the original ctree algorithm from R's partykit/party packages.

## Executive Summary

The current citrees implementation has **three fundamental departures** from the original ctree that affect statistical validity:

| Gap | Impact | Priority |
|-----|--------|----------|
| No asymptotic p-values | 10-100x slower for large n | HIGH |
| Wrong test statistics | May have variable selection bias | CRITICAL |
| Split selection uses impurity | Not statistically principled | HIGH |

---

## CRITICAL: Test Statistic Framework

### Original ctree (Strasser-Weber Framework)

The original ctree uses **linear statistics** based on the Strasser-Weber (1999) theorem:

```
T = Σᵢ g(Yᵢ) ⊗ h(Xᵢⱼ)   ∈ ℝ^(p×q)
```

Where:
- `g(Y)` = influence function on response (e.g., class indicators, residuals)
- `h(X)` = transformation of covariate (e.g., identity, ranks)
- Under H₀ (independence), T follows asymptotic multivariate normal

**Key insight**: The test statistic is standardized using the **conditional expectation and covariance** under the permutation distribution:

```
c = max_k |T_k - μ_k| / σ_k    (c_max statistic)
```

This standardization ensures **unbiased variable selection** regardless of measurement scale.

### Current citrees Implementation

citrees uses ad-hoc correlation measures:
- **Classifier**: Multiple correlation η² = SSB/SST (ANOVA-like)
- **Regressor**: Pearson correlation r

**Problem**: These measures are NOT equivalent to the standardized linear statistics. They can have different distributions for different covariate scales, potentially introducing selection bias.

### Required Fix

Implement the proper linear statistics framework:

```python
def compute_linear_statistic(x, y, influence_func):
    """Compute linear statistic T and its conditional moments."""
    h = x  # or rank(x) for ordinal data
    g = influence_func(y)  # class indicators, residuals, etc.

    n = len(y)
    T = np.sum(np.outer(g, h), axis=0)

    # Conditional expectation under H0 (permutation symmetry)
    mu = np.sum(g) * np.sum(h) / n

    # Conditional covariance (Strasser-Weber formula)
    sigma = compute_permutation_covariance(g, h, n)

    # Standardized statistic
    c = np.abs(T - mu) / np.sqrt(np.diag(sigma))
    return c, mu, sigma
```

---

## HIGH: Asymptotic P-Values

### Original ctree

Can compute p-values **without Monte Carlo** using the asymptotic distribution:

```
P(c_max ≥ observed) ≈ 1 - Φ_p(observed; 0, Σ_standardized)
```

Where Φ_p is the multivariate normal CDF. For univariate case, this is just `2 * (1 - Φ(c))`.

**Advantage**: O(1) p-value computation vs O(B) for Monte Carlo.

### Current citrees

Only supports Monte Carlo permutation tests (B = 100 to 10,000 resamples).

**Impact**:
- For α = 0.05: need ~20 resamples minimum
- For α = 0.001 with Bonferroni (100 features): need ~100,000 resamples per feature
- Total: 10 million permutations per split!

### Required Fix

Add asymptotic p-value computation as default for large samples:

```python
from scipy.stats import norm

def asymptotic_pvalue(statistic, standardized=True):
    """Compute asymptotic p-value for max-type statistic."""
    if standardized:
        # Two-sided p-value from standard normal
        return 2 * (1 - norm.cdf(np.abs(statistic)))
    else:
        raise NotImplementedError("Need to standardize first")

def select_pvalue_method(n_samples, n_resamples):
    """Use asymptotic for large n, Monte Carlo for small n."""
    if n_samples > 100 and n_resamples == "auto":
        return "asymptotic"
    return "monte_carlo"
```

---

## HIGH: Split Selection Method

### Original ctree

After selecting a variable, ctree selects the split point using **maximally selected statistics**:

1. For each candidate cutpoint c:
   - Compute two-sample test statistic T(c)
   - Standardize: c(c) = |T(c) - μ(c)| / σ(c)

2. Select cutpoint maximizing c(c)

3. **Critically**: Adjust p-value for multiple cutpoint testing using the distribution of `max_c c(c)`

### Current citrees

Uses **impurity measures** (Gini, MSE) for split selection, which is CART-style, not conditional inference.

**Problem**: This breaks the statistical framework. Variable selection uses permutation tests, but split selection uses impurity. This inconsistency may:
- Introduce bias in split point selection
- Invalidate the statistical interpretation

### Required Fix

Use the same permutation test framework for splits:

```python
def select_split_statistical(x, y, influence_func):
    """Select split using maximally selected statistics."""
    unique_values = np.unique(x)
    cutpoints = (unique_values[:-1] + unique_values[1:]) / 2

    best_statistic = -np.inf
    best_cutpoint = None

    for c in cutpoints:
        # Binary indicator for x <= c
        indicator = (x <= c).astype(float)

        # Compute two-sample linear statistic
        g = influence_func(y)
        T = np.sum(g * indicator)

        # Standardize under permutation distribution
        n = len(y)
        n1 = np.sum(indicator)
        n0 = n - n1

        mu = np.sum(g) * n1 / n
        var = (n1 * n0 / (n * (n - 1))) * np.sum((g - g.mean())**2)

        c_stat = np.abs(T - mu) / np.sqrt(var) if var > 0 else 0

        if c_stat > best_statistic:
            best_statistic = c_stat
            best_cutpoint = c

    return best_cutpoint, best_statistic
```

---

## MEDIUM: Missing Value Handling

### Original ctree

Supports **MIA (Missing Incorporated in Attributes)**:
- Missing values treated as a separate category
- Can split on "missing vs non-missing"
- Handles both numeric and categorical missingness

### Current citrees

No missing value handling. Will crash or produce undefined behavior with NaN.

### Required Fix

```python
def handle_missing_values(x, method="mia"):
    """Handle missing values in features."""
    if method == "mia":
        # Create indicator for missingness
        is_missing = np.isnan(x)
        x_filled = np.where(is_missing, np.nanmedian(x), x)
        return x_filled, is_missing
    elif method == "omit":
        return x[~np.isnan(x)], None
```

---

## MEDIUM: Influence Functions

### Original ctree

Uses a general framework of **influence functions** g(Y) that adapt to response type:

| Response Type | Influence Function g(Y) |
|--------------|-------------------------|
| Numeric | Identity or residuals |
| Binary | Class indicator {0, 1} |
| Multiclass | One-hot encoded indicators |
| Ordinal | Scores (e.g., 1, 2, 3, ...) |
| Censored | Log-rank scores |

### Current citrees

Hard-coded for classification (multiple correlation) and regression (Pearson correlation).

### Required Fix

```python
def get_influence_function(y, response_type="auto"):
    """Get appropriate influence function for response type."""
    if response_type == "auto":
        response_type = infer_response_type(y)

    if response_type == "numeric":
        return y - np.mean(y)  # Centered response
    elif response_type == "binary":
        return (y == 1).astype(float) - np.mean(y == 1)
    elif response_type == "multiclass":
        n_classes = len(np.unique(y))
        indicators = np.zeros((len(y), n_classes))
        for j, label in enumerate(np.unique(y)):
            indicators[:, j] = (y == label).astype(float)
        return indicators - indicators.mean(axis=0)
    elif response_type == "ordinal":
        return y.astype(float)  # Use raw scores
```

---

## LOW: Quadratic Test Statistics

### Original ctree (v1.2-20+)

Supports quadratic forms for multivariate responses:

```
c_quad = (T - μ)ᵀ Σ⁻¹ (T - μ) ~ χ²(rank(Σ))
```

**Use case**: When testing multiple aspects of the response jointly.

### Current citrees

Not supported.

---

## Verification Checklist

To verify citrees achieves unbiased variable selection, run this simulation:

```python
def test_unbiased_selection():
    """Verify 1/p selection probability under H0."""
    np.random.seed(42)
    n_simulations = 1000
    n_features = 6
    n_samples = 200

    # Features with different scales (should not matter)
    selected_features = []

    for _ in range(n_simulations):
        X = np.column_stack([
            np.random.normal(0, 1, n_samples),      # Standard normal
            np.random.normal(0, 10, n_samples),     # Wide normal
            np.random.uniform(0, 1, n_samples),     # Uniform [0,1]
            np.random.choice([0, 1], n_samples),    # Binary
            np.random.choice(range(10), n_samples), # 10 categories
            np.random.choice(range(100), n_samples) # 100 categories
        ])
        y = np.random.choice([0, 1], n_samples)  # Random binary (H0 true)

        clf = ConditionalInferenceTreeClassifier(max_depth=1)
        clf.fit(X, y)

        if hasattr(clf, 'tree_') and 'feature' in clf.tree_:
            selected_features.append(clf.tree_['feature'])

    # Under H0, each feature should be selected ~1/6 of the time
    counts = np.bincount(selected_features, minlength=n_features)
    proportions = counts / len(selected_features)

    print("Selection proportions (expect ~0.167 for each):")
    print(proportions)

    # 95% CI should include 1/6 for unbiased selection
    expected = 1 / n_features
    se = np.sqrt(expected * (1 - expected) / len(selected_features))
    ci_lower = expected - 1.96 * se
    ci_upper = expected + 1.96 * se
    print(f"95% CI for unbiased: [{ci_lower:.3f}, {ci_upper:.3f}]")
```

---

## Implementation Roadmap

### Phase 1: Statistical Correctness (Critical)
1. Implement proper linear statistics with conditional moments
2. Add asymptotic p-value computation
3. Use statistical tests for split selection (not impurity)

### Phase 2: Performance & Usability
4. Add missing value handling (MIA)
5. Implement influence functions for different response types
6. Add quadratic test statistics option

### Phase 3: Validation
7. Run unbiased selection simulation
8. Benchmark against R's partykit::ctree
9. Document statistical properties

---

## References

- Hothorn, Hornik, Zeileis (2006). "Unbiased Recursive Partitioning: A Conditional Inference Framework"
- Strasser, Weber (1999). "On the asymptotic theory of permutation statistics"
- partykit vignette: https://cran.r-project.org/web/packages/partykit/vignettes/ctree.pdf
- libcoin: https://cran.r-project.org/web/packages/libcoin/
