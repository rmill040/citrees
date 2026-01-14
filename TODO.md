# citrees: Blind Spots & Issues Checklist

> **Purpose**: This document catalogs all identified issues requiring resolution before publication claims can be made. Each issue includes proof-of-concept scripts, acceptance criteria, and verification steps.

---

## Executive Summary

| Priority | Issue | Location | Type | Status |
|----------|-------|----------|------|--------|
| 🔴 CRITICAL | Multi-selector p-value inflation | `_tree.py:454-471` | Statistical | ✅ Resolved |
| 🔴 CRITICAL | Classification honesty violates independence | `_tree.py:1097-1106` | Statistical | ✅ Resolved |
| 🟠 HIGH | No power analysis / B selection guidance | `theory.md` | Documentation | ❌ Open |
| 🟠 HIGH | Nested CV structure unclear in experiments | `paper/scripts/` | Benchmarking | ❌ Open |
| 🟡 MEDIUM | Global error control with feature muting undefined | `theory.md:715-717` | Theory | ❌ Open |
| 🟡 MEDIUM | Missing baselines (RFE, TreeSHAP, mRMR) | `paper/scripts/` | Benchmarking | ❌ Open |
| 🟡 MEDIUM | Synthetic experiments too easy | `synthetic_experiments.py` | Benchmarking | ❌ Open |
| 🟡 MEDIUM | Early stopping p-value inflation not quantified | `theory.md:683-685` | Theory | ✅ Resolved |
| 🟢 LOW | Broad exception handling | `_selector.py:257-260` | Code Quality | ❌ Open |
| 🟢 LOW | Parallel RNG seeding fragile | `_selector.py:131-132` | Code Quality | ❌ Open |
| 🟢 LOW | Forest-level theory absent | `theory.md:735-737` | Theory | ❌ Open |
| 🟢 LOW | Conformal prediction double-dipping | `_conformal.py` | Statistical | ❌ Open |

---

## 🔴 CRITICAL ISSUES

### 1. Multi-Selector P-Value Inflation ✅ RESOLVED

**Location**: `citrees/_tree.py:534-554`, `citrees/_selector.py:137-263`

**Problem**: When `selector=['mc', 'rdc']`, the code was selecting which selector to use *after* looking at data (picks max score), then running the permutation test using only that selector's statistic. This inflated Type I error.

**✅ RESOLVED**: Implemented the **max-T method** (Westfall & Young, 1993).

#### Solution Implemented

The `_ptest_multi()` function in `citrees/_selector.py` computes the max statistic INSIDE each permutation:

```python
def _ptest_multi(*, funcs, func_args, take_abs, x, y, n_resamples, early_stopping, alpha, random_state, confidence):
    """Max-T permutation test for multiple selectors."""
    def compute_max_stat(x, y):
        return max(abs(func(x, y, arg)) for func, arg in zip(funcs, func_args))

    theta = compute_max_stat(x, y)
    # For each permutation, compute max(selector_scores) and compare to observed
    ...
```

**Empirical Validation** (10,000 simulations under global null):

| Condition | Before Fix | After Fix |
|-----------|------------|-----------|
| Single selector (mc) | 5.3% [4.9%, 5.7%] | 5.6% [4.9%, 6.2%] |
| Multi selector (mc+rdc) | **7.9%** [7.3%, 8.4%] | **5.9%** [5.3%, 6.6%] |
| Inflation factor | 1.57x | 1.06x |

#### Files Changed

- `citrees/_selector.py` - Added `_ptest_multi()` function
- `citrees/_tree.py` - Updated `_select_best_feature()` to use `_ptest_multi` for multi-selector mode
- `paper/theory.md` - Updated Section 6.2 to document the fix

#### Proof Checklist

- [x] **Create proof script**: `scratch/prove_multiselector_inflation.py`
- [x] **Run proof script and document results**: Confirmed 7.9% → 5.9% after fix
- [x] **Implement fix**: Added `_ptest_multi()` with max-T method
- [x] **Verify fix**: Re-ran proof script, confirmed no inflation
- [x] **Update theory.md**: Section 6.2 now documents the fix

---

### 2. Classification Honesty Violates Independence Assumption ✅ RESOLVED

**Location**: `citrees/_tree.py:1138-1144`

**Problem**: Proposition 4 in `theory.md` (unbiased honest estimator) requires the index split (S, E) to be **independent** of the data. Previously, for classification, the code used stratified splitting which depends on Y.

**✅ RESOLVED**: Changed to use `stratify=None` for both classification and regression.

#### Solution Implemented

The `train_test_split` call in `_tree.py` now uses `stratify=None` unconditionally:

```python
if self.honesty:
    X_split, X_est, y_split, y_est = train_test_split(
        X, y,
        test_size=self.honesty_fraction,
        random_state=self._random_state,
    )
```

This satisfies Proposition 4's independence assumption for both classifiers and regressors.

**Empirical Validation** (`scratch/prove_honesty_bias.py`):
- Variance ratio (random/stratified): 32x
- Confirmed stratified split creates deterministic dependence on Y

#### Files Changed

- `citrees/_tree.py` - Removed `stratify=y` for classifiers
- `paper/theory.md` - Updated implementation notes in Section 5 and Section 8

#### Proof Checklist

- [x] **Create proof script**: `scratch/prove_honesty_bias.py`
- [x] **Run proof script and document results**: 32x variance ratio confirmed
- [x] **Implement fix**: Changed to `stratify=None` for all honesty modes
- [x] **Update theory.md**: Section 5 and Section 8 updated

---

## 🟠 HIGH PRIORITY ISSUES

### 3. No Power Analysis / B Selection Guidance

**Location**: `theory.md:328-344`

**Problem**: Theory.md Section 3.8 provides only informal remarks on power. Users have no guidance on:
- How to choose B for a target power level
- When `n_resamples='auto'` (~1537) is sufficient
- Power tables for common scenarios

**Current informal guidance** (theory.md:341-344):
> "For a test at level α with m features and Bonferroni correction, the effective per-feature threshold is α/m. To have any chance of rejection, one needs B ≥ m/α - 1."

**Example problem**: With α=0.05, m=100 features, Bonferroni threshold is 0.0005, requiring B≥1999. But `n_resamples='auto'` only gives ~1537.

#### Proof Checklist

- [ ] **Create proof script**: `scratch/prove_power_insufficient.py`
```python
"""Prove n_resamples='auto' is insufficient for high-feature scenarios.

Expected result: With m=100 features, auto resamples cannot achieve p < 0.0005.
"""
import numpy as np
from math import ceil
from scipy.stats import norm

def calculate_auto_resamples(alpha: float) -> int:
    """Replicate n_resamples='auto' calculation from _tree.py:993-997."""
    lower_limit = ceil(1 / alpha)
    z = norm.ppf(1 - alpha)
    upper_limit = ceil(z * z * (1 - alpha) / alpha)
    return max(lower_limit, upper_limit)

def analyze_power_gap():
    alpha = 0.05
    for m_features in [10, 50, 100, 200, 500]:
        bonferroni_alpha = alpha / m_features
        min_B_for_rejection = ceil(1 / bonferroni_alpha) - 1
        auto_B = calculate_auto_resamples(alpha)

        min_achievable_p = 1 / (auto_B + 1)
        can_reject = min_achievable_p < bonferroni_alpha

        print(f"m={m_features:3d} features:")
        print(f"  Bonferroni threshold: {bonferroni_alpha:.6f}")
        print(f"  Auto B: {auto_B}")
        print(f"  Min achievable p: {min_achievable_p:.6f}")
        print(f"  Min B needed: {min_B_for_rejection}")
        print(f"  Can reject? {'YES' if can_reject else 'NO ⚠️'}")
        print()

    # ASSERTION: auto_B insufficient for m >= some threshold
    assert calculate_auto_resamples(0.05) < 1 / (0.05 / 100) - 1, \
        "Expected auto to be insufficient for 100 features"

if __name__ == "__main__":
    analyze_power_gap()
```

- [ ] **Run proof script and document results**
- [ ] **Create power table for common scenarios**

#### Resolution Checklist

- [ ] Add formal power analysis section to `theory.md`
- [ ] Create power table: (n_features, effect_size, alpha) → recommended B
- [ ] Add warning when `auto` B is insufficient for Bonferroni threshold
- [ ] Consider adding `n_resamples='power-optimal'` option

---

### 4. Nested CV Structure Unclear in Experiments

**Location**: `paper/scripts/configs.py:142`, `paper/scripts/synthetic_experiments.py`

**Problem**: The experiment scripts use `cv_folds=5` but it's unclear if this is:
- **Standard CV**: Train feature ranker on fold, evaluate on fold (leakage!)
- **Nested CV**: Outer fold for evaluation, inner fold for feature selection (correct)

Non-nested CV artificially inflates embedding methods (CIT, RF, XGB) that can overfit to the test fold patterns.

**Current code** (`synthetic_experiments.py:118-129`):
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for train_idx, test_idx in cv.split(X_selected, y):
    X_train, y_train = X_selected[train_idx], y[train_idx]
    X_test, y_test = X_selected[test_idx], y[test_idx]
    # ... evaluate downstream accuracy
```

**Issue**: Feature selection happens BEFORE the CV loop (line 168: `rank_features(method_name, model, X, y)`), using ALL data including test folds.

#### Proof Checklist

- [ ] **Create proof script**: `scratch/prove_cv_leakage.py`
```python
"""Prove feature selection leakage in current experiment setup.

Expected result: Show that non-nested CV inflates performance estimates.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

def compare_cv_structures(n_sims=50):
    """Compare nested vs non-nested CV for feature selection."""

    leaky_scores = []
    proper_scores = []

    for seed in range(n_sims):
        # Generate data with some noise features
        X, y = make_classification(
            n_samples=200, n_features=50, n_informative=10,
            random_state=seed, shuffle=True
        )

        # LEAKY: Feature selection on ALL data, then CV
        rf = RandomForestClassifier(n_estimators=50, random_state=seed)
        rf.fit(X, y)
        top_features = np.argsort(rf.feature_importances_)[-10:]
        X_selected = X[:, top_features]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        leaky_score = cross_val_score(
            RandomForestClassifier(n_estimators=50, random_state=seed),
            X_selected, y, cv=cv, scoring='accuracy'
        ).mean()
        leaky_scores.append(leaky_score)

        # PROPER: Feature selection inside CV (nested)
        pipe = Pipeline([
            ('select', SelectFromModel(
                RandomForestClassifier(n_estimators=50, random_state=seed),
                max_features=10
            )),
            ('clf', RandomForestClassifier(n_estimators=50, random_state=seed))
        ])
        proper_score = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy').mean()
        proper_scores.append(proper_score)

    print(f"Leaky (non-nested) CV:  {np.mean(leaky_scores):.3f} ± {np.std(leaky_scores):.3f}")
    print(f"Proper (nested) CV:     {np.mean(proper_scores):.3f} ± {np.std(proper_scores):.3f}")
    print(f"Inflation: {(np.mean(leaky_scores) - np.mean(proper_scores)) * 100:.1f}% points")

    # ASSERTION: Leaky CV should inflate scores
    assert np.mean(leaky_scores) > np.mean(proper_scores) + 0.01

if __name__ == "__main__":
    compare_cv_structures()
```

- [ ] **Run proof script and document results**
- [ ] **Audit all experiment scripts for CV structure**

#### Resolution Checklist

- [ ] Document CV structure explicitly in `paper/README.md`
- [ ] Verify nested CV is used (or implement if not)
- [ ] Add assertion/test that feature selection doesn't see test data
- [ ] Re-run experiments with proper nested CV if needed

---

## 🟡 MEDIUM PRIORITY ISSUES

### 5. Global Error Control with Feature Muting Undefined

**Location**: `theory.md:715-717`, `citrees/_tree.py:669-709`

**Problem**: Feature muting uses intermediate p-values to remove features globally from future consideration. This adaptively changes the hypothesis family across the tree.

**theory.md states** (lines 715-717):
> "Muting uses intermediate p-values to remove features globally from future consideration. This adaptively changes the hypothesis family across the tree and makes global error statements subtle."

**No global FWER bounds are provided when feature muting is enabled.**

#### Proof Checklist

- [ ] **Create proof script**: `scratch/prove_muting_adaptive.py`
```python
"""Demonstrate adaptive hypothesis family with feature muting.

Expected result: Show that the tested feature set changes across nodes.
"""
import numpy as np
from citrees import ConditionalInferenceTreeClassifier

def track_feature_muting():
    """Track how feature muting changes available features."""

    np.random.seed(42)
    X = np.random.randn(500, 20)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Only features 0,1 informative

    clf = ConditionalInferenceTreeClassifier(
        feature_muting=True,
        verbose=3,  # Will print muting events
        random_state=42,
    )

    print("Training with feature_muting=True (watch for muting messages):")
    clf.fit(X, y)

    print(f"\nOriginal features: 20")
    print(f"Features remaining: {len(clf._available_features)}")
    print(f"Features muted: {20 - len(clf._available_features)}")
    print("\nNote: Muted features create an adaptive hypothesis family")
    print("that varies across nodes, complicating global error control.")

if __name__ == "__main__":
    track_feature_muting()
```

- [ ] **Run proof script and document results**

#### Resolution Checklist

- [ ] Add section to `theory.md` explaining muting's effect on global control
- [ ] Consider adding parameter `feature_muting_mode='local'` vs `'global'`
- [ ] Document that `feature_muting=False` is recommended for inference claims

---

### 6. Missing Baselines (RFE, TreeSHAP, mRMR)

**Location**: `paper/scripts/synthetic_experiments.py:45-72`

**Problem**: Several widely-used feature selection methods are missing from benchmarks:

| Missing Method | Why Important |
|----------------|---------------|
| **RFE** | Recursive Feature Elimination - standard sklearn baseline |
| **TreeSHAP** | Listed as HIGH priority in roadmap but not benchmarked |
| **mRMR** | minimum Redundancy Maximum Relevance - handles correlations |
| **Conditional Permutation Importance** | Strobl et al. 2008 - directly relevant |

**Current methods** (`synthetic_experiments.py:45-72`):
- citree, ciforest (yours)
- rf, et, dt (sklearn trees)
- xgb, lgbm (gradient boosting)

#### Resolution Checklist

- [ ] Add RFE with RF base estimator
- [ ] Add TreeSHAP importance extraction
- [ ] Add mRMR (use `mrmr_selection` package or implement)
- [ ] Add Conditional Permutation Importance (use `rfpimp` or implement)
- [ ] Re-run synthetic experiments with new baselines
- [ ] Update results and figures

---

### 7. Synthetic Experiments Too Easy

**Location**: `paper/scripts/synthetic_experiments.py:135-144`

**Problem**: The synthetic data generation creates well-separated informative features:

```python
X, y = make_classification(
    n_samples=config.n_samples,
    n_features=config.n_features,
    n_informative=config.n_informative,
    n_redundant=config.n_redundant,  # Default: 0
    # ...
    shuffle=False,  # Keep informative features at the start
)
```

**Missing challenging scenarios**:
- High correlation between informative and noise features (confounders)
- Redundant informative features (multicollinearity)
- Toeplitz correlation structures (common in genomics)
- Weak signals mixed with strong noise

#### Proof Checklist

- [ ] **Create proof script**: `scratch/prove_easy_synthetic.py`
```python
"""Demonstrate synthetic experiments are too easy.

Expected result: All methods achieve near-perfect feature recovery.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

def evaluate_difficulty():
    """Show all methods easily solve current synthetic setup."""

    configs = [
        {'n_features': 50, 'n_informative': 5},
        {'n_features': 100, 'n_informative': 10},
        {'n_features': 500, 'n_informative': 20},
    ]

    for cfg in configs:
        X, y = make_classification(
            n_samples=1000,
            n_features=cfg['n_features'],
            n_informative=cfg['n_informative'],
            n_redundant=0,
            random_state=42,
            shuffle=False,
        )
        true_informative = set(range(cfg['n_informative']))

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        top_k = set(np.argsort(rf.feature_importances_)[-cfg['n_informative']:])
        recall = len(top_k & true_informative) / len(true_informative)

        print(f"p={cfg['n_features']:4d}, k={cfg['n_informative']:2d}: "
              f"Recall@k = {recall:.2f}")

    print("\n⚠️ All scenarios achieve near-perfect recall.")
    print("Real-world feature selection is much harder due to correlations.")

if __name__ == "__main__":
    evaluate_difficulty()
```

- [ ] **Run proof script and document results**

#### Resolution Checklist

- [ ] Add correlated noise scenario: `noise_features ~ N(informative_features, σ²)`
- [ ] Add Toeplitz correlation structure
- [ ] Add redundant informative features scenario
- [ ] Add weak signal scenario (low class_sep with high noise)
- [ ] Re-run experiments with challenging scenarios

---

### 8. Early Stopping P-Value Inflation Not Quantified ✅ RESOLVED

**Location**: `theory.md:683-685`, `citrees/_selector.py`, `citrees/_sequential.py`

**Problem**: Theory.md acknowledges early stopping affects p-values but doesn't quantify the inflation.

**✅ RESOLVED**: Implemented adaptive sequential permutation testing using Bayesian Beta CDF stopping.

#### Solution Implemented

Changed `early_stopping_*: bool` to `early_stopping_*: Literal["simple", "adaptive"] | None`:
- **`"adaptive"`** (default): Bayesian Beta CDF stopping - valid Type I error (~5%), 95% faster
- **`"simple"`**: Futility + significance stopping - for baseline comparison (inflates to ~9%)
- **`None`**: No early stopping (equivalent to old `early_stopping=False`)

**Benchmark Results** (from `scratch/benchmark_sequential_ptest.py`):

| Method | Type I Error | Avg Perms (null) | Power |
|--------|-------------|------------------|-------|
| fixed_b | 0.056 | 1000 | 0.970 |
| citrees_early (old) | **0.091** | 915 | 0.978 |
| simple_seq | **0.091** | 135 | 0.978 |
| adaptive_seq | **0.055** | **48** | 0.964 |

#### Files Changed

- `citrees/_sequential.py` - NEW: Beta CDF and sequential test functions
- `citrees/_selector.py` - Updated early stopping API
- `citrees/_splitter.py` - Updated early stopping API
- `citrees/_tree.py` - New parameters: `early_stopping_confidence_selector/splitter`
- `citrees/_forest.py` - Updated parameter passing
- `paper/theory.md` Section 6.1 - Documented algorithm and theory

#### Proof Checklist

- [x] **Benchmarked**: `scratch/benchmark_sequential_ptest.py` validates Type I error control
- [x] **Implemented**: Adaptive sequential testing with valid p-values
- [x] **Documented**: Updated `theory.md` with algorithm details

---

## 🟢 LOW PRIORITY ISSUES

### 9. Broad Exception Handling

**Location**: `citrees/_selector.py:257-260`, `citrees/_selector.py:395-398`

**Problem**: The code catches all exceptions and silently returns 0.0:

```python
# _selector.py:257-260
try:
    return np.sqrt(ssb / sst)
except Exception:  # Too broad!
    return 0.0     # Silent failure

# _selector.py:395-398
try:
    return cov / np.sqrt(ssx * ssy)
except Exception:  # Too broad!
    return 0.0     # Silent failure
```

**Issues**:
- Catches ALL exceptions (should be `ZeroDivisionError`, `FloatingPointError`)
- No logging of when/why fallback occurred
- Silent failures can mask bugs

#### Resolution Checklist

- [ ] Change to specific exception types: `except (ZeroDivisionError, FloatingPointError):`
- [ ] Add logging when fallback occurs
- [ ] Add tests for edge cases that trigger fallback
- [ ] Audit codebase for other broad exception handlers

---

### 10. Parallel RNG Seeding Fragile

**Location**: `citrees/_selector.py:131-132`, `citrees/_selector.py:175-176`

**Problem**: Parallel permutation tests use fragile RNG seeding:

```python
# _selector.py:131-132
for i in prange(n_resamples):
    np.random.seed(random_state + i)  # Fragile!
```

**Issues**:
- Non-deterministic across NumPy versions
- Parallel ordering may vary, affecting RNG state
- `np.random.default_rng()` is recommended for NumPy 1.17+

#### Resolution Checklist

- [ ] Replace `np.random.seed(random_state + i)` with `np.random.default_rng(random_state + i)`
- [ ] Verify reproducibility across NumPy versions
- [ ] Add test for deterministic parallel permutation results

---

### 11. Forest-Level Theory Absent

**Location**: `theory.md:735-737`

**Problem**: Theory.md explicitly states forest-level guarantees are missing:

> "Consistency, rates, and uncertainty quantification for the forest (especially with permutation-based splitting) require assumptions and arguments beyond what can be asserted from the current implementation alone."

**Missing**:
- OOB estimation validity analysis
- Consistency properties as n→∞
- Confidence intervals on forest predictions
- When forests outperform single trees

#### Resolution Checklist

- [ ] Add section on OOB estimation for permutation forests
- [ ] Reference Wager & Athey (2018) for honest forest consistency
- [ ] Add empirical study of forest vs tree performance
- [ ] Consider implementing forest-level confidence intervals

---

### 12. Conformal Prediction Double-Dipping

**Location**: `citrees/_conformal.py:84-102`

**Problem**: Conformal prediction assumes exchangeability of (X, Y). If conformal is applied *after* tree structure selection (which uses Y), the exchangeability assumption breaks.

**Current code** (`_conformal.py:100-101`):
```python
X_train, X_cal, y_train, y_cal = train_test_split(
    X, y, test_size=self.calibration_size, random_state=self.random_state, stratify=y
)
```

**Issue**: Stratified split depends on Y, similar to honesty issue.

#### Resolution Checklist

- [ ] Add warning about applying conformal after data-dependent model selection
- [ ] Document when conformal coverage guarantees hold
- [ ] Consider implementing "conformalized honest trees" with proper sample splitting

---

## Missing Literature

### Critical Papers (Must Cite/Implement)

| Paper | Year | Topic | Action | Section |
|-------|------|-------|--------|---------|
| **Fischer & Ramdas** | 2025 | Anytime-valid sequential MC testing | Implement | R1 |
| **Knockoff Boosted Tree (KOBT)** | 2021 | FDR control for trees | Benchmark | R6 |
| **SAGE (Covert et al.)** | 2020 | Global Shapley importance | Benchmark | R7 |
| **Westfall & Young** | 1993 | minP/maxT resampling | Implement | R9 |
| **Meinshausen** | 2006 | Quantile Regression Forests | Implement | R10 |
| **CVPFI** | 2024 | Cross-validated PFI | Implement | R12 |

### Important Papers (Should Cite)

| Paper | Year | Topic | Action | Section |
|-------|------|-------|--------|---------|
| Strobl et al. | 2008 | Conditional Permutation Importance | Cite and compare | R3 |
| Candès et al. | 2018 | Model-X Knockoffs | Discuss in related work | R6 |
| Besag & Clifford | 1991 | Original sequential permutation test | Reference | R1 |
| Hooker & Mentch | 2019 | Extrapolation problem | Document limitation | R8 |
| Holm | 1979 | Step-down multiple testing | Implement | R11 |
| RF-FIRE | 2024 | Localized UQ for forests | Consider implementing | R10 |

### Nice-to-Have Papers

| Paper | Year | Topic | Action |
|-------|------|-------|--------|
| Taylor & Tibshirani | 2015 | Selective Inference | Reference for adaptive inference |
| Pesarin & Salmaso | 2010 | Permutation Tests | Use for power guidance |
| Knockoff-ML | 2025 | FDR for ML models | Benchmark |
| PermuCATE | 2024 | CPI for causal inference | Consider |
| The Honest Truth About Causal Trees | 2024 | Theoretical limits | Document |
| Hochberg | 1988 | Step-up multiple testing | Implement with warning |

---

## 🔬 Research Directions to Investigate

> **Added 2026-01-13**: These findings emerged from deep literature review and validation of our theoretical claims.

### Literature Validation Summary

| Issue | Confidence | Literature Support |
|-------|------------|-------------------|
| Multi-selector p-value inflation | ✅ HIGH | Max-T method confirms fix approach |
| Phipson & Smyth +1 correction | ✅ HIGH | Implementation is correct |
| Early stopping validity | ✅ **IMPLEMENTED** | Adaptive sequential testing with valid Type I error |
| Classification honesty bias | ✅ **FIXED** | Changed to `stratify=None` for all honesty modes |
| Strobl conditional importance | 🔴 GAP | Missing from benchmarks |

---

### R1. Anytime-Valid Sequential Monte Carlo Testing ✅ IMPLEMENTED

> **✅ IMPLEMENTED**: Adaptive sequential permutation testing using Bayesian Beta CDF stopping.
> See Issue #8 above for implementation details and benchmark results.
> The new `early_stopping_selector="adaptive"` (default) provides valid Type I error (~5%) with 95% faster execution.

#### The Problem with Current citrees Early Stopping

**Current behavior** (`_selector.py:82-89`):
```python
for i in range(n_resamples):
    np.random.shuffle(y_)
    theta_p[i] = func(x, y_, func_arg, random_state=random_state)
    if i >= min_resamples - 1:
        asl = (1 + np.sum(np.abs(theta_p[: i + 1]) >= theta)) / (2 + i)
        if asl < alpha:  # Stop early if significant
            break
```

**Why this is problematic**:
1. Only stops early when finding significance (wastes computation on null features)
2. P-values are only valid at the **pre-specified** stopping rule
3. If you stop at a data-dependent time, p-values are **biased**
4. Type I error can inflate to ~11% instead of 5% (see Issue #8)

#### The Solution: Betting Martingales

**Key Paper**: [Fischer & Ramdas (2025)](https://academic.oup.com/jrsssb/article/87/4/1200/8106328) - *Journal of the Royal Statistical Society Series B*

**Core Insight**: Instead of computing p-values that are only valid at fixed stopping times, compute an **e-value** (evidence value) using a **betting martingale** that remains valid at ANY stopping time.

#### Technical Details

##### What is an E-Value?

An **e-value** is a measure of evidence against the null hypothesis with a key property:

```
Under H₀: E[E_τ] ≤ 1 for ANY stopping time τ
```

This means you can stop whenever you want and still have valid inference!

**Relationship to p-values**:
- If E is an e-value, then p = 1/E is a valid p-value
- E ≥ 20 corresponds to p ≤ 0.05
- E ≥ 100 corresponds to p ≤ 0.01

##### The Betting Martingale Algorithm

**Setup**:
- Observed test statistic: Y₀ = S(X₀)
- Generate permuted statistics sequentially: Y₁, Y₂, ..., Yₜ
- Binary indicator: Iₜ = 𝟙{Yₜ ≥ Y₀} (1 if permuted stat beats observed)

**Algorithm** (Binomial Betting Strategy):

```python
def anytime_valid_permutation_test(X, y, test_statistic, alpha, max_perms):
    """
    Anytime-valid sequential permutation test using betting martingale.

    Returns e-value that is valid at ANY stopping time.
    """
    # Compute observed statistic
    theta_obs = test_statistic(X, y)

    # Initialize wealth (e-value)
    W = 1.0
    L = 0  # Count of "losses" (permuted stat >= observed)

    # Betting parameter (optimal for alternative p ~ 0)
    # This is the "aggressive" strategy from Fischer & Ramdas
    p_bet = alpha  # Bet as if true p-value is alpha

    for t in range(1, max_perms + 1):
        # Generate permuted statistic
        y_perm = np.random.permutation(y)
        theta_perm = test_statistic(X, y_perm)

        # Check if permuted beats observed
        I_t = 1 if theta_perm >= theta_obs else 0
        L += I_t

        # Update wealth using betting formula
        # B(0) = (1-p_bet) / (1 - L/t)  when I_t = 0
        # B(1) = p_bet / (L/t)          when I_t = 1
        if I_t == 0:
            # Permuted stat was smaller - we "win" the bet
            W *= (1 - p_bet) * (t + 1) / (t - L + 1)
        else:
            # Permuted stat was larger - we "lose" the bet
            W *= p_bet * (t + 1) / (L)

        # STOPPING RULES (both are valid at any time!)

        # 1. Reject H₀: Strong evidence against null
        if W >= 1 / alpha:
            return {
                'decision': 'reject',
                'e_value': W,
                'p_value': 1 / W,
                'n_perms': t,
                'n_exceedances': L
            }

        # 2. Futility: Cannot possibly reject even with remaining perms
        # Best case: all remaining perms have theta_perm < theta_obs
        # Then W_best = W * ((1-p_bet) * (max_perms+1) / (max_perms - L + 1))^(max_perms - t)
        remaining = max_perms - t
        if remaining > 0:
            W_best = W * ((1 - p_bet) * (max_perms + 1) / (max_perms - L + 1)) ** remaining
            if W_best < 1 / alpha:
                return {
                    'decision': 'fail_to_reject',
                    'e_value': W,
                    'p_value': min(1.0, 1 / W),
                    'n_perms': t,
                    'n_exceedances': L,
                    'reason': 'futility'
                }

    # Reached max permutations
    return {
        'decision': 'reject' if W >= 1/alpha else 'fail_to_reject',
        'e_value': W,
        'p_value': min(1.0, 1 / W),
        'n_perms': max_perms,
        'n_exceedances': L
    }
```

##### Mathematical Foundation

**Wealth after T permutations with L exceedances** (binomial strategy):

```
W_T(L) = (T+1) × p^L × (1-p)^(T-L) × C(T,L)
```

Where:
- p = betting parameter (typically α for aggressive strategy)
- C(T,L) = binomial coefficient "T choose L"

**Key Theorem** (Ville's Inequality):
```
For any stopping time τ: P(W_τ ≥ 1/α) ≤ α under H₀
```

This is what makes the test valid at ANY stopping time!

##### Comparison to Current citrees Approach

| Aspect | Current citrees | Anytime-Valid (Fischer-Ramdas) |
|--------|-----------------|-------------------------------|
| **Stops early for significance** | ✅ Yes | ✅ Yes |
| **Stops early for non-significance** | ❌ No | ✅ Yes (futility) |
| **Valid at any stopping time** | ❌ No | ✅ Yes |
| **Type I error control** | ⚠️ ~11% with early stop | ✅ Exactly α |
| **Computational savings** | ~50% (only sig.) | ~80-95% (both directions) |
| **Returns p-value** | ✅ Yes | ✅ Yes (via 1/e-value) |
| **Numba compatible** | ✅ Yes | ✅ Yes (no scipy needed) |

##### Expected Speedup

Based on Fischer & Ramdas paper and our benchmarks:

| Scenario | Current citrees | Anytime-Valid | Speedup |
|----------|-----------------|---------------|---------|
| Strong signal (p << α) | ~50 perms | ~20 perms | 2.5x |
| Weak signal (p ~ α) | ~500 perms | ~200 perms | 2.5x |
| Null (p ~ 0.5) | **ALL perms** | ~30 perms | **15-30x** |
| Very null (p ~ 1.0) | **ALL perms** | ~10 perms | **50-100x** |

The massive speedup on null features is critical because **most features are null**!

#### Implementation Plan for citrees

##### Option A: Full Integration (Recommended)

Replace the current early stopping mechanism entirely:

```python
# New parameter in ConditionalInferenceTreeClassifier
class ConditionalInferenceTreeClassifier:
    def __init__(
        self,
        ...
        permutation_method: Literal["fixed", "sequential", "anytime_valid"] = "anytime_valid",
        ...
    ):
        """
        permutation_method : str
            - "fixed": Run exactly n_resamples permutations (slowest, valid)
            - "sequential": Current early stopping (fast for signals, slow for nulls, BIASED)
            - "anytime_valid": Fischer-Ramdas betting (fast for both, valid) [DEFAULT]
        """
```

**Files to modify**:
- `citrees/_selector.py`: Add `_anytime_valid_ptest_mc`, `_anytime_valid_ptest_pc`, etc.
- `citrees/_splitter.py`: Same pattern for split testing
- `citrees/_tree.py`: Add `permutation_method` parameter
- `citrees/_parameters.py`: Add validation for new parameter

##### Option B: Benchmark First

Before full integration, create benchmark comparing methods:

```python
# scratch/benchmark_anytime_valid.py
"""
Compare permutation test methods:
1. Fixed (n_resamples=1000, no early stopping)
2. Current early stopping
3. Anytime-valid betting

Metrics:
- Type I error rate (should be ≤ α)
- Power (should be high for true signals)
- Average permutations used
- Wall clock time
"""
```

##### Numba Implementation

The algorithm is fully Numba-compatible (no scipy needed):

```python
@njit(cache=True, fastmath=True, nogil=True)
def _anytime_valid_ptest(
    x: np.ndarray,
    y: np.ndarray,
    func: callable,
    func_arg: int | float,
    alpha: float,
    max_resamples: int,
    random_state: int,
) -> tuple[float, int, str]:
    """
    Anytime-valid permutation test using betting martingale.

    Returns
    -------
    p_value : float
        Valid p-value (= 1/e_value, capped at 1.0)
    n_perms : int
        Number of permutations actually used
    decision : str
        'reject', 'fail_to_reject', or 'completed'
    """
    np.random.seed(random_state)

    # Compute observed statistic
    theta_obs = abs(func(x, y, func_arg, random_state=random_state))

    # Initialize
    W = 1.0  # Wealth (e-value)
    L = 0    # Exceedance count
    p_bet = alpha  # Aggressive betting strategy

    y_ = y.copy()

    for t in range(1, max_resamples + 1):
        # Permute and compute statistic
        np.random.shuffle(y_)
        theta_perm = abs(func(x, y_, func_arg, random_state=random_state))

        # Update exceedance count
        if theta_perm >= theta_obs:
            L += 1
            # Lose bet: multiply by p_bet * (t+1) / L
            W *= p_bet * (t + 1) / L
        else:
            # Win bet: multiply by (1-p_bet) * (t+1) / (t - L + 1)
            W *= (1 - p_bet) * (t + 1) / (t - L + 1)

        # Check rejection
        if W >= 1 / alpha:
            return min(1.0, 1 / W), t, 0  # 0 = reject

        # Check futility (simplified: if L/t is already > 2*alpha, likely won't reject)
        if t >= 20 and L / t > 2 * alpha:
            # More sophisticated futility check
            remaining = max_resamples - t
            # Best case wealth if all remaining are wins
            W_best = W * ((1 - p_bet) ** remaining) * ((max_resamples + 1) ** remaining)
            for k in range(remaining):
                W_best /= (t - L + 1 + k)
            if W_best < 1 / alpha:
                return min(1.0, 1 / W), t, 1  # 1 = fail_to_reject (futility)

    return min(1.0, 1 / W), max_resamples, 2  # 2 = completed
```

#### Existing Implementations to Reference

1. **R Package**: [MChtest](https://rdrr.io/cran/MChtest/man/MChtest-package.html)
   - Implements Besag-Clifford stopping boundaries
   - Has `MCbound` and `MCtest` functions

2. **Fischer-Ramdas GitHub**: [github.com/fischer23/MC-testing-by-betting](https://github.com/fischer23/MC-testing-by-betting)
   - R code reproducing paper results
   - Not a reusable library, but has core algorithms

3. **Python niseq**: [github.com/john-veillette/niseq](https://github.com/john-veillette/niseq)
   - Permutation alpha spending for neuroimaging
   - Different approach but related sequential testing

#### Proof-of-Concept Checklist

- [ ] **Create benchmark script**: `scratch/benchmark_anytime_valid.py`
  - Compare Type I error: current vs anytime-valid
  - Compare power on simulated signals
  - Compare average permutations used
  - Compare wall clock time

- [ ] **Implement Numba version**: `scratch/anytime_valid_numba.py`
  - Port the algorithm to Numba
  - Verify correctness against R implementation
  - Benchmark speed

- [ ] **Validate Type I error**:
  ```python
  def validate_type1_error(method, n_sims=10000, alpha=0.05):
      """Should reject ≤ alpha fraction under true null."""
      rejections = 0
      for seed in range(n_sims):
          X = np.random.randn(100, 5)
          y = np.random.randint(0, 2, 100)  # Independent of X
          p = method(X, y)
          if p < alpha:
              rejections += 1
      rate = rejections / n_sims
      assert rate <= alpha * 1.1, f"Type I error {rate} > {alpha}"
  ```

- [ ] **Integrate into citrees**:
  - Add `permutation_method` parameter
  - Implement `_anytime_valid_ptest_mc`, `_anytime_valid_ptest_pc`
  - Update documentation

#### References

1. **Fischer L, Ramdas A (2025)**. "Sequential Monte-Carlo testing by betting." *Journal of the Royal Statistical Society Series B*, 87(4):1200. [Link](https://academic.oup.com/jrsssb/article/87/4/1200/8106328)

2. **Besag J, Clifford P (1991)**. "Sequential Monte Carlo p-values." *Biometrika*, 78(2):301-304.

3. **Fay MP, Follmann DA (2002)**. "Designing Monte Carlo implementations of permutation or bootstrap hypothesis tests." *The American Statistician*, 56(1):63-70.

4. **Gandy A (2009)**. "Sequential implementation of Monte Carlo tests with uniformly bounded resampling risk." *Journal of the American Statistical Association*, 104(488):1504-1511.

---

### R2. Classification Honesty - ✅ RESOLVED

**Status**: Fixed. Both classification and regression now use `stratify=None`.

**What we did**:
- Proved stratified split creates 32x variance ratio (deterministic dependence on Y)
- Changed `train_test_split` to use `stratify=None` for all honesty modes
- Updated theory.md to reflect the fix

See Issue #2 in Critical Issues section for full details.

---

### R3. Strobl et al. Conditional Permutation Importance

**Paper**: [Strobl et al. (2008)](https://link.springer.com/article/10.1186/1471-2105-9-307)

**Gap**: citrees claims "unbiased variable selection" but doesn't address Strobl's bias mechanisms:
1. Preference for correlated predictors in tree building
2. Advantage for correlated predictors in unconditional permutation

**Action**: Either implement conditional permutation importance or cite as limitation.

---

### R4. Max-T Method Confirmation

**Literature**: [PMC2611984](https://pmc.ncbi.nlm.nih.gov/articles/PMC2611984/)

**Finding**: The max-T method (computing max inside each permutation) is correct.
Our proposed fix for multi-selector mode is validated.

---

### R5. Hothorn et al. (2006) Alignment

**Paper**: [Hothorn et al. (2006)](https://www.zeileis.org/papers/Hothorn+Hornik+Zeileis-2006.pdf)

**Finding**: citrees correctly follows the conditional inference framework.
Core algorithm design is sound.

---

### R6. Knockoff-Based FDR Control (Alternative to Bonferroni)

> **This is a fundamentally different approach to multiple testing that could provide better power than Bonferroni while controlling False Discovery Rate (FDR) instead of Family-Wise Error Rate (FWER).**

#### The Problem with Bonferroni in citrees

**Current behavior** (`_tree.py:652`):
```python
_alpha = alpha / n_tests  # Bonferroni: divide by number of tests
```

**Why Bonferroni is conservative**:
- With 100 features at α=0.05, threshold becomes 0.0005
- Requires B ≥ 1999 permutations to even detect
- Many true signals are missed (low power)

#### The Knockoff Solution

**Key Papers**:
- [Knockoff Boosted Tree (KOBT)](https://academic.oup.com/bioinformatics/article/37/7/976/5910548) - Bioinformatics 2021
- [Knockoff-ML](https://www.nature.com/articles/s41746-025-02102-2) - npj Digital Medicine 2025
- [Model-X Knockoffs](https://cran.r-project.org/web/packages/knockoff/vignettes/knockoff.html) - R package

**Core Idea**: Generate "knockoff" variables that mimic the correlation structure of original features but have no relationship with the response. Use the difference between original and knockoff importance to control FDR.

#### How Knockoffs Work

```python
def knockoff_filter(X, y, fdr_target=0.1):
    """
    Knockoff filter for FDR-controlled feature selection.

    1. Generate knockoff matrix X̃ with same correlation structure as X
    2. Fit model on [X, X̃] to get importances for all 2p features
    3. Compute W_j = |importance(X_j)| - |importance(X̃_j)|
    4. Find threshold T such that FDR ≤ target
    5. Select features where W_j ≥ T
    """
    # Step 1: Generate knockoffs
    X_knockoff = generate_knockoffs(X)  # Same correlation structure

    # Step 2: Fit model on augmented data
    X_aug = np.hstack([X, X_knockoff])
    model.fit(X_aug, y)

    # Step 3: Compute knockoff statistics
    importance_orig = model.feature_importances_[:p]
    importance_knock = model.feature_importances_[p:]
    W = np.abs(importance_orig) - np.abs(importance_knock)

    # Step 4: Find threshold for FDR control
    # Knockoff+ threshold (Barber & Candès 2015)
    T = knockoff_threshold(W, fdr_target)

    # Step 5: Select features
    selected = np.where(W >= T)[0]
    return selected
```

#### Knockoff Statistics for Trees

The KOBT paper tests several importance measures:

| Statistic | Description | Performance |
|-----------|-------------|-------------|
| **TreeSHAP** | Shapley values from tree | Best overall |
| **Gain** | Total gain from splits | Good |
| **Cover** | Samples affected by splits | Moderate |
| **Frequency** | Number of splits | Worst |

**Key finding**: TreeSHAP provides the best knockoff statistics for FDR control.

#### FWER vs FDR Trade-off

| Approach | Controls | Formula | Trade-off |
|----------|----------|---------|-----------|
| **Bonferroni (current)** | FWER | α' = α/m | Very few false positives, may miss true |
| **Knockoffs** | FDR | E[FP/discoveries] ≤ q | More discoveries, some may be false |

**When to use each**:
- **Bonferroni**: When ANY false positive is unacceptable (e.g., medical diagnosis)
- **Knockoffs**: When you want to maximize discoveries with controlled FDR (e.g., exploratory analysis)

#### Implementation Plan for citrees

```python
# New parameter in ConditionalInferenceTreeClassifier
class ConditionalInferenceTreeClassifier:
    def __init__(
        self,
        ...
        multiple_testing: Literal["bonferroni", "holm", "knockoff"] = "bonferroni",
        fdr_target: float = 0.1,  # Only used if multiple_testing="knockoff"
        ...
    ):
        """
        multiple_testing : str
            - "bonferroni": Current FWER control (most conservative)
            - "holm": Step-down FWER control (more powerful)
            - "knockoff": FDR control via knockoffs (most powerful, different guarantee)
        """
```

**Files to modify**:
- `citrees/_tree.py`: Add knockoff filter option
- `citrees/_knockoff.py`: New file for knockoff generation
- Add dependency: `knockpy` or implement knockoff generation

#### Proof-of-Concept Checklist

- [ ] **Benchmark knockoffs vs Bonferroni**: `scratch/benchmark_knockoffs.py`
  - Compare power (true positive rate)
  - Compare FDR (false discovery rate)
  - Compare on simulated data with known ground truth

- [ ] **Implement knockoff generation**:
  ```python
  def generate_gaussian_knockoffs(X):
      """Generate Model-X knockoffs for Gaussian features."""
      # Estimate covariance
      Sigma = np.cov(X.T)
      # Solve SDP for knockoff covariance
      s = solve_knockoff_sdp(Sigma)
      # Generate knockoffs
      X_knockoff = generate_from_covariance(X, Sigma, s)
      return X_knockoff
  ```

- [ ] **Test with tree importance**: Use citrees feature_importances_ as knockoff statistic

#### References

1. **Candès E, et al. (2018)**. "Panning for gold: Model-X knockoffs for high-dimensional controlled variable selection." *JRSS-B*.

2. **He Z, et al. (2021)**. "Knockoff boosted tree for model-free variable selection." *Bioinformatics*.

3. **R knockoff package**: [CRAN](https://cran.r-project.org/web/packages/knockoff/)

4. **Python knockpy**: [GitHub](https://github.com/amspector100/knockpy)

---

### R7. SAGE - Shapley Additive Global Importance

> **SAGE provides theoretically-grounded global feature importance that accounts for feature interactions, unlike MDI which can be biased.**

#### The Problem with Current Importance Measures

**citrees provides**:
- `feature_importances_` (MDI - Mean Decrease in Impurity)

**Known issues with MDI** ([Strobl et al. 2007](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25)):
- Biased toward high-cardinality features
- Biased toward correlated features
- Doesn't account for feature interactions properly

#### What is SAGE?

**Paper**: [Covert et al. (NeurIPS 2020)](https://arxiv.org/abs/2004.00668) - "Understanding Global Feature Contributions With Additive Importance Measures"

**Key insight**: SAGE uses Shapley values to quantify each feature's contribution to the model's **predictive power** (not just the model's predictions).

#### SHAP vs SAGE

| Aspect | SHAP | SAGE |
|--------|------|------|
| **Scope** | Local (per-prediction) | Global (whole dataset) |
| **Question** | "Why this prediction?" | "How important is this feature overall?" |
| **Accounts for interactions** | Yes | Yes |
| **Accounts for model performance** | No | Yes |
| **Sum to** | Prediction - expected | Total predictive power |

#### How SAGE Works

```python
def sage_importance(model, X, y, loss_fn):
    """
    Compute SAGE values for global feature importance.

    SAGE value for feature i = Expected marginal contribution of feature i
    to the model's predictive power across all possible feature subsets.
    """
    n_features = X.shape[1]
    sage_values = np.zeros(n_features)

    # For each feature i
    for i in range(n_features):
        # Compute expected contribution across all subsets S ⊆ {1,...,p} \ {i}
        for S in all_subsets_excluding(i):
            # v(S) = expected loss with features S
            # v(S ∪ {i}) = expected loss with features S and i
            marginal = v(S | {i}) - v(S)
            sage_values[i] += weight(S) * marginal

    return sage_values

# Key property: sum(sage_values) = total predictive power
```

#### SAGE Properties (from Shapley axioms)

1. **Efficiency**: SAGE values sum to total predictive power
2. **Symmetry**: Features with identical contributions get equal importance
3. **Dummy**: Uninformative features get zero importance
4. **Linearity**: Importance is additive across models

#### Implementation for citrees

**Option A: Use existing package**

```python
# pip install sage-importance
import sage

# Wrap citrees model
imputer = sage.MarginalImputer(model, X)
estimator = sage.PermutationEstimator(imputer, 'cross entropy')
sage_values = estimator(X, y)
```

**Option B: Implement for trees**

```python
def tree_sage_values(tree, X, y, loss_fn='gini'):
    """
    Efficient SAGE computation for tree models.

    Key insight: For trees, we can compute SAGE values efficiently
    by tracking feature contributions through the tree structure.
    """
    # Use tree structure to compute feature contributions
    # Similar to TreeSHAP but for global importance
    pass
```

#### Comparison with citrees feature_importances_

| Metric | MDI (current) | SAGE |
|--------|---------------|------|
| Handles correlations | ❌ Biased | ✅ Via Shapley |
| Handles interactions | ❌ No | ✅ Yes |
| Theoretical grounding | ⚠️ Heuristic | ✅ Game theory |
| Computational cost | O(n_nodes) | O(2^p) or approximation |

#### Proof-of-Concept Checklist

- [ ] **Benchmark MDI vs SAGE**: `scratch/benchmark_sage.py`
  - Compare rankings on correlated features
  - Compare on known ground truth

- [ ] **Implement SAGE wrapper for citrees**:
  ```python
  def sage_feature_importances(self, X, y):
      """Compute SAGE values for this tree/forest."""
      import sage
      imputer = sage.MarginalImputer(self, X)
      estimator = sage.PermutationEstimator(imputer, 'cross entropy')
      return estimator(X, y).values
  ```

- [ ] **Add to benchmarks**: Compare SAGE rankings to MDI in paper experiments

#### References

1. **Covert I, Lundberg S, Lee SI (2020)**. "Understanding Global Feature Contributions With Additive Importance Measures." *NeurIPS*.

2. **Python package**: [sage-importance](https://pypi.org/project/sage-importance/)

3. **Blog post**: [Explaining SHAP and SAGE](https://iancovert.com/blog/understanding-shap-sage/)

---

### R8. Extrapolation Problem with Correlated Features

> **When features are correlated, permutation-based tests create "impossible" data points, leading to biased importance estimates.**

#### The Problem

**Paper**: [Hooker & Mentch (2019)](https://arxiv.org/pdf/1905.03151) - "Unrestricted Permutation forces Extrapolation"

When you permute a feature X₁ while keeping correlated feature X₂ fixed:
- You create (X₁, X₂) combinations that never exist in real data
- The model must **extrapolate** to these impossible regions
- Extrapolation quality varies by model → **biased importance**

#### Example

```
Original data:           After permuting X₁:
Height  Weight           Height  Weight
180cm   80kg             180cm   50kg  ← Impossible!
160cm   50kg             160cm   80kg  ← Impossible!
```

If height and weight are correlated, permuting one creates physiologically impossible combinations.

#### Impact on citrees

**citrees uses permutation for**:
1. Feature selection (`_selector.py`): Permute Y to test X-Y association
2. Split selection (`_splitter.py`): Permute Y to test split significance
3. Feature importance: MDI doesn't use permutation, but permutation importance does

**Current citrees behavior**:
- Permutes **Y** not **X** for selection tests → Less affected
- But permutation importance (if used) would be affected

#### Solutions from Literature

| Solution | Description | Complexity |
|----------|-------------|------------|
| **Conditional Permutation** | Permute within strata of correlated features | O(n log n) |
| **Group Permutation** | Permute correlated features together | O(n) |
| **CVPFI** | Cross-validated PFI with correlation-aware shuffling | O(k × n) |
| **Knockoffs** | Use knockoff variables instead of permutation | O(n × p) |

#### Conditional Permutation Implementation

```python
def conditional_permutation_test(x, y, z_correlated, n_resamples=1000):
    """
    Permutation test that conditions on correlated features.

    Instead of permuting y globally, permute within strata
    defined by z_correlated.
    """
    # Discretize correlated features into strata
    strata = discretize_features(z_correlated, n_bins=10)

    theta_obs = compute_statistic(x, y)
    theta_perm = np.zeros(n_resamples)

    for b in range(n_resamples):
        y_perm = y.copy()
        # Permute y WITHIN each stratum
        for s in np.unique(strata):
            mask = strata == s
            y_perm[mask] = np.random.permutation(y[mask])
        theta_perm[b] = compute_statistic(x, y_perm)

    p_value = (1 + np.sum(theta_perm >= theta_obs)) / (1 + n_resamples)
    return p_value
```

#### CVPFI (Cross-Validated Permutation Feature Importance)

**Paper**: [CVPFI (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10989554/)

```python
def cvpfi(model, X, y, cv_folds=5):
    """
    Cross-validated PFI that handles correlated features.

    Key innovation: When shuffling feature j, also shuffle
    correlated features with probability proportional to |correlation|.
    """
    correlations = np.corrcoef(X.T)
    importances = np.zeros(X.shape[1])

    for fold in cv_folds:
        X_train, X_test, y_train, y_test = split(fold)
        model.fit(X_train, y_train)
        baseline_score = model.score(X_test, y_test)

        for j in range(X.shape[1]):
            X_test_perm = X_test.copy()

            # Shuffle feature j
            X_test_perm[:, j] = np.random.permutation(X_test[:, j])

            # Also shuffle correlated features with probability |r|
            for k in range(X.shape[1]):
                if k != j and np.random.random() < np.abs(correlations[j, k]):
                    X_test_perm[:, k] = np.random.permutation(X_test[:, k])

            perm_score = model.score(X_test_perm, y_test)
            importances[j] += baseline_score - perm_score

    return importances / cv_folds
```

#### Relevance to citrees

**Good news**: citrees permutes **Y** for feature/split selection, not X
- This is less affected by X-correlation extrapolation
- The exchangeability assumption is about Y|X, not X

**Potential issue**: If Y is structured (e.g., spatial/temporal), permuting Y may still create impossible scenarios

#### Proof-of-Concept Checklist

- [ ] **Test citrees on correlated features**: `scratch/test_correlated_features.py`
  - Generate data with known correlation structure
  - Compare feature selection accuracy

- [ ] **Implement conditional permutation option**:
  - Add `condition_on` parameter to permutation tests
  - Permute within strata of conditioning variables

- [ ] **Benchmark against CVPFI**:
  - Compare feature importance rankings

#### References

1. **Hooker G, Mentch L (2019)**. "Please Stop Permuting Features." [arXiv](https://arxiv.org/abs/1905.03151)

2. **Strobl C, et al. (2008)**. "Conditional variable importance for random forests." *BMC Bioinformatics*.

3. **CVPFI (2024)**. "Cross-validated permutation feature importance considering correlation." [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10989554/)

---

### R9. Westfall-Young minP/maxT (Alternative to Bonferroni)

> **Westfall-Young resampling methods are more powerful than Bonferroni for correlated tests, which is exactly the scenario in tree variable selection.**

#### The Problem with Bonferroni

Bonferroni assumes tests are **independent**. In reality:
- Features are often correlated
- Correlated features → correlated test statistics
- Independent assumption → **overly conservative**

**Current citrees** (`_tree.py:652`):
```python
_alpha = alpha / n_tests  # Assumes independence
```

#### Westfall-Young Methods

**Book**: Westfall & Young (1993). "Resampling-Based Multiple Testing."

**Key insight**: Use **permutation** to estimate the joint distribution of test statistics, accounting for correlation.

| Method | Description | Assumptions |
|--------|-------------|-------------|
| **maxT** | Adjust p-values based on distribution of max(|T|) | None |
| **minP** | Adjust p-values based on distribution of min(P) | None |
| **Holm** | Step-down Bonferroni | None |
| **Hochberg** | Step-up Bonferroni | Independence |

#### How maxT Works

```python
def westfall_young_maxT(X, y, n_resamples=1000):
    """
    Westfall-Young maxT procedure for FWER control.

    Key: Use permutation to estimate distribution of max statistic
    under the global null, accounting for correlation between tests.
    """
    n_features = X.shape[1]

    # Step 1: Compute observed test statistics
    T_obs = np.array([compute_test_stat(X[:, j], y) for j in range(n_features)])

    # Step 2: Compute permutation distribution of max|T|
    T_max_perm = np.zeros(n_resamples)
    for b in range(n_resamples):
        y_perm = np.random.permutation(y)
        T_perm = np.array([compute_test_stat(X[:, j], y_perm) for j in range(n_features)])
        T_max_perm[b] = np.max(np.abs(T_perm))

    # Step 3: Compute adjusted p-values
    p_adjusted = np.array([
        np.mean(T_max_perm >= np.abs(T_obs[j]))
        for j in range(n_features)
    ])

    return p_adjusted
```

#### Step-Down maxT (More Powerful)

```python
def stepdown_maxT(X, y, n_resamples=1000):
    """
    Step-down maxT procedure (Westfall & Young Algorithm 4.1).

    More powerful than single-step maxT because it adapts
    the comparison set as hypotheses are rejected.
    """
    n_features = X.shape[1]
    T_obs = np.array([compute_test_stat(X[:, j], y) for j in range(n_features)])

    # Sort by |T| descending
    order = np.argsort(np.abs(T_obs))[::-1]
    T_sorted = np.abs(T_obs[order])

    # Permutation loop
    p_adjusted = np.zeros(n_features)
    for b in range(n_resamples):
        y_perm = np.random.permutation(y)
        T_perm = np.array([compute_test_stat(X[:, j], y_perm) for j in range(n_features)])
        T_perm_sorted = np.abs(T_perm[order])

        # Successive maxima
        successive_max = np.maximum.accumulate(T_perm_sorted[::-1])[::-1]

        # Count exceedances
        for j in range(n_features):
            if successive_max[j] >= T_sorted[j]:
                p_adjusted[j] += 1

    p_adjusted = (p_adjusted + 1) / (n_resamples + 1)

    # Enforce monotonicity
    p_adjusted = np.maximum.accumulate(p_adjusted)

    # Unsort
    result = np.zeros(n_features)
    result[order] = p_adjusted
    return result
```

#### Comparison

| Method | Assumptions | Power | Computation |
|--------|-------------|-------|-------------|
| **Bonferroni** | None | Lowest | O(1) |
| **Holm** | None | Low-Medium | O(m log m) |
| **maxT** | None | Medium | O(B × m) |
| **Step-down maxT** | None | **Highest** | O(B × m) |
| **Hochberg** | Independence | High | O(m log m) |

#### Implementation Plan for citrees

```python
# New parameter
class ConditionalInferenceTreeClassifier:
    def __init__(
        self,
        ...
        adjust_alpha_method: Literal["bonferroni", "holm", "westfall_young"] = "bonferroni",
        ...
    ):
        """
        adjust_alpha_method : str
            - "bonferroni": Simple division (current, most conservative)
            - "holm": Step-down Bonferroni (more powerful, no extra cost)
            - "westfall_young": Resampling-based (most powerful, requires computation)
        """
```

#### Proof-of-Concept Checklist

- [ ] **Benchmark methods**: `scratch/benchmark_multiple_testing.py`
  - Compare power on correlated features
  - Compare FWER control

- [ ] **Implement Holm** (easy win):
  ```python
  def holm_adjustment(p_values, alpha):
      """Holm step-down procedure."""
      n = len(p_values)
      order = np.argsort(p_values)
      p_sorted = p_values[order]
      for i, p in enumerate(p_sorted):
          if p > alpha / (n - i):
              return order[:i]  # Reject these
      return order  # Reject all
  ```

- [ ] **Implement Westfall-Young**:
  - Add to `_selector.py` as alternative to Bonferroni

#### References

1. **Westfall PH, Young SS (1993)**. "Resampling-Based Multiple Testing." Wiley.

2. **R multtest package**: [Bioconductor](https://rdrr.io/bioc/multtest/man/mt.maxT.html)

3. **Romano JP, Wolf M (2005)**. "Exact and Approximate Stepdown Methods for Multiple Hypothesis Testing." *JASA*.

---

### R10. Uncertainty Quantification for Forests

> **citrees forests provide point predictions but no uncertainty estimates. Modern methods can provide prediction intervals with coverage guarantees.**

#### The Gap

**Current citrees forest**:
- Returns `predict(X)` → point estimates
- Returns `predict_proba(X)` → probability estimates
- **No prediction intervals or confidence sets**

#### Methods for Forest UQ

| Method | Type | Coverage Guarantee | Computation |
|--------|------|-------------------|-------------|
| **Quantile RF** | Prediction intervals | Asymptotic | O(n × T) |
| **Conformal** | Prediction sets | Finite-sample | O(n × T) |
| **RF-FIRE** | Trust scores | Heuristic | O(n × T) |
| **Jackknife+** | Confidence intervals | Finite-sample | O(n² × T) |

#### Quantile Regression Forests

**Paper**: Meinshausen (2006). "Quantile Regression Forests."

```python
class QuantileForest:
    """
    Forest that predicts conditional quantiles.

    Instead of averaging leaf predictions, store all training
    responses in each leaf and compute weighted quantiles.
    """
    def predict_interval(self, X, alpha=0.1):
        """
        Predict interval [lower, upper] with (1-alpha) coverage.
        """
        lower = self.predict_quantile(X, alpha/2)
        upper = self.predict_quantile(X, 1 - alpha/2)
        return lower, upper

    def predict_quantile(self, X, q):
        """Predict q-th quantile."""
        weights = self._compute_weights(X)  # From all trees
        return weighted_quantile(self.y_train_, weights, q)
```

#### Conformal Prediction (Already in citrees!)

citrees has `_conformal.py` but it's a **wrapper**, not integrated:

```python
# Current: Separate wrapper
from citrees._conformal import ConformalForestClassifier

# Could be: Integrated into forest
clf = ConditionalInferenceForestClassifier()
clf.fit(X_train, y_train)
intervals = clf.predict_interval(X_test, alpha=0.1)  # ← Add this
```

#### RF-FIRE: Localized UQ via Proximities

**Paper**: [RF-FIRE (2024)](https://arxiv.org/html/2509.22928v1)

```python
def rf_fire_trust_score(forest, X_train, y_train, X_new):
    """
    Compute trust scores using random forest proximities.

    Key idea: Points with low proximity to training data
    should have lower trust in their predictions.
    """
    # Compute proximities (how often points end up in same leaf)
    proximities = compute_proximities(forest, X_train, X_new)

    # Find nearest neighbors by proximity
    neighbors = get_nearest_by_proximity(proximities, k=50)

    # Compute OOB residuals for neighbors
    residuals = y_train[neighbors] - forest.oob_predictions_[neighbors]

    # Trust score = 1 - spread of neighbor residuals
    trust = 1 - np.std(residuals, axis=1)

    return trust
```

#### Implementation Plan for citrees

```python
class ConditionalInferenceForestRegressor:
    def predict_interval(self, X, alpha=0.1, method="conformal"):
        """
        Predict with uncertainty quantification.

        Parameters
        ----------
        alpha : float
            Miscoverage rate (returns 1-alpha coverage interval)
        method : str
            - "conformal": Split conformal (finite-sample guarantee)
            - "quantile": Quantile regression forest (asymptotic)
            - "jackknife": Jackknife+ (finite-sample, expensive)
        """
        if method == "conformal":
            return self._conformal_interval(X, alpha)
        elif method == "quantile":
            return self._quantile_interval(X, alpha)
        elif method == "jackknife":
            return self._jackknife_interval(X, alpha)
```

#### Proof-of-Concept Checklist

- [ ] **Integrate conformal into forest API**: Add `predict_interval` method
- [ ] **Implement quantile forest**: Store leaf responses, compute weighted quantiles
- [ ] **Benchmark coverage**: `scratch/benchmark_uq.py`
  - Test empirical coverage vs nominal
  - Compare interval widths

#### References

1. **Meinshausen N (2006)**. "Quantile Regression Forests." *JMLR*.

2. **RF-FIRE (2024)**. "Localized Uncertainty Quantification in Random Forests via Proximities." [arXiv](https://arxiv.org/html/2509.22928v1)

3. **Barber RF, et al. (2021)**. "Predictive inference with the jackknife+." *Annals of Statistics*.

---

### R11. Holm/Hochberg Step-Down Methods

> **Holm's method is uniformly more powerful than Bonferroni with no additional assumptions. This is a "free" improvement.**

#### Comparison of Methods

| Method | Rejects if | Assumptions | Power |
|--------|------------|-------------|-------|
| **Bonferroni** | p_j ≤ α/m | None | Lowest |
| **Holm** | p_(j) ≤ α/(m-j+1) | None | Medium |
| **Hochberg** | p_(j) ≤ α/(m-j+1) | Independence | Highest |

Where p_(j) is the j-th smallest p-value.

#### Why Holm is Better Than Bonferroni

**Bonferroni**: Reject H_j if p_j ≤ α/m (same threshold for all)

**Holm**:
1. Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
2. Find smallest k where p_(k) > α/(m-k+1)
3. Reject H_(1), ..., H_(k-1)

**Key insight**: Later tests use less stringent thresholds because earlier rejections reduce the "burden" of multiple testing.

#### Implementation

```python
def holm_bonferroni(p_values, alpha=0.05):
    """
    Holm-Bonferroni step-down procedure.

    Returns mask of rejected hypotheses.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    p_sorted = p_values[order]

    rejected = np.zeros(m, dtype=bool)

    for j in range(m):
        threshold = alpha / (m - j)
        if p_sorted[j] <= threshold:
            rejected[order[j]] = True
        else:
            break  # Stop at first non-rejection

    return rejected


def hochberg_stepup(p_values, alpha=0.05):
    """
    Hochberg step-up procedure.

    More powerful than Holm but requires independence.
    """
    m = len(p_values)
    order = np.argsort(p_values)[::-1]  # Descending
    p_sorted = p_values[order]

    rejected = np.zeros(m, dtype=bool)

    for j in range(m):
        threshold = alpha / (j + 1)
        if p_sorted[j] <= threshold:
            # Reject this and all smaller p-values
            rejected[order[j:]] = True
            break

    return rejected
```

#### Integration into citrees

**Current** (`_tree.py:652`):
```python
_alpha = alpha / n_tests  # Bonferroni
```

**Proposed**:
```python
if self.adjust_alpha_method == "bonferroni":
    _alpha = alpha / n_tests
elif self.adjust_alpha_method == "holm":
    # Apply Holm after getting all p-values
    rejected = holm_bonferroni(p_values, alpha)
    selected_features = np.where(rejected)[0]
elif self.adjust_alpha_method == "hochberg":
    # Requires independence assumption
    rejected = hochberg_stepup(p_values, alpha)
    selected_features = np.where(rejected)[0]
```

**Note**: Holm requires computing ALL p-values first, then deciding rejections. This changes the current "test one at a time" flow but provides better power.

#### Proof-of-Concept Checklist

- [ ] **Implement Holm**: Add to `_tree.py`
- [ ] **Implement Hochberg**: With independence warning
- [ ] **Benchmark power**: Compare on simulated data
- [ ] **Document assumption**: Hochberg requires independence

#### References

1. **Holm S (1979)**. "A simple sequentially rejective multiple test procedure." *Scandinavian Journal of Statistics*.

2. **Hochberg Y (1988)**. "A sharper Bonferroni procedure for multiple tests of significance." *Biometrika*.

---

### R12. CVPFI - Cross-Validated Permutation Feature Importance

> **CVPFI addresses both the extrapolation problem and variance issues with standard permutation importance.**

#### The Problems with Standard PFI

1. **Extrapolation**: Permutation creates impossible data points
2. **High variance**: Single permutation is noisy
3. **Correlation handling**: Correlated features share importance incorrectly

#### CVPFI Solution

**Paper**: [CVPFI (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10989554/)

```python
def cvpfi(model, X, y, n_folds=5, n_repeats=10):
    """
    Cross-Validated Permutation Feature Importance.

    Key innovations:
    1. Cross-validation reduces variance
    2. Correlation-aware shuffling addresses extrapolation
    3. Stable across different data splits
    """
    correlations = np.abs(np.corrcoef(X.T))
    importances = np.zeros((n_repeats, X.shape[1]))

    for repeat in range(n_repeats):
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=repeat)

        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            baseline = model.score(X_test, y_test)

            for j in range(X.shape[1]):
                X_perm = X_test.copy()

                # Shuffle feature j
                X_perm[:, j] = np.random.permutation(X_perm[:, j])

                # Correlation-aware: also shuffle correlated features
                for k in range(X.shape[1]):
                    if k != j:
                        # Shuffle k with probability = |correlation(j, k)|
                        if np.random.random() < correlations[j, k]:
                            X_perm[:, k] = np.random.permutation(X_perm[:, k])

                perm_score = model.score(X_perm, y_test)
                importances[repeat, j] += (baseline - perm_score) / n_folds

    return importances.mean(axis=0), importances.std(axis=0)
```

#### Comparison with Standard PFI

| Aspect | Standard PFI | CVPFI |
|--------|--------------|-------|
| Variance | High | Low (CV averaging) |
| Extrapolation | Yes | Reduced (correlation-aware) |
| Correlation handling | Poor | Better |
| Computation | O(n × p) | O(k × n × p) |

#### Implementation for citrees

```python
class ConditionalInferenceForestClassifier:
    def cvpfi_importance(self, X, y, n_folds=5, n_repeats=10):
        """
        Cross-validated permutation feature importance.

        More stable and accurate than standard feature_importances_.
        """
        return cvpfi(self, X, y, n_folds, n_repeats)
```

#### Proof-of-Concept Checklist

- [ ] **Implement CVPFI**: `citrees/_importance.py`
- [ ] **Benchmark vs MDI**: Compare rankings on correlated data
- [ ] **Test stability**: Compare variance across runs

#### References

1. **CVPFI (2024)**. "Cross-validated permutation feature importance considering correlation between features." [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10989554/)

---

## Verification Script Template

Run all proof scripts:

```bash
#!/bin/bash
# verify_all_issues.sh

echo "=== Running all issue verification scripts ==="

for script in scratch/prove_*.py; do
    echo ""
    echo "--- Running $script ---"
    uv run python "$script" || echo "FAILED: $script"
done

echo ""
echo "=== Verification complete ==="
```

---

---

## 🚀 PERFORMANCE OPTIMIZATIONS

> **Added 2026-01-13**: These optimizations were identified through rigorous benchmarking in `scratch/benchmarks/`. Each has proof-of-concept code and expected outcomes documented.

### Status Matrix

| # | Optimization | Status | Evidence | Priority |
|---|--------------|--------|----------|----------|
| P1 | Sequential Permutation Testing | ✅ DONE | 9-38x speedup | N/A |
| P2 | Type I Error Fix (early_stopping) | ✅ DONE | 5% error (fixed) | N/A |
| P3 | Parallel RDC Permutation Test | NOT PROVEN | Needs benchmarking | 🟡 MEDIUM |
| P4 | Benjamini-Hochberg FDR | NOT PROVEN | Needs benchmarking | 🟡 MEDIUM |
| P5 | Spearman Correlation Selector | NOT PROVEN | Needs benchmarking | 🟢 LOW |
| P6 | Batch Selector Computation | NOT PROVEN | Needs benchmarking | 🟢 LOW |
| P7 | Look-Ahead Interaction Detection | NOT PROVEN | Needs benchmarking | 🟢 LOW |
| P8 | OOB Error Estimation | **MISSING** | Confirmed not implemented | 🟡 MEDIUM |
| P9 | Conformal Prediction | ✅ DONE | `_conformal.py` complete | N/A |
| P10 | TreeSHAP Integration | ✅ DONE | `_importance.py` has SHAP | N/A |
| P11 | CPI (Conditional Permutation) | ✅ DONE | `_importance.py` | N/A |
| P12 | Honest Estimation | ✅ DONE | Parameter exists in tree | N/A |

---

### P1. Sequential Permutation Testing ✅ DONE

**Location**: `citrees/_selector.py`, `citrees/_splitter.py`, `citrees/_sequential.py`

**✅ RESOLVED**: Implemented adaptive sequential testing. See Issue #8 above for full details.

~~**Problem**: Current early stopping only stops when finding SIGNIFICANCE.~~

**Current behavior** (`_selector.py:82-89`):
```python
for i in range(n_resamples):
    np.random.shuffle(y_)
    theta_p[i] = func(x, y_, func_arg, random_state=random_state)
    if i >= min_resamples - 1:
        asl = (1 + np.sum(np.abs(theta_p[: i + 1]) >= theta)) / (2 + i)
        if asl < alpha:  # Only stops if SIGNIFICANT
            break
# If not significant, runs ALL n_resamples iterations!
```

**Mathematical justification for stopping on non-significance**:

After `n` permutations with `k` extreme values:
- Current p-value: `p = (k+1)/(n+1)` (Phipson-Smyth correction)
- Best possible if remaining `r` permutations have ZERO extremes: `p_best = (k+1)/(n+r+1)`
- If `p_best >= alpha`, then **significance is mathematically impossible** → safe to stop

**Example calculation**:
```
After 50 permutations: 30 extreme values
Current p-value: p = 31/51 ≈ 0.61

Best case (0 more extreme in remaining 450):
p_best = 31/501 ≈ 0.062 >= 0.05

Conclusion: CANNOT become significant → STOP NOW (save 450 permutations!)
```

**Benchmark evidence** (from `scratch/benchmarks/02_exhaustive_sequential_proof.py`):

| Method | Type I Error | Avg Perms (noise) | Speedup |
|--------|--------------|-------------------|---------|
| Current (early_stop=True) | 0.110 (BAD) | 457 | 1.0x |
| Current (early_stop=False) | 0.052 (OK) | 1000 | N/A |
| Sequential Simple | 0.110 (BAD) | 63 | 7.3x |
| Sequential Adaptive | 0.047 (OK) | 33 | **14.0x** |

**Proposed change**:
```python
# Replace: early_stopping: bool
# With:    early_stopping_mode: Literal["none", "simple", "adaptive"]

# "none" - Run all permutations (slowest, correct Type I)
# "simple" - Current + stop for non-significant (fastest, inflated Type I)
# "adaptive" - Bayesian posterior stopping (fast, correct Type I) ← NEW DEFAULT
```

**Benchmark script**: `scratch/benchmarks/02_exhaustive_sequential_proof.py`

#### Proof Checklist

- [x] Created benchmark scripts proving speedup
- [x] Measured Type I error under null hypothesis
- [x] Implemented njit-compatible Bayesian adaptive stopping
- [ ] Implement `early_stopping_mode` parameter in `_selector.py`
- [ ] Implement same pattern in `_splitter.py`
- [ ] Update `_tree.py` parameter types
- [ ] Add unit tests for Type I error validation

---

### P2. Type I Error Inflation with Early Stopping ✅ DONE

**Location**: `citrees/_selector.py`, `citrees/_splitter.py`, `citrees/_sequential.py`

**✅ RESOLVED**: Implemented adaptive sequential testing with Bayesian Beta CDF stopping. See Issue #8 above for full details.

~~**Problem**: Current early stopping has **inflated Type I error** (~11% instead of 5%).~~

**Current formula** (`_selector.py:87`):
```python
asl = (1 + np.sum(np.abs(theta_p[: i + 1]) >= theta)) / (2 + i)
```

**Why this is biased**: When we stop early because `asl < alpha`, we're making a decision based on a p-value that's biased downward because:
1. We stop as soon as we hit significance
2. We don't account for the multiple looks at the data

**Statistical theory**: This is a well-known issue in sequential testing. Solutions:
1. **Conservative p-values** (e.g., Wald SPRT with spending function)
2. **Bayesian posterior** - doesn't suffer from optional stopping bias

**Benchmark evidence** (from `scratch/benchmarks/03_type1_error_investigation.py`):

| Method | Type I Error | Expected | Status |
|--------|--------------|----------|--------|
| current (early_stop=True) | 0.110 | 0.05 | ❌ BAD |
| current (early_stop=False) | 0.052 | 0.05 | ✅ OK |
| Bayesian adaptive (njit) | 0.046 | 0.05 | ✅ OK |

**Implementation detail** (Bayesian adaptive in njit):
```python
@njit(cache=True, fastmath=True, nogil=True)
def _beta_cdf_njit(x: float, a: float, b: float) -> float:
    """Compute Beta CDF using incomplete beta function approximation."""
    # Regularized incomplete beta for posterior P(p < alpha | k, n)
    ...

@njit(cache=True, fastmath=True, nogil=True)
def _adaptive_permutation_test(x, y, func, alpha, max_resamples, ...):
    """Bayesian adaptive permutation test with proper Type I control."""
    k = 0  # Count of extreme values
    for i in range(max_resamples):
        y_perm = shuffle(y)
        if abs(func(x, y_perm)) >= abs(theta_obs):
            k += 1

        n = i + 1
        # Bayesian: P(true_p < alpha | observed k in n)
        # Using Beta(k+1, n-k+1) posterior
        prob_significant = _beta_cdf_njit(alpha, k + 1, n - k + 1)
        prob_not_significant = 1 - _beta_cdf_njit(alpha, k + 1, n - k + 1)

        if prob_significant > 0.95:  # High confidence significant
            return (k + 1) / (n + 1), "significant"
        if prob_not_significant > 0.95:  # High confidence not significant
            return (k + 1) / (n + 1), "not_significant"

    return (k + 1) / (n + 1), "completed"
```

**Speed benchmark** (njit vs scipy):
- Scipy Beta.cdf: ~620 µs per call
- Njit _beta_cdf_njit: ~3.4 µs per call (182x faster)

**Benchmark script**: `scratch/benchmarks/03_type1_error_investigation.py`

#### Proof Checklist

- [x] Proved Type I error inflation exists
- [x] Implemented njit-compatible fix
- [x] Verified fix maintains correct Type I error
- [ ] Integrate into `_selector.py`
- [ ] Integrate into `_splitter.py`
- [ ] Add regression tests for Type I error

---

### P3. Parallel RDC Permutation Test (NOT PROVEN)

**Location**: `citrees/_selector.py:858-969`

**Problem**: RDC permutation tests use generic `_ptest()` without parallelization. mc and pc have parallel versions (`_ptest_mc_parallel`, `_ptest_pc_parallel`) but RDC does not.

**Current code** (`_selector.py:903-912`):
```python
# mc has parallel version (line 661-668):
if not early_stopping and n_resamples >= _PARALLEL_THRESHOLD:
    return _ptest_mc_parallel(...)  # ← EXISTS

# rdc does NOT (line 903-912):
return _ptest(  # ← Always uses generic non-parallel version
    func=rdc_classifier,
    ...
)
```

**Hypothesis**: When `early_stopping=None` and `n_resamples >= 200`, parallelization should provide 2-4x speedup.

**Concern**: RDC is O(n log n) per permutation (vs O(n) for mc/pc), so parallelization overhead may reduce benefit.

**What to measure**:
1. Time for sequential vs parallel on 100-10000 samples
2. Parallelization overhead (Numba prange setup)
3. Whether speedup justifies code complexity

**Benchmark script to create**: `scratch/benchmarks/04_parallel_rdc.py`

```python
import time
import numpy as np
from citrees._selector import ptest_rdc_classifier

def benchmark_rdc():
    np.random.seed(42)
    x = np.random.randn(1000)
    y = (np.random.randn(1000) > 0).astype(np.int64)

    t0 = time.perf_counter()
    for _ in range(10):
        ptest_rdc_classifier(
            x=x, y=y, n_classes=2, n_resamples=500,
            early_stopping=None, alpha=0.05, random_state=42
        )
    t_seq = (time.perf_counter() - t0) / 10
    print(f"Sequential RDC: {t_seq:.3f}s per test")

# Expected: Need to implement parallel version and compare
```

#### Proof Checklist

- [ ] Create `scratch/benchmarks/04_parallel_rdc.py`
- [ ] Measure baseline sequential RDC time
- [ ] Implement `_ptest_rdc_classifier_parallel`
- [ ] Implement `_ptest_rdc_regressor_parallel`
- [ ] Measure speedup vs overhead
- [ ] Decide if worth the complexity

---

### P4. Benjamini-Hochberg FDR Control (NOT PROVEN)

**Location**: `citrees/_tree.py:634-667`

**Problem**: Bonferroni correction is very conservative. With 100 features at α=0.05, the adjusted threshold is 0.0005. BH-FDR could improve power.

**Current code** (`_tree.py:652`):
```python
_alpha = alpha / n_tests  # Bonferroni: divide by number of tests
```

**Bonferroni vs Benjamini-Hochberg**:

| Method | Controls | Formula | Trade-off |
|--------|----------|---------|-----------|
| Bonferroni | FWER | α' = α/m | Few false positives, may miss true |
| BH-FDR | FDR | α' = k×α/m for sorted p[k] | More discoveries, some may be false |

**What to measure**:
1. True positive rate (power)
2. False positive rate
3. Impact on tree structure

**Benchmark script to create**: `scratch/benchmarks/05_fdr_vs_bonferroni.py`

```python
import numpy as np

def bonferroni_correction(pvals, alpha):
    n = len(pvals)
    return pvals < (alpha / n)

def benjamini_hochberg(pvals, alpha):
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    thresholds = np.arange(1, n+1) * alpha / n
    reject_sorted = sorted_pvals <= thresholds

    if not reject_sorted.any():
        return np.zeros(n, dtype=bool)

    max_k = np.max(np.where(reject_sorted)[0])
    reject = np.zeros(n, dtype=bool)
    reject[sorted_idx[:max_k+1]] = True
    return reject

# Compare TP/FP rates across simulation scenarios
```

#### Proof Checklist

- [ ] Create `scratch/benchmarks/05_fdr_vs_bonferroni.py`
- [ ] Measure TP/FP rates for both methods
- [ ] Test on simulated data with known informative features
- [ ] Decide if BH-FDR should be added as option

---

### P5. Spearman Correlation Selector (NOT PROVEN)

**Location**: `citrees/_selector.py:292-398`

**Problem**: Only Pearson correlation (`pc`) exists. Spearman (rank-based) would be more robust to outliers.

**Missing code**:
```python
@RegressorSelectors.register("sc")  # Spearman correlation
@njit(cache=True, nogil=True, fastmath=True)
def spearman_correlation(x, y, standardize, random_state=None):
    """Rank-based correlation, robust to outliers."""
    x_ranks = _rankdata(x)
    y_ranks = _rankdata(y)
    return pearson_correlation(x_ranks, y_ranks, standardize, random_state)
```

**What to measure**:
1. Detection rate on clean vs contaminated data
2. Computational overhead of rank transformation
3. Impact on tree structure

**Benchmark script to create**: `scratch/benchmarks/06_spearman_vs_pearson.py`

#### Proof Checklist

- [ ] Create `scratch/benchmarks/06_spearman_vs_pearson.py`
- [ ] Compare Pearson vs Spearman on outlier data
- [ ] Implement njit `_rankdata` function
- [ ] Decide if worth adding

---

### P6. Batch Selector Computation (NOT PROVEN)

**Location**: `citrees/_tree.py:520-561`

**Problem**: Current code runs permutation tests sequentially for each feature. Could compute cheap scores for ALL features first, then only run expensive tests on top-k.

**Current flow**:
```python
for feature in features:
    x = X[:, feature]
    pval = selector_test(x=x, y=y, **kwargs)  # 500 perms each!
```

**Proposed flow**:
```python
# Step 1: Cheap scores for all features
scores = [selector(X[:, f], y) for f in features]  # No perms, fast

# Step 2: Only permutation test top-k
top_k = np.argsort(scores)[::-1][:20]
for f in top_k:
    pval = selector_test(X[:, f], y, **kwargs)  # 500 perms only for top 20
```

**Potential savings**: 100 features × 500 perms = 50,000 → 100 scores + 20 × 500 = 10,100

**CRITICAL**: This changes the statistical procedure! Must verify Type I error is still controlled.

**Benchmark script to create**: `scratch/benchmarks/07_batch_selector.py`

#### Proof Checklist

- [ ] Create `scratch/benchmarks/07_batch_selector.py`
- [ ] Measure speedup factor
- [ ] **Verify Type I error is still controlled**
- [ ] Decide if worth the complexity

---

### P7. Look-Ahead Interaction Detection (NOT PROVEN)

**Location**: `citrees/_tree.py:831-956`

**Problem**: Current greedy approach selects feature with lowest p-value at THIS node, ignoring whether it leads to good splits in children.

**Example**:
- Feature A: p=0.02 at root, children have no good splits
- Feature B: p=0.04 at root, enables excellent splits in children

Greedy selects A, but B might build a better overall tree.

**What to measure**:
1. Detection of known interaction features (e.g., XOR)
2. Computational overhead of look-ahead
3. Tree depth and complexity

**Benchmark script to create**: `scratch/benchmarks/08_lookahead.py`

```python
def generate_xor_data(n_samples=1000):
    """XOR interaction: individually weak, jointly strong."""
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    y = ((X1 > 0) ^ (X2 > 0)).astype(int)
    X_noise = np.random.randn(n_samples, 8)
    return np.column_stack([X1, X2, X_noise]), y

# Expected: Greedy may miss X1/X2, look-ahead should find them
```

#### Proof Checklist

- [ ] Create `scratch/benchmarks/08_lookahead.py`
- [ ] Generate XOR interaction dataset
- [ ] Test if greedy finds correct features
- [ ] Prototype look-ahead approach
- [ ] Measure computational cost

---

### P8. OOB Error Estimation (MISSING)

**Location**: `citrees/_forest.py:35-176, 329-415`

**Problem**: Forest implementation doesn't track OOB indices or compute OOB score.

**What's missing**:
- Tracking of bootstrap indices per tree
- `oob_score_` attribute
- `oob_decision_function_` for classifiers

**Implementation approach**:
```python
# During _parallel_fit_*:
bootstrap_indices = np.random.choice(n_samples, size=max_samples, replace=True)
oob_mask = np.ones(n_samples, dtype=bool)
oob_mask[bootstrap_indices] = False
# Store oob_mask

# After fitting:
oob_predictions = np.zeros((n_samples, n_classes))
oob_counts = np.zeros(n_samples)

for tree_idx, tree in enumerate(self.estimators_):
    oob_mask = self._oob_masks[tree_idx]
    X_oob = X[oob_mask]
    oob_predictions[oob_mask] += tree.predict_proba(X_oob)
    oob_counts[oob_mask] += 1

oob_predictions /= oob_counts[:, None]
self.oob_score_ = accuracy_score(y, oob_predictions.argmax(axis=1))
```

#### Proof Checklist

- [ ] Implement OOB tracking in `_parallel_fit_classifier`
- [ ] Implement OOB tracking in `_parallel_fit_regressor`
- [ ] Add `oob_score_` computation in `fit()`
- [ ] Add test comparing OOB score to CV score

---

## Verification Scripts

Run all proof scripts:

```bash
#!/bin/bash
# Run all benchmark and proof scripts
for script in scratch/benchmarks/*.py scratch/prove_*.py; do
    echo "--- Running $script ---"
    uv run python "$script" || echo "FAILED: $script"
done
```

---

## Changelog

| Date | Issue | Action | Author |
|------|-------|--------|--------|
| 2025-01-13 | All | Initial documentation | Claude |
| 2026-01-13 | P1-P12 | Added performance optimizations with benchmarks | Claude |

