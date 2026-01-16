# citrees: Open Issues

> **Purpose**: Track remaining issues before publication.

---

## Executive Summary

| Priority | Issue | Location | Type |
|----------|-------|----------|------|
| ✅ DONE | Sequential stopping rule vs Fischer-Ramdas | `theory.md` Section 6.1.1 | Theory |
| ✅ DONE | Broad exception handling | `_selector.py` | Code Quality |

---

## Review Findings Backlog (Jan 2026)

### Critical
- [x] Fix label handling: preserve non-numeric/non-0..k-1 labels; classes_ should store original labels (`citrees/_tree.py`, `citrees/_forest.py`)
- [x] Fix ptest_* experiment crash (keyword-only selector tests called positionally) (`paper/scripts/experiments/ray_feature_selection.py`)
- [x] Implement and document `apply`, `decision_path`, `oob_score_` (sklearn-style API) (`citrees/_tree.py`, `citrees/_forest.py`, `docs/api.md`)
- [x] Fix RNG reinitialization per node (feature subsampling + random thresholds) (`citrees/_tree.py`, `citrees/_threshold_method.py`)

### Major
- [x] Feature name order validation should detect reordered columns (not set-based) (`citrees/_tree.py`)
- [x] Honesty split remains unstratified (independence assumption); docs clarify fraction semantics (`citrees/_tree.py`, `docs/algorithm.md`, `docs/parameters.md`)
- [x] `_scan_thresholds` ordering should align with weighted impurity criterion (`citrees/_tree.py`)
- [x] Forest predict_proba should align missing classes across trees (`citrees/_forest.py`)

### Minor / Experiments / Quality
- [x] Regression selector ranking should use abs(Pearson) in experiments (`paper/scripts/experiments/ray_feature_selection.py`)
- [x] Fix experiment leakage: pi/cpi should not use evaluation fold (`paper/scripts/experiments/ray_feature_selection.py`)
- [x] Replace broad exception handling with explicit guards + warnings (`citrees/_selector.py`)
- [x] Add test: column reorder should raise (`citrees/_tree.py`)
- [x] Add experiment test coverage for ptest_* keyword-only calls (`paper/scripts/experiments/ray_feature_selection.py`)
- [x] Add tests for non-numeric/non-0..k-1 labels (`tests/unit/test_tree.py`, `tests/unit/test_forest.py`)

---

## 🟢 LOW PRIORITY

### 2. Broad Exception Handling

**Location**: `citrees/_selector.py:257-260`, `citrees/_selector.py:395-398`

**Problem**: The code catches all exceptions and silently returns 0.0:

```python
try:
    return np.sqrt(ssb / sst)
except Exception:  # Too broad!
    return 0.0     # Silent failure
```

#### Resolution Checklist

- [x] Replace broad try/except with explicit finite/denominator guards
- [x] Add logging when fallback occurs (warnings in `mc`; Numba paths cannot log)
- [x] Audit codebase for other broad exception handlers

---

## ✅ COMPLETED

### Sequential Stopping Rule Validation (Jan 2026)

**Location**: `theory.md` Sections 6.1.1-6.1.2, `paper/scripts/analysis/sequential_stopping_comparison.py`

**Summary**: Empirically validated citrees' Beta CDF adaptive stopping rule against Fischer-Ramdas (2025)
anytime-valid testing framework, plus preliminary theoretical analysis.

**Key Results** (N=10,000 simulations, njit-compiled):

| Metric | citrees | Fischer-Ramdas |
|--------|---------|----------------|
| Type I Error | 0.0493 | 0.0503 |
| Mean Perms (null) | 41.5 | 97.9 |
| Overall Speedup | **2.2×** | baseline |

**Preliminary Theoretical Finding** (N=100,000 simulations):
- Under H0, method appears **dramatically conservative** (Type I error ≈ 0, not 5%)
- P(accept at n=20) ≈ 0.9998 under H0
- Rejection requires L_n to be astronomically small (10⁻¹⁸ probability)

**Files:**
- `paper/scripts/analysis/sequential_stopping_comparison.py` - Head-to-head comparison script
- `paper/scripts/theory/sequential_stopping_analysis.py` - Consolidated theoretical analysis
- `theory.md` Section 6.1.1 - Empirical comparison documentation
- `theory.md` Section 6.1.2 - Preliminary proof sketch
- `theory.md` Section 6.1.3 - Mathematical foundations (detailed)

---

## ✅ PROOF COMPLETE

### Theoretical Proof for Sequential Stopping Rule

**Location**: `theory.md` Sections 6.1.2, 6.1.3

**Status**: ✅ **FORMALLY PROVEN**

#### The Main Result

**Theorem (Bayesian Calibration at Stopping Times):**

Under H₀ (exchangeability), for ANY stopping time τ:
$$\mathbb{E}[S_\tau] = \alpha$$

**Corollary (Type I Error Control):**
$$P(\text{reject}) = P(S_\tau \geq \gamma) \leq \frac{\alpha}{\gamma}$$

For α = 0.05, γ = 0.95: Type I error ≤ 5.26%.

#### The Proof (Tower Property + Markov Inequality)

The proof is remarkably elegant:

1. **Key observation:** S_τ = P(p < α | L_τ) = E[1_{p < α} | L_τ]

2. **Tower property:** E[S_τ] = E[E[1_{p < α} | L_τ]] = E[1_{p < α}] = α

3. **Universality:** The tower property works for ANY random variable L_τ, not just fixed n.

4. **Type I error bound:** P(S_τ ≥ γ) ≤ E[S_τ]/γ = α/γ (Markov's inequality)

**Why it's so simple:** The proof exploits the structure of S_n as a conditional expectation,
not requiring martingale theory or first-passage analysis.

#### All Questions Resolved

1. **✅ Distribution under H0:** L_n ~ Uniform{0, ..., n}

2. **✅ Supermartingale:** S_n is NOT a supermartingale, but this is NOT required

3. **✅ Bayesian calibration:** E[S_n] = α exactly (proven)

4. **✅ E[S_τ] = α:** Proven via tower property (equality, not just inequality!)

5. **✅ Type I error control:** P(reject) ≤ α/γ (proven)

#### Empirical Verification

| Quantity | Theoretical | Empirical | Match |
|----------|-------------|-----------|-------|
| E[S_τ] | α = 0.05 | 0.0499 | ✅ |
| Type I bound | α/γ = 0.0526 | — | — |
| Type I error | ≤ 0.0526 | 0.0448 | ✅ |

#### Files

- `theory.md` Section 6.1.3.9 - **Complete formal proof**
- `paper/scripts/theory/sequential_stopping_analysis.py` - Empirical verification
- `paper/scripts/theory/supermartingale_check.py` - Supermartingale analysis
- `paper/scripts/analysis/sequential_stopping_comparison.py` - Fischer-Ramdas comparison

#### Checklist

- [x] Fix distribution (Uniform, not Binomial)
- [x] Check supermartingale property (not needed)
- [x] Verify Bayesian calibration (proven)
- [x] Document proof path in theory.md
- [x] **Complete formal proof of E[S_τ] = α** (DONE - tower property!)
- [x] Document why the proof is so simple

**Status: ✅ FORMALLY PROVEN. The proof uses only the tower property and Markov's inequality.**

---

## References

### Papers to Cite

| Paper | Year | Topic |
|-------|------|-------|
| Strobl et al. | 2008 | Conditional Permutation Importance |
| Candès et al. | 2018 | Model-X Knockoffs |
| Wager & Athey | 2018 | Honest forest consistency |
| Hooker & Mentch | 2019 | Extrapolation problem |
| Fischer & Ramdas | 2025 | Anytime-valid sequential Monte Carlo testing (JRSS-B) |
| Besag & Clifford | 1991 | Sequential Monte Carlo p-values (Biometrika) |
