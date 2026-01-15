# citrees: Open Issues

> **Purpose**: Track remaining issues before publication.

---

## Executive Summary

| Priority | Issue | Location | Type |
|----------|-------|----------|------|
| 🟡 MEDIUM | Global error control with feature muting undefined | `theory.md` | Theory |
| 🟡 MEDIUM | Synthetic experiments too easy | `paper/scripts/` | Benchmarking |
| 🟢 LOW | Broad exception handling | `_selector.py` | Code Quality |
| 🟢 LOW | Forest-level theory absent | `theory.md` | Theory |

---

## 🟡 MEDIUM PRIORITY

### 1. Global Error Control with Feature Muting Undefined

**Location**: `theory.md:715-717`, `citrees/_tree.py:669-709`

**Problem**: Feature muting uses intermediate p-values to remove features globally from future consideration. This adaptively changes the hypothesis family across the tree.

**No global FWER bounds are provided when feature muting is enabled.**

#### Resolution Checklist

- [ ] Add section to `theory.md` explaining muting's effect on global control
- [ ] Document that `feature_muting=False` is recommended for inference claims

---

### 2. Synthetic Experiments Too Easy

**Location**: `paper/scripts/`

**Problem**: The synthetic data generation creates well-separated informative features. Missing challenging scenarios:
- High correlation between informative and noise features (confounders)
- Redundant informative features (multicollinearity)
- Toeplitz correlation structures (common in genomics)
- Weak signals mixed with strong noise

#### Resolution Checklist

- [ ] Add correlated noise scenario
- [ ] Add Toeplitz correlation structure
- [ ] Add redundant informative features scenario
- [ ] Add weak signal scenario
- [ ] Re-run experiments with challenging scenarios

---

## 🟢 LOW PRIORITY

### 3. Broad Exception Handling

**Location**: `citrees/_selector.py:257-260`, `citrees/_selector.py:395-398`

**Problem**: The code catches all exceptions and silently returns 0.0:

```python
try:
    return np.sqrt(ssb / sst)
except Exception:  # Too broad!
    return 0.0     # Silent failure
```

#### Resolution Checklist

- [ ] Change to specific exception types: `except (ZeroDivisionError, FloatingPointError):`
- [ ] Add logging when fallback occurs
- [ ] Audit codebase for other broad exception handlers

---

### 4. Forest-Level Theory Absent

**Location**: `theory.md:735-737`

**Problem**: Theory.md explicitly states forest-level guarantees are missing:

> "Consistency, rates, and uncertainty quantification for the forest require assumptions and arguments beyond what can be asserted from the current implementation alone."

**Missing**:
- OOB estimation validity analysis
- Consistency properties as n→∞
- Confidence intervals on forest predictions

#### Resolution Checklist

- [ ] Add section on OOB estimation for permutation forests
- [ ] Reference Wager & Athey (2018) for honest forest consistency
- [ ] Consider implementing forest-level confidence intervals

---

## References

### Papers to Cite

| Paper | Year | Topic |
|-------|------|-------|
| Strobl et al. | 2008 | Conditional Permutation Importance |
| Candès et al. | 2018 | Model-X Knockoffs |
| Wager & Athey | 2018 | Honest forest consistency |
| Hooker & Mentch | 2019 | Extrapolation problem |
