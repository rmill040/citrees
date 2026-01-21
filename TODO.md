# citrees: Open Issues

> **Purpose**: Track remaining issues before publication.

---

## Executive Summary

| Priority | Issue                                 | Location              | Type         |
| -------- | ------------------------------------- | --------------------- | ------------ |
| ✅ DONE  | Early stopping documented (heuristic) | `paper/docs/paper.md` | Theory       |
| ✅ DONE  | Broad exception handling              | `_selector.py`        | Code Quality |

---

## Review Findings Backlog (Jan 2026)

### Critical

- [x] Fix label handling: preserve non-numeric/non-0..k-1 labels; classes\_
      should store original labels (`citrees/_tree.py`, `citrees/_forest.py`)
- [x] Fix ptest\_\* experiment crash (keyword-only selector tests called
      positionally) (`paper/scripts/experiments/ray_feature_selection.py`)
- [x] Implement and document `apply`, `decision_path`, `oob_score_`
      (sklearn-style API) (`citrees/_tree.py`, `citrees/_forest.py`,
      `docs/api.md`)
- [x] Fix RNG reinitialization per node (feature subsampling + random
      thresholds) (`citrees/_tree.py`, `citrees/_threshold_method.py`)

### Major

- [x] Feature name order validation should detect reordered columns (not
      set-based) (`citrees/_tree.py`)
- [x] Honesty split remains unstratified (independence assumption); docs clarify
      fraction semantics (`citrees/_tree.py`, `docs/algorithm.md`,
      `docs/parameters.md`)
- [x] `_scan_thresholds` ordering should align with weighted impurity criterion
      (`citrees/_tree.py`)
- [x] Forest predict_proba should align missing classes across trees
      (`citrees/_forest.py`)

### Minor / Experiments / Quality

- [x] Regression selector ranking should use abs(Pearson) in experiments
      (`paper/scripts/experiments/ray_feature_selection.py`)
- [x] Fix experiment leakage: pi/cpi should not use evaluation fold
      (`paper/scripts/experiments/ray_feature_selection.py`)
- [x] Replace broad exception handling with explicit guards + warnings
      (`citrees/_selector.py`)
- [x] Add test: column reorder should raise (`citrees/_tree.py`)
- [x] Add experiment test coverage for ptest\_\* keyword-only calls
      (`paper/scripts/experiments/ray_feature_selection.py`)
- [x] Add tests for non-numeric/non-0..k-1 labels (`tests/unit/test_tree.py`,
      `tests/unit/test_forest.py`)

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
- [x] Add logging when fallback occurs (warnings in `mc`; Numba paths cannot
      log)
- [x] Audit codebase for other broad exception handlers

---

## ✅ COMPLETED

### Early stopping documentation (paper stance)

Early stopping is treated as a computational heuristic. Paper-facing p-value
claims are made only in fixed-$B$ mode. See `paper/docs/paper.md` (Appendix
A.5).

**If we include early stopping in the paper at all:** support it via calibration
figures (not optional-stopping theorems) using
`paper/scripts/theory/generate_sequential_stopping_calibration.py`.

---

## References

### Papers to Cite

| Paper            | Year | Topic                                                 |
| ---------------- | ---- | ----------------------------------------------------- |
| Strobl et al.    | 2008 | Conditional Permutation Importance                    |
| Candès et al.    | 2018 | Model-X Knockoffs                                     |
| Wager & Athey    | 2018 | Honest forest consistency                             |
| Hooker & Mentch  | 2019 | Extrapolation problem                                 |
| Fischer & Ramdas | 2025 | Anytime-valid sequential Monte Carlo testing (JRSS-B) |
| Besag & Clifford | 1991 | Sequential Monte Carlo p-values (Biometrika)          |
