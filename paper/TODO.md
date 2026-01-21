# Paper TODO — Ray Experiments (Open Issues)

This document captures remaining open issues in the Ray experiments pipeline and analysis stack.

---

## 1) Code Correctness Issues

### 1.5 Rankings artifacts store large embedding predictions (potential bloat)
**Status:** ASSESSED - NOT AN ISSUE

**Where:**
- `paper/scripts/experiments/ray_feature_selection.py` → `embedding_selector(...)`

**Assessment:**
The `embedding_selector()` function only returns `ranking` (feature indices sorted by importance).
It does NOT store predictions. The X_test/y_test parameters are unused in the function.
The rankings parquet only contains `feature_ranking: list[int]`, which is minimal.

---

## 2) Missing Analysis / Tables / Figures

### 2.1 No S3 → local aggregation script
**Status:** DONE

**Solution:**
Created `paper/scripts/analysis/download_and_aggregate.py` that:
1. Lists S3 keys under `s3://{S3_BUCKET}/rankings/` and `s3://{S3_BUCKET}/metrics/`
2. Downloads and concatenates to canonical files:
   - `clf_rankings.parquet` / `reg_rankings.parquet`
   - `clf_evaluation.parquet` / `reg_evaluation.parquet`

Usage:
```bash
S3_BUCKET=my-bucket uv run python paper/scripts/analysis/download_and_aggregate.py --task-type classification
S3_BUCKET=my-bucket uv run python paper/scripts/analysis/download_and_aggregate.py --dry-run  # preview
```

---

### 2.2 No formal aggregation policy for folds/seeds/k/models
**Status:** DONE

**Solution:**
Documented aggregation policy in `paper/docs/README.md` (Aggregation Policy section):

| Dimension | Policy |
|-----------|--------|
| Folds | Mean across folds |
| Seeds | Mean across seeds |
| k values | Fixed k=10 (or best-k if specified) |
| Downstream models | Report per-model or average |

Final metric per (dataset, method) = mean over folds → mean over seeds → fixed k.
The `load_and_pivot_results()` function in `stats.py` implements this aggregation.

---

### 2.3 Missing runtime / cost tables
**Status:** DONE

**Solution:**
Added runtime analysis functions to `paper/scripts/analysis/stats.py`:
- `generate_runtime_summary()` - generates CSV and LaTeX tables with mean, std, median, min, max
- `plot_runtime_bars()` - bar chart with error bars
- `plot_runtime_violin()` - box plot with method type coloring
- `analyze_runtime()` - entry point called from main()

Output files: `{clf,reg}_runtime_summary.csv`, `{clf,reg}_runtime_bars.png`, `{clf,reg}_runtime_violin.png`

---

### 2.8 Method count for Kendall's W may be wrong if some methods missing
**Status:** FIXED

**Where:**
- `paper/scripts/analysis/stats.py` → `run_statistical_analysis(...)`

**Assessment:**
The `friedman_test()` function (line 499-515) already returns the correct count of available
method columns via `len(method_cols)`. The `run_statistical_analysis()` function (line 1265)
correctly uses this value for Kendall's W calculation.

---

## 3) Optional Enhancements

### 3.1 Reliability / resume semantics
**Status:** DONE

**Solution:**
Added `--skip-existing` flag to `_driver.py` common parser. Both `ray_feature_selection.py` and
`ray_eval.py` now support per-task S3 existence checks for extra safety against race conditions.

Usage:
```bash
# Bulk filtering via S3 listing (existing)
uv run python ray_feature_selection.py --only-missing

# Per-task check for extra safety (new)
uv run python ray_feature_selection.py --skip-existing

# Both together for maximum reliability
uv run python ray_feature_selection.py --only-missing --skip-existing
```

### 3.2 Track dataset metadata in evaluation
**Status:** DONE

**Solution:**
Added `get_dataset_metadata()` helper to `_common.py` that extracts metadata from parquet schema:
- `dataset_source`: "real" or "synthetic"
- `dataset_type`: "standard", "bias", "nonlinear", "correlated", "redundant", "toeplitz", "weak_signal", "real"
- `dataset_family`: "standard", "high_cardinality", "friedman", "toeplitz", "confounded", "weak", "openml", "uci", "other"
- `n_informative`: number of informative features (synthetic only)

Both Stage 1 (rankings) and Stage 2 (metrics) now include these columns in output parquet files.

---

## Notes on Environment

- `S3_BUCKET` must be set on head + workers (Ray EC2 templates already do this).
- `GIT_SHA` should be set so artifacts aren't labeled `unknown`.
