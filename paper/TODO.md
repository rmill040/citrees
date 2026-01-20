# Paper TODO — Ray Experiments (Open Issues)

This document captures remaining open issues in the Ray experiments pipeline and analysis stack.

---

## 1) Code Correctness Issues

### 1.5 Rankings artifacts store large embedding predictions (potential bloat)
**Status:** NOT ASSESSED

**Where:**
- `paper/scripts/experiments/ray_feature_selection.py` → `embedding_selector(...)`

**Problem:**
Embedding predictions can be large and are stored in rankings parquet but not used in analysis.

**Remedy:**
- Remove embedding predictions from rankings unless explicitly needed.
- Or gate with a flag (e.g., `--store-embeddings` default False).

---

## 2) Missing Analysis / Tables / Figures

### 2.1 No S3 → local aggregation script
**Status:** MISSING

**Problem:**
Ray Stage 1/2 outputs are written as many per-dataset/per-seed parquet files on S3. Analysis scripts expect aggregated parquet files under `paper/results/`. No official script exists to download and assemble them.

**Remedy:**
Create `paper/scripts/analysis/download_and_aggregate.py` that:
1. Lists S3 keys under `s3://{S3_BUCKET}/rankings/` and `s3://{S3_BUCKET}/metrics/`
2. Downloads and concatenates to canonical files (`clf_evaluation.parquet`, `reg_evaluation.parquet`)

---

### 2.2 No formal aggregation policy for folds/seeds/k/models
**Status:** MISSING

**Problem:**
Stage 2 generates results per fold × k × downstream_model. The reduction policy to a single score per dataset per method is not specified.

**Remedy:**
Define and document a clear policy:
1. **Folds:** mean across folds
2. **Seeds:** mean across seeds
3. **k values:** fixed k (e.g., 10) or best-k per method
4. **Downstream models:** average or report per-model

---

### 2.3 Missing runtime / cost tables
**Status:** MISSING

**Problem:**
Both Stage 1 and Stage 2 record `elapsed_seconds` but no standard table/figure uses them.

**Remedy:**
- Add a table of median/mean runtime by method and dataset
- Add a runtime plot (box/violin or bar with error bars)

---

### 2.8 Method count for Kendall's W may be wrong if some methods missing
**Status:** NOT FIXED

**Where:**
- `paper/scripts/analysis/stats.py` → `run_statistical_analysis(...)`

**Problem:**
Kendall's W uses `k_methods = len(methods)` even if some method columns are absent for a metric.

**Remedy:**
Compute `k_methods` from the actually-available columns for each metric.

---

## 3) Optional Enhancements

### 3.1 Reliability / resume semantics
**Status:** PARTIAL

Stage 1/2 overwrite outputs in S3. Add a `--skip-existing` flag to avoid recomputing finished configs.

### 3.2 Track dataset metadata in evaluation
**Status:** PARTIAL

Stage 2 results include `n_samples`/`n_features` but not dataset provenance (source=real/synthetic, dataset family, etc.).

---

## Notes on Environment

- `S3_BUCKET` must be set on head + workers (Ray EC2 templates already do this).
- `GIT_SHA` should be set so artifacts aren't labeled `unknown`.
