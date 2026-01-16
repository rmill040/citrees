# Ray Experiments Pipeline Refactor (Rankings → Metrics)

This document describes, in detail, the current Ray pipeline, the issues found during review, and a concrete refactor
target with measurable success criteria.

Primary scripts:
- Stage 1 (feature selection → rankings): `paper/scripts/experiments/ray_feature_selection.py`
- Stage 2 (downstream evaluation → metrics): `paper/scripts/experiments/ray_eval.py`
- Progress reporting (S3 listing): `paper/scripts/experiments/check_progress.py`

---

## 1) Current pipeline: what it does today

### 1.1 Stage 1: feature selection → rankings

For each tuple:
`(task_type, dataset, method_config, seed)`:

1. Load dataset from parquet (`paper/data/*.parquet`).
2. CV split construction:
   - Classification: `StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)`
   - Regression: `KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)`
3. For each fold (currently `N_SPLITS`):
   - Slice raw data first: `X_train_raw = X[train_idx]`, `X_test_raw = X[test_idx]`
   - Fit preprocessing on train only (no leakage):
     - `scaler.fit(X_train_raw)`
     - `X_train = scaler.transform(X_train_raw)`
     - `X_test = scaler.transform(X_test_raw)`
   - Compute a feature ranking for the fold using the selected method.
   - Save:
     - `fold_idx`
     - `feature_ranking`
     - optionally `embedding_*` predictions/probabilities for embedding methods
4. Upload a parquet file to:
   - `s3://{S3_BUCKET}/rankings/{task_type}/{dataset}/{method_id}_seed{seed}.parquet`

Resume semantics:
- Stage 1 overwrites existing S3 objects (no skip-by-existence), so reruns recompute and rewrite artifacts.

### 1.2 Stage 2: rankings → downstream evaluation metrics

For each tuple:
`(task_type, dataset, method_config, seed)`:

1. Read rankings parquet from:
   - `s3://{S3_BUCKET}/rankings/{task_type}/{dataset}/{method_id}_seed{seed}.parquet`
2. Load dataset from local parquet.
3. For each fold row in rankings parquet:
   - Use `fold_idx` and reconstruct `(train_idx, test_idx)` deterministically using the same CV spec as Stage 1.
   - For each `k` in `[5, 10, 25, 50, 100, n_features]` (bounded by `n_features`):
     - Take top-k features from `feature_ranking`.
     - Fit `StandardScaler` on `X_train[:, top_k]` and transform train/test.
     - Fit downstream models and compute metrics.
4. Upload metrics parquet to:
   - `s3://{S3_BUCKET}/metrics/{task_type}/{dataset}/{method_id}_seed{seed}.parquet`

Resume semantics:
- Stage 2 overwrites existing S3 objects (no skip-by-existence), so reruns recompute and rewrite artifacts.
- If the rankings object does not exist, returns `no_rankings`.

### 1.3 Progress checking

`check_progress.py` lists `s3://{S3_BUCKET}/{stage}/{task_type}/...` keys and counts `(method, seed)` completions per
dataset.

Status: fixed — it now uses `load_config().experiment.n_seeds`, matching Stage 1/2.

---

## 2) Issues & why they matter

This section is intentionally explicit about **the failure mode**, **the impact**, and **how to verify the issue exists**.

### 2.1 CV leakage in Stage 1 (scaling happens before splitting)

**Status**
- Implemented fix: Stage 1 fits `StandardScaler` per fold using `X_train` only.

**Previously (failure mode)**
- The scaler used information from test folds (mean/variance) during training folds.

**Impact (why it mattered)**
- Rankings can be optimistically biased and irreproducible relative to a “proper” per-fold preprocessing pipeline.
- This undermines comparisons between methods, especially scale-sensitive methods (distance-based, RDC, etc.).

**How to verify (still recommended)**
- Create a dataset where feature distributions differ strongly between folds (e.g., add a fold-specific shift),
  then compare rankings between:
  - global scaling before CV (previous), and
  - per-fold scaling fit on training only (correct).

**Success criteria**
- Stage 1 never fits preprocessors on samples outside `train_idx` for a given fold.
- A regression test demonstrates that the scaler is fit only on train data and Stage 1 results are stable given a seed.

---

### 2.2 Rankings artifacts are unnecessarily large (train/test indices stored per fold)

**Status**
- Implemented fix: rankings artifacts no longer store per-sample train/test indices.

**Previously (failure mode)**
- For large `n_samples`, these dominate artifact size and S3 transfer time.

**Impact (why it mattered)**
- Higher cost (S3 storage + bandwidth).
- Slower Stage 2 (download/parquet read overhead).
- Higher memory overhead in pandas.

**How to verify (still recommended)**
- Compare parquet file sizes with/without index arrays on a representative dataset.

**Success criteria**
- Rankings parquet stores only `fold_idx` and `feature_ranking` (plus optional method-specific metadata).
- Stage 2 reconstructs the fold splits deterministically from `(task_type, seed, N_SPLITS)` and the dataset.
- Artifact size is reduced substantially (expected: O(n_splits * n_samples) → O(n_splits * n_features)).

---

### 2.3 CPU oversubscription + runaway parallelism inside Ray tasks

**Status**
- Implemented fix (Stage 1): tasks now request Ray `num_cpus` per method and threaded estimators use `n_jobs`/`thread_count`
  consistent with that allocation.

**Previously (failure mode)**
- Ray may schedule many tasks concurrently per node because the custom resource is not tied to real CPU capacity.
- Each task attempts to use all CPU cores because `n_jobs=-1`.

**Impact**
- Severe thrashing, lower throughput, high variance in runtime, and potential OOM.

**How to verify**
- On a single worker node, observe CPU utilization and process/thread counts during load.
- Runtime should improve (or at least become more stable) when:
  - task concurrency is bounded correctly, and
  - internal threading is limited.

**Success criteria**
- Task resource declarations reflect real compute usage:
  - use `num_cpus` (and optionally `num_gpus`) in `@ray.remote(...)`, AND/OR
  - ensure custom resources represent “slots” with realistic counts (e.g., 1 per node) and rely on CPU resources.
- Internal parallelism is controlled:
  - either set model `n_jobs=1`, or set `n_jobs` based on allocated CPUs,
  - set thread env vars (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc.) if needed.
- Throughput is stable and scales roughly with total available CPUs.

---

### 2.4 Downstream metrics can hard-fail due to ROC AUC edge cases

**Status**
- Implemented fix: ROC AUC now uses a safe wrapper and returns `NaN` when undefined (e.g., single-class folds).

**Previously (failure mode)**
- One bad fold can mark the entire `(method_id, dataset, seed)` config as `failed`.

**Impact**
- Lost results, wasted compute; noisy “failed” rate.

**How to verify**
- Use an imbalanced dataset and small folds; some folds may lack the minority class.

**Success criteria**
- Metrics collection never fails the whole config for a known-degenerate metric.
- Record `roc_auc = NaN` (and optionally a “metric_warning” field) when undefined.
- The config still writes metrics and reports `done` unless a truly fatal error occurs.

---

### 2.5 Results schemas are missing key metadata (harder to aggregate and debug)

**Status**
- Implemented fix (metrics): output metric rows now include `dataset`, `task_type`, `seed`, and `method_id`.

**Failure mode**
- Aggregation needs to parse S3 keys or carry external context.

**Impact**
- Harder to join, filter, and debug results reliably.

**Success criteria**
- Every row in metrics parquet is “self-describing”, including:
  - identifiers: `dataset`, `task_type`, `seed`, `method_id`
  - fold context: `fold_idx`, `k`, `downstream_model`
  - run metadata: `n_samples`, `n_features`, `elapsed_seconds`, `git_sha` (optional), `artifact_version`

---

### 2.6 Duplication and lack of shared utilities (harder to maintain)

**Current behavior**
- Stage 1 and Stage 2 duplicate:
  - `DATA_DIR` logic
  - `get_datasets()`, `load_dataset()`
  - S3 helpers `s3_file_exists()`, `upload_to_s3()`

**Impact**
- Behavior diverges over time (region handling, error handling, schema changes).

**Success criteria**
- Shared experiment helpers live in one module under `paper/scripts/experiments/` (or `paper/scripts/utils/`):
  - dataset loading
  - S3 IO helpers with consistent error handling
  - artifact path building
  - common metadata stamping

---

## 3) Refactor goals / non-goals

### 3.1 Goals

1. Correctness:
   - Eliminate Stage 1 preprocessing leakage.
2. Practicality:
   - Make artifacts smaller, simpler, and self-describing.
3. Reliability:
   - Make metrics collection robust to expected edge cases (e.g., undefined AUC).
4. Scalability:
   - Prevent CPU oversubscription; align Ray scheduling with real resources.
5. Maintainability:
   - Reduce duplication; clarify stage responsibilities; make code easier to follow.

### 3.2 Non-goals (explicitly out of scope)

- Changing the scientific definition of the evaluation metrics (unless required for robustness).
- Replacing pandas/parquet with an entirely different data plane (Ray Datasets, Delta Lake, etc.).
- Full experiment orchestration framework (Airflow, Prefect) unless desired later.

---

## 4) Target “v2” pipeline design

### 4.1 Artifact conventions (paths + versions)

Keep existing locations, but add an explicit version marker either:
- In the path (recommended for safe migration):
  - `rankings_v2/{...}`
  - `metrics_v2/{...}`
or
- In the parquet schema only:
  - add `artifact_version = 2` column and keep same paths (riskier: overwrites old).

**Recommendation**
- Use `*_v2` prefixes first for easy A/B comparisons and rollback.

### 4.2 Rankings (Stage 1) schema: minimal + deterministic

Required columns per row:
- `fold_idx: int`
- `feature_ranking: list[int]`

Optional columns (method-dependent):
- `embedding_train_preds`, `embedding_test_preds`, `embedding_train_proba`, `embedding_test_proba`
- `selection_elapsed_seconds` (fold-level or config-level)

Config-level metadata to add (either repeated in every row or stored in a small companion “manifest”):
- `dataset: str`
- `task_type: str`
- `seed: int`
- `method_id: str`
- `method: str`
- `params: dict` (if reasonable) or `params_json: str`
- `n_samples: int`
- `n_features: int`
- `artifact_version: int`
- `created_at_utc: str` (ISO8601)
- `git_sha: str` (optional)

### 4.3 Splits: reconstruct instead of store indices

Stage 1 and Stage 2 must share split logic:
- `cv = StratifiedKFold(..., random_state=seed)` for classification
- `cv = KFold(..., random_state=seed)` for regression
- Fold order must be stable.

Stage 2 reconstructs `(train_idx, test_idx)` using the same CV and uses `fold_idx` to select the correct split.

### 4.4 Per-fold preprocessing: eliminate leakage, keep consistency

Recommended per fold:
1. Slice raw data first: `X_train_raw = X[train_idx]`, `X_test_raw = X[test_idx]`.
2. Fit scaler on train only:
   - `scaler.fit(X_train_raw)`
   - `X_train = scaler.transform(X_train_raw)`
   - `X_test = scaler.transform(X_test_raw)` (needed for embedding predictions)
3. Run selection method using `X_train` (and possibly `X_test` for embedding outputs).

This preserves the current intent (“everything is standardized”) while fixing leakage.

### 4.5 Metrics (Stage 2) schema: self-describing

One row per `(fold_idx, k, downstream_model)`:
- identifiers: `dataset`, `task_type`, `seed`, `method_id`
- context: `fold_idx`, `k`, `downstream_model`
- `n_features_total`, `n_features_selected`
- metrics:
  - classification: `accuracy`, `f1_weighted`, `roc_auc_ovr_weighted` (or consistent naming)
  - regression: `r2`, `rmse`, `mae`
- runtime:
  - `eval_elapsed_seconds` (config-level repeated or fold-level)
- metadata: `artifact_version`, `created_at_utc`, `git_sha` (optional)

Robustness rule:
- If a metric is undefined (e.g., ROC AUC with single-class `y_test`), write `NaN` and proceed.

---

## 5) Concurrency + resource model (Ray)

### 5.1 Principle: “One parallelism source”

Either:
- Let Ray manage parallelism → set estimator `n_jobs=1`, and use `num_cpus` to scale, OR
- Let estimators multithread → set `num_cpus` accordingly and restrict task concurrency.

Mixing `n_jobs=-1` with unbounded task concurrency is the failure mode.

### 5.2 Recommended approach (simple + stable)

1. Use custom resources (`selection`, `evaluation`) for **routing**, not for concurrency control:
   - `@ray.remote(resources={"selection": 1})`
   - `@ray.remote(resources={"evaluation": 1})`
2. Declare **real CPU needs** per task using Ray `num_cpus` (recommended: per-config, per-method):
   - `process_config.options(num_cpus=selection_cpus).remote(...)`
3. Align internal threading with Ray scheduling:
   - For thread-parallel libraries: set `n_jobs` / `thread_count` to `selection_cpus`.
   - For single-threaded methods: keep `selection_cpus=1`.

With this pattern, Ray naturally behaves like a worker-queue system: tasks are submitted, and workers “pull” them when
they have the declared resources. On a 32-vCPU node, if you request:
- `selection_cpus_default=1`: up to 32 lightweight tasks can pack per node.
- `selection_cpus_threaded=8`: up to 4 threaded tasks can pack per node.
- `selection_cpus_cif_large=32`: at most 1 large CIF task runs per node.

The CPU policy is configured in `paper/scripts/infra/config.yaml` and is deterministic:

1. If `selection_cpus_overrides` has an entry for the base method, use it.
2. Else if method is `cif`, use the dataset-size rule (`n_samples * n_features` threshold).
3. Else if method is in the “threaded” bucket, use `selection_cpus_threaded`.
4. Else use `selection_cpus_default`.

### 5.3 Success criteria

- CPU usage is high but stable under load (no massive context switching).
- Runtime variance shrinks (p50 ≈ p90 in the same order of magnitude).
- Total time scales with total CPUs when adding workers.

---

## 6) Reliability + observability improvements

### 6.1 S3 existence checks: avoid swallowing real errors

Status: fixed — `s3_file_exists()` now returns `False` only for “not found”, and other errors fail the task.

Previously, catching all exceptions and returning `False` could hide:
- permission errors (403)
- network issues
- region misconfiguration

Refactor behavior:
- Return `False` only for “not found” (404 / NoSuchKey).
- Re-raise or log-and-fail for other errors (so the pipeline does not silently recompute forever).

### 6.2 Write-time validation

Optional but recommended:
- After writing parquet to S3, read parquet metadata (or at least verify size > 0) to reduce the chance of “corrupt but
present” artifacts causing permanent skips.

### 6.3 Add structured status outputs

Instead of only `{"status": "...", ...}`, consider including:
- `error_type`, `traceback` (truncated), `hostname`, `pid`, `ray_node_id`
- `elapsed_seconds`

This helps debugging.

---

## 7) Detailed implementation plan (with acceptance criteria)

This plan is written to be executed in small PR-sized steps, each with a clear “done” definition.

### Step A — Shared utilities module

Create `paper/scripts/experiments/_common.py` (name flexible) containing:
- `get_datasets(task_type) -> list[str]`
- `load_dataset(name, task_type) -> (X, y)`
- `artifact_paths(...) -> rankings_path, metrics_path`
- `s3_file_exists(s3_path) -> bool` with correct error handling
- `upload_parquet_to_s3(rows_or_df, s3_path) -> None`
- `download_parquet_from_s3(s3_path) -> pd.DataFrame`

Acceptance criteria:
- Stage 1 and Stage 2 import shared functions.
- Behavior matches current pipeline (before logic changes), but code duplication is reduced.

### Step B — Fix Stage 1 leakage (per-fold preprocessing)

Change Stage 1’s `run_selection()` to:
- compute CV splits on raw `X` (not globally scaled),
- fit scaler on `X_train` only per fold,
- pass scaled per-fold data into selection methods.

Status:
- Implemented in `paper/scripts/experiments/ray_feature_selection.py`.
- A deterministic repro/test is still recommended to guard against regressions.

Acceptance criteria:
- No `StandardScaler().fit_transform(X)` occurs before fold splitting in Stage 1.
- A small deterministic test or repro script demonstrates that global-vs-fold scaling yields different rankings on a
constructed dataset (and “v2” uses fold scaling).

### Step C — Stop storing fold indices in rankings artifacts

Update Stage 1 artifact schema:
- remove `train_indices` and `test_indices`.
Stage 2:
- reconstruct indices using the shared CV logic and `fold_idx`.

Status:
- Implemented in `paper/scripts/experiments/ray_feature_selection.py` and `paper/scripts/experiments/ray_eval.py`.
- Validate artifact size reduction on a representative dataset (recommended).

Acceptance criteria:
- Stage 2 reproduces the same `(train_idx, test_idx)` shapes per fold and runs end-to-end without needing stored indices.
- Rankings artifacts are much smaller (verify on one real dataset).

### Step D — Make AUC robust (never hard-fail on undefined AUC)

Implement a helper:
- `safe_roc_auc_score(...) -> float | np.nan`

Status:
- Implemented in `paper/scripts/experiments/ray_eval.py` (`safe_roc_auc_score`).

Acceptance criteria:
- AUC exceptions are handled and recorded as `NaN`.
- Config completes and writes metrics parquet unless a truly fatal error occurs.

### Step E — Add metadata columns to metrics parquet

Update Stage 2 to stamp identifiers + run metadata into each result row.

Status:
- Implemented in `paper/scripts/experiments/ray_eval.py` (adds `dataset`, `task_type`, `seed`, `method_id` to each row).

Acceptance criteria:
- `dataset`, `task_type`, `seed`, `method_id` exist as columns in the output.
- Downstream aggregation can be done without parsing S3 keys.

### Step F — Fix oversubscription (Ray + estimator threading)

Pick one policy (recommended: “Ray owns placement; estimator threading matches allocation”):
- Request Ray `num_cpus` per task (and per method when needed).
- For threaded estimators, set `n_jobs` / `thread_count` to the same value.
- Keep custom resources (`selection`, `evaluation`) as routing constraints (not concurrency controls).

Status:
- Implemented in `paper/scripts/experiments/ray_feature_selection.py`:
  - `selection_num_cpus(method, n_samples, n_features)` defines per-config CPU needs from `paper/scripts/infra/config.yaml`:
    - `selection_cpus_overrides` (optional): per-method override mapping (base method name → CPU count). This takes
      precedence over all other rules.
    - `selection_cpus_default` (lightweight methods)
    - `selection_cpus_threaded` (rf/et/xgb/lgbm/cat + wrappers)
    - `selection_cpus_cif` vs `selection_cpus_cif_large` chosen by `n_samples * n_features >= selection_cif_large_threshold`
  - Dataset shapes are read once on the driver from parquet metadata (fallback: load dataset) to make the CIF size rule
    deterministic and cheap.
  - Stage 1 submits tasks with `process_config.options(num_cpus=...)`.
  - Threaded methods pass `n_jobs` / `thread_count` through to their underlying estimators.

Acceptance criteria:
- No estimator uses `n_jobs=-1` inside Ray tasks.
- Ray dashboard shows tasks scheduled according to CPU resources.

### Step G — Update progress checker to match config

Update `check_progress.py` to:
- read `n_seeds` from `paper/scripts/infra/config.yaml` via `load_config()`, OR
- accept `--n-seeds` CLI arg.

Status:
- Implemented in `paper/scripts/experiments/check_progress.py` (reads `n_seeds` via `load_config()`).

Acceptance criteria:
- Progress output matches actual configured seeds for a run.

---

## 8) “Success looks like” checklist (final)

### Correctness
- [x] No leakage: Stage 1 preprocessing is fit only on training folds.
- [x] Splits are deterministic and consistent between Stage 1 and Stage 2.

### Artifacts
- [x] Rankings artifacts do not store per-sample indices.
- [x] Metrics artifacts contain identifying metadata columns.
- [ ] Artifact versioning enables side-by-side comparison of old vs new outputs.

### Robustness
- [x] Undefined metrics (AUC edge cases) do not fail entire configs.
- [x] S3 errors are not silently swallowed (only “not found” returns False).

### Scalability
- [x] Ray task scheduling is aligned with CPU resources (`num_cpus` + sane `n_jobs`).
- [ ] Throughput improves predictably with more worker CPUs.

### Maintainability
- [ ] Shared utilities remove duplicated code between Stage 1 and Stage 2.
- [ ] The pipeline is easier to read (clear stage responsibilities and consistent naming).

---

## 9) Migration strategy (recommended)

This repo is currently configured to **overwrite** canonical `rankings/` and `metrics/` outputs on rerun (no skip-by-
existence).

Recommended workflow when overwriting:

1. Run one dataset × few methods × few seeds as a smoke test.
2. Compare:
   - artifact sizes,
   - runtime stability,
   - summary metrics distributions (expect some changes due to leakage fix).
3. When confident, scale up to the full experiment grid.
