# Analysis Lockdown Plan (Paper)

Last updated: 2026-02-10

This is the canonical internal execution document for:

1. Locking the statistical analysis contract before final headline claims.
2. Running theory-backed simulations independent of full benchmark completion.
3. Converting near-complete benchmark artifacts into paper-grade tables/figures.

This is not manuscript text. It is an operational plan and QA contract.

---

## 1) Scope and Immediate Priorities

### 1.1 Goals

1. Prevent post-hoc drift in estimands, tests, and multiplicity handling.
2. Ensure fair method comparisons when multiple hyperparameter configurations
   exist.
3. Generate reviewer-resilient evidence for:
   - inferential calibration scope,
   - ranking quality,
   - runtime/robustness tradeoffs.
4. Keep mathematical/theory writing unblocked while benchmark jobs finish.

### 1.2 Blocker Status

- Not a blocker for theory/math sections and formal claims drafting.
- Blocker for final empirical claims and final Experiments/Results wording.

### 1.3 Current Risks to Resolve

1. Confirmatory endpoint is specified in manuscript but not hard-enforced in
   analysis scripts.
2. `method` in artifacts is config-hash level (`method_id`), while headline
   comparisons should likely be family level (`method_base`) unless explicitly
   doing config-level analysis.
3. Default aggregation can mix folds, seeds, k values, and downstream models if
   data are not pre-filtered.
4. Attrition (`failed`, `skipped`, `no_rankings`) can bias results if not
   explicitly reported.

---

## 2) Confirmatory Analysis Contract (Must Lock Before Final Claims)

### 2.1 Real-Data Primary Endpoints

Classification:

- Dataset source: real only.
- Downstream model: `lr`.
- Metric: `balanced_accuracy`.
- Feature budget: `k=25`.

Regression:

- Dataset source: real only.
- Downstream model: `ridge`.
- Metric: `r2`.
- Feature budget: `k=25`.

Everything else is exploratory/robustness analysis.

### 2.2 Unit of Analysis and Aggregation

1. Compute per-dataset, per-method endpoint by averaging across folds and seeds.
2. Perform cross-dataset inference on those dataset-level paired summaries.
3. Do not pool heterogeneous endpoints in confirmatory analysis.

Definition:

- Let `S_{d,m}` be the primary endpoint score for dataset `d`, method `m`, after
  averaging across all available folds and seeds for that dataset-method pair.

### 2.3 Method Identity Policy

1. Confirmatory tier compares method families using `method_base`.
2. Config-hash-level (`method` / `method_id`) analyses are exploratory only.
3. If a family has multiple configs, pre-register one of:
   - fixed default config per family,
   - pre-declared config-selection rule using only training-side criteria,
   - report all configs as exploratory without family-level superiority claims.

### 2.4 Hypotheses and Tests

For each task-specific primary endpoint:

1. Omnibus null:
   - `H0`: all compared methods have equal performance distribution across
     datasets.
   - Primary omnibus: Friedman test when assumptions and method count permit.
2. Pairwise nulls:
   - `H0_{m1,m2}`: paired dataset-level differences center at zero.
   - Pairwise test: paired Wilcoxon signed-rank when sample size is adequate.
3. Small-sample fallback:
   - For low dataset count regimes (notably regression real datasets), use exact
     sign/permutation paired tests where Wilcoxon is underpowered/unstable.

### 2.5 Multiplicity Control

Define inferential families before final run:

1. Family A (classification primary endpoint pairwise tests).
2. Family B (regression primary endpoint pairwise tests).

Within each family:

- Use Holm adjustment for pairwise p-values.
- Report adjusted p-values and significance at `alpha=0.05`.

Exploratory analyses:

- Either adjust separately within each exploratory family or clearly label as
  descriptive/exploratory without confirmatory claims.

### 2.6 Effect Sizes and Uncertainty

For confirmatory tables:

1. Paired median delta per method vs reference (for example, vs `cit` and/or vs
   top baseline).
2. Interquartile range (IQR) of paired deltas.
3. Hodges-Lehmann estimate for paired shift when feasible.
4. Optional paired standardized effect size based on differences (not unpaired
   pooled-variance Cohen's d).
5. Bootstrap CI over datasets for descriptive means/ranks (clearly labeled
   descriptive).

### 2.7 Missingness, Failures, Timeouts

Mandatory reporting table per stage and task:

1. Expected runs by method x dataset x seed.
2. Completed runs.
3. Failed runs.
4. Skipped (already exists).
5. No-rankings / missing prerequisite artifact.

Primary analysis default:

- Use completed runs only, but include attrition table and sensitivity checks.

Sensitivity checks:

1. Best-case vs worst-case imputation bounds for missing runs (if feasible).
2. Re-run with methods restricted to high-completion subsets.
3. Explicitly disclose if attrition is materially imbalanced across methods.

---

## 3) Exploratory and Robustness Analyses

These are allowed and encouraged, but must be labeled non-confirmatory:

1. Per-downstream-model analyses beyond `lr`/`ridge`.
2. Per-k analyses across `k in {5,10,25,50,100,p}`.
3. Additional metrics:
   - classification: `accuracy`, `f1_macro`, `auc/roc_auc`.
   - regression: `mse`, `mae`, `rmse`.
4. Runtime distributions and scaling behavior.
5. Stability analyses (for example Nogueira/Jaccard/Kendall variants) with exact
   definitions.

Claims policy:

- Use robustness language ("consistent with", "similar ordering", "sensitivity
  result").
- Do not replace confirmatory endpoint claims with exploratory results.

---

## 4) Theory-Linked Simulation Program (Can Run Before Full Benchmarks)

These simulations are the fastest way to strengthen the statistical narrative
now.

### 4.1 Simulation S1: Complete Global Null Calibration (Stage A Root/Fixed Node)

Question:

- Do fixed-B Stage A permutation p-values behave as super-uniform under the
  intended complete global null?

Design:

1. Generate datasets where labels are exchangeable with respect to covariates.
2. Evaluate root/fixed-node Stage A screening tests only.
3. Use fixed-B resampling settings aligned with theorem assumptions.

Factors:

- `n` (sample size), `p` (feature count), class balance, number of candidate
  thresholds.

Outputs:

1. ECDF/QQ diagnostics of p-values.
2. Empirical type-I rejection rate at `alpha in {0.01, 0.05, 0.10}`.
3. Root split probability under null.

Acceptance criteria:

- Empirical rejection at or below nominal within Monte Carlo error bands.

### 4.2 Simulation S2: Selection Bias Under High-Cardinality Noise

Question:

- Under complete global null settings containing high-cardinality noise, do
  CART-like greedy mechanisms over-select spurious predictors relative to
  conditional inference variants?

Design:

1. Null data with mixed cardinality noise features.
2. Compare rank/split selection rates for high-cardinality noise features.

Outputs:

1. Noise selection rate by method at top-k.
2. Relative reduction for citrees methods vs CART-style baselines.

Acceptance criteria:

- Clear directional separation consistent with bias-mitigation claims.

### 4.3 Simulation S3: Early-Stopping Calibration Drift

Question:

- How much type-I calibration drift appears when using sequential/adaptive
  stopping vs full fixed-B testing?

Design:

1. Repeat S1 under three modes:
   - full fixed-B,
   - adaptive stopping,
   - simple stopping.

Outputs:

1. Empirical type-I rates by mode.
2. Runtime speedup by mode.
3. Calibration-vs-speed tradeoff plots.

Interpretation rule:

- Early-stopped outputs are algorithmic/statistical surrogates unless
  calibration remains within acceptable bounds and scope is explicitly limited.

### 4.4 Simulation S4: Power Curves Under Controlled Alternatives

Question:

- What is the detection power as signal strength and sample size vary?

Design:

1. Synthetic alternatives with known informative features.
2. Vary signal amplitude, noise scale, and feature correlation structure.

Outputs:

1. Power curves by method and setting.
2. Minimum n for target power thresholds.

### 4.5 Simulation S5: Max-T Multi-Selector Validity

Question:

- Does multi-selector max-T control family-wise error at nominal alpha under
  null combinations?

Design:

1. Evaluate combined selector sets used in practice:
   - classification combinations allowed by scale constraints.
   - regression combinations (`pc`, `dc`, `rdc`).

Outputs:

1. Empirical FWER under null.
2. Power comparison vs single-selector alternatives.

### 4.6 Simulation S6: Stage B Statistic Sensitivity (Optional but Recommended)

Question:

- Are results robust to split-statistic choices that may prefer unbalanced
  splits?

Design:

1. Controlled scenarios with known balanced vs unbalanced signal structure.
2. Compare candidate Stage B statistic variants (if implemented).

Outputs:

1. Split balance diagnostics.
2. Downstream ranking/performance impact.

---

## 5) Benchmark Data Integration Plan (When Nearly Complete Results Arrive)

### 5.1 Data Freeze Procedure

1. Record freeze timestamp and git SHA for scripts used.
2. Record pipeline config snapshot (seeds, methods, grids, timeout policy).
3. Export immutable local parquet snapshots used for analysis.

### 5.2 Ingestion and QA

1. Run aggregation download.
2. Validate required columns and schema versions.
3. Compute expected-vs-observed counts by stage/task/method/dataset/seed.
4. Generate attrition report before any inferential table.

Suggested commands:

```bash
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/download_and_aggregate.py --task all
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/synthetic_analysis.py \
  --results-dir paper/results/rankings/classification \
  --data-dir paper/data/classification/synthetic \
  --output paper/results/synthetic_analysis.parquet
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/stats.py
```

### 5.3 Confirmatory Runbook (Real Data)

1. Filter to real datasets only.
2. Classification confirmatory filter:
   - `downstream_model == "lr"`,
   - `k == 25`,
   - metric `balanced_accuracy`.
3. Regression confirmatory filter:
   - `downstream_model == "ridge"`,
   - `k == 25`,
   - metric `r2`.
4. Aggregate to dataset-level paired summaries.
5. Run omnibus + pairwise + multiplicity.
6. Emit:
   - primary inferential table,
   - paired delta table (median/IQR),
   - rank table for endpoint.

### 5.4 Exploratory Runbook

1. Per-model analyses.
2. Per-k analyses.
3. Secondary metrics.
4. Runtime analyses.
5. Sensitivity analyses under completion filters.

---

## 6) Required Script Changes (Tracking Checklist)

### 6.1 `paper/scripts/analysis/stats.py`

- [ ] Add explicit "confirmatory mode" entry point.
- [ ] Support deterministic filtering by:
  - task,
  - dataset source,
  - downstream model,
  - k,
  - metric.
- [ ] Use `method_base` for confirmatory method identity.
- [ ] Keep config-level (`method`) analysis explicitly exploratory.
- [ ] Add paired-delta summaries (median/IQR and optional Hodges-Lehmann).
- [ ] Add small-sample exact paired test fallback.
- [ ] Guard CD/Nemenyi paths when method count exceeds supported range.
- [ ] Ensure runtime analysis filters use consistent method identity.

### 6.2 `paper/scripts/analysis/download_and_aggregate.py` or companion script

- [ ] Add attrition/completeness audit outputs.
- [ ] Persist expected-vs-observed run counts by stage/task/method/dataset/seed.

### 6.3 `paper/scripts/analysis/synthetic_analysis.py`

- [ ] Align default k-grid with manuscript language.
- [ ] Explicitly output metrics needed by final synthetic tables/figures.
- [ ] Keep dataset-type stratifications reproducible and documented.

### 6.4 `paper/arxiv/sections/05_experiments.tex`

- [ ] Match seed count and protocol details to current pipeline defaults or
      update pipeline defaults to match manuscript.
- [ ] Replace any vague multiplicity wording with exact procedure names.
- [ ] Add explicit failure/timeout handling policy.

---

## 7) Deliverables Map (Artifact -> Manuscript Slot)

### 7.1 Confirmatory Core

1. Primary endpoint comparison table (classification).
2. Primary endpoint comparison table (regression).
3. Pairwise delta summary table vs reference methods.
4. Multiplicity-adjusted pairwise table.

### 7.2 Calibration and Theory Support

1. Stage A complete-global-null calibration figure.
2. Early-stopping calibration-vs-speed figure.
3. Selection-bias high-cardinality-null figure.
4. Max-T validity figure/table (if multi-selector claims appear).

### 7.3 Robustness

1. Per-k robustness plot grid.
2. Per-downstream-model robustness tables.
3. Runtime summary + distribution plots.
4. Stability table (if retained in paper scope).

### 7.4 Transparency

1. Attrition/completion table.
2. Config and data freeze appendix note.
3. Script versioning and reproducibility note.

---

## 8) Claim Discipline Rules (Non-Negotiable)

1. No calibrated p-value claims beyond fixed-node/root Stage A fixed-B scope.
2. No method-superiority headline from exploratory metrics alone.
3. No omission of attrition policy in final empirical writeup.
4. No family-level claims if analysis is actually config-hash-level unless
   explicitly framed as such.
5. No changes to confirmatory endpoint definitions after viewing final aggregate
   comparisons without documented rationale.

---

## 9) Execution Timeline (Suggested)

Phase 0 (immediate, before full data finalization):

1. Lock manuscript analysis contract text.
2. Implement confirmatory-mode analysis path.
3. Run simulations S1-S3 at minimum.

Phase 1 (as benchmark data finalizes):

1. Freeze and QA benchmark artifacts.
2. Produce confirmatory core tables.
3. Produce attrition and sensitivity outputs.

Phase 2 (final paper assembly):

1. Add exploratory robustness panels.
2. Tighten claims to exactly supported scope.
3. Final reviewer-watchout QA pass.

---

## 10) Quick Command Reference

```bash
# 1) Aggregate artifacts from S3
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/download_and_aggregate.py --task all

# 2) Build synthetic metrics parquet
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/synthetic_analysis.py \
  --results-dir paper/results/rankings/classification \
  --data-dir paper/data/classification/synthetic \
  --output paper/results/synthetic_analysis.parquet

# 3) Run full statistical pipeline (confirmatory + exploratory once implemented)
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/stats.py
```
