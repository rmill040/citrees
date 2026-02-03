# Figures + Tables Plan (Reproducible Map)

This file maps each paper-facing figure/table to:

1. the script that generates it,
2. the data artifact(s) it writes, and
3. the claim/story it supports.

**Canonical output directories:**

- Figures: `paper/results/figures/`
- Data caches: `paper/results/cache/`
- Tables: `paper/results/tables/`

---

## A. Theory / calibration figures (p-values)

### A1. Selection bias demo (why testing matters)

- Outputs:
  - `paper/results/figures/selection_bias_demo.png`
  - `paper/results/cache/selection_bias_demo_data.parquet`
- Script: `paper/scripts/theory/generate_selection_bias_demo.py`
- Claim/story:
  - Greedy split optimization can favor “high-cardinality” noise under a
    complete global null; Stage A permutation screening (fixed node, fixed-`B`)
    prevents “split unless significant” at the root.
  - Also reports $\Pr(\text{root splits})$ under the complete null (should be
    $\le \alpha_{\mathrm{sel}}$ with Bonferroni in fixed-`B` mode).
  - Caption requirements (avoid scope creep):
    - State the simulated null is a **complete global null** (label
      exchangeability under the permutation scheme).
    - Note the inferential claim is **fixed-node/root Stage A in fixed-`B`**
      mode; the Stage B/internal-node adaptivity caveats do not apply here.
- Status:
  - Should be kept in main text (motivation figure).
  - Generated on 2026-01-20 (older defaults). Defaults were increased on
    2026-01-21; rerun to update.

### A2. Fixed-$B$ Monte Carlo p-value calibration (+1 correction)

- Outputs:
  - `paper/results/figures/fixedB_pvalue_calibration.png`
  - `paper/results/cache/fixedB_pvalue_calibration_data.parquet`
- Script: `paper/scripts/theory/generate_fixedB_pvalue_calibration.py`
- Claim/story:
  - Empirical backstop for Theorem 1 (super-uniformity of the +1 Monte Carlo
    permutation p-value).
  - Caption requirements:
    - Explicitly state the permutation-calibration target (exchangeability under
      the permutation scheme) and that this is **fixed-`B`** (no optional
      stopping).
- Status:
  - Appendix figure (calibration/sanity check).
  - Generated on 2026-01-20 (older defaults). Defaults were increased on
    2026-01-21; rerun to update.

### A3. Adaptive sequential stopping calibration (continuous-null idealization)

- Outputs:
  - `paper/results/figures/sequential_stopping_calibration.png`
  - `paper/results/cache/sequential_stopping_calibration_data.parquet`
- Script: `paper/scripts/theory/generate_sequential_stopping_calibration.py`
- Claim/story:
  - Empirical calibration of the early-stopping heuristic under a controlled
    null model.
  - Explicitly _not_ a claim that the returned $\widehat p_\tau$ is a classical
    p-value under optional stopping.
- Status:
  - Appendix figure (calibration/sanity check).
  - Generated on 2026-01-20.

### A4. Martingale identity sanity check (developer note)

- Output: console only
- Script: `paper/scripts/theory/supermartingale_check.py`
- Claim/story:
  - Numerically checks the one-step identity $\mathbb{E}[S_{n+1}\mid L_n,n]=S_n$
    under the continuous-null mixture model.
- Status:
  - Not a paper figure; keep as a reproducibility check for theory development.
  - Generated on 2026-01-21 (console only).

---

## B. Main benchmark figures (synthetic experiments)

These are self-contained synthetic experiments for feature-selection behavior
and scaling.

**Regenerate in one command:**

```bash
uv sync --group paper
UV_CACHE_DIR=$PWD/.uv-cache uv run python paper/scripts/analysis/generate_figures.py --profile paper
```

Notes:

- `--profile paper` uses larger synthetic datasets than `--profile quick`
  (intended for publication-quality figures).
- `--profile huge` is provided for “very large” runs (slow; mainly for
  stress-testing stability).
- Profile defaults live in `paper/scripts/analysis/generate_figures.py` (see
  `PROFILES`).
- Profile sizes were bumped on 2026-01-21 (paper/huge). Re-run figures after
  updating if you want outputs to reflect the larger defaults.
- In restricted/sandboxed environments, multiprocessing backends may be
  unavailable; the script will fall back to `n_jobs=1`.
- Use `--only ...` to regenerate a subset, e.g.:
  `UV_CACHE_DIR=$PWD/.uv-cache uv run python paper/scripts/analysis/generate_figures.py --profile paper --only feature_selection signal`

### B1. Feature selection behavior (classification)

- Outputs:
  - `paper/results/figures/feature_selection_clf.png`
  - `paper/results/cache/feature_selection_data.parquet`
  - `paper/results/tables/feature_selection_table.tex`
  - `paper/results/figures/informative_ratio.png`
- Script: `paper/scripts/analysis/generate_figures.py`
- Claim/story:
  - How embedding methods split on informative vs noise features in a controlled
    setting.
- Status:
  - Main text (core behavioral figure) + appendix table.

### B1b. Feature selection behavior (regression)

- Outputs:
  - `paper/results/figures/regression_comparison.png`
  - `paper/results/cache/regression_data.parquet`
- Script: `paper/scripts/analysis/generate_figures.py`
- Claim/story:
  - Controlled regression toy experiment: split quality on known-informative
    features.
- Status:
  - Appendix or “sanity check” figure (not real-data performance).

### B2. Synthetic robustness slices (one question per figure)

- Outputs (each has a `_data.parquet` companion):
  - `paper/results/figures/signal_strength.png`
  - `paper/results/figures/sample_size.png`
  - `paper/results/figures/high_dimensional.png`
  - `paper/results/figures/correlated_features.png`
  - `paper/results/figures/redundant_features.png`
  - `paper/results/figures/complexity_vs_accuracy.png`
  - `paper/results/figures/multiclass.png`
  - `paper/results/figures/imbalanced.png`
- Script: `paper/scripts/analysis/generate_figures.py`
- Claim/story:
  - Synthetic “stress tests” to show when/why citrees behaves well or fails.
- Status (suggested ordering):
  - Main text: `signal_strength`, `sample_size`, `high_dimensional`,
    `correlated_features`, `redundant_features`
  - Appendix: `multiclass`, `imbalanced`, `complexity_vs_accuracy`

### B3. Runtime / scalability

- Outputs:
  - `paper/results/figures/timing_speedup.png`
  - `paper/results/figures/timing_bars.png`
  - `paper/results/cache/timing_data.parquet`
- Script: `paper/scripts/analysis/generate_figures.py`
- Claim/story:
  - Cost of permutation testing and speedups from early stopping / parallelism.
- Status:
  - Main text: one runtime figure (pick either speedup curve or bars); appendix
    for the other.

---

## C. Real-data figures (Stage 1/Stage 2 pipeline)

These figures depend on the full experiment pipeline (S3-backed in the current
setup). They should only be used in the paper after verifying the underlying
parquet artifacts correspond to the reported benchmark protocol.

- Existing artifacts in `paper/results/` are **synthetic** (from
  `generate_figures.py`).
- Real-data figures are not checked in and must be generated from S3-backed
  Stage 2 metrics.
- Scripts:
  - Stage 1 (rankings): `paper/scripts/pipeline/stage1.py`
  - Stage 2 (downstream eval): `paper/scripts/pipeline/stage2.py`
  - Analysis aggregation: `paper/scripts/analysis/stats.py`
- Paper rule:
  - Every number/curve must have a traceable pipeline: config → parquet inputs →
    analysis script → figure output.

### C1. Per-model / per-k analysis outputs (after evaluation parquets exist)

Once `paper/results/clf_evaluation.parquet` and
`paper/results/reg_evaluation.parquet` exist, `stats.py` emits:

- Overall aggregates:
  - `paper/results/tables/clf_*` and `paper/results/figures/clf_*`
  - `paper/results/tables/reg_*` and `paper/results/figures/reg_*`
- Per‑downstream‑model:
  - `paper/results/tables/clf_{model}_*` and
    `paper/results/figures/clf_{model}_*` (e.g., `clf_lr_*`, `clf_svm_*`,
    `clf_knn_*`)
  - `paper/results/tables/reg_{model}_*` and
    `paper/results/figures/reg_{model}_*` (e.g., `reg_ridge_*`, `reg_svr_*`,
    `reg_knn_*`)
- Per‑model‑per‑k:
  - `paper/results/tables/clf_{model}_k{k}_*` and
    `paper/results/figures/clf_{model}_k{k}_*` (e.g., `clf_lr_k10_*`)
  - `paper/results/tables/reg_{model}_k{k}_*` and
    `paper/results/figures/reg_{model}_k{k}_*` (e.g., `reg_ridge_k25_*`)

These outputs are generated by `paper/scripts/analysis/stats.py` and should be
mapped into figures/tables once the evaluation parquets are available.
