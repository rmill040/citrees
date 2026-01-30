# Reviewer Watchouts (stats-focused) — post-results checklist

This document is an internal checklist of predictable reviewer questions and
failure modes for the citrees manuscript. Use it when results finish to ensure
the final paper reads like a careful statistical submission (clear estimands,
calibrated claims, and defensible experimental comparisons).

**Scope note:** This is not manuscript text. It is an internal QA + planning
artifact to keep us from (i) overclaiming beyond what we prove, and (ii)
post-hoc changing evaluation choices after seeing results.

---

## 0) “Paper contract” checks (do these before inspecting results)

### 0.1 Primary endpoints are locked

**Where stated:** `paper/arxiv/sections/05_experiments.tex`
(\cref{sec:experiments-analysis-plan}).

**Risk:** If the primary endpoint changes after results are known, a
statistician reviewer will interpret it as post-hoc selection (“metric shopping”
/ “garden of forking paths”).

**Action:**

- Confirm the primary endpoint choices (task, metric, downstream model, and
  chosen k) are final _before_ looking at aggregate comparisons.
- If changes are necessary, update the manuscript and record the decision
  rationale in this file.

### 0.2 The paper’s “two roles” framing matches results

**Where stated:** `paper/arxiv/sections/01_introduction.tex` (“Two roles:
screening vs. embedding”).

**Risk:** Reviewers will ask: “If the only clean inference is root/fixed-node,
why should I care about the full tree/forest ranking outputs?”

**Action:** Ensure results include:

- at least one root/fixed-node screening view (clean interpretability), and
- full tree/forest ranking views (empirical performance/stability).

---

## 1) Inferential scope / p-value calibration (most common stats-reviewer pushback)

### 1.1 What null are Stage A p-values calibrated for?

**Where stated:** Assumptions A0.1–A0.5 in
`paper/arxiv/appendices/appendix_A_setup.tex`; scope in
`paper/arxiv/sections/03_method.tex` (\cref{sec:pvalue-scope} and
Table~\ref{tab:pvalue-scope}).

**Reviewer question to anticipate:** “Are your per-feature permutation p-values
valid only under a nodewise global null, or also under feature-specific nulls
when other features are associated with Y?”

**Risk:** Readers may incorrectly infer broad per-feature validity even when the
permutation scheme conditions on all X and exchangeability only holds under a
complete/global null.

**What to check once results arrive:**

- Any language in results/captions implying “Type I error control per feature”
  outside a complete nodewise global null.
- Any plots that look like “feature-level calibration” but were generated under
  a setting where other features have signal.

**Mitigations / manuscript actions (if needed):**

- Make the null being simulated explicit in every calibration plot caption:
  “complete nodewise global null” vs “single-feature null with restricted
  permutations” (if implemented).
- If showing non-global-null simulations, phrase them as empirical behavior
  studies, not calibration claims.

### 1.2 Fixed-B only: no optional-stopping validity for early-stopped p-values

**Where stated:** `paper/arxiv/sections/06_discussion.tex` (Early stopping) and
`paper/arxiv/appendices/appendix_H_methods.tex` (Sequential permutation
testing).

**Reviewer question:** “You use early stopping—are the resulting p-values still
valid? What is the inferential interpretation?”

**Risk:** Even if we say “heuristic,” figures/tables might inadvertently label
early-stopped values as p-values in a way that implies calibration.

**What to check:**

- Any figure/table that reports early-stopped values without a clear
  “algorithmic statistic” label.
- Any comparison that mixes fixed-B runs and early-stopped runs without
  explaining that they target different objects.

**Mitigations:**

- Keep early-stopping plots explicitly framed as “calibration sanity checks” and
  do not claim Type I error guarantees.
- Consider a separate appendix subsection: “Early stopping as a speed/accuracy
  tradeoff” with empirical calibration diagnostics only.

### 1.3 Distinguish two “early stopping” mechanisms (scan termination vs sequential resampling)

**Where stated:** `paper/arxiv/sections/03_method.tex` and
`paper/arxiv/appendices/appendix_H_methods.tex`.

**Risk:** Reviewers (and readers) conflate:

- early-terminated candidate scans (may change which feature is selected), with
- sequential resampling (changes distribution of the reported running estimate).

**What to check:**

- In results, ensure labels clearly separate these mechanisms.
- If you report speedups, specify which mechanism produced them.

---

## 2) Stage B and internal nodes: prevent accidental overclaims

### 2.1 Stage B “p-values” are post-selection/adaptive

**Where stated:** `paper/arxiv/sections/03_method.tex` (Stage B scope note) and
`paper/arxiv/sections/06_discussion.tex` (What we do/do not claim).

**Reviewer question:** “Why report a p-value at all for Stage B if it is not
calibrated?”

**Risk:** Results might accidentally read like we are doing valid splitting
inference (or that Stage B provides calibrated confidence).

**What to check:**

- Any captions or text interpreting Stage B values as hypothesis tests.
- Any comparisons that treat Stage B p-values as on the same footing as Stage A.

**Mitigations:**

- If Stage B values are reported, rename them “split scores” in plots/tables.
- Consider moving Stage B p-value outputs to appendix only, unless they are
  essential for runtime/ablation narratives.

### 2.2 Internal-node p-values are not classical p-values

**Risk:** Very common reviewer complaint: “Your tree is adaptive; internal node
tests reuse data; p-values are not calibrated.”

**Action:** Ensure any internal-node “p-values” shown are labeled as algorithmic
statistics and are not used for inferential claims.

---

## 3) Stage B statistic choice and split bias concerns

### 3.1 Unweighted impurity test statistic can bias toward unbalanced splits

**Where stated:** `paper/arxiv/appendices/appendix_H_methods.tex`
(\cref{app:methods-splitters}).

**Reviewer question:** “Why is the split test based on unweighted impurity sum
rather than weighted impurity? Does it favor pathological splits?”

**What to check:**

- Whether results show unusually deep trees with tiny leaves unless constrained.
- Sensitivity to `min_samples_leaf` and `min_impurity_decrease`.

**Mitigations:**

- Add an ablation comparing weighted vs unweighted split statistics (even if
  framed as algorithmic).
- If unweighted is retained, justify as an engineering choice and show it does
  not harm ranking/performance in the regimes you care about.

---

## 4) Benchmark design: endpoints, aggregation, and multiplicity

### 4.1 Do not average across k and downstream models as a headline score (unless justified)

**Risk:** Averaging across k values mixes different inferential questions
(small-subset selection vs full-feature performance). Averaging across models
makes the estimand unclear and can hide interactions.

**Action:** Keep the primary endpoint fixed and treat other summaries as
secondary robustness checks.

### 4.2 Cross-dataset aggregation must be paired and interpretable

**Reviewer question:** “How did you aggregate across datasets of wildly
different difficulty?”

**What to check:**

- Use paired, per-dataset comparisons (win/loss rates, paired differences, rank
  comparisons).
- If you report means, report distributional summaries (median/IQR) too.

### 4.3 Multiple comparisons across many methods

**Risk:** With 20+ methods, a reviewer may demand some multiple-comparison
discipline for claims like “method A is best.”

**Mitigations:**

- Prefer statements like “top tier” (average ranks / CD diagrams) over claiming
  strict dominance everywhere.
- If doing pairwise tests vs many baselines, state correction method or treat as
  exploratory.

---

## 5) Baseline fairness and tuning (predictable, high-friction reviewer topic)

### 5.1 Hyperparameter tuning parity

**Reviewer question:** “Did you tune baselines? If yes, how much effort per
method? If no, why are defaults fair?”

**Action items once results are in:**

- Produce a table summarizing each method’s tuning regime:
  - fixed defaults vs grid,
  - grid size,
  - compute budget / timeouts,
  - failures dropped vs penalized.
- Ensure the manuscript explicitly states this policy (main text or appendix).

### 5.2 Failure/timeout policy

**Risk:** Quietly excluding failed runs can bias results.

**Action:** Decide and document:

- what counts as a failure,
- how it affects ranks/aggregates,
- and report failure rates by method.

---

## 6) Dataset provenance, preprocessing, and leakage

### 6.1 Dataset table is required for credibility

**Reviewer expectation:** Names + (n, p) + task type + source/citation +
preprocessing notes.

**Action:** Add a dataset summary table (appendix is fine) and explicitly
describe:

- missing-value handling,
- categorical encoding (if any),
- standardization and where it is fit (train-fold only),
- any dataset-method exclusions.

### 6.2 Leakage prevention must be “code-exact”

**Where stated:** `paper/arxiv/appendices/appendix_H_methods.tex`
(\cref{app:methods-benchmark}).

**What to check:** Ensure the pipeline:

- fits standardization on training fold only,
- applies transforms to held-out fold,
- and does not use test fold during ranking.

---

## 7) Stability: don’t promise it without reporting it

**Motivation mentions stability:** `paper/arxiv/sections/01_introduction.tex`.

**Risk:** If the final paper talks about stability but reports only
accuracy-type metrics, reviewers may call it incomplete.

**Action (choose one):**

- Add stability metrics (e.g., Kendall’s tau on rankings, Jaccard@k) and define
  them precisely; or
- remove/soften “stable across resamples” language and frame stability as future
  work.

---

## 8) Runtime: report tradeoffs, not only speedups

**Reviewer question:** “Speedups at what cost (accuracy, ranking quality,
calibration)?”

**Action once results are in:**

- Pair runtime plots with quality plots (same ablation axis).
- If early stopping is used, show a calibration/behavior sanity check in the
  appendix and keep claims limited.

---

## 9) Presentation / narrative hygiene when inserting figures

### 9.1 Every figure must answer one question

**Rule of thumb:** Each main-text figure should correspond to exactly one
headline claim (“intended takeaway”), stated in the caption and supported by the
panel.

### 9.2 Avoid overclaiming words in captions

Flag words that imply inference unless fully justified:

- “significant”, “controls”, “valid p-values”, “Type I error controlled”, “FWER
  controlled” (unless in fixed-node, fixed-B, nodewise global-null context).

---

## 10) When results arrive: final “QA run”

Before freezing the manuscript:

- Confirm the figures/tables inserted match the pre-specified endpoint section.
- Cross-check every inferential claim against
  `paper/arxiv/sections/04_theory.tex` and the assumption ledger
  `paper/arxiv/appendices/appendix_A_setup.tex`.
- Ensure no plot labels Stage B/internal-node quantities as calibrated p-values.
- Add a dataset table and baseline fairness table (even if in appendix).
