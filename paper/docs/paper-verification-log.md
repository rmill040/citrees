# Paper Verification Log

Working log for getting the manuscript into a coherent, defensible state.
This is a live tracker of:

- what has already been verified,
- what still needs verification,
- what has been changed to match the code or the actual claim scope.

Last updated: 2026-03-25

## Status legend

- `DONE`: reviewed and updated
- `IN PROGRESS`: reviewed, with follow-up still needed
- `TODO`: not yet reviewed carefully

## Completed checks

### 1. Mathematical scope

- `DONE` Reduced the main theory to the defensible core:
  - fixed-node complete-null permutation calibration,
  - fixed-$B$ +1 Monte Carlo p-value super-uniformity,
  - Bonferroni control for the fixed-node Stage~A decision,
  - root no-split consequence.
- `DONE` Removed or demoted derivative statements that made the section read
  inflated:
  - per-feature/subsampling bounds moved to appendix note,
  - multi-selector validity moved to appendix note,
  - forest-level probability bounds removed from headline theory.
- `DONE` Aligned assumptions ledger and claims tracker with the reduced claim
  set.

### 2. Runtime / complexity

- `DONE` Rewrote the complexity section to use a smaller set of node-level
  counts and clearer worst-case versus expected-work language.
- `DONE` Verified that the implementation performs a descendant-safe
  constant-feature pruning pass over all currently available features before
  `max_features` subsampling.
- `DONE` Updated the manuscript to reflect that:
  - `max_features` caps Stage~A testing width,
  - it does not cap the initial constant-feature preprocessing pass,
  - `feature_muting` mainly reduces future subtree candidate sets,
  - early stopping / scanning reduce realized work, not worst-case order.
- `DONE` Tightened the runtime-results prose so the observed flatness in `p`
  is presented as empirical behavior on the tested range, not as a theorem-like
  complexity claim.

## Current pass

### 3. Front matter coherence: title / abstract / introduction / conclusion

- `IN PROGRESS` Trimmed the front matter so it no longer advertises the
  derivative “cardinality-free” bound as a headline theory result.
- `IN PROGRESS` Removed unsupported empirical superlatives from the abstract,
  introduction, discussion, and conclusion; those sections now describe the
  real-data aggregate plots as exploratory context and stop short of pairwise
  superiority claims that are not yet frozen.
- `IN PROGRESS` Remaining verification needed:
  - check that all empirical superlatives in abstract/introduction/conclusion
    are directly supported by the final tables/figures,
  - check that dataset counts, method counts, and effect sizes match the final
    analysis outputs exactly.

### 4. Experiments section coherence

- `IN PROGRESS` Found an internal dataset-count mismatch in
  `sections/05_experiments.tex`:
  - the real-data dataset paragraph and the downstream-performance captions did
    not agree,
  - the filesystem currently contains 23 real classification parquet files and
    8 real regression parquet files.
- `IN PROGRESS` Removed unverified dataset counts from headline captions and
  summary sentences until they can be matched directly to the finalized
  analysis outputs.
- `IN PROGRESS` Removed the main-text pairwise table from the real-data
  subsection and rewrote that subsection as exploratory aggregate evidence,
  because the current endpoint exports are not yet trustworthy for confirmatory
  pairwise claims.
- `IN PROGRESS` Remaining verification needed:
  - method counts in classification/regression comparisons,
  - benchmark completeness percentages and denominators,
  - every exact rank/effect-size claim in the real-data results section.

## Open findings

### A. Empirical headline claims still need direct verification

The following types of claims should be checked directly against the finalized
analysis outputs before the paper is frozen:

- “strongest statistically principled feature-ranking method we tested”
- “substantially outperforming legacy R implementations (+6 rank positions)”
- benchmark counts such as “15 methods across 32 datasets”
- exact ablation percentages such as depth increases or precision changes
- exact dataset counts used in each headline comparison

These may be true, but they should be treated as unverified until matched
against the final tables/figures used in the paper.

### B. Current real-data exports show concrete integrity risks

During the experiments pass, two specific issues surfaced in the local result
artifacts:

- `pairwise_cif_significance.csv` uses different effective dataset counts
  across competitors (`n_datasets` varies across rows), so those pairwise
  results are not on a common paired comparison set.
- `per_dataset_balanced_accuracy_lr_k25.csv` currently includes synthetic
  dataset rows, so it is not a clean real-only primary-endpoint export.

These issues may disappear once the reruns and rebuilt summaries land, but they
are the reason the manuscript now treats the aggregate real-data plots as
exploratory and defers exact pairwise claims.

### C. Scope language must remain consistent everywhere

The paper should continue to avoid implying:

- calibrated Stage~B threshold inference,
- calibrated internal-node p-values,
- optional-stopping validity for early-stopped outputs,
- global feature-selection error control for the full adaptive workflow.

## Next verification queue

### 5. Experiments section coherence

- `TODO` Verify that every headline result in `sections/05_experiments.tex`
  maps to an actual figure/table and uses the same aggregation unit as the
  analysis pipeline.
- `TODO` Verify that benchmark counts, endpoint choices, and rank summaries are
  internally consistent.
- `TODO` Check that runtime claims are paired with the right ablation setting
  and are not mixing incomparable configs.

### 6. Methods / Appendix H code-exact audit

- `TODO` Verify that Appendix H matches the actual implementation for:
  - Stage~B semantics,
  - scanning/muting/threshold generation order,
  - benchmark protocol details,
  - ranking and evaluation pipeline details.

### 7. Discussion / limitations

- `TODO` Check that every limitation discussed is reflected consistently in the
  methods and experiments sections.
- `TODO` Remove any remaining statements that read stronger than the evidence.

### 8. Title / framing

- `TODO` Revisit the title once the empirical and theoretical scope are frozen.
  The title should reflect:
  - practical implementation and benchmarks,
  - limited fixed-node calibration results,
  - not a broad new inference theory for adaptive forests.

## Build status

- `DONE` `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
  succeeds after the theory and runtime cleanup passes.
