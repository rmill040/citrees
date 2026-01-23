# Immediate Next Steps (Paper)

This file is the short-term execution plan. It complements:

- `paper/WRITING_CHECKLIST.md` (process + proof QA checklist)
- `paper/CLAIMS_INDEX.md` (one row per formal claim in the manuscript)

## A) Manuscript correctness (make core theory airtight)

1. Proof QA pass for all current claims (DRAFT -> QA)

- Target: every row in `paper/CLAIMS_INDEX.md` moves from `DRAFT` to `QA`.
- Files: `paper/arxiv/sections/04_theory.tex` and
  `paper/arxiv/appendices/appendix_A_setup.tex` through
  `paper/arxiv/appendices/appendix_G_multiselector.tex`.
- Checklist items per claim:
  - Conditioning is explicit (e.g., conditional on $(X_t,U)$).
  - Tested families and resample budgets are label-independent (A0.3).
  - Tie handling matches the stated convention (A0.5).
  - Scope note is explicit: fixed-node/root only; no adaptive-tree p-value
    claims.

2. Decide scope for Stage B + internal nodes (and lock the language)

- Ensure every mention of Stage B p-values is framed as post-selection/adaptive.
- If we want any additional inference for Stage B/internal nodes, add a separate
  section that is explicitly “future work” (selective inference / honesty /
  sample splitting).

## B) Methods appendix migration (finish Appendix H -> LaTeX)

Goal: make Appendix H (`paper/arxiv/appendices/appendix_H_methods.tex`) a clean,
implementation-aligned reference so the main paper stays readable.

1. Complete Stage B details (match implementation semantics precisely)

- Verify/encode in `paper/arxiv/appendices/appendix_H_methods.tex`:
  - left-tail convention for splitters
  - unweighted-vs-weighted impurity usage (testing vs scanning vs
    min-impurity-decrease)
  - threshold generation methods (exact/random/percentile/histogram),
    small-unique fallback, `max_thresholds` semantics
  - min-leaf filtering, and the “no valid thresholds” behavior

2. Add forest sampling + OOB semantics

- Add to `paper/arxiv/appendices/appendix_H_methods.tex`:
  - sampling modes (classification stratified vs balanced, etc.)
  - `max_samples` semantics (bootstrap draw then subsample)
  - OOB scoring definition (which samples count; requirement that bootstrap is
    enabled)

3. Add evaluation protocol + metrics (rank-then-evaluate)

- Add a subsection to `paper/arxiv/appendices/appendix_H_methods.tex`
  describing:
  - ranking outputs (what is a “ranking”)
  - downstream models (at a high level)
  - the top-k sweep and aggregation policy (seeds/folds/models)
  - stability metrics (e.g., Kendall tau, Jaccard@k) with precise definitions

4. Ensure “heuristics” are correctly scoped

- Keep early stopping / scanning / muting explicitly labeled as heuristics.
- If we want to claim anything about them, add separate calibration results (and
  keep them out of the fixed-node theorem chain).

## C) Experiments section (turn skeleton into paper-grade)

1. Write the experiment protocol in `paper/arxiv/sections/05_experiments.tex`

- List datasets (synthetic + real), seeds, splits, and compute budgets.
- List baselines and the fairness policy (what is fixed vs tuned).
- Define the primary ranking metrics and downstream metrics.

2. Decide the “minimum viable” figure/table set for arXiv

- Use `paper/notes/figures_plan.md` as the map.
- Decide what goes in main text vs appendix.

3. arXiv packaging rule for figures

- Final manuscript figures should live under `paper/arxiv/` (self-contained
  arXiv bundle).
- `paper/results/` remains a generation/cache directory (see
  `paper/results/README.md`).

## D) Background / related work (state of the field)

- Expand `paper/arxiv/sections/02_background.tex` to explicitly compare:
  - R `party`/`partykit` conditional inference implementations
  - CART/sklearn trees/forests (why high-cardinality bias arises)
  - feature selection baselines used in experiments

## E) Housekeeping (keep the workflow clean)

- Keep `paper/README.md` accurate as directories evolve.
- Keep `paper/CLAIMS_INDEX.md` updated whenever a claim is
  added/removed/renamed.
- Prefer “one appendix per claim” (proofs) and “one appendix per topic”
  (methods), rather than mega-appendices.
