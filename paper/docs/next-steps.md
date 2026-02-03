# Outstanding Next Steps (Paper)

This file lists only the remaining, actionable work for the arXiv manuscript and
experiment writeup. Supporting trackers:

- `paper/docs/claims-index.md` (one row per formal claim)
- `paper/docs/writing-checklist.md` (proof QA checklist)

## A) Scope lock (Stage B + internal nodes)

- For this arXiv paper, do **not** claim calibrated p-values beyond fixed-node
  (especially root) Stage~A screening in fixed-$B$ mode.
- Treat Stage~B threshold tests, internal-node tests, and early-stopped
  permutation outputs as algorithmic statistics; ensure plots/tables/captions
  use consistent naming (avoid implying calibration).
- Optional future work (separate scope): if we ever extend inference to Stage~B
  / internal nodes, pick one:
  - fixed feature chosen label-independently (not Stage~A selected), or
  - sample splitting / honesty (Stage~A selects on one split, Stage~B tests on
    an independent split), or
  - full selective-inference development (larger scope).

## B) Methods appendix (Appendix H) remaining

- Make the benchmark protocol code-exact in
  `paper/arxiv/appendices/appendix_H_methods.tex`: folds/seeds, top-$k$,
  scaling, downstream models + hyperparams, metrics, aggregation.
- Decide whether to include ranking stability metrics (Kendall tau / Jaccard@k):
  - if yes, define precisely in Appendix H and ensure the pipeline produces them
  - if no, remove mentions to avoid promising unavailable results
- Spot-check Stage~B semantics against implementation during the code-exact
  audit (sanity pass).
- Keep early stopping / scanning / muting explicitly scoped as heuristics (and
  keep them out of the fixed-node theorem chain).

## C) Experiments section (turn skeleton into paper-grade)

- Write the experiment protocol in `paper/arxiv/sections/05_experiments.tex`:
  datasets, seeds/splits, compute budgets, baselines + fairness policy, primary
  metrics, and an explicit “optimizations + ablations” mapping (knobs vs
  figures/tables).
- Decide the minimum viable figure/table set for arXiv using
  `paper/docs/figures-plan.md`.
- Ensure every calibration figure/caption states the null being simulated (e.g.,
  **complete global null** / exchangeability target) and matches the fixed-node,
  fixed-$B$ scope of the Stage~A theory.
- Ensure final manuscript figures live under `paper/arxiv/` for a self-contained
  arXiv bundle.

## D) Background tightening (after experiments are drafted)

- Tighten `paper/arxiv/sections/02_background.tex` so every citation supports
  something we benchmark or a limitation we discuss.

## E) Housekeeping

- Keep `paper/README.md` accurate as directories evolve.
- Keep `paper/docs/claims-index.md` updated whenever a claim is
  added/removed/renamed.
- Keep appendix organization clean (avoid mega-appendices).

## F) Final alignment

- Make experiments and methods mutually consistent (Appendix H is code-exact;
  `sections/05_experiments.tex` promises only what the pipeline produces).
- Confirm author list/order and affiliation formatting in
  `paper/arxiv/main.tex`.
