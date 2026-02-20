# Outstanding Next Steps (Paper)

This file lists only the remaining, actionable work for the arXiv manuscript and
experiment writeup. Supporting trackers:

- `paper/docs/claims-index.md` (one row per formal claim)
- `paper/docs/writing-checklist.md` (proof QA checklist)
- `paper/docs/analysis-lockdown-plan.md` (statistical contract, simulations, and
  benchmark runbook)

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
- Ensure the final arXiv PDF contains at least one explicit fixed-node/root
  Stage~A calibration/sanity figure under the complete global null (a stats
  reviewer will expect to see this even if the theory is correct).
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

## G) Data cleaning before analysis

- **Remove `mtry="all"` r_cforest configs from analysis.** The `r_cforest` grid
  included `mtry: ["sqrt", "all"]`. The `mtry="all"` variants are intractable on
  mid/large datasets (isolet, gisette, comm_violence, community_crime) — single
  tasks ran 15+ hours without completing on c6a.8xlarge instances. Drop these
  configs (`r_cforest__4c600a4a16bac398`, `r_cforest__2775412ac549973e`) from
  all downstream analysis; only keep `mtry="sqrt"` variants. The grid has
  already been patched in `paper/scripts/pipeline/config.py`.

## H) Analysis lock + simulations

- Use `paper/docs/analysis-lockdown-plan.md` as the canonical tracker for:
  - confirmatory endpoint lock and multiplicity policy,
  - method identity policy (`method_base` vs config-level),
  - attrition/missingness reporting,
  - theory-linked simulation set (S1-S6),
  - benchmark data freeze and final runbook.
