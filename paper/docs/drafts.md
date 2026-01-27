# citrees: Draft Staging Notes (Trimmed)

This file is an internal staging document. As of 2026-01-26, the stable,
paper-facing mathematics and algorithm documentation have been migrated into
the arXiv manuscript source of truth under `paper/arxiv/`.

## Where the content went

- Main manuscript: `paper/arxiv/main.tex`
- Method overview: `paper/arxiv/sections/03_method.tex`
- Theory statements: `paper/arxiv/sections/04_theory.tex`
- Experiments overview: `paper/arxiv/sections/05_experiments.tex`
- Discussion/limitations: `paper/arxiv/sections/06_discussion.tex`
- Proof appendices: `paper/arxiv/appendices/appendix_*.tex`
- Algorithm + metrics appendix (implementation-aligned): `paper/arxiv/appendices/appendix_H_methods.tex`
- CART bias motivation: `paper/arxiv/appendices/appendix_I_cart_bias.tex`
- Reproducibility + implementation notes: `paper/arxiv/appendices/appendix_implementation.tex`

## Use these docs for ongoing work

- Claims tracker: `paper/docs/claims-index.md`
- Writing checklist: `paper/docs/writing-checklist.md`
- Experiments pipeline + method list: `paper/docs/experiments.md`
- Figures/tables plan: `paper/docs/figures-plan.md`
- Next steps: `paper/docs/next-steps.md`

## Remaining TODOs (paper-tightening)

1. Replace placeholder experiment text with concrete tables/figures and the
   minimum narrative needed to support each claim.
2. Add a short ``recommended configs'' subsection (default fixed-$B$ vs
   heuristic early stopping; when to disable feature muting).
3. Final notation pass: ensure consistent symbols across sections
   (node index $t$, feature set $F_t$, thresholds $C_{t,j}$, budgets $B$, etc.).
