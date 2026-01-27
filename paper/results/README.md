# Results artifacts (generated)

This directory contains generated artifacts from experiment/analysis scripts
under `paper/scripts/`.

Canonical subdirectories (used by scripts and referenced by
`paper/docs/figures-plan.md`):

- `paper/results/figures/`: PNG figures (paper-facing or diagnostic).
- `paper/results/tables/`: tables (CSV/LaTeX).
- `paper/results/cache/`: cached intermediate data and theory/calibration outputs
  (typically parquet) to speed up figure regeneration.

Notes:

- Treat these as **outputs**, not source-of-truth prose or theory.
- The arXiv manuscript should eventually include only the final figures it needs
  under `paper/arxiv/` for a self-contained arXiv bundle.
