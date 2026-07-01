# Results Artifacts

This directory contains generated outputs from the paper analysis pipeline.

Treat these as outputs, not prose authority.

For paper-facing status:

- use `paper/docs/results-finalization.md` for locked numbers and package
  decisions
- use `paper/results/tables/README.md` for canonical vs supporting vs superseded
  table status

The benchmark rebuild consumes `paper_real_evaluation.parquet`, the joined
real-data downstream evaluation surface for all paper-facing methods. The
synthetic top-k rebuild consumes `synthetic_topk_composition.parquet`, the
joined synthetic recovery surface. Paper-facing CSV tables should be generated
from these joined surfaces rather than from method-specific sidecar files.

Subdirectories:

- `figures/`: analysis-generated figures
- `tables/`: CSV and LaTeX tables
- `cache/`: cached intermediates and theory/calibration outputs

Exploratory artifacts should not drive manuscript claims unless a paper-facing
doc explicitly promotes them.
