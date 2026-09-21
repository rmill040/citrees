# Results

This directory keeps only the result files needed to verify reported manuscript
claims.

Most experiment outputs are untracked. Regenerate them locally when rebuilding
the arXiv analysis.

Subdirectories:

- `figures/`: ignored local figure outputs
- `tables/`: tracked CSVs used to verify manuscript claims
- `cache/`: the two rankings parquets mirrored from the S3 campaign by
  `paper/analysis/aggregate_pipeline_artifacts.py` (ignored, regenerable)
- `rdc-projection-sensitivity/`: tracked rankings, downstream metrics,
  summaries, and execution receipt for the RDC projection-count analysis

Do not add broad analysis tables here unless a submission cites them.
