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

The aggregators read a local mirror at `../data/rankings/` and
`../data/metrics/`, one parquet per method configuration, dataset, and seed.
It holds only configurations in the current grid (4,458 classification and
2,400 regression cells on 2026-09-23), accumulated across the benchmark
campaigns; the canonical rerun, campaign d805868f at source commit 05ee3cd7, is
`reference/v3/benchmark/campaign-d805868f/` in S3.
- `rdc-projection-sensitivity/`: tracked rankings, downstream metrics,
  summaries, and execution receipt for the RDC projection-count analysis

Do not add broad analysis tables here unless a submission cites them.
