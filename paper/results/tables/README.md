# Result Tables

Tracked tables:

- `paper_benchmark_method_aggregate.csv`: CIF aggregate ranks reported in the
  manuscript.
- `paper_presentation_practical_controls_summary.csv`: adaptive-stopping
  runtime and score-delta summary reported in the manuscript.
- `cit_runtime_ablation_decoupled_summary.csv`: CIT runtime ablation with
  decoupled knobs on the fixed kernels (image sha256:8dba68ec at d6c0696,
  c6a.8xlarge, 805 fits, 23 datasets). Source of the rewritten CIT rows.
- `paper_h2h_rerun_summary.csv`: head-to-head forest fitting time (median
  seconds per cell and configuration: citrees recommended, no adjustment,
  max-type, partykit cforest on 1 and 32 cores, sklearn RF) on the fixed
  kernels (image sha256:a4fc734d at e78d84a, c6a.8xlarge). Source of the JSS
  performance-real and performance-scaling tables.
- `paper_maxt_extension_{method_aggregate,pairwise,swap_aggregate}.csv`: the
  max-type ranking extension of the benchmark (arXiv Table
  `tab:maxt-benchmark-rankings`). Built by
  `paper/analysis/build_maxt_extension_tables.py`.
- `cit_autobudget_ablation_summary.csv`: CIT at the `auto` permutation budget with
  and without adaptive stopping (fixed kernels, c6a.8xlarge, 345 fits).
- `paper_threshold_test_calibration_5seeds.csv`: Stage B null false-split rates
  pooled over seeds 0-4 (10,000 replications per cell) with Clopper-Pearson
  intervals.
- `locked_cells` are listed in `paper/benchmark/config/locked_cells.csv`, not here.
- `paper_mirrored_knob_ablation_summary.csv` and
  `paper_threshold_ablation_summary.csv`: CIF knob and threshold-search
  ablation summaries behind the arXiv CIF runtime table.
- `cif_mechanism_ablation_paired_foldseed_vs_default.csv` and
  `cif_mechanism_ablation_paired_foldseed_dataset_deltas_vs_default.csv`: CIF
  component ablation (one tree, no bootstrap, no muting, split-count ranking)
  paired against the selected CIF configuration at the fold x seed replicate
  level on 22 classification and 8 regression real-data benchmark datasets
  (isolet excluded by author decision). Built by `paper/analysis/build_cif_mechanism_ablation_tables.py`
  from `s3://<citrees bucket>/repairs/benchmark-rerun/source-05ee3cd7.../campaign-d805868f.../_control/cif-mechanism-ablation/metrics`.

Everything else should be regenerated locally and left untracked.
