# Result Tables

Tracked tables:

- `paper_benchmark_method_aggregate.csv`: CIF aggregate ranks reported in the
  manuscript.
- `paper_presentation_practical_controls_summary.csv`: adaptive-stopping
  runtime and score-delta summary reported in the manuscript.
- `cit_cif_runtime_ablation_summary.csv`: CIT/CIF runtime and ranking deltas
  per setting change (arXiv runtime ablation tables).
- `paper_mirrored_knob_ablation_summary.csv` and
  `paper_threshold_ablation_summary.csv`: CIF knob and threshold-search
  ablation summaries behind the arXiv CIF runtime table.
- `cif_mechanism_ablation_paired_foldseed_vs_default.csv` and
  `cif_mechanism_ablation_paired_foldseed_dataset_deltas_vs_default.csv`: CIF
  component ablation (one tree, no bootstrap, no muting, split-count ranking)
  paired against the selected CIF configuration at the fold x seed replicate
  level on the eight classification datasets with at least five paired
  replicates. Built by `paper/analysis/build_cif_mechanism_ablation_tables.py`
  from `s3://citrees-856480643277/repairs/benchmark-rerun/source-05ee3cd7.../campaign-d805868f.../_control/cif-mechanism-ablation/metrics`.

Everything else should be regenerated locally and left untracked.
