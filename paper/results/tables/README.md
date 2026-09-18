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
- `cif_autobudget_ablation_summary.csv`, `cif_njobs1_ablation_summary.csv`: CIF at
  the `auto` permutation budget and with a single worker, with and without
  adaptive stopping (fixed kernels, c6a.8xlarge). arXiv Section 5 runtime
  ablations.
- `wide_runtime_ablation_summary.csv`: CIT and CIF runtime ablation on the
  benchmark-size real datasets (at least 500 observations; fixed kernels,
  c6a.8xlarge). arXiv Section 5 wide-dataset check.
- `scaling_curves.csv`: fitting time of CIT and CIF in n and p with and without
  adaptive stopping (fixed kernels, c6a.8xlarge). arXiv Appendix E.
- `paper_threshold_test_scaling_real_5seeds.csv`, `paper_threshold_test_power_5seeds.csv`:
  Stage B scaling, real-subset, and power studies pooled over seeds 0-4
  (complete-node design). arXiv Appendix I, JSS threshold-test subsection.
- `paper_stageb_only_pooled.csv`, `paper_stageb_only_size_matched.csv`: the
  Stage-B-only study (one predictor, Stage A gate off, three seeds, nominal
  levels 0.02-0.50) with Clopper-Pearson intervals and power interpolated to a
  matched realized size. Built by `paper/analysis/build_stageb_only_pooled.py`.
  arXiv `tab:stageB-only`.
- `paper_benchmark_lodo_{aggregate,selected_configs,config_stability}_allmethod.csv`:
  leave-one-dataset-out configuration selection on the all-method panels (21 and
  8 datasets). Built by `build_lodo_config_sensitivity_tables.py --panel
  all-method`. arXiv `tab:lodo-config-sensitivity-allmethod`.
- `paper_cif_permutation_importance_check.csv`: mechanism-matched CIF check.
  CIF refit per Stage 1 fold and ranked by out-of-bag permutation importance
  (`cif_oobperm`, the mechanism cforest uses) or by training-row scikit-learn
  permutation importance (`cif_perm`), evaluated with the Stage 2 evaluator;
  directed pairwise aggregates against cforest and against split-ranked CIF on
  the same datasets. Built by
  `paper/analysis/build_cif_permutation_importance_check.py` from the EC2 run
  under `reference/v3/ablations/cif-perm-check/`. arXiv `tab:cif-perm-check`.
- Review-round-3 control experiments (EC2, 2026-09-17; built by
  `paper/analysis/build_review3_tables.py` from `reference/v3/ablations/review3/`):
  `paper_stageb_fixed_budget.csv` (per-threshold Bonferroni at a fixed 999
  permutations per candidate: size, power, depth; arXiv `tab:stageB-fixed-budget`),
  `paper_stageb_scaling_3arm.csv` (one-run timing of the scaled, fixed, and
  max-type rules), `paper_behavior_scanning_control.csv` (JSS matched-behavior
  positive control with scanning on), `paper_performance_equal_work.csv` (the
  88-shard exhaustive performance campaign at 999 permutations per predictor in
  both libraries; JSS `tab:performance-reference` exhaustive rows), and, when
  their boxes land, `paper_nhanes_controls.csv`, `paper_cif_all_recovery.csv`,
  `paper_stageb_power_replicates.csv`.
- `paper_weak_signal_cardinality.csv`, `paper_support_size.csv`: weak-signal
  mixed-cardinality designs (top-ten composition, support, downstream score by
  ranker and signal) and ranker support sizes on the synthetic benchmark
  datasets (EC2, 2026-09-17/18; `paper/benchmark/experiments/weak_signal_cardinality.py`,
  built by `build_review3_tables.py`). arXiv `tab:weak-signal-cardinality`.
- `paper_performance_decomposition.csv`: the recommended forest at the JSS
  reference condition under variants that isolate its cost (serial and parallel
  fits, threshold test off, adjustment off, max-type), with tree depth and
  internal node counts (EC2, 2026-09-18;
  `paper/jss/replication/performance_decomposition.py`). JSS runtime section.
- `paper_noadjust_calibration.csv`, `paper_noadjust_stageb.csv`: complete-node
  false-split rate and Stage-B-only size and power of the per-threshold rule
  without the Bonferroni adjustment over candidates, the "No adjustment" column
  of the JSS timing tables (EC2, 2026-09-18; `threshold_test.py
  --tests bonferroni_unadjusted`).
