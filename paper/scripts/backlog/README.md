# Backlog Script Inventory

This file tracks paper-side scripts that are not on the current arXiv
submission path. The goal is to archive them later, not delete them blindly.

## Triage Rule

Backlog a script when all of the following are true:

- it is not in the closed rebuild path from `paper/docs/experiments.md`
- it does not build one of the three local arXiv figure assets
- it is not part of the current support package
- it is not required by `citrees-exp`, the experiment pipeline, or paper tests

Keep that rule conservative before submission. If a script is upstream
provenance for a locked table, keep it in place for now.

## Keep In Place

These are still on the live paper path and should not be backlogged yet:

- `paper/scripts/analysis/build_dataset_characteristics_table.py`
- `paper/scripts/analysis/build_benchmark_package_tables.py`
- `paper/scripts/analysis/build_benchmark_heterogeneity_tables.py`
- `paper/scripts/analysis/build_high_p_saturation_tables.py`
- `paper/scripts/analysis/build_top_ranking_tables.py`
- `paper/scripts/analysis/build_synthetic_topk_tables.py`
- `paper/scripts/analysis/build_knob_ablation_summary_tables.py`
- `paper/scripts/analysis/build_threshold_ablation_summary_tables.py`
- `paper/scripts/analysis/build_manuscript_summary_tables.py`
- `paper/scripts/analysis/build_mechanism_summary_tables.py`
- `paper/scripts/analysis/build_fixed_panel_omnibus_table.py`
- `paper/scripts/analysis/build_fixed_panel_pairwise_ci_table.py`
- `paper/scripts/analysis/build_lodo_config_sensitivity_tables.py`
- `paper/scripts/analysis/build_cif_mechanism_ablation_tables.py`
- `paper/scripts/analysis/analyze_synthetic_ground_truth.py`
- `paper/scripts/analysis/aggregate_pipeline_artifacts.py`
- `paper/scripts/analysis/fig_benchmark_k_trajectory.py`
- `paper/scripts/analysis/fig_synthetic_topk_focus_curves.py`
- `paper/scripts/analysis/fig_mechanism_forest_feature_counts.py`
- `paper/scripts/analysis/run_mechanism_dimension_sweep_shard.py`
- `paper/scripts/experiments/mirrored_knob_ablation.py`
- `paper/scripts/experiments/threshold_search_ablation.py`
- `paper/scripts/experiments/cif_mechanism_ablation.py`
- `paper/scripts/theory/generate_calibration_support_package.py`
- `paper/scripts/theory/muting_power_theory.py`
- `paper/scripts/theory/study_batched_adaptive_stopping.py`
- all of `paper/scripts/api/`, `paper/scripts/cli/`, `paper/scripts/config/`,
  `paper/scripts/pipeline/`, `paper/scripts/adapters/`, `paper/scripts/utils/`,
  and the active EC2 helpers in `paper/scripts/infra/`

Notes:

- `build_knob_ablation_summary_tables.py` reads
  `paper/results/tables/mirrored_knob_ablation.csv`, so
  `paper/scripts/experiments/mirrored_knob_ablation.py` is still provenance for
  a locked paper-facing summary.
- `build_threshold_ablation_summary_tables.py` reads
  `paper/results/tables/threshold_search_ablation.csv`, so
  `paper/scripts/experiments/threshold_search_ablation.py` is also still live
  provenance.

## Backlog Now: Unused Experiment Runners

These do not appear on the current rebuild path or support-package list. They
now live under `paper/scripts/backlog/experiments/`:

- `paper/scripts/backlog/experiments/study_alpha_selector_sweep.py`
- `paper/scripts/backlog/experiments/study_bootstrap_feature_subsampling.py`
- `paper/scripts/backlog/experiments/study_multi_selector_correction.py`
- `paper/scripts/backlog/experiments/study_forest_size_sweep.py`
- `paper/scripts/backlog/experiments/study_noise_feature_robustness.py`
- `paper/scripts/backlog/experiments/study_legacy_optimization_ablation.py`
- `paper/scripts/backlog/experiments/study_ptest_power_grid.py`
- `paper/scripts/backlog/experiments/study_real_data_ablation.py`
- `paper/scripts/backlog/experiments/study_resamples_honesty_sweep.py`
- `paper/scripts/backlog/experiments/study_sample_size_sweep.py`
- `paper/scripts/backlog/experiments/study_runtime_scaling.py`
- `paper/scripts/backlog/experiments/study_selector_strictness_continuum.py`

These are worth keeping as backlog rather than deleting because they still
encode study design and raw-result provenance, even if the current paper no
longer packages their outputs.

### Second-Pass Review

This second pass reviewed the experiment backlog against current paper docs,
paper-facing rebuild scripts, and downstream consumers under `paper/scripts`.

| Current script | Verdict | Why it is not live now | Better archive name |
| --- | --- | --- | --- |
| `alpha_sweep.py` | backlog | only feeds `figures_ablation.py`, which is itself backlog-only | `study_alpha_selector_sweep.py` |
| `bootstrap_vs_subsampling.py` | backlog | dedicated side study, not in closed rebuild path or support package | `study_bootstrap_feature_subsampling.py` |
| `max_t_selector.py` | backlog | synthetic-only selector study, not cited by current paper package | `study_multi_selector_correction.py` |
| `n_estimators_sweep.py` | backlog | only feeds backlog ablation figures, not current manuscript tables | `study_forest_size_sweep.py` |
| `noise_robustness.py` | backlog | synthetic robustness study outside current support package | `study_noise_feature_robustness.py` |
| `optimization_ablation.py` | backlog | superseded in practice by the mirrored-knob and threshold ablation package | `study_legacy_optimization_ablation.py` |
| `power_analysis.py` | backlog | duplicated by the newer calibration support path in `paper/scripts/theory/generate_calibration_support_package.py`; only downstream consumer now is backlog-only `figures_ablation.py` | `study_ptest_power_grid.py` |
| `real_dataset_ablation.py` | backlog | earlier real-data ablation surface, superseded by benchmark package + mirrored knob summaries | `study_real_data_ablation.py` |
| `resamples_and_honesty.py` | backlog | side sweep not in support package and only used by backlog figures | `study_resamples_honesty_sweep.py` |
| `sample_size_curves.py` | backlog | exploratory sample-size study, not packaged for the current paper | `study_sample_size_sweep.py` |
| `scaling_curves.py` | backlog | runtime wall-clock study is explicitly outside paper-grade cross-method claims | `study_runtime_scaling.py` |
| `strictness_continuum.py` | backlog | exploratory continuum study, not part of the closed manuscript layer | `study_selector_strictness_continuum.py` |

Practical conclusion:

- `power_analysis.py` should stay on the backlog list.
- none of the scripts above should be deleted before submission
- the unused experiment runners have now been moved into
  `paper/scripts/backlog/experiments/`

## Archived Non-Experiment Helpers

The exploratory analysis/story layer has now been moved into
`paper/scripts/backlog/analysis/`:

- `paper/scripts/backlog/analysis/study_cif_config_sensitivity.py`
- `paper/scripts/backlog/analysis/fig_boundary_story_compact.py`
- `paper/scripts/backlog/analysis/fig_historical_cif_vs_r.py`
- `paper/scripts/backlog/analysis/fig_dataset_winner_heatmap.py`
- `paper/scripts/backlog/analysis/fig_exploratory_effective_rank.py`
- `paper/scripts/backlog/analysis/fig_exploratory_feature_agreement.py`
- `paper/scripts/backlog/analysis/fig_fixed_panel_summary.py`
- `paper/scripts/backlog/analysis/fig_feature_selection_vs_endpoint.py`
- `paper/scripts/backlog/analysis/fig_synthetic_precision_retention.py`
- `paper/scripts/backlog/analysis/fig_stability_accuracy_scatter.py`
- `paper/scripts/backlog/analysis/fig_ablation_suite.py`
- `paper/scripts/backlog/analysis/fig_benchmark_exploration_suite.py`
- `paper/scripts/backlog/analysis/fig_critical_difference_diagrams.py`

These are intentionally kept, but separated from the live manuscript builders.

## Maintenance Helpers

Operational audits and one-off repair scripts now live under
`paper/scripts/maintenance/`:

- `paper/scripts/maintenance/audit_dataset_k_trajectories.py`
- `paper/scripts/maintenance/audit_grid_artifacts.py`
- `paper/scripts/maintenance/audit_hash_alias_manifest.py`
- `paper/scripts/maintenance/repair_hash_alias_canonicalization.py`
- `paper/scripts/maintenance/repair_clf_pi_cpi_balanced_accuracy.py`
- `paper/scripts/maintenance/run_dt_rt_ranked_feature_check.py`

These are not on the current submission path, but they are still useful for
artifact hygiene and should not be deleted casually.

## Archived Theory And Wrappers

The superseded theory layer now lives under `paper/scripts/backlog/theory/`:

- `paper/scripts/backlog/theory/generate_fixed_b_pvalue_calibration.py`
- `paper/scripts/backlog/theory/generate_selection_bias_demo.py`
- `paper/scripts/backlog/theory/generate_sequential_stopping_calibration.py`
- `paper/scripts/backlog/theory/generate_muting_power_figures.py`
- `paper/scripts/backlog/theory/study_muting_power_gap.py`
- `paper/scripts/backlog/theory/study_sequential_stopping_analysis.py`
- `paper/scripts/backlog/theory/study_sequential_stopping_comparison.py`
- `paper/scripts/backlog/theory/study_supermartingale_check.py`

The leftover wrapper layer now lives under backlog namespaces as well:

- `paper/scripts/backlog/cli/run_legacy_cli.py`
- `paper/scripts/backlog/data_generation/generate_synthetic_datasets.py`
- `paper/scripts/backlog/infra/launch_legacy_ec2_batch_job.py`

## Naming And Structure Debt

The current paper tree mixes four different script roles in one flat namespace:
paper-facing builders, exploratory figures, audits and repairs, and archived
study runners.

The main structural cleanup is now in place:

1. Keep the closed rebuild path where it is.
2. Keep unused study runners in `paper/scripts/backlog/experiments/`.
3. Keep unused exploratory figures in `paper/scripts/backlog/analysis/`.
4. Keep one-off audits and repair scripts in `paper/scripts/maintenance/`.
5. Keep superseded theory generators in `paper/scripts/backlog/theory/`.

Suggested naming cleanup:

- reserve `build_*` for table-producing paper-facing scripts
- reserve `fig_*` for figure-producing paper-facing scripts
- reserve `study_*` for experiment runners that generate raw study outputs
- use `audit_*` for read-only verification tools
- use `repair_*` for one-off result mutation scripts
- avoid mixing `fig_*` and `figures_*` for the same job class
- rename `generate_fixedB_pvalue_calibration.py` to
  `generate_fixed_b_pvalue_calibration.py` if it survives the backlog move

Current live-path naming:

| Live script | Status |
| --- | --- |
| `build_dataset_characteristics_table.py` | renamed |
| `build_benchmark_package_tables.py` | renamed |
| `build_benchmark_heterogeneity_tables.py` | renamed |
| `build_high_p_saturation_tables.py` | renamed |
| `build_top_ranking_tables.py` | renamed |
| `build_synthetic_topk_tables.py` | renamed |
| `build_knob_ablation_summary_tables.py` | renamed |
| `build_threshold_ablation_summary_tables.py` | renamed |
| `build_manuscript_summary_tables.py` | renamed |
| `build_mechanism_summary_tables.py` | renamed |
| `build_fixed_panel_pairwise_ci_table.py` | renamed |
| `aggregate_pipeline_artifacts.py` | renamed |
| `analyze_synthetic_ground_truth.py` | renamed |
| `run_mechanism_dimension_sweep_shard.py` | renamed |
| `fig_mechanism_forest_feature_counts.py` | renamed |

Recommended structural cleanup after submission:

```text
paper/scripts/
  analysis/
  backlog/
    analysis/
    cli/
    data_generation/
    experiments/
    infra/
    theory/
  maintenance/
  theory/
```

That split would separate:

- manuscript-facing build scripts
- exploratory and archived study runners
- one-off audit and repair utilities
- theory generators that still support the paper versus superseded theory work

## Post-Submission Move Order

Remaining low-risk cleanup order:

1. Decide whether any archived backlog scripts can be deleted outright.
2. If desired, split the live `analysis/` namespace further into `build/` and
   `figures/`.
3. Only then consider renaming live paper-facing builders.
