# Tables Manifest

This directory mixes current paper-facing tables with older exploratory and
historical artifacts. Do not treat every file here as manuscript authority.

## Canonical Paper-Facing Tables

These are the tables intended for current manuscript and support-doc use:

- `dataset_characteristics.csv`
- `paper_benchmark_best_configs.csv`
- `paper_benchmark_selected_config_details.csv`
- `paper_benchmark_method_aggregate.csv`
- `paper_benchmark_stratified.csv`
- `paper_benchmark_complete_case_membership.csv`
- `paper_benchmark_sensitivity_support.csv` - dataset counts for the main, learner-by-k, matched-rank, and seed-sensitivity benchmark checks
- `paper_benchmark_downstream_sensitivity.csv` - all-method downstream-learner sensitivity summary used by the appendix
- `paper_benchmark_learner_k_sensitivity.csv` - all-method learner-by-k sensitivity summary used by the appendix
- `paper_benchmark_seed_sensitivity.csv` - all-method seed sensitivity summary used by the appendix
- `paper_benchmark_seed_complete_membership.csv` - dataset membership for the shared-support seed sensitivity checks
- `paper_benchmark_fixed_panel_membership.csv` - membership for the 14-dataset benchmark
- `paper_benchmark_fixed_panel_aggregate.csv` - aggregate summary for the 14-dataset benchmark
- `paper_benchmark_fixed_panel_pairwise_ci.csv` - matched-dataset uncertainty summaries for the 14-dataset benchmark
- `paper_benchmark_fixed_panel_omnibus.csv` - omnibus test summaries for the 14-dataset benchmark
- `paper_benchmark_lodo_selected_configs.csv` - leave-one-dataset-out selected configurations
- `paper_benchmark_lodo_aggregate.csv` - leave-one-dataset-out aggregate sensitivity summary
- `paper_benchmark_lodo_config_stability.csv` - leave-one-dataset-out selected-config stability summary
- `paper_benchmark_spread.csv`
- `k_trajectory_ranks.csv` - classification rank-by-k matrix used by the main-text trajectory figure
- `regression_k_trajectory_ranks.csv` - regression rank-by-k matrix used by the main-text trajectory figure
- `paper_benchmark_pairwise_aggregate.csv` - directed all-vs-all method comparisons; the `baseline` column is the compared method, not a restricted baseline set
- `paper_benchmark_pairwise_stratified.csv` - directed all-vs-all comparisons by downstream model and number of selected features; the `baseline` column is the compared method
- `paper_presentation_benchmark_summary.csv`
- `paper_heterogeneity_method_summary.csv`
- `paper_heterogeneity_cif_pairwise_breadth.csv`
- `paper_high_p_delta_vs_endpoint_overall.csv`
- `paper_high_p_cif_endpoint_summary.csv`
- `paper_high_p_cif_best_observed_k_summary.csv`
- `paper_high_p_endpoint_aggregate.csv`
- `paper_high_p_delta_vs_endpoint_cells.csv`
- `paper_high_p_endpoint_method_presence.csv`
- `paper_high_p_endpoint_pairwise.csv`
- `paper_high_p_endpoint_spread.csv`
- `top_ranking_best_configs.csv`
- `top_ranking_best_config_details.csv`
- `top_ranking_curve_summary.csv`
- `top_ranking_summary.csv`
- `top_ranking_by_dataset_type.csv`
- `synthetic_topk_best_configs.csv`
- `synthetic_topk_best_config_details.csv`
- `synthetic_topk_composition_summary.csv`
- `synthetic_topk_composition_curve_summary.csv`
- `synthetic_topk_composition_by_dataset_type.csv`
- `synthetic_topk_composition_curve_by_dataset_type.csv`
- `synthetic_topk_composition_by_dataset.csv`
- `paper_mirrored_knob_ablation_summary.csv`
- `paper_threshold_ablation_summary.csv`
- `paper_cit_runtime_ablation_summary.csv`
- `paper_presentation_practical_controls_summary.csv`
- `paper_mechanism_candidate_set_summary.csv`
- `paper_mechanism_frequency_summary.csv`
- `paper_mechanism_grid_combined_aggregate_summary.csv`
- `paper_mechanism_grid_tree_classification_aggregate_summary.csv`
- `paper_mechanism_grid_tree_regression_aggregate_summary.csv`
- `paper_mechanism_grid_forest_classification_aggregate_summary_1000trees.csv`
- `paper_mechanism_grid_forest_regression_aggregate_summary_1000trees.csv`
- `paper_mechanism_grid_forest_classification_cif_vs_cif_all_deltas_1000trees.csv`
- `paper_mechanism_grid_forest_regression_cif_vs_cif_all_deltas_1000trees.csv`
- `paper_mechanism_grid_tree_classification_display_slice_ranking.csv`
- `paper_mechanism_grid_tree_regression_display_slice_ranking.csv`
- `paper_mechanism_grid_forest_classification_display_slice_ranking_1000trees.csv`
- `paper_mechanism_grid_forest_regression_display_slice_ranking_1000trees.csv`
- `paper_mechanism_grid_tree_classification_feature_counts.csv`
- `paper_mechanism_grid_tree_regression_feature_counts.csv`
- `paper_mechanism_grid_forest_classification_feature_counts.csv`
- `paper_mechanism_grid_forest_regression_feature_counts.csv`

## Supporting-Only (Refreshed)

These remain outside the closed main-text table package:

- `calibration_summary.csv`
- `cit_runtime_ablation_raw.csv`
- `cit_runtime_ablation_dataset_summary.csv`

## Superseded Or Supporting-Only Examples

These files may be informative, but they should not be cited as the current
paper contract without an explicit re-justification:

- `benchmark_contract_*.csv`
- `clf_ranking_balanced_accuracy*`
- `reg_ranking_r2*`
- `synthetic_ranking_precision_at_10*`
- `friedman_synthetic*`
- `dataset_trajectory_summary.csv`
- `paper_heterogeneity_dataset_profile.csv`
  - superseded by the more specific `paper_heterogeneity_cif_dataset_profile.csv`
- `paper_heterogeneity_cif_pairwise_summary.csv`
  - superseded by `paper_heterogeneity_cif_pairwise_breadth.csv`
- `paper_mechanism_candidate_set_runs.csv`
  - useful for traceability, but the candidate-set sweep is tiny and only
    supporting

## Historical Audit Outputs

These files audit upstream grids or historical artifacts rather than define the
current packaged paper benchmark:

- `grid_artifact_audit_summary.json`
- `grid_config_audit.csv`
- `grid_dataset_audit.csv`
- `grid_hash_aliases.csv`
- `grid_hash_aliases_summary.json`
- `hash_alias_canonicalization_actions.csv`
- `hash_alias_canonicalization_summary.json`

## Default Rule

If a table is not listed under "Canonical Paper-Facing Tables" above, treat it
as exploratory, historical, or supporting-only until a paper-facing doc says
otherwise.
