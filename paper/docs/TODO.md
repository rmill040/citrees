# Paper TODO

This note tracks result layers that the current manuscript is not surfacing well.
The main issue is not lack of analysis. The repo already contains much more
benchmark detail than the paper shows. The manuscript currently over-compresses
that detail into a small number of pooled point estimates.

## Committee-Prioritized Execution Order

These items reflect the converged committee vote and are the ordered execution
plan for the next paper pass.

### Phase 0: Rewrite Section 4 Before Adding More Results

- Major rewrite of Section 4 (`paper/arxiv/sections/04_experiments.tex`).
  The current methods / experimental-design section is too sparse for the
  amount of pooling and sensitivity logic the paper now needs to explain.
  Before adding new benchmark and boundary displays, Section 4 should make the
  following explicit:
  - rankings are built on the training fold only
  - downstream evaluation uses the matching held-out fold
  - folds and seeds are averaged within a fixed
    `(dataset, downstream model, k)` cell
  - pooled benchmark summaries then average within dataset over downstream
    models and supported standard budgets
  - the pooled benchmark is complemented by a downstream x k sensitivity layer
  - the 14-dataset classification benchmark is a complete-coverage credibility
    surface
  - config selection is global within task, with leave-one-dataset-out used as
    a sensitivity check rather than full nested config selection

### Phase 1: Main-Text Additions

- Add one main-text classification downstream x k sensitivity display from
  `paper/results/tables/paper_benchmark_pairwise_stratified.csv`.
  This is the first result-layer addition after the Section 4 rewrite. It is
  the clearest fix for the current over-pooled benchmark story and should focus
  on CIF vs `r_ctree`, `r_cforest`, and CIT.

- Add a downstream-stratified high-p boundary display from
  `paper/results/tables/paper_high_p_cif_endpoint_summary.csv` and
  `paper/results/tables/paper_high_p_cif_best_observed_k_summary.csv`, or use
  `paper/results/figures/boundary_story.png` after cleanup.
  This should make the classification boundary legible by downstream learner,
  not only in pooled counts.

- Replace the tiny CIF breadth table with a stronger heterogeneity /
  consistency summary using
  `paper/results/tables/paper_heterogeneity_method_summary.csv` and
  `paper/results/tables/paper_heterogeneity_cif_pairwise_breadth.csv`.
  The goal is a robustness layer that supports “worth using and broadly
  stable,” not “best everywhere.”

### Phase 2: Appendix Support Package

- Add an appendix stability package:
  - `paper/results/tables/paper_benchmark_fixed_panel_pairwise_ci.csv`
  - `paper/results/tables/paper_benchmark_fixed_panel_omnibus.csv`
  - `paper/results/tables/paper_benchmark_lodo_aggregate.csv`
  - `paper/results/tables/paper_benchmark_lodo_config_stability.csv`
  This package should back the real-data validation story without turning the
  main text into a second benchmark paper.

- Add stronger classification mechanism scaling support in the appendix,
  ideally from
  `paper/results/figures/paper_mechanism_grid_forest_classification_dimension_curves_1000trees.png`
  plus the matching forest mechanism tables. This should broaden the
  candidate-exposure story beyond the current single sparse-case snapshot.

## Immediate priorities

- Explain the evaluation layout more explicitly in Section 4.
  The code uses fold-wise cross-fitted ranking/evaluation:
  - Stage 1 builds rankings on the training fold only.
  - Stage 2 evaluates the top-k set on the matching held-out fold.
  What is not yet explicit in the paper is that config selection is global within
  task, not fully nested per dataset. The current leave-one-dataset-out analysis
  is a sensitivity check for that choice, not a replacement for nested config selection.

- Add a visible downstream-model sensitivity layer for the real benchmark.
  Right now the main benchmark tables average over downstream models and standard
  feature budgets before presenting the headline result. That is too pooled for
  the strength of the current claims.
  Existing artifacts that can support this:
  - `paper/results/tables/paper_benchmark_stratified.csv`
  - `paper/results/tables/paper_benchmark_extended_stratified.csv`
  - `paper/results/tables/paper_benchmark_pairwise_stratified.csv`
  Recommended manuscript action:
  - add one main-text downstream-model x k sensitivity display, ideally a heatmap
    or compact matrix for CIF vs `r_ctree`, `r_cforest`, and CIT
  - move the full stratified table to the appendix

- Add a visible statement about what is being averaged.
  Section 4 currently says that scores are averaged across folds, seeds,
  downstream models, and available k values, but the paper does not explain why
  that pooling is acceptable or where the reader can inspect the stratified surface.
  The text should explicitly distinguish:
  - fold/seed averaging within a fixed dataset x downstream x k cell
  - downstream/k averaging within a dataset
  - dataset averaging across the benchmark surface

- Surface seed/fold variability, or explicitly state that it is not driving the benchmark.
  Raw evaluation outputs retain `seed` and `fold_idx`, but the paper-facing
  benchmark summaries collapse them immediately.
  Existing related artifacts:
  - `paper/results/clf_evaluation.parquet`
  - `paper/results/reg_evaluation.parquet`
  - `paper/results/tables/variance_decomposition.csv`
  - `paper/results/tables/exploratory_variance_decomposition.csv`
  Current status:
  - we do not have a paper-facing seed-variance summary in the manuscript
  - we do have exploratory variance decomposition suggesting seed contributes
    very little on the pooled benchmark surface
  Recommended manuscript action:
  - either build a small appendix summary for seed/fold spread by method family
    and task, or explicitly note that the benchmark contracts average over
    repeated seeds/folds and that a variance decomposition check finds dataset
    and k dominate the variation

- Replace ugly summary tables with the right object for the job.
  Several current tables are just prose or tiny summaries forced into tabular form.
  The fix should not be "shrink the font harder."
  Recommended direction:
  - keep one compact top-line benchmark table in main text
  - add one sensitivity display for downstream x k heterogeneity
  - convert tiny summary tables into prose if they do not need grid structure
  - move detailed stratified tables to appendix

## Existing result layers we are barely using

### Real-data benchmark

- `paper_benchmark_method_aggregate.csv`
  Already used only indirectly through the pooled benchmark summary.

- `paper_benchmark_stratified.csv`
  Mean score and mean rank by downstream model and k on the complete-case surface.
  This is the cleanest direct answer to "how robust is the headline ranking across LR/SVM/KNN?"

- `paper_benchmark_extended_stratified.csv`
  Same idea, but over all observed k values rather than only the standard set.
  Useful if the paper wants to compare the standard benchmark surface against the
  extended observed surface.

- `paper_benchmark_pairwise_stratified.csv`
  CIF-vs-baseline deltas by downstream model and k. This is probably the best
  source for a main-text sensitivity heatmap.

- `paper_benchmark_pairwise_aggregate.csv`
  Pooled CIF-vs-baseline deltas over all downstreams and supported standard k.
  Useful for benchmark summary prose, but currently overused relative to the
  stratified artifacts.

- `paper_benchmark_spread.csv`
  Cross-method spread by k and pooled over all standard k. This could help justify
  why some pooled rank gaps are meaningful or narrow.

### 14-dataset benchmark and config-selection sensitivity

- `paper_benchmark_fixed_panel_pairwise_ci.csv`
  We currently cite only a few confidence intervals from this file in prose.
  The full table is richer and could be surfaced in the appendix.

- `paper_benchmark_fixed_panel_omnibus.csv`
  Already used partly in prose, but not shown directly.

- `paper_benchmark_lodo_aggregate.csv`
  Leave-one-dataset-out performance summary across the 14-dataset benchmark.
  This is useful to show that global config selection is not causing the CIF result.

- `paper_benchmark_lodo_config_stability.csv`
  Stronger evidence for config-selection stability than the manuscript currently shows.
  For classification, CIF is reselected on all 14 held-out datasets.

- `paper_benchmark_lodo_selected_configs.csv`
  Detailed held-out config choices. Probably appendix-only, but currently unused.

### Benchmark heterogeneity

- `paper_heterogeneity_method_summary.csv`
  Contains top-1 share, top-3 share, top-half share, and best/worst dataset positions.
  This is much better than the current tiny breadth table if we want to discuss
  consistency without flattening everything into one sentence.

- `paper_heterogeneity_cif_pairwise_by_dataset.csv`
  Dataset-level CIF-vs-baseline deltas. Good source for a compact appendix table
  or a supporting figure showing where CIF wins are concentrated.

- `paper_heterogeneity_cif_pairwise_breadth.csv`
  Already partially used in the manuscript, but only as a very compressed summary.

- `paper_heterogeneity_cif_dataset_profile.csv`
  Candidate source for a "where CIF helps most" appendix subsection.

### High-p boundary diagnostics

- `paper_high_p_cif_endpoint_summary.csv`
  Stratifies the endpoint-vs-k100 story by downstream model. The manuscript currently
  reports only pooled counts.

- `paper_high_p_cif_best_observed_k_summary.csv`
  Shows the first-best-k distribution by downstream model. This is stronger than
  the current table because it reveals whether the intermediate-budget story is
  concentrated in one downstream learner.

- `paper_high_p_endpoint_pairwise.csv`
  Compares CIF to baselines specifically at the endpoint. Useful if the paper
  wants to make the "using all features usually does not help" statement more precise.

- `paper_high_p_endpoint_spread.csv`
  Quantifies cross-method spread at `k=100` versus endpoint. Could support the
  claim that endpoint performance compresses or destabilizes.

- `paper_high_p_endpoint_inventory.csv`
  Support accounting for the endpoint surface. Useful if the high-p subsection
  needs cleaner methodological bookkeeping.

- `paper_high_p_cif_endpoint_examples.csv`
- `paper_high_p_cif_best_observed_k_examples.csv`
  Useful for appendix examples if we want concrete dataset/downstream cases.

### Synthetic ranking diagnostics

- `synthetic_topk_composition_by_dataset_type.csv`
  Breaks the top-k composition story out by synthetic family, not just pooled task.

- `synthetic_topk_composition_curve_by_dataset_type.csv`
  Natural source for a stratified synthetic figure if the paper wants to show
  which synthetic families drive the subset-construction story.

- `top_ranking_by_dataset_type.csv`
  Breaks top-1 and precision-style metrics out by synthetic family. This would
  make the "subset construction vs exact top-of-list recovery" contrast more informative.

- `top_ranking_curve_summary.csv`
  Useful if the manuscript wants a curve-based top-ranking story rather than
  only the current compact head-of-list summary.

### Mechanism studies

- `paper_mechanism_candidate_set_summary.csv`
  Small but useful supporting table for the candidate-set exposure story.

- `paper_mechanism_frequency_summary.csv`
  Single-tree selection frequency summary. Helpful if the paper wants the
  "Stage A works when all features are available" point to be more explicit.

- `paper_mechanism_grid_combined_aggregate_summary.csv`
  Broader summary over the mechanism grid. Currently not surfaced.

- `paper_mechanism_grid_forest_classification_aggregate_summary_1000trees.csv`
- `paper_mechanism_grid_forest_regression_aggregate_summary_1000trees.csv`
- `paper_mechanism_grid_forest_classification_cif_vs_cif_all_deltas_1000trees.csv`
- `paper_mechanism_grid_forest_regression_cif_vs_cif_all_deltas_1000trees.csv`
  These provide a much broader backing for the mechanism section than the single
  forest feature-count figure currently shown.

## Existing figure assets we are not using

The current manuscript uses only:
- `k_trajectory.png`
- `synthetic_topk_focus_curves.png`
- `paper_mechanism_grid_forest_classification_feature_counts_p1000_i2_1000trees.png`

Potentially useful unused figure assets:

- `paper/results/figures/dataset_winners_heatmap.png`
  Candidate replacement for one of the ugly breadth-style tables.

- `paper/results/figures/fixed_panel_story.png`
  Candidate appendix figure for the 14-dataset benchmark surface.

- `paper/results/figures/boundary_story.png`
  Candidate appendix figure if the boundary section needs a visual summary.

- `paper/results/figures/clf_cd_balanced_accuracy_clean.png`
- `paper/results/figures/reg_cd_r2_clean.png`
  Candidate appendix critical-difference diagrams.

- `paper/results/figures/paper_mechanism_grid_forest_classification_dimension_curves_1000trees.png`
- `paper/results/figures/paper_mechanism_grid_forest_regression_dimension_curves_1000trees.png`
- `paper/results/figures/paper_mechanism_grid_forest_classification_density_curves_1000trees.png`
- `paper/results/figures/paper_mechanism_grid_forest_regression_density_curves_1000trees.png`
  These may be stronger than the single current mechanism plot if the goal is
  to show how the candidate-set limit scales with p or signal density.

- `paper/results/figures/synthetic_dgp_heatmap.png`
  Candidate appendix synthetic-family summary.

## New analysis that may still need to be built

- A paper-facing seed/fold variability table or figure.
  We currently have raw `seed` and `fold_idx` columns plus exploratory variance
  decomposition, but no manuscript-ready summary of repeatability within a fixed
  dataset x downstream x k cell.

- A main-text heterogeneity visualization for CIF vs historical baselines across
  downstream model and feature budget.
  The data already exists in `paper_benchmark_pairwise_stratified.csv`, but the
  current paper does not visualize it.

- A cleaner appendix benchmark package that mirrors the pooled main-text claims
  with stratified tables:
  - by downstream model
  - by k
  - by dataset

## Suggested paper structure changes

- Main text
  - keep one compact pooled benchmark summary table
  - add one sensitivity heatmap or compact matrix based on
    `paper_benchmark_pairwise_stratified.csv`
  - keep the current k-trajectory figure
  - keep the synthetic top-k figure
  - keep one mechanism figure, but consider whether a dimension-curve figure is
    stronger than the current count plot

- Appendix
  - full stratified benchmark table
  - full 14-dataset pairwise CI table
  - leave-one-dataset-out aggregate and config-stability tables
  - high-p stratified summaries by downstream model
  - optional synthetic-family stratification table/figure

## Short version

The paper is not short on results. It is short on visible stratification.
The immediate fix is to stop letting pooled point estimates carry the entire
benchmark story by themselves.
