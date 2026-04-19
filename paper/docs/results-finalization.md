# Results Finalization

This document locks the paper-facing results surface before narrative
rewriting. It inventories what each experiment is, what it supports, and what
it does not justify. If a claim is not supported here, it is not ready for the
paper.

## 1. Canonical Sources

Locked source of truth for the closed layers in this document:

- `paper/results/tables/dataset_characteristics.csv`
- `paper/results/tables/paper_benchmark_best_configs.csv`
- `paper/results/tables/paper_benchmark_selected_config_details.csv`
- `paper/results/tables/paper_benchmark_method_aggregate.csv`
- `paper/results/tables/paper_benchmark_stratified.csv`
- `paper/results/tables/paper_benchmark_complete_case_membership.csv`
- `paper/results/tables/paper_benchmark_fixed_panel_membership.csv`
- `paper/results/tables/paper_benchmark_fixed_panel_aggregate.csv`
- `paper/results/tables/paper_benchmark_fixed_panel_pairwise_ci.csv`
- `paper/results/tables/paper_benchmark_fixed_panel_omnibus.csv`
- `paper/results/tables/paper_benchmark_lodo_selected_configs.csv`
- `paper/results/tables/paper_benchmark_lodo_aggregate.csv`
- `paper/results/tables/paper_benchmark_lodo_config_stability.csv`
- `paper/results/tables/paper_benchmark_spread.csv`
- `paper/results/tables/paper_presentation_benchmark_summary.csv`
- `paper/results/tables/paper_benchmark_pairwise_aggregate.csv`
- `paper/results/tables/paper_heterogeneity_method_summary.csv`
- `paper/results/tables/paper_heterogeneity_cif_pairwise_breadth.csv`
- `paper/results/tables/paper_high_p_delta_vs_endpoint_overall.csv`
- `paper/results/tables/paper_high_p_cif_endpoint_summary.csv`
- `paper/results/tables/paper_high_p_cif_best_observed_k_summary.csv`
- `paper/results/tables/paper_high_p_endpoint_aggregate.csv`
- `paper/results/tables/paper_high_p_delta_vs_endpoint_cells.csv`
- `paper/results/tables/paper_high_p_endpoint_method_presence.csv`
- `paper/results/tables/paper_high_p_endpoint_pairwise.csv`
- `paper/results/tables/paper_high_p_endpoint_spread.csv`
- `paper/results/tables/top_ranking_best_configs.csv`
- `paper/results/tables/top_ranking_best_config_details.csv`
- `paper/results/tables/top_ranking_curve_summary.csv`
- `paper/results/tables/top_ranking_summary.csv`
- `paper/results/tables/top_ranking_by_dataset_type.csv`
- `paper/results/tables/synthetic_topk_best_configs.csv`
- `paper/results/tables/synthetic_topk_best_config_details.csv`
- `paper/results/tables/synthetic_topk_composition_summary.csv`
- `paper/results/tables/synthetic_topk_composition_curve_summary.csv`
- `paper/results/tables/synthetic_topk_composition_by_dataset_type.csv`
- `paper/results/tables/synthetic_topk_composition_curve_by_dataset_type.csv`
- `paper/results/tables/synthetic_topk_composition_by_dataset.csv`
- `paper/results/tables/paper_mirrored_knob_ablation_summary.csv`
- `paper/results/tables/paper_threshold_ablation_summary.csv`
- `paper/results/tables/paper_presentation_practical_controls_summary.csv`
- `paper/results/tables/paper_mechanism_candidate_set_summary.csv`
- `paper/results/tables/paper_mechanism_frequency_summary.csv`
- `paper/results/tables/paper_mechanism_grid_combined_aggregate_summary.csv`
- `paper/results/tables/paper_mechanism_grid_tree_classification_aggregate_summary.csv`
- `paper/results/tables/paper_mechanism_grid_tree_regression_aggregate_summary.csv`
- `paper/results/tables/paper_mechanism_grid_forest_classification_aggregate_summary_1000trees.csv`
- `paper/results/tables/paper_mechanism_grid_forest_regression_aggregate_summary_1000trees.csv`
- `paper/results/tables/paper_mechanism_grid_forest_classification_cif_vs_cif_all_deltas_1000trees.csv`
- `paper/results/tables/paper_mechanism_grid_forest_regression_cif_vs_cif_all_deltas_1000trees.csv`
- `paper/docs/experiments.md` (for documented skipped cells / known exclusions)

Supporting-only source outside the closed main-text package:

- `paper/results/tables/calibration_summary.csv`

Important note:

- `paper/results/tables/calibration_summary.csv` remains appendix/supporting-only
  by design.
- file names and some table column names still use `endpoint`; paper-facing
  prose should prefer `full feature set`, `full-feature budget`, or `k=p`

## 2. Locked Experiment Inventory

### 2.1 Null diagnostics

What the experiment is:

- root-level Stage A checks under a complete nodewise permutation null,
- selection-bias demonstration,
- calibration table,
- and fixed-`B` power-resolution sanity checks.

What the refreshed table says:

- fixed-`B` empirical rejection rates in the refreshed artifact are
  `0.0409`, `0.0463`, `0.0438`, `0.0503`, and `0.0495` for MC at
  `B=49,99,199,499,999` with `10,000` null simulations per setting
- fixed-`B` PC at `B=199` is `0.0504` with `10,000` null simulations
- adaptive MC at `B=199` is `0.0476` with `10,000` null simulations
- CIT root split rates are `0.0120`, `0.0230`, `0.0290`, `0.0100`, and `0.0200`
  for `(n,p)=(200,10),(200,50),(200,100),(500,10),(500,100)` with `1,000`
  null simulations per condition
- with `B=49`, the minimum attainable p-value is `1/50 = 0.02`, so rejection at
  `alpha = 0.01` is impossible regardless of signal

Interpretation boundary:

- the selection-bias demonstration is a figure-level motivating object
- the refreshed calibration table is a supporting numeric null-diagnostic
  surface, not a main-text paper anchor

Paper-safe interpretation:

- the refreshed fixed-`B` Stage A calibration is broadly theorem-aligned on the
  current authored script
- the selection-bias demonstration is a qualitative motivating figure, not a
  separate numeric benchmark
- adaptive stopping can be shown descriptively, but not as a theorem-backed
  calibration statement
- the main practical fixed-`B` issue is resolution, not a different power-curve
  shape
- this layer should remain appendix/supporting-only in the master story even
  after refresh, because it supports the fixed-node theorem boundary rather
  than the central benchmark claim

Do not say:

- adaptive stopping is formally calibrated
- these diagnostics establish end-to-end tree or forest validity

### 2.2 Main real-data classification benchmark

What the experiment is:

- real-data rank-then-evaluate benchmark on `23` classification datasets
- standard budgets `k = 5, 10, 25, 50, 100`
- downstreams `lr`, `svm`, `knn`
- one best global config per method family within task

What the canonical aggregate says:

- CIF is `4th/15` by mean rank on the complete-case classification aggregate
- CIF mean rank is `5.4758`
- CIF mean score is `0.8194`
- the top three are `lgbm = 4.6750`, `xgb = 5.0386`, `cat = 5.2606`
- on the 14-dataset classification benchmark, CIF is `5th/15` with
  mean rank `5.6286`

What the CIF trajectory says:

- LR: `6.4545` at `k=5` to `4.6786` at `k=100`
- SVM: `5.2955` at `k=5` to `5.0000` at `k=100`
- KNN: `4.9773` at `k=5` to `4.0714` at `k=100`
- KNN is `1st` at `k=100`
- the complete-case spread surface tightens as `k` grows:
  mean cross-method range drops from `0.2560` at `k=5` to `0.1837` at `k=100`

Support accounting:

- complete-case support is `22, 21, 15, 15, 14` across `k = 5, 10, 25, 50, 100`
- the exact dataset membership behind those shrinking supports is now exposed in
  `paper_benchmark_complete_case_membership.csv`, so the 14-dataset benchmark is
  auditable even though it is still not the primary estimand
- `paper_benchmark_fixed_panel_membership.csv` and
  `paper_benchmark_fixed_panel_aggregate.csv` make the 14-dataset benchmark
  explicit rather than leaving it implicit in prose
- `paper_benchmark_fixed_panel_omnibus.csv` adds the 14-dataset Friedman
  summary, so the manuscript can cite one omnibus test on the same
  dataset-mean 14-dataset surface
- the leave-one-dataset-out config-selection summaries show that the
  classification CIF headline is not coming from unstable config choice:
  CIF remains essentially in the same position and the same CIF config is
  selected on all `14/14` held-out 14-dataset benchmark datasets
- `paper_benchmark_selected_config_details.csv` records each selected config's
  resolved parameters, candidate grid size, and runner-up gap within family

Important caveats:

- config selection is benchmark-internal family-level tuning, not an external
  meta-selection guarantee
- small amounts of documented skip logic affect support on a few
  high-dimensional cells, especially for `r_ctree`, `r_cforest`, and some
  CIT-RDC runs
- `dexter` is still retained for non-R methods; it only drops out of the
  stricter all-method complete-case layers because the R baselines are excluded
  there

Paper-safe interpretation:

- CIF is competitive, not the leader
- CIF often becomes more competitive as `k` grows, especially for LR and KNN
- this is a benchmark-level trajectory over changing support, not a 14-dataset
  same-dataset longitudinal claim

Do not say:

- CIF is the best overall classification method
- the `k` trajectory is a clean same-dataset time series

### 2.3 Classification breadth and heterogeneity

What the experiment is:

- paired CIF-vs-baseline dataset summaries over all supported downstream-model
  and standard-`k` cells
- plus dataset-level heterogeneity summaries

What the canonical pairwise aggregate says on the looser all-supported layer:

- CIF is positive against `10/14` classification baselines by mean delta
- strongest positive deltas:
  - `cpi +0.1285` on `23` datasets
  - `r_ctree +0.0876` on `22` datasets
  - `r_cforest +0.0715` on `22` datasets
  - `pi +0.0694` on `23` datasets
  - `ptest_rdc +0.0689` on `23` datasets
  - `boruta +0.0574` on `23` datasets
  - `ptest_mc +0.0428` on `23` datasets
  - `cit +0.0315` on `23` datasets
- near-zero or negative versus stronger ensembles:
  - `et +0.0059`
  - `rfe +0.0062`
  - `rf -0.0043`
  - `cat -0.0056`
  - `xgb -0.0097`
  - `lgbm -0.0114`

What the stricter `22`-dataset breadth layer says:

- CIF is positive against `10/14` classification baselines by mean delta
- strongest positive deltas:
  - `cpi +0.1305`, `20/22` wins
  - `r_ctree +0.0876`, `22/22` wins
  - `r_cforest +0.0715`, `19/22` wins
  - `ptest_rdc +0.0706`, `20/22` wins
  - `pi +0.0700`, `16/22` wins
  - `boruta +0.0615`, `16/22` wins
  - `ptest_mc +0.0453`, `21/22` wins
  - `cit +0.0301`, `21/22` wins
- near-zero or negative versus stronger ensembles:
  - `et +0.0063`, `13/22` wins
  - `rfe +0.0018`, `15/22` wins
  - `rf -0.0030`, `7/22` wins and `1` tie
  - `cat -0.0038`, `8/22` wins and `1` tie
  - `xgb -0.0086`, `11/22` wins and `1` tie
  - `lgbm -0.0106`, `9/22` wins and `1` tie

What the stricter heterogeneity layer says:

- CIF is top-half on `21/22` classification datasets
- CIF is top-3 on `7/22`
- CIF is top-1 on `3/22`
- dataset means are computed over each dataset's available all-method
  complete-case downstream-by-`k` cells, not over a perfectly rectangular
  `3 x 5` cell panel
- mean classification support at that layer is `11.86` cells per dataset, not
  `15`

Paper-safe interpretation:

- CIF is broadly reliable across datasets, but not a dominant winner
- its clearest broad wins are over historical or inference-style baselines
- the ensemble comparison is materially tighter

Do not say:

- CIF broadly beats the strongest ensemble baselines
- CIF is the dataset-by-dataset leader

### 2.4 High-p saturation and full-feature checks

What the experiment is:

- secondary high-`p` analysis on datasets with `p > 100`
- classification cohort size `15`, regression cohort size `6`
- full-feature completeness `14/15` in classification and `6/6` in regression
- some methods emit truncated ranking surfaces on certain high-dimensional
  datasets, so this layer should be read as a diagnostic of the executed
  ranking surface rather than a guarantee that every method contributes a
  distinct ordering all the way to `p`
- the all-method cell-level high-`p` surface and method-level endpoint
  presence are now exposed directly in
  `paper_high_p_delta_vs_endpoint_cells.csv` and
  `paper_high_p_endpoint_method_presence.csv`

What the canonical overall saturation table says:

- classification all-method mean `score_k - score_{k=p}`:
  - `-0.1974` at `k=5`
  - `-0.1275` at `k=10`
  - `-0.0605` at `k=25`
  - `-0.0281` at `k=50`
  - `-0.0062` at `k=100`
- regression all-method mean `score_k - score_{k=p}`:
  - `+0.0504` at `k=5`
  - `+0.0988` at `k=10`
  - `+0.0541` at `k=25`
  - `+0.0907` at `k=50`
  - `+0.0864` at `k=100`

What the CIF-specific full-feature summary says:

- classification mean `k=p` minus `k=100` score is `-0.0369`
- regression mean `k=p` minus `k=100` score is `-0.3084`
- classification score improves at `k=p` on `23/45` cells
- classification `k=p` matches the best observed score on `8/45` cells
- classification `k=p` is the first best observed budget on only `5/45` cells
- regression score improves at `k=p` on `5/18` cells
- regression `k=p` matches the best observed score on `2/18` cells
- regression `k=p` is the first best observed budget on only `2/18` cells

What the CIF best-observed-budget summary says:

- classification first best observed budget:
  - below `100`: `5/45`
  - at `100`: `5/45`
  - intermediate between `100` and `k=p`: `30/45`
  - full feature set: `5/45`
- regression first best observed budget:
  - below `100`: `9/18`
  - at `100`: `2/18`
  - intermediate: `5/18`
  - full feature set: `2/18`

What the full-feature-only aggregate says:

- CIF is `5th/15` on the full-feature-only classification aggregate
- CIF is `15th/16` on the full-feature-only regression aggregate
- in classification, the mean cross-method score range collapses from `0.1837`
  at `k=100` to `0.0393` at `k=p`
- full-feature pairwise comparisons are mostly ties in classification, with
  `r_cforest` the only clearly separated baseline
- full-feature pairwise comparisons in regression are too degenerate and
  small-support to support strong pairwise claims

What the full-feature pairwise surface says:

- in classification, full-feature CIF-vs-baseline deltas are mostly ties, with
  `r_cforest` the only clearly separated baseline
- in regression, the full-feature pairwise surface is almost entirely degenerate
  ties and should not carry comparative claims

Paper-safe interpretation:

- high-`p` is a saturation check, not a second main benchmark
- in classification, useful operating points often appear between `100` and
  `p`, not necessarily at `p`
- in regression, using all features often hurts
- full-feature-only ranks are too compressed or degenerate to be a strong
  comparative surface

Do not say:

- CIF needs all features to rank well
- full-feature-only aggregate ranks are a main result
- `23/45` means the full feature set is usually the best budget

### 2.5 Synthetic ground-truth recovery

What the experiment is:

- synthetic feature-recovery benchmark with known informative features
- separate config-selection rule chosen by mean informative recovery over
  `k = 5, 10, 25, 50, 100` with effective `k=min(k,p)`
- eight synthetic classification suites and eight synthetic regression suites

What the canonical summary says:

- the primary synthetic ranking surface is the across-`k` curve, not any single
  budget slice
- CIF classification mean informative recovery over `k = 5, 10, 25, 50, 100`
  is `0.3443`, rank `8/15`
- CIF regression mean informative recovery over the same budget curve is
  `0.3901`, rank `10/16`

What the CIF regime split says:

- classification favorable:
  - `toeplitz = 0.5280`
  - `bias = 0.5176`
  - `standard_easy = 0.5096`
- classification weak:
  - `standard_hard = 0.0484`
  - `confounder = 0.2798`
  - `weak_signal = 0.2003`
- regression favorable:
  - `toeplitz = 0.5336`
  - `standard_easy = 0.4918`
  - `weak_signal = 0.4864`
- regression harder:
  - `standard_hard = 0.3143`
  - `confounder = 0.3309`
  - `friedman = 0.3218`
  - `heteroscedastic = 0.3136`

Paper-safe interpretation:

- CIF is not a synthetic feature-discovery leader
- correlated structure is favorable
- weak signal and standard-hard classification are unfavorable on informative recovery
- redundancy needs a separate composition read because lower informative-only
  recovery can still hide proxy-heavy top-`k` sets

Do not say:

- synthetic recovery makes CIF a clear winner
- pooled `standard` is a homogeneous synthetic regime

### 2.6 Synthetic top-k feature composition

What the experiment is:

- synthetic top-`k` composition audit using the same one-best-config-per-family
  rule as the top-ranking diagnostics
- evaluates `k = 1, 2, 5, 10, 20, 25, 50, 100`
- this is a decomposition of the same selected-config synthetic ranking
  surface, not a third independent benchmark
- trend summaries should be read from the across-`k` curve over the standard
  budgets `5, 10, 25, 50, 100`; individual `k` rows are supporting detail only
- decomposes the returned top-`k` positions into:
  - informative features
  - redundant proxies
  - correlated-noise confounders
  - pure noise
  - missing / unreturned positions

Important audit note:

- for requested budgets above a dataset's total feature count, the composition
  is now evaluated on the effective budget `min(k, p)`
- dataset-size clipping is tracked separately through
  `dataset_size_cap_share`, so a 50-feature dataset at requested `k=100` is no
  longer misread as having `missing_share = 0.5`

What the overall focus comparison says:

- classification mean over `k = 5, 10, 25, 50, 100`:
  - CIF `informative_share = 0.3443`
  - CIF `signal_or_redundant_share = 0.4087`
  - CIF `pure_noise_share = 0.5297`
  - CIF `correlated_noise_share = 0.0616`
  - RF `0.3769 / 0.4433 / 0.4947 / 0.0620`
  - ET `0.3830 / 0.4510 / 0.4857 / 0.0633`
- regression mean over `k = 5, 10, 25, 50, 100`:
  - CIF `informative_share = 0.3901`
  - CIF `signal_or_redundant_share = 0.4535`
  - CIF `pure_noise_share = 0.4933`
  - CIF `correlated_noise_share = 0.0532`
  - RF `0.4064 / 0.4649 / 0.4892 / 0.0460`
  - ET `0.4109 / 0.4723 / 0.4794 / 0.0483`
- the trajectory itself also matters:
  - classification CIF informative share falls from `0.639` at `k=5` to
    `0.0964` at `k=100`, while pure-noise share rises from `0.208` to `0.8286`
  - regression CIF informative share falls from `0.764` at `k=5` to `0.1121`
    at `k=100`, while pure-noise share rises from `0.111` to `0.8138`

What the regime split adds:

- classification `confounder`:
  - CIF failures are mostly correlated-noise occupancy, not pure-noise occupancy
  - mean over standard `k`: CIF `informative_share = 0.2798`,
    `correlated_noise_share = 0.4930`
  - RF `0.3147 / 0.4280`, ET `0.3107 / 0.4257`
  - CIT is unusually better on this suite: `0.4795 / 0.2781`
- classification `redundant`:
  - CIF informative-only recovery is not leading (`0.3138`)
  - but `signal_or_redundant_share = 0.8285`, close to RF `0.8852` and
    ET `0.8942`
  - this is a proxy-heavy top-`k`, not a pure-noise failure
- classification `standard_hard` and `weak_signal`:
  - CIF top-`k` purity is genuinely weaker
  - mean pure-noise share on `standard_hard`: CIF `0.9516` vs RF `0.9021`,
    ET `0.8747`
  - mean pure-noise share on `weak_signal`: CIF `0.7997` vs RF `0.6078`,
    ET `0.5381`
- regression `redundant`:
  - CIF again fills top-`k` with informative-or-redundant features rather than
    pure noise
  - mean `signal_or_redundant_share = 0.8355`
  - informative-only share `0.3286` vs RF `0.3245`, ET `0.3189`
- regression `confounder`:
  - CIF is not especially robust to correlated-noise attraction
  - mean over standard `k`: CIF `0.3309 / 0.4257`
  - RF `0.3677 / 0.3710`, ET `0.3600 / 0.3636`, CIT `0.5268 / 0.2089`

Paper-safe interpretation:

- this does **not** support a blanket claim that permutation-based feature selection selects
  less noise than RF or ET
- CIF's synthetic weakness on classification is not just a top-1 issue; it also
  shows up in lower top-`k` purity on `standard_hard` and `weak_signal`
- redundancy is a different failure mode: CIF often returns useful proxies
  rather than pure noise
- confounder suites are dominated by correlated-noise attraction across methods;
  CIF is not uniquely robust there
- correlated-structure settings such as `toeplitz` remain comparatively
  favorable

Do not say:

- CIF generally selects less noise than RF or ET on these synthetics
- confounder failures are mainly pure-noise errors
- lower informative-only recovery on redundant suites means CIF is mostly
  choosing junk

### 2.7 Fixed-design mechanism diagnostics

What the experiment is:

- fixed-design mechanism studies
- not a new benchmark layer; these are diagnostic mechanism studies
- separates:
  - candidate-set coverage under different CIF `max_features` regimes
  - full-support feature-count behavior on simple fixed designs
- completed grid with `n=250` fixed, `p ∈ {100, 500, 1000}`,
  `n_informative ∈ {1, 2, 5, 10}`
- completed tree methods:
  - classification: `cit`, `dt`, `rt`
  - regression: `cit`, `dt`, `rt`
- completed forest methods:
  - classification: `cif`, `cif_all`, `rf`, `et`
  - regression: `cif`, `cif_all`, `rf`, `et`

What the candidate-set sweep says:

- fixed `p=100`, `n_informative=2` classification design with `n_seeds=2`
- CIF default forest-style `max_features="sqrt"` means 10 sampled features per
  node and root informative-feature coverage `0.1909`
- on that checked design:
  - `sqrt_bootstrap_1tree`: `top1_hit_rate = 0.0`, `mean_true_top2 = 0.0`
  - `all_bootstrap_1tree`: `top1_hit_rate = 1.0`, `mean_true_top2 = 1.0`
  - `all_no_bootstrap_1tree`: `top1_hit_rate = 1.0`, `mean_true_top2 = 1.0`
  - `sqrt_bootstrap_5tree`: `top1_hit_rate = 0.0`, `mean_true_top2 = 0.0`
  - `all_bootstrap_5tree`: `top1_hit_rate = 1.0`, `mean_true_top2 = 1.0`

What the completed fixed-`n=250` grids say:

- single-tree classification:
  - `cit` is perfectly pure over the completed grid:
    `informative_split_share = 1.000`,
    `mean_total_splits_per_fit = 2.217`,
    `distinct_false_features_used = 0.0`
  - `dt` is intermediate:
    `informative_split_share = 0.676`,
    `mean_total_splits_per_fit = 6.017`,
    `distinct_false_features_used = 10.33`
  - `rt` is the noise-spraying baseline:
    `informative_split_share = 0.061`,
    `mean_total_splits_per_fit = 62.28`,
    `distinct_false_features_used = 191.0`
  - `cit` stays at `informative_split_share = 1.000` for
    `n_informative = 1, 2, 5, 10`; it simply uses as many informative splits
    as the easy classification design needs
  - `dt` is clean only in the easiest one-signal case; by `n_informative=10`
    it falls to `0.429`
  - `rt` gets worse as `p` grows:
    `informative_split_share = 0.120` at `p=100`,
    `0.041` at `p=500`,
    `0.022` at `p=1000`
  - purity is not the same as covering every true feature:
    `cit` only uses about `60%` of the informative features on average,
    versus `0.708` coverage share for `dt` and `0.875` for `rt`
  - this should be read as sparse sufficient-set behavior, not exhaustive
    signal recovery

- single-tree regression:
  - `cit` remains very clean while continuing to split:
    `informative_split_share = 0.935`,
    `mean_total_splits_per_fit = 12.7`,
    `distinct_false_features_used = 0.67`
  - `dt` and `rt` both fully grow and spend almost all split mass on noise:
    `dt = 0.073`, `rt = 0.084`
  - `cit` degrades only modestly with width:
    `informative_split_share = 0.990` at `p=100`,
    `0.917` at `p=500`,
    `0.898` at `p=1000`
  - again, coverage and purity separate:
    `cit` covers about `77.5%` of the informative features on average,
    while `dt` and `rt` cover `96.7%` / `99.2%`
  - the latter is not a quality win by itself; it comes from much deeper trees
    that also spend most split mass on noise

- forest classification at `1000` trees:
  - `cif_all` is much cleaner than default `cif`:
    `informative_split_share = 0.917` vs `0.333`
  - the same contrast holds against CART-style forests:
    `rf = 0.084`, `et = 0.065`
  - false-feature spread tells the same story:
    `distinct_false_features_used = 93.4` for `cif_all`,
    `265.1` for `cif`,
    `528.8` for both `rf` and `et`
  - the `cif` degradation is strongest in sparse high-`p` settings:
    - `p=1000, i=1`: `cif = 0.099`, `cif_all = 1.000`
    - `p=1000, i=2`: `cif = 0.090`, `cif_all = 1.000`
    - `p=500, i=1`: `cif = 0.093`, `cif_all = 0.989`
  - averaged by width, `cif` falls sharply as `p` grows:
    `0.539` at `p=100`,
    `0.265` at `p=500`,
    `0.193` at `p=1000`
  - `cif_all` stays high over the same sweep:
    `0.954`, `0.899`, `0.898`
  - the gap narrows as the informative fraction rises, but it remains large:
    by `n_informative=10`, `cif = 0.507` and `cif_all = 0.808`
  - coverage alone is not a useful forest summary:
    `rf` and `et` touch essentially all informative features by `1000` trees,
    but still spend most split mass on noise
  - even `cif_all` can remain extremely pure while touching only one of two
    informative features on easy sparse designs, so the right read is again
    "clean sufficient subset" rather than "complete true-feature recovery"

- forest regression at `1000` trees:
  - `cif_all` improves informative split share over default `cif`:
    `0.423` vs `0.278`
  - both conditional-inference forests still beat CART-style forests on this
    purity metric:
    `rf = 0.077`, `et = 0.084`
  - unlike classification, the all-feature control is not a clean rescue:
    `cif_all` uses nearly the full false feature set
    (`distinct_false_features_used = 528.75`)
    while default `cif` already uses `367.67`
  - the regression delta surface is mixed:
    split-share gains are largest in sparse wide settings, but
    false-feature reduction is usually negative because `cif_all` overgrows

- fixed-design frequency studies remain useful as secondary context:
  - easy shuffled classification:
    `cit` / `cif_all` spread false top-`k` mass over only `9` false variables,
    versus `32` for `rf` / `et`
  - symmetric two-signal Gaussian:
    all methods recover both true features in the small-budget check, but
    `cit` / `cif_all` still spread false mass over only `8` / `10` false
    variables versus `26` / `29` for `rf` / `et`

Paper-safe interpretation:

- these studies are the closest thing we have to the original motivation:
  full-support counts over known ground-truth features
- the selection principle itself looks strong in trees:
  `cit` is extremely pure in classification and remains very clean in
  regression
- purity and coverage are distinct estimands here:
  noisy methods can touch more true features simply because they overgrow
- the main classification-forest failure mode is not
  "permutation-based feature selection is bad"; it is that sparse candidate sampling can
  bury the signal before feature selection acts
- the direct candidate-set sweep is suggestive rather than decisive because it
  is tiny (`n_seeds=2`)
- `cif_all` is a mechanism control, not an automatic recommendation that the
  production default should be `max_features=None`
- the concentration / diffusion studies still add useful context, but the
  completed fixed-`n=250` grids are the stronger mechanism evidence
- the forest story is task-dependent:
  - in classification, `cif_all` is much cleaner than default `cif`
  - in regression, `cif_all` still improves informative split share
    (`0.423` vs `0.278`), but it also overgrows badly and uses nearly the full
    false feature set
- they do **not** overturn the locked synthetic benchmark design

Do not say:

- the main synthetic benchmark is explained by `max_features`
- CIF generally selects less noise than RF or ET
- these fixed-design studies replace the benchmark evidence

Recommended displays:

- Figure: single-tree classification feature-count bar chart on
  `make_classification_n250_p1000_i10`
  - methods: `cit`, `dt`, `rt`
  - x-axis: feature index
  - y-axis: split count or tree-use count
  - highlight the 10 informative features
  - why: this is the cleanest "pure vs mixed vs spray-noise" panel

- Figure: forest classification feature-count bar chart on
  `make_classification_n250_p1000_i2` at `1000` trees
  - methods: `cif`, `cif_all`, `rf`, `et`
  - highlight the 2 informative features
  - why: this illustrates the candidate-set-width story under the same selection
    logic

- Figure: informative split share over `(p, n_informative)` for completed
  classification studies
  - panel A: `cit`, `dt`, `rt`
  - panel B: `cif`, `cif_all`, `rf`, `et` at `1000` trees
  - why: this separates the tree story from the forest operating-regime story

- Figure: informative split share over `(p, n_informative)` for completed
  regression trees
  - methods: `cit`, `dt`, `rt`
  - why: this shows that regression trees are not just repeating the
    classification pattern; `cit` keeps splitting while staying mostly on
    signal

- Figure: informative split share over `(p, n_informative)` for completed
  regression forests at `1000` trees
  - methods: `cif`, `cif_all`, `rf`, `et`
  - why: this shows the forest candidate-set story is weaker and messier in
    regression than in classification

- Table: aggregate mechanism summary for each completed study
  - columns: method, mean total splits, mean true split events, mean noise split
    events, informative split share, informative-feature coverage share,
    distinct false features used
  - why: purity and coverage should be shown together so "pure" is not
    misread as "recovers every true feature"

- Table: `cif` vs `cif_all` delta table at `1000` trees
  - columns: `p`, `n_informative`, `cif informative_split_share`,
    `cif_all informative_split_share`, share gain, false-feature reduction
  - why: this is the shortest compact table for the candidate-set-width
    mechanism

Canonical mechanism artifacts:

- aggregate summaries:
  - `paper_mechanism_grid_combined_aggregate_summary.csv`
  - `paper_mechanism_grid_tree_classification_aggregate_summary.csv`
  - `paper_mechanism_grid_tree_regression_aggregate_summary.csv`
  - `paper_mechanism_grid_forest_classification_aggregate_summary_1000trees.csv`
  - `paper_mechanism_grid_forest_regression_aggregate_summary_1000trees.csv`
  - `paper_mechanism_grid_forest_classification_cif_vs_cif_all_deltas_1000trees.csv`
  - `paper_mechanism_grid_forest_regression_cif_vs_cif_all_deltas_1000trees.csv`
- best display slices from the completed grids are:
  - tree classification: `make_classification_n250_p1000_i10`
  - forest classification: `make_classification_n250_p1000_i2` at `1000` trees
  - tree regression if needed: `make_regression_n250_p1000_i2`
  - forest regression is better shown through the aggregate table and trend
    curves than through a single feature-count main panel
- those slice choices are data-driven; they come from the display-slice ranking
  tables in `paper/results/tables/`

### 2.8 Top-ranking diagnostics

What the experiment is:

- synthetic top-of-ranking diagnostics where ground truth is known
- one best synthetic config per family, selected by mean informative recovery
  over `k = 5, 10, 25, 50, 100` with effective `k=min(k,p)`
- the supporting trend table here is now a head-of-list summary over
  `k = 1, 2`, not a duplicate of the standard `5, 10, 25, 50, 100`
  recovery curve

What the canonical summary says:

- CIF classification:
  - mean precision over `k = 1, 2`: `0.6500`, head-of-list rank `11/15`
  - mean precision over `k = 5, 10, 25, 50, 100`: `0.3443`, rank `8/15`
  - `top1_hit_rate = 0.640`
  - `any_hit_at_2_rate = 0.795`
  - `MRR = 0.7616`
  - `mean_first_true_rank = 3.085`
  - top1 rank position `11/15`
  - MRR rank position `10/15`
- CIF regression:
  - mean precision over `k = 1, 2`: `0.8163`, head-of-list rank `12/16`
  - mean precision over `k = 5, 10, 25, 50, 100`: `0.3901`, rank `10/16`
  - `top1_hit_rate = 0.810`
  - `any_hit_at_2_rate = 0.945`
  - `MRR = 0.8935`
  - `mean_first_true_rank = 1.275`
  - top1 rank position `13/16`
  - MRR rank position `11/16`

What the CIF regime split adds:

- classification top-hit strength on `toeplitz`, `bias`, `nonlinear`, and
  `standard_easy`
- classification top-hit weakness on `standard_hard`, `confounder`,
  `weak_signal`, and especially `redundant`
- regression is better at getting a true feature near the top, but still not a
  leading top-ranker

Paper-safe interpretation:

- CIF is better at recovering multiple useful features than at exact
  top-of-ranking discovery

Do not say:

- CIF is a strong exact top-of-ranking method

### 2.9 Practical knob evidence

What the experiment is:

- mirrored runtime-plus-quality proxy ablations for the practical controls
  actually used in the benchmark
- targeted threshold-search ablation
- in the locked CIT/CIF benchmark grid, only selector family and honesty varied;
  the other practical controls were fixed by design
- the real-data side of this layer uses a small proxy panel rather than the
  full canonical benchmark surface

What the canonical mirrored ablation says:

- disabling adaptive stopping is:
  - `11.4734x` slower on real classification
  - `6.2215x` slower on synthetic classification
  - `11.6918x` slower on real regression
  - `5.1000x` slower on synthetic regression
- associated quality changes are small:
  - real classification downstream `-0.0045`
  - synthetic classification downstream `-0.0051`, with only a very small
    decline in mean recovery over standard `k`
    (`delta_precision_over_standard_k = -0.0014`)
  - real regression downstream `+0.0004`
  - synthetic regression downstream `-0.0032`, with only a small decline in
    the same curve-based recovery summary
    (`delta_precision_over_standard_k = -0.0051`)
- this toggle disables both selector-side and splitter-side adaptive stopping,
  so it evaluates the implemented stopping regime rather than an isolated Stage
  A switch

What the strictness comparison says:

- removing Bonferroni is much faster:
  - `0.0286x` runtime on real classification
  - `0.0186x` on synthetic classification
  - `0.0137x` on real regression
  - `0.0248x` on synthetic regression
- but it also makes trees deeper and less sparse:
  - depth increases by about `3.14` / `5.95` / `3.00` / `7.00`
  - features used increase by about `20.64` / `13.28` / `1.62` / `35.27`
- the synthetic recovery curve moves only modestly relative to that structural
  shift:
  - classification `+0.0213`
  - regression `+0.0009`
- so this is a strictness-versus-depth regime change, not a clean practical
  speedup

What the weaker-control rows say:

- `cif_no_scan`, `cif_no_mute`, and `cif_no_bootstrap` do not show comparably
  clean wins
- `cif_no_scan` is still worth reporting: with adaptive stopping already on, it
  is `1.1045x` slower on real classification, `1.0213x` slower on synthetic
  classification, `1.0795x` slower on real regression, and `1.2175x` slower on
  synthetic regression
- associated downstream-score changes are small and mixed:
  `-0.0015`, `-0.0054`, `+0.0006`, and `-0.0055`
- `cif_no_threshold_scan` is more mixed still:
  `0.6876x`, `1.0696x`, `0.7793x`, and `0.9258x` across the same four surfaces,
  with small/mixed downstream changes
- paper-safe reading: feature scanning is a measured but modest supporting gain
  inside the early-stopping regime, while threshold scanning does not show a
  stable supporting speed benefit

What the threshold ablation says:

- implementation note:
  - in the current code, the `exact` threshold generator ignores
    `max_thresholds` and always returns all exact midpoints
  - therefore `exact_256` is not a distinct configuration; it is a duplicate of
    `exact_all` and should not be interpreted separately
- `exact_all` relative to `histogram_256`:
  - `1.8711x` slower on real classification
  - `5.0083x` slower on synthetic classification
  - `10.6324x` slower on real regression
  - `5.1917x` slower on synthetic regression
- `histogram_32` relative to `histogram_256`:
  - `0.0999x` on real classification
  - `0.0427x` on synthetic classification
  - `0.0963x` on real regression
  - `0.0394x` on synthetic regression
- its curve-based synthetic recovery shifts are small and favorable:
  - classification `+0.0150`
  - regression `+0.0033`
- `exact_all` is mixed even on the same curve-based synthetic summary:
  - classification `-0.0125`
  - regression `+0.0167`
- this is a targeted supporting surface rather than a rerun of the full main
  benchmark

Paper-safe interpretation:

- adaptive stopping is the clearest practical win
- bounded histogram thresholding is the second clearest practical win
- dropping Bonferroni changes the procedure, not just the runtime
- scan, mute, and bootstrap should remain background controls rather than
  primary discoveries
- these runtime ratios are within-method comparisons under the collected setup,
  not hardware-independent benchmark claims

Do not say:

- every knob was individually validated
- the executed regime is globally optimal
- the practical shortcuts preserve the theorem

### 2.10 Regression mirror

What the experiment is:

- regression version of the main benchmark and related supporting summaries

What the canonical aggregate says:

- CIF is `2nd/16` by mean rank on the main regression aggregate
- CIF mean rank is `6.2313`
- CIF mean score is `0.3294`
- support is only `8, 8, 7, 7, 6` across the standard `k` values
- on the looser all-supported pairwise layer, CIF has positive mean deltas
  against `14/15` regression baselines
- pairwise aggregate deltas are positive against `14/15` regression baselines,
  but that layer is descriptive only because of the small dataset count
- the complete-case spread surface widens with `k`:
  mean cross-method range grows from `1.0571` at `k=5` to `1.8834` at `k=100`

What the regression pairwise breadth says:

- CIF has positive mean deltas against `14/15` regression baselines on the
  all-supported layer
- largest positive deltas:
  - `r_cforest +0.7211`
  - `cpi +0.6198`
  - `cit +0.3598`
  - `lgbm +0.3576`
- closest comparisons:
  - `cat +0.0251`
  - `rf +0.0026`
  - `et -0.0238`
  - `xgb +0.0598`

What the heterogeneity summary says:

- CIF is top-half on `8/8`
- CIF is top-3 on `2/8`
- CIF is top-1 on `1/8`

Paper-safe interpretation:

- regression broadens the picture, but it is not the main benchmark focus
- CIF is competitive there, but the support ceiling is too low for stronger
  claims

Do not say:

- regression confirms the classification story at equal strength

### 2.11 CIF-vs-R supporting comparison

What the experiment is:

- comparison against historical `r_ctree` and `r_cforest` references
- with documented high-dimensional failures and exclusions still recorded in the
  executed grid

What the canonical pairwise aggregate says:

- classification:
  - `r_ctree +0.0876` on `22` datasets
  - `r_cforest +0.0715` on `22` datasets
- regression:
  - `r_ctree +0.2265` on `8` datasets
  - `r_cforest +0.7211` on `8` datasets

Paper-safe interpretation:

- this is useful supporting evidence that the practical Python implementation
  improves on the historical conditional-inference references
- it should remain an anchor, not the whole paper
- any manuscript use of this comparison should preserve the skip/exclusion
  caveat rather than treating it as a perfectly symmetric all-dataset bakeoff

Do not say:

- this is mainly a Python-versus-R paper
