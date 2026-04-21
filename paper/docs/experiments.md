# Paper Experiments

This is the short paper-facing experiment runbook. Use it for:

- rebuilding the closed paper-facing tables from saved artifacts,
- remembering the inferential boundary on p-values,
- remembering the runtime boundary on ablation and wall-clock claims.

This file intentionally omits the old EC2 launcher cookbook.

## Closed Rebuild Path

For manuscript numbers, rebuild the paper-facing analysis layer from the saved
artifacts already under `paper/results/`.

```bash
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_dataset_characteristics_table.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_benchmark_package_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_benchmark_heterogeneity_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_high_p_saturation_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_top_ranking_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_synthetic_topk_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_knob_ablation_summary_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_threshold_ablation_summary_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_manuscript_summary_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_mechanism_summary_tables.py
```

After rebuilding:

- refresh paper prose only after checking the rebuilt tables.

## Canonical Benchmark Envelope

The packaged benchmark used by the current paper is:

- real-data classification: `23` datasets
- real-data regression: `8` datasets
- standard budgets: `k = 5, 10, 25, 50, 100`
- all-downstream reporting:
  - classification: `lr`, `svm`, `knn`
  - regression: `ridge`, `svr`, `knn`
- one best global config per method family within task

Dataset-specific `k=p` endpoint checks belong to the high-`p` saturation layer,
not the main benchmark.

What we did **not** run:

- we did not evaluate every integer `k`
- we did not evaluate every `5` features

For datasets with `p > 100`, the evaluation pipeline also runs a sparse
high-`p` extension:

- `150, 200, 300, 500, 750, 1000`
- `0.25p, 0.5p, 0.75p`
- `p`

Those extra budgets are for the high-`p` boundary layer, not the main
benchmark summary.

## Actual Pipeline Contract

This is the plain-English version of what the code actually does.

### Stage 1: ranking

- Stage 1 builds a ranking separately for each random seed and CV fold.
- The ranking is fit on the training fold only.
- Features are standardized using training-fold statistics.
- The fitted transform is then applied to the held-out fold.

### Stage 2: evaluation

- Stage 2 reconstructs the same seed and fold split.
- For each stored fold ranking, it evaluates downstream models on the matching
  held-out fold.
- For each feature budget `k`, it takes the top-`k` prefix of the stored
  ranking, re-standardizes those selected features on the training fold, then
  fits a fresh downstream model.

This means the real-data benchmark is a fold-wise rank-then-evaluate design.

### Config selection

- Several families are run under multiple configs.
- The benchmark does not report every config.
- Instead, for each method family and task, the analysis picks one global
  config using the real-data benchmark.
- That choice is global within task.
- It is not nested separately within each dataset.

The leave-one-dataset-out analysis is a sensitivity check for that choice. It
is not a replacement for nested config selection.

### Averaging contract

The benchmark uses three averaging steps:

1. Within a fixed `(dataset, downstream model, k)` cell, scores are averaged
   over folds and seeds.
2. For pooled benchmark summaries, those cell scores are averaged within a
   dataset over downstream models and supported standard budgets.
3. The pooled tables then average those dataset-level summaries over datasets.

## Supporting Studies Used In The Paper

The current support package includes:

- fixed-node/root calibration refresh
- 14-dataset paired and omnibus benchmark summaries
- leave-one-dataset-out config-selection sensitivity summaries
- mirrored practical-knob ablations
- threshold-search ablation
- synthetic top-`k` composition diagnostics
- fixed-design mechanism diagnostics

These studies are part of the locked support package, but they are not all
reruns of the main benchmark.
When in doubt, treat anything outside those locked outputs as exploratory or historical by
default.

Support-only rebuilds, when those locked outputs need to be refreshed:

```bash
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_fixed_panel_omnibus_table.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_fixed_panel_pairwise_ci_table.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_lodo_config_sensitivity_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/theory/generate_calibration_support_package.py
```

## Result Layers We Already Have

The following result layers are already available in the locked outputs.

### Broader classification benchmark

Main artifacts:

- `paper/results/tables/paper_presentation_benchmark_summary.csv`
- `paper/results/tables/paper_benchmark_method_aggregate.csv`
- `paper/results/tables/paper_benchmark_pairwise_aggregate.csv`

What this layer says:

- CIF is `4th/15` on the broader classification benchmark.
- CIF mean rank is about `5.47`.
- CIF mean balanced accuracy is about `0.819`.
- CIF is positive against the historical conditional-inference
  references:
  - `r_ctree`: `22/22` datasets, mean delta `+0.0876`
  - `r_cforest`: `19/22` datasets, mean delta `+0.0715`
  - `CIT`: `21/22` datasets, mean delta `+0.0301`
- The strongest generic ensembles rank above CIF in the pooled classification
  summary.

### 14-dataset classification benchmark with complete coverage

Main artifacts:

- `paper/results/tables/paper_benchmark_fixed_panel_membership.csv`
- `paper/results/tables/paper_benchmark_fixed_panel_aggregate.csv`
- `paper/results/tables/paper_benchmark_fixed_panel_pairwise_ci.csv`
- `paper/results/tables/paper_benchmark_fixed_panel_omnibus.csv`

What this layer says:

- The 14 datasets are the classification datasets that remain complete across
  all downstream models and standard budgets.
- This layer keeps the same dataset membership across all downstream models and
  standard budgets.
- The omnibus summary is:
  - classification: `Kendall's W = 0.64`
- CIF remains positive against the historical baselines on this layer.

### Downstream-model × k sensitivity

Main artifact:

- `paper/results/tables/paper_benchmark_pairwise_stratified.csv`

What this layer says:

- The broader classification benchmark can be broken out by downstream model
  and standard feature budget.
- CIF stays positive against `CIT`, `r_ctree`, and `r_cforest` across the
  classification cells, not just after pooling.

### Heterogeneity / consistency

Main artifacts:

- `paper/results/tables/paper_heterogeneity_method_summary.csv`
- `paper/results/tables/paper_heterogeneity_cif_pairwise_breadth.csv`
- `paper/results/tables/paper_heterogeneity_cif_pairwise_by_dataset.csv`

What this layer says:

- On classification:
  - top half on `21/22` datasets
  - top 3 on `7/22`
  - top 1 on `3/22`
- CIF's broadest clean wins are again against the historical
  conditional-inference baselines and CIT.

### High-p boundary

Main artifacts:

- `paper/results/tables/paper_high_p_cif_endpoint_summary.csv`
- `paper/results/tables/paper_high_p_cif_best_observed_k_summary.csv`
- `paper/results/tables/paper_high_p_endpoint_pairwise.csv`
- `paper/results/tables/paper_high_p_endpoint_inventory.csv`

What this layer says:

- On classification, CIF usually does not first hit its best score at the
  endpoint.
- Classification all-downstream summary:
  - under `100`: `5/45`
  - at `100`: `5/45`
  - intermediate beyond `100`: `30/45`
  - at endpoint `p`: `5/45`
- Endpoint performance is usually worse than `k=100` on classification:
  - mean score change from `k=100` to endpoint: about `-0.0369`
  - mean rank change: about `+3.07`
- At the downstream level, `lr` improves at the endpoint more often than
  `knn` or `svm`.

### Synthetic ranking diagnostics

Main artifacts:

- `paper/results/tables/synthetic_topk_composition_summary.csv`
- `paper/results/tables/synthetic_topk_composition_by_dataset_type.csv`
- `paper/results/tables/top_ranking_summary.csv`
- `paper/results/tables/top_ranking_by_dataset_type.csv`
- `paper/results/figures/synthetic_topk_focus_curves.png`

What this layer says:

- The synthetic layer is about known informative features, not downstream
  prediction.
- CIF is better at subset construction than exact head-of-list recovery.
- The top-`k` composition curves keep that distinction visible across `k`.

### Mechanism diagnostics

Main artifacts:

- `paper/results/tables/paper_mechanism_grid_combined_aggregate_summary.csv`
- `paper/results/tables/paper_mechanism_grid_forest_classification_aggregate_summary_1000trees.csv`
- `paper/results/tables/paper_mechanism_grid_forest_classification_cif_vs_cif_all_deltas_1000trees.csv`
- `paper/results/figures/paper_mechanism_grid_forest_classification_feature_counts_p1000_i2_1000trees.png`
- `paper/results/figures/paper_mechanism_grid_forest_classification_dimension_curves_1000trees.png`

What this layer says:

- The mechanism is sparse candidate exposure in forests, not generic tree
  weakness.
- In the combined aggregate summary:
  - `cif` forest classification informative split share is about `0.332`
  - `cif_all` pushes that to about `0.917`
- This quantifies how much informative exposure changes when feature
  subsampling is removed in sparse high-`p` cases.

### Stability and seed/fold variability

Main artifacts:

- `paper/results/tables/paper_benchmark_lodo_aggregate.csv`
- `paper/results/tables/paper_benchmark_lodo_config_stability.csv`
- `paper/results/tables/variance_decomposition.csv`

What this layer says:

- CIF's global benchmark-selected config is reselected on all `14/14`
  held-out datasets in the 14-dataset classification check.
- Most variance sits at the dataset level.
- `k` contributes some variation.
- `seed` is tiny:
  - about `0.0024%` in `variance_decomposition.csv`

## Inferential Scope

Only fixed-node/root Stage A permutation p-values computed in fixed-`B` mode
under the nodewise complete permutation null are treated as calibrated in the
paper.

Not theorem-backed by default:

- Stage B threshold tests
- internal-node tests
- early-stopped permutation outputs
- end-to-end adaptive learner behavior

Calibration figures and tables stay supporting-only unless they are explicitly
re-locked in a paper-facing doc.

## Runtime Scope

Do not use pipeline wall-clock times for cross-method runtime claims.

The distributed experiment pipeline mixes:

- heterogeneous EC2 instances,
- Docker overhead,
- S3 I/O,
- JIT warm-up,
- R/Python bridge overhead,
- and end-to-end serialization costs.

Paper-grade runtime claims should come only from the dedicated practical-control
ablations and should be phrased as within-method comparisons under the
collected setup, not hardware-independent method-vs-method benchmarks.

Within-CIF speed-study provenance lives under `scratch/speed-study/`.

## Where Operational Details Live

If you need to rerun infrastructure-heavy jobs:

- use `citrees-exp --help`
- inspect `paper/scripts/infra/`
- use `paper/docs/infrastructure.md` only as a short operational pointer
