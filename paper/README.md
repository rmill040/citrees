# Paper Directory

This directory now has a deliberately small documentation stack.

If you are drafting or editing the paper, do not wander through every markdown
file. Start from the consolidated brief and only drop to the lower-level docs
when you need the governing rule or a locked number.

## Read These

1. `paper/README.md`
2. `paper/docs/analysis-contract.md`
3. `paper/docs/results-finalization.md`
4. `paper/docs/motivation.md`

Authority still flows in that order:

1. `analysis-contract.md`
2. `results-finalization.md`
3. `paper/docs/motivation.md`

Use them this way:

- `analysis-contract.md`: decides whether a claim is headline-eligible.
- `results-finalization.md`: decides which numbers, figures, and tables are
  locked.
- `paper/docs/motivation.md`: decides the motivation, hierarchy, and
  compression of those locked results.

## Support-Only Docs

- `paper/docs/experiments.md`
  Rebuild path and manuscript-facing caveats only.
- `paper/results/tables/README.md`
  Canonical vs supporting vs superseded table manifest.

These can support drafting, but they should not drive the paper's story.

## Operational-Only Docs

- `paper/docs/infrastructure.md`
- `paper/scripts/backlog/README.md`
- `paper/arxiv/README.md`
- `paper/results/README.md`

These are workflow notes, not paper authorities.

## Not Authority

Anything outside the authority and support lists above is history, workflow
support, or generated output. Do not use it as source of truth for claims,
structure, or numbers.

## Current Paper In One Paragraph

This is an optimization-first paper about conditional inference. The principled
core is a narrow fixed-node Stage A screening guarantee. The practical result
is that adaptive stopping and bounded histogram thresholding make the method
usable without materially changing quality. The empirical result is that CIF is
the strongest conditional-inference method in the benchmark and remains
competitive with common tree ensembles, while clearly improving on the classic
conditional-inference baselines. High-`p`, synthetic, and mechanism diagnostics
then explain the operating boundary.

## Tone

The paper should be assertive but bounded.

- state the positive claim first, then the boundary
- keep caveats in the limitations paragraph instead of scattering them through
  every result paragraph
- do not narrate the paper as an apology for not being `1st/15`
- use rank positions as table support, not as the voice of the paper

The empirical headline is:

- CIF is the strongest conditional-inference method in the benchmark,
- it clearly improves on the classic conditional-inference baselines,
- and it remains competitive with the common tree ensembles people actually use.

## Main-Text Package

The main paper should be small.

Keep in main text:

- one compact introduction with background folded in
- one lean method section
- one short theory section
- one experimental-design section
- one benchmark-and-controls section
- one boundary-diagnostics section
- one discussion
- one short conclusion

Main-text figures:

- `paper/results/figures/k_trajectory.png`
- `paper/results/figures/regression_k_trajectory.png`
- `paper/results/figures/high_p_boundary_summary.png`
- `paper/results/figures/synthetic_topk_focus_curves.png`
- `paper/results/figures/paper_mechanism_grid_forest_classification_feature_counts_p1000_i2_1000trees.png`

Main-text tables:

- main real-data rank table from `paper_benchmark_method_aggregate.csv`
- matched conditional-inference comparison table from
  `paper_benchmark_pairwise_aggregate.csv`
- CIF breadth table from `paper_heterogeneity_method_summary.csv` and
  `paper_heterogeneity_cif_pairwise_breadth.csv`
- CIT runtime table from `paper_cit_runtime_ablation_summary.csv`
- CIF runtime table from `paper_mirrored_knob_ablation_summary.csv`,
  `paper_threshold_ablation_summary.csv`, and
  `paper_presentation_practical_controls_summary.csv`
- synthetic recovery table from `synthetic_topk_composition_summary.csv`

Do not let the paper regrow:

- composite figures
- calibration in main text
- CIF-vs-R as a headline layer
- long method-by-method prose
- regression displays beyond the compact trajectory figure
- regime-by-regime synthetic walkthroughs

## Build The Manuscript

```bash
cd paper/arxiv
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Rebuild Paper-Facing Tables

```bash
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_paper_data_surfaces.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_dataset_characteristics_table.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_benchmark_package_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/fig_benchmark_k_trajectory.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_benchmark_heterogeneity_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_high_p_saturation_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_top_ranking_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_synthetic_topk_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_knob_ablation_summary_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_threshold_ablation_summary_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_cit_runtime_ablation_summary_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_manuscript_summary_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_mechanism_summary_tables.py
```

The CIF ranking ablation table is regenerated from its fold-level ablation
metrics when those metrics are available:

```bash
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_cif_mechanism_ablation_tables.py --input-uri <local-or-s3-metrics-prefix>
```

After rebuilding, reconcile against:

- `paper/docs/results-finalization.md`
- `paper/results/tables/README.md`
- this README
