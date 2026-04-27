# arXiv Manuscript

- `main.tex` is the manuscript entrypoint.
- `sections/` contains the main-paper content.
- `appendices/` contains proofs and supporting details.

For claims, numbers, and paper structure, do not draft from this README.
Use:

1. `../README.md`
2. `../docs/analysis-contract.md`
3. `../docs/results-finalization.md`
4. `../docs/motivation.md`

Build:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Live preview:

```bash
latexmk -pdf -pvc -interaction=nonstopmode -halt-on-error main.tex
```

Source bundle contents for submission:

- `main.tex`
- `macros.tex`
- `references.bib`
- `main.bbl`
- `sections/`
- `appendices/`
- referenced figures:
  - `figures/benchmark_pairwise_sensitivity.png`
  - `figures/high_p_boundary_summary.png`
  - `figures/k_trajectory.png`
  - `figures/regression_k_trajectory.png`
  - `figures/paper_mechanism_grid_forest_classification_dimension_curves_1000trees.png`
  - `figures/paper_mechanism_grid_forest_classification_feature_counts_p1000_i2_1000trees.png`
  - `figures/synthetic_topk_focus_curves.png`

Do not zip this directory by hand. It contains ignored scratch and build
outputs. From the repository root, build the deterministic source bundle with:

```bash
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_arxiv_source_bundle.py
```

The bundler rebuilds the manuscript, includes the generated `main.bbl`, copies
only the files above, and excludes scratch, PDF, unused figures, and LaTeX
build products.
