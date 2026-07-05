# arXiv Manuscript

- `main.tex`: manuscript entrypoint.
- `main.pdf`: published arXiv PDF.
- `sections/`: main-paper content.
- `appendices/`: proofs and supporting details.
- `figures/`: figures referenced by the manuscript.

Do not edit these files unless preparing a deliberate new arXiv version.

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
- `sections/`
- `appendices/`
- referenced figures:
  - `figures/benchmark_pairwise_sensitivity.png`
  - `figures/high_p_boundary_summary.png`
  - `figures/classification_k_trajectory.png`
  - `figures/paper_mechanism_grid_forest_classification_dimension_curves_1000trees.png`
  - `figures/paper_mechanism_grid_forest_classification_feature_counts_p1000_i2_1000trees.png`
  - `figures/paper_mechanism_grid_forest_regression_dimension_curves_1000trees.png`
  - `figures/regression_benchmark_pairwise_sensitivity.png`
  - `figures/regression_k_trajectory.png`
  - `figures/synthetic_topk_focus_curves.png`

Do not zip this directory by hand. It contains ignored scratch and build
outputs. From the repository root, build the deterministic source bundle with:

```bash
uv run python paper/analysis/build_arxiv_source_bundle.py
```

The bundler rebuilds the manuscript, copies only the files above, and excludes
scratch, PDF, unused figures, `main.bbl`, and LaTeX build products. arXiv
regenerates the bibliography from `references.bib`.
