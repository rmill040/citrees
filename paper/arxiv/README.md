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
  - `figures/paper_mechanism_grid_forest_classification_dimension_curves_1000trees.png`
  - `figures/paper_mechanism_grid_forest_classification_feature_counts_p1000_i2_1000trees.png`
  - `figures/synthetic_topk_focus_curves.png`

Exclude build products and unused figures from the arXiv source bundle:
`*.aux`, `*.blg`, `*.log`, `*.out`, `*.fls`, `*.fdb_latexmk`, and `main.pdf`.
