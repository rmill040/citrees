# arXiv Manuscript

- `main.tex`: manuscript entrypoint (body, references, appendices A to H).
- `main.pdf`: published arXiv PDF.
- `sections/`: main-paper content, one file per section or results subsection.
- `appendices/`: appendices in compiled order A setup (`appendix_A_setup.tex`),
  B proofs (`appendix_B_exchangeability.tex`), C benchmark configurations and
  coverage (`appendix_D_methods.tex`), D datasets (`appendix_F_datasets.tex`), E
  benchmark robustness (`appendix_robustness.tex`), F where CIF is weaker
  (`appendix_weaker.tex`), G runtime and memory accounting
  (`appendix_E_complexity.tex`), and H threshold test comparison and cardinality
  designs (`appendix_threshold_tests.tex`).
- `macros.tex`: macros and theorem environments.
- `figures/`: figures referenced by the manuscript.

Do not edit these files unless preparing a deliberate new arXiv version.

Build from this directory (the local `latexmkrc` builds only `main.tex`):

```bash
latexmk
```

Figures referenced by `main.tex`:

- `figures/benchmark_k_trajectory.png`
- `figures/paper_mechanism_grid_forest_classification_feature_counts_p1000_i2_1000trees.png`
- `figures/benchmark_pairwise_sensitivity.png`
- `figures/regression_benchmark_pairwise_sensitivity.png`
- `figures/high_p_boundary_summary.png`
- `figures/synthetic_topk_focus_curves.png`
- `figures/paper_mechanism_grid_forest_classification_dimension_curves_1000trees.png`
- `figures/paper_mechanism_grid_forest_regression_dimension_curves_1000trees.png`

Do not zip this directory by hand. It contains ignored scratch and build
outputs. From the repository root, build the deterministic source bundle with:

```bash
uv run python paper/analysis/build_arxiv_source_bundle.py --build-pdf --write
```

The bundler rebuilds the paper, then copies `main.tex`, `macros.tex`,
`references.bib`, every TeX file that `main.tex` inputs, and the figures those
files include. It excludes scratch, `main.pdf`, `main.bbl`, and LaTeX build
products. arXiv regenerates the bibliography from `references.bib`.
