# arXiv Manuscript

- `main.tex`: manuscript entrypoint (body, references, appendices A to F).
- `main.pdf`: published arXiv PDF.
- `sections/`: main-paper content, one file per section or results subsection.
- `appendices/`: appendices in compiled order A setup (`appendix_A_setup.tex`),
  B proofs (`appendix_B_exchangeability.tex`), C benchmark configurations and
  coverage (`appendix_D_methods.tex`), D datasets (`appendix_F_datasets.tex`), E
  runtime and memory accounting (`appendix_E_complexity.tex`), and F feature use
  in sampled forests (`appendix_H_candidate_availability.tex`).
- `supplement.tex` and `supplement/`: the supplement, sections S1 benchmark
  stability, S2 wide data and synthetic recovery, S3 threshold test comparison
  tables, and S4 weak-signal cardinality designs.
- `macros.tex`: macros and theorem environments shared by both documents.
- `figures/`: figures referenced by the manuscript and the supplement.

Do not edit these files unless preparing a deliberate new arXiv version.

The supplement is an arXiv ancillary file. It cross-references paper labels
through `xr-hyper` and `main.aux`, so `main.tex` must be built first; the paper
points to it only as "(Supplement, Section S1)" and never references supplement
labels.

Build both documents from this directory (the local `latexmkrc` builds
`main.tex` and then `supplement.tex`):

```bash
latexmk
```

Figures referenced by `main.tex`:

- `figures/classification_k_trajectory.png`
- `figures/regression_k_trajectory.png`
- `figures/paper_mechanism_grid_forest_classification_feature_counts_p1000_i2_1000trees.png`

Figures referenced by `supplement.tex`:

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

The bundler rebuilds both documents, then copies `main.tex`, `macros.tex`,
`references.bib`, every TeX file that `main.tex` inputs, and the figures those
files include, and adds the compiled supplement as `anc/supplement.pdf`. It
excludes scratch, `main.pdf`, `main.bbl`, the supplement sources, figures used
only by the supplement, and LaTeX build products. arXiv regenerates the
bibliography from `references.bib`.
