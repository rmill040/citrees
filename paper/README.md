# Paper Artifacts

This directory contains the manuscript, benchmark, and submission artifacts for
`citrees`. It supports two related papers:

- **arXiv methods and benchmark manuscript**: the full technical paper on the
  conditional-inference tree/forest method, theory, implementation details, and
  empirical benchmark results.
- **JOSS software paper**: the short software-focused paper for the reusable
  Python package, documentation, tests, API design, and community readiness.

## Submission Status

Current as of 1 July 2026:

- **arXiv**: submitted on 1 July 2026. The arXiv identifier is pending. Update
  `paper/arxiv/`, `paper/joss/paper.md`, and any repository metadata only after
  arXiv issues the real identifier.
- **JOSS**: submitted on 1 July 2026 and awaiting editorial screening/review.
  Update `paper/joss/paper.md` and this README after JOSS creates the review
  issue, archive DOI, or publication metadata.

Do not add placeholder arXiv IDs, DOIs, review issue URLs, archive URLs, or
publication metadata. Use real identifiers only after they exist.

## Directory Map

| Path             | Purpose                                                                              |
| ---------------- | ------------------------------------------------------------------------------------ |
| `paper/arxiv/`   | LaTeX source for the full methods and benchmark manuscript.                          |
| `paper/joss/`    | Markdown source and bibliography for the JOSS software paper.                        |
| `paper/docs/`    | Claim ledger, result-finalization notes, experiment notes, and active review checks. |
| `paper/results/` | Generated benchmark tables and figures used by the manuscript.                       |
| `paper/data/`    | Benchmark datasets and local data used by the paper pipeline.                        |
| `paper/scripts/` | Experiment, analysis, infrastructure, and artifact-management code.                  |

Important generated or local-only outputs are ignored by Git, including arXiv
LaTeX build files, JOSS draft PDFs/JATS output, local S3 syncs, and paper result
caches.

## Paper Scope

The arXiv manuscript is the source for detailed scientific claims. It focuses
on:

- conditional-inference trees and forests for feature ranking;
- permutation-test-based node selection and split selection;
- adaptive sequential stopping and bounded threshold search as practical runtime
  controls;
- real-data, synthetic, and mechanism-oriented benchmark evidence.

The JOSS paper is intentionally narrower. It focuses on:

- the public Python package and scikit-learn-style API;
- installation, documentation, tests, and package maintainability;
- the software need relative to existing Python tree libraries;
- high-level evidence from the companion benchmark manuscript.

## Result Headline

The benchmark headline currently used across the paper package is:

- CIF ranks 4th among 17 classification methods on the main real-data
  classification comparison.
- CIF ranks 3rd among 18 regression methods on the main real-data regression
  comparison.
- These benchmark summaries are descriptive because configuration selection and
  reporting use the same benchmark surface.

For exact numbers, table membership, caveats, and claim eligibility, use the
authority files below rather than this README.

## Claim Authority

Read these before changing paper claims, numbers, tables, or result prose:

1. `paper/docs/analysis-contract.md`
2. `paper/docs/results-finalization.md`
3. `paper/docs/motivation.md`

Use them this way:

- `analysis-contract.md`: decides whether a claim is headline-eligible.
- `results-finalization.md`: records locked numbers, figures, and tables.
- `motivation.md`: records the motivation, hierarchy, and compression of those
  locked results.

Supporting references:

- `paper/docs/experiments.md`: rebuild path and experiment caveats.
- `paper/docs/final-review-checklist.md`: active visual or presentation checks
  not covered by automated tests.
- `paper/results/tables/README.md`: canonical, supporting, and superseded table
  manifest.
- `paper/arxiv/README.md`: arXiv build and source-bundle notes.
- `paper/joss/README.md`: JOSS paper format and build notes.
- `paper/scripts/README.md`: paper-side script map.

## Script Layout

`paper/scripts/` has several different kinds of code:

- `analysis/`, `theory/`, and `experiments/`: paper-facing table, figure,
  diagnostic, and experiment builders.
- `api/`, `cli/`, `pipeline/`, `adapters/`, `config/`, `infra/`, and `utils/`:
  the distributed benchmark pipeline and its command-line tooling.
- `data_generation/`: dataset-generation helpers.
- `maintenance/`: operational audit and repair helpers for paper artifacts.
  These are not part of the package API or normal paper rebuild path.
- `archive/`: archived exploratory scripts and provenance. Keep these unless a
  separate cleanup pass verifies that they are no longer useful.

## Build Commands

Build the arXiv manuscript:

```bash
cd paper/arxiv
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Build the deterministic arXiv source bundle from the repository root:

```bash
uv run python paper/scripts/analysis/build_arxiv_source_bundle.py
```

Build the JOSS draft PDF:

```bash
docker run --rm \
  --volume "$PWD/paper/joss:/data" \
  --user "$(id -u):$(id -g)" \
  --env JOURNAL=joss \
  openjournals/inara
```

Run paper-side tests:

```bash
uv run pytest tests/paper -q
```

## Rebuild Paper-Facing Tables

The closed paper-facing rebuild path is:

```bash
uv run python paper/scripts/analysis/build_paper_data_surfaces.py
uv run python paper/scripts/analysis/build_dataset_characteristics_table.py
uv run python paper/scripts/analysis/build_benchmark_package_tables.py
uv run python paper/scripts/analysis/fig_benchmark_k_trajectory.py
uv run python paper/scripts/analysis/build_benchmark_heterogeneity_tables.py
uv run python paper/scripts/analysis/build_high_p_saturation_tables.py
uv run python paper/scripts/analysis/build_top_ranking_tables.py
uv run python paper/scripts/analysis/build_synthetic_topk_tables.py
uv run python paper/scripts/analysis/build_knob_ablation_summary_tables.py
uv run python paper/scripts/analysis/build_threshold_ablation_summary_tables.py
uv run python paper/scripts/analysis/build_cit_runtime_ablation_summary_tables.py
uv run python paper/scripts/analysis/build_manuscript_summary_tables.py
uv run python paper/scripts/analysis/build_mechanism_summary_tables.py
```

The CIF ranking ablation table is regenerated from fold-level ablation metrics
when those metrics are available:

```bash
uv run python paper/scripts/analysis/build_cif_mechanism_ablation_tables.py --input-uri <local-or-s3-metrics-prefix>
```

After rebuilding, reconcile the generated outputs against:

- `paper/docs/results-finalization.md`
- `paper/results/tables/README.md`
- this README
