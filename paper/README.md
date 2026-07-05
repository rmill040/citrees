# Paper

This directory contains the `citrees` paper materials:

- **arXiv manuscript**: the published methods and benchmark paper.
- **JOSS paper**: the shorter software paper for the Python package.

## Status

Current as of 4 July 2026:

- **arXiv**: version 1 is published as `arXiv:2607.01417`. The
  `paper/arxiv/` source records that published version and is frozen unless a
  deliberate new arXiv version is prepared.
- **JOSS**: draft source lives in `paper/joss/`. Update `paper/joss/paper.md`
  and this README after JOSS creates the review issue, archive DOI, or
  publication metadata.

Do not add placeholder DOIs, review issue URLs, archive URLs, or publication
metadata.

## Layout

| Path             | Purpose                                                                              |
| ---------------- | ------------------------------------------------------------------------------------ |
| `paper/arxiv/`   | Frozen LaTeX source and PDF for the published arXiv manuscript.                      |
| `paper/joss/`    | Markdown source and bibliography for the JOSS software paper.                        |
| `paper/results/` | Small set of tracked result tables used to verify paper claims.                     |
| `paper/data/`    | Benchmark datasets used by the experiment pipeline.                                  |
| `paper/benchmark/` | Importable benchmark CLI, pipeline, API, storage, and infrastructure code.         |
| `paper/analysis/` | Paper table and figure builders.                                                    |
| `paper/theory/` | Calibration and stopping-analysis support code.                                      |
| `paper/data_generation/` | Synthetic dataset generator.                                                |
| `paper/maintenance/` | Operational audit helpers for benchmark outputs.                               |

Generated PDFs, LaTeX build files, local S3 syncs, figure outputs, result
caches, and broad analysis tables are ignored.

## Scope

The arXiv manuscript is the source for detailed scientific claims. It focuses
on:

- conditional-inference trees and forests for feature ranking;
- permutation-test-based node selection and split selection;
- adaptive sequential stopping and bounded threshold search as practical runtime
  controls;
- real-data, synthetic, and mechanism-oriented benchmark evidence.

The JOSS paper is shorter and software-focused. It covers:

- the public Python package and scikit-learn-style API;
- installation, documentation, tests, and package maintainability;
- the software need relative to existing Python tree libraries;
- high-level evidence from the arXiv benchmark.

## JOSS Result Support

The JOSS paper cites two benchmark summaries:

- CIF ranks 4th among 17 classification methods on the main real-data
  classification comparison.
- CIF ranks 3rd among 18 regression methods on the main real-data regression
  comparison.
- These benchmark summaries are descriptive because configuration selection and
  reporting use the same benchmark surface.

The JOSS benchmark claims are backed by the two CSVs listed in
`paper/results/tables/README.md`.

## Code Layout

`paper/benchmark/` is importable code used by `citrees-exp`. It contains the
distributed benchmark pipeline, API worker/server, S3 adapters, infrastructure
helpers, and focused benchmark experiments.

`paper/analysis/`, `paper/theory/`, `paper/data_generation/`, and
`paper/maintenance/` are repo-side paper utilities. They are not shipped in the
wheel and are not part of the public package API.

## Build

Build the arXiv manuscript:

```bash
cd paper/arxiv
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Build the deterministic arXiv source bundle from the repository root:

```bash
uv run python paper/analysis/build_arxiv_source_bundle.py --check
```

Use `--write --build-pdf` only when deliberately preparing a new arXiv version.

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

## Results

We track only the result tables needed by the JOSS paper. Broader arXiv
analysis tables and figures are regenerated locally.
