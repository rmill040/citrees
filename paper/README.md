# Paper

This directory contains the published `citrees` methods and benchmark
manuscript, the Journal of Statistical Software article in preparation, their
supporting analyses, and the experiment pipeline.

## Status

Version 1 is published as `arXiv:2607.01417`. The `paper/arxiv/` source is the
working copy of the next arXiv version; `TODO.md` at the repository root lists
what remains before it and the JSS article are submitted.

## Layout

| Path                     | Purpose                                                                    |
| ------------------------ | -------------------------------------------------------------------------- |
| `paper/arxiv/`           | LaTeX source and PDF of the arXiv manuscript.                              |
| `paper/jss/`             | JSS manuscript source and standalone replication materials.                |
| `paper/results/`         | Small set of tracked result tables used to verify paper claims.            |
| `paper/data/`            | Benchmark datasets used by the experiment pipeline.                        |
| `paper/benchmark/`       | Importable benchmark CLI, pipeline, API, storage, and infrastructure code. |
| `paper/analysis/`        | Paper table and figure builders.                                           |
| `paper/data_generation/` | Synthetic dataset generator.                                               |
| `paper/maintenance/`     | Operational audit helpers for benchmark outputs.                           |
| `paper/theory/`          | Exact null size of the adaptive stopping rule (Pólya-urn recursion).       |

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

## Result Support

Every number in the two manuscripts traces to a table under
`paper/results/tables/` (index and provenance in
`paper/results/tables/README.md`) or to the reference release in S3 named there.
The manuscripts, not this file, state the results. The benchmark ranks are
descriptive because configuration selection and reporting use the same benchmark
surface.

## Code Layout

`paper/benchmark/` is importable code used by `citrees-exp`. It contains the
distributed benchmark pipeline, API worker/server, S3 adapters, infrastructure
helpers, and focused benchmark experiments.

`paper/analysis/`, `paper/data_generation/`, and `paper/maintenance/` are
repo-side paper utilities. They are not shipped in the wheel and are not part of
the public package API.

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

Run paper-side tests:

```bash
uv run pytest tests/paper -q
```

## Results

We track only the result tables needed to verify manuscript claims. Broader
analysis tables and figures are regenerated locally.
