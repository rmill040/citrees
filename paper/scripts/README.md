# Paper Script Map

This directory contains the code used to build, audit, and document the paper
artifacts. It is separate from the public `citrees` package API.

## Active Paper Path

- `analysis/`: paper-facing table and figure builders.
- `experiments/`: focused study runners that feed locked paper summaries.
- `theory/`: support-package generators for calibration and stopping analyses.
- `api/`, `cli/`, `pipeline/`, `adapters/`, `config/`, `infra/`, `utils/`:
  distributed benchmark pipeline and command-line tooling.
- `data_generation/`: dataset-generation helpers.

The closed table rebuild sequence is documented in `paper/docs/experiments.md`.

## Support And Provenance

- `maintenance/`: operational audit and repair helpers for generated paper
  artifacts. These are not part of the package API or the normal rebuild path.
- `archive/`: older exploratory scripts and wrappers retained for provenance.
  These are not manuscript authority unless a paper-facing doc explicitly
  promotes their outputs.

## Naming Rules

- Use `build_*` for table-producing paper-facing scripts.
- Use `fig_*` for figure-producing paper-facing scripts.
- Use `study_*` for experiment runners.
- Use `audit_*` for read-only verification tools.
- Use `repair_*` for one-off result mutation scripts.

Before adding a new script, decide whether it belongs on the active paper path,
in `maintenance/`, or in `archive/`.
