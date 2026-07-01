# Archived Paper Scripts

This directory contains paper-side scripts that are no longer on the current
arXiv/JOSS rebuild path but are still useful as experiment provenance. Keep them
out of the normal rebuild path, and do not delete them unless a focused cleanup
pass verifies that their outputs and design notes are no longer needed.

## Archive Rule

Archive a script when all of the following are true:

- it is not in the closed rebuild path from `paper/docs/experiments.md`;
- it does not build a current arXiv figure asset;
- it is not part of the locked support package;
- it is not required by `citrees-exp`, the benchmark pipeline, or paper tests.

If a script is upstream provenance for a locked table, keep it in the live tree
until the dependency is removed.

## Live Script Boundaries

These script groups remain on the active paper path:

- `paper/scripts/analysis/`
- `paper/scripts/experiments/`
- `paper/scripts/theory/`
- `paper/scripts/api/`
- `paper/scripts/cli/`
- `paper/scripts/config/`
- `paper/scripts/pipeline/`
- `paper/scripts/adapters/`
- `paper/scripts/utils/`
- active EC2 helpers in `paper/scripts/infra/`

For the exact rebuild sequence, use `paper/docs/experiments.md`.

## Archived Experiment Runners

These older study runners are retained for provenance:

- `paper/scripts/archive/experiments/study_alpha_selector_sweep.py`
- `paper/scripts/archive/experiments/study_bootstrap_feature_subsampling.py`
- `paper/scripts/archive/experiments/study_multi_selector_correction.py`
- `paper/scripts/archive/experiments/study_forest_size_sweep.py`
- `paper/scripts/archive/experiments/study_noise_feature_robustness.py`
- `paper/scripts/archive/experiments/study_legacy_optimization_ablation.py`
- `paper/scripts/archive/experiments/study_ptest_power_grid.py`
- `paper/scripts/archive/experiments/study_real_data_ablation.py`
- `paper/scripts/archive/experiments/study_resamples_honesty_sweep.py`
- `paper/scripts/archive/experiments/study_sample_size_sweep.py`
- `paper/scripts/archive/experiments/study_runtime_scaling.py`
- `paper/scripts/archive/experiments/study_selector_strictness_continuum.py`

They are not current manuscript builders.

## Archived Analysis, Theory, And Wrappers

Archived analysis/story scripts live in `paper/scripts/archive/analysis/`.
Archived theory generators live in `paper/scripts/archive/theory/`. Older CLI,
data-generation, and EC2 wrappers live under `paper/scripts/archive/cli/`,
`paper/scripts/archive/data_generation/`, and `paper/scripts/archive/infra/`.

These files can still explain how older outputs were generated, but they should
not be used as authority for current manuscript claims. Use
`paper/docs/analysis-contract.md` and `paper/docs/results-finalization.md` for
claim authority.

## Naming Convention

The paper tree uses these prefixes deliberately:

- `build_*`: table-producing paper-facing builders
- `fig_*`: figure-producing paper-facing builders
- `study_*`: experiment runners that generate raw study outputs
- `audit_*`: read-only verification tools
- `repair_*`: one-off result mutation scripts

Avoid adding new mixed-purpose scripts here. If a script becomes current again,
move it back to the live tree and document the rebuild path that consumes it.
