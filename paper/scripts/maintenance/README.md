# Paper Artifact Maintenance Scripts

This directory contains operational audit and repair helpers for paper
experiment artifacts.

These scripts are not part of the public `citrees` API, not part of the normal
package test path, and not required to use the library. They are kept for paper
provenance and for targeted checks against generated benchmark outputs.

Examples:

- `audit_grid_artifacts.py` checks generated ranking and metric artifact
  coverage.
- `audit_hash_alias_manifest.py` checks artifact hash alias bookkeeping.
- `repair_clf_pi_cpi_balanced_accuracy.py` documents a targeted repair after the
  classification PI/CPI metric changed.
- `run_dt_rt_ranked_feature_check.py` verifies decision-tree and random-tree
  ranked-feature outputs.

Do not use this directory as a source of truth for manuscript claims. For claims
and locked result numbers, use `paper/docs/analysis-contract.md` and
`paper/docs/results-finalization.md`.
