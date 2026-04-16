# Paper Experiments

This is the short paper-facing experiment runbook. Use it for:

- rebuilding the closed paper-facing tables from saved artifacts,
- remembering the inferential boundary on p-values,
- remembering the runtime boundary on ablation and wall-clock claims.

This file intentionally omits the old EC2 launcher cookbook.

## Closed Rebuild Path

For manuscript numbers, rebuild the paper-facing analysis layer from the saved
artifacts already under `paper/results/`.

```bash
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_dataset_inventory.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_paper_benchmark_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_dataset_heterogeneity_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_high_p_endpoint_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_top_ranking_diagnostics.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_synthetic_topk_composition.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_mirrored_knob_ablation_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_threshold_ablation_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_presentation_summary_tables.py
UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_screening_mechanism_tables.py
```

After rebuilding:

- refresh manuscript prose only after checking the rebuilt tables.

## Canonical Benchmark Envelope

The packaged benchmark used by the current paper is:

- real-data classification: `23` datasets
- real-data regression: `8` datasets
- standard budgets: `k = 5, 10, 25, 50, 100`
- all-downstream reporting:
  - classification: `lr`, `svm`, `knn`
  - regression: `ridge`, `svr`, `knn`
- one best global config per method family within task

Dataset-specific `k=p` endpoint checks belong to the high-`p` saturation layer,
not the main benchmark surface.

## Supporting Studies Used In The Paper

The current support package includes:

- fixed-node/root calibration refresh
- mirrored practical-knob ablations
- threshold-search ablation
- synthetic top-`k` composition diagnostics
- fixed-design mechanism diagnostics

Those studies are allowed to support the paper, but they are not all canonical
reruns of the main benchmark. Keep that distinction explicit in prose.
When in doubt, treat anything outside those locked outputs as exploratory or historical by
default.

## Inferential Scope

Only fixed-node/root Stage A permutation p-values computed in fixed-`B` mode
under the nodewise complete permutation null are treated as calibrated in the
paper.

Not theorem-backed by default:

- Stage B threshold tests
- internal-node tests
- early-stopped permutation outputs
- end-to-end adaptive learner behavior

Calibration figures and tables stay supporting-only unless they are explicitly
re-locked in a paper-facing doc.

## Runtime Scope

Do not use pipeline wall-clock times for cross-method runtime claims.

The distributed experiment pipeline mixes:

- heterogeneous EC2 instances,
- Docker overhead,
- S3 I/O,
- JIT warm-up,
- R/Python bridge overhead,
- and end-to-end serialization costs.

Paper-grade runtime claims should come only from the dedicated practical-control
ablations and should be phrased as within-method comparisons under the
collected setup, not hardware-independent method-vs-method benchmarks.

Within-CIF speed-study provenance lives under `scratch/speed-study/`.

## Where Operational Details Live

If you need to rerun infrastructure-heavy jobs:

- use `citrees-exp --help`
- inspect `paper/scripts/infra/`
- use `paper/docs/infrastructure.md` only as a short operational pointer
