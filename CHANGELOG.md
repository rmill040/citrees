# Changelog

All notable changes to `citrees` will be documented here.

This project follows semantic-versioning-style release labels where practical,
with version numbers matching `pyproject.toml` and GitHub release tags.

## Unreleased

### Added

- `threshold_test="maxt"`: opt-in joint max-type Stage B test on the minimum
  standardized child impurity over all candidate thresholds (default remains
  `"bonferroni"`).

### Changed

- Forests divide the Numba thread pool across tree-fitting workers instead of
  multiplying it.
- Permutation kernels give identical results for every Numba thread count and
  `n_jobs`.
- `early_stopping_confidence_*` must be at least 0.95.
- `dcor` is imported lazily by the distance-correlation selector.
- RDC permutation kernels reuse per-thread buffers instead of allocating per
  permutation.

### Fixed

- `n_resamples="auto"` produced a non-finite count at the extremes of the valid
  `alpha` range.

## 0.1.0 - 2026-07-01

### Added

- Initial Python package metadata for conditional inference trees and forests.
- Scikit-learn-style classifier and regressor APIs for conditional inference
  trees and forests.
- Permutation-test selectors for classification and regression, including
  multiple correlation, Pearson correlation, distance correlation, randomized
  dependence coefficient, and mutual information where applicable.
- Fixed-budget and adaptive permutation-test options with Phipson-Smyth
  corrected Monte Carlo p-values.
- Forest support with bootstrapping, optional out-of-bag scoring, feature
  subsampling, and parallel training.
- Optional sample-split leaf re-estimation.
- Unit, integration, and paper-pipeline tests.
- MkDocs documentation and paper experiment support.
