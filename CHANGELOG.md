# Changelog

All notable changes to `citrees` will be documented here.

This project follows semantic-versioning-style release labels where practical,
with version numbers matching `pyproject.toml` and GitHub release tags.

## 0.1.0 - 2026-07-01

### Added

- Initial Python package metadata for conditional inference trees and forests,
  including JOSS submission files.
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
- MkDocs documentation and JOSS paper draft.
