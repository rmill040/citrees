# Changelog

All notable changes to citrees will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive MkDocs documentation
- RDC (Randomized Dependence Coefficient) selector for both classification and regression
- Multi-selector mode: combine multiple selectors (e.g., `selector=['mc', 'rdc']`)
- Honest estimation for unbiased leaf predictions
- Conformal prediction wrappers (`ConformalClassifier`, `ConformalRegressor`, `CQR`)
- SHAP integration via `SHAPExplainer` class
- Conditional Permutation Importance (CPI) for correlated features
- Parallel permutation tests for improved performance
- Feature muting to accelerate training
- Feature scanning to prioritize promising features

### Changed
- Clarified that citrees is inspired by but not a direct port of R's partykit::ctree

### Fixed
- Documentation now uses pseudocode instead of hardcoded line numbers

## [0.1.0] - 2024-01-01

### Added
- Initial release
- `ConditionalInferenceTreeClassifier` and `ConditionalInferenceTreeRegressor`
- `ConditionalInferenceForestClassifier` and `ConditionalInferenceForestRegressor`
- Selectors: `mc` (multiple correlation), `mi` (mutual information), `pc` (Pearson), `dc` (distance correlation)
- Splitters: `gini`, `entropy`, `mse`, `mae`
- Threshold methods: `exact`, `random`, `percentile`, `histogram`
- Bonferroni correction for multiple testing
- Early stopping for permutation tests
- scikit-learn compatible API

[Unreleased]: https://github.com/rmill040/citrees/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rmill040/citrees/releases/tag/v0.1.0
