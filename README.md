# citrees

**Conditional Inference Trees and Forests for Python**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Paper**: This repository contains the `citrees` package, benchmark code, and
results for Milletich, Downes, Goley, and Hirst (2026),
[_Conditional Inference Trees and Forests for Feature Selection_](https://arxiv.org/abs/2607.01417)
([PDF](https://arxiv.org/pdf/2607.01417)).

citrees implements conditional-inference-style decision trees and forests using
permutation-based screening before threshold selection. Unlike traditional
CART-style trees that greedily optimize split criteria, citrees separates
variable selection from split point selection to reduce the classic
high-cardinality split-selection bias mechanism. Fixed-B p-value calibration is
nodewise; adaptive tree and forest rankings remain empirical model outputs.

**Note**: citrees is _inspired by_ the conditional inference framework (Hothorn
et al., 2006) but is not a direct port of R's `partykit::ctree`. We implement
the core principles—permutation-based variable selection and statistical
stopping rules—while adding our own extensions like RDC selectors and feature
muting. In particular, Stage A screening is designed to mitigate the classic
high-cardinality selection bias mechanism of greedy impurity optimization (with
the usual fixed-node/root scope caveats for adaptive trees).

## Why citrees?

Traditional decision trees (CART, ID3, C4.5) suffer from **variable selection
bias**:

| Problem                | CART Behavior                           | citrees Solution                                                                             |
| ---------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Selection bias**     | Favors high-cardinality features        | Fixed-node Stage A permutation screening mitigates this multiplicity mechanism               |
| **Spurious splits**    | Finds "good" splits by chance           | Stage A must accept a feature before threshold search; Stage B is an algorithmic split score |
| **Overfitting**        | Requires pruning/cross-validation       | Principled stopping via hypothesis tests                                                     |
| **Feature importance** | Biased toward frequently-split features | Still uses impurity-decrease importance; Stage A mitigates a key root-level bias mechanism   |

## Installation

```bash
# From source
git clone https://github.com/rmill040/citrees.git
cd citrees
pip install -e .

# With uv (recommended)
uv sync
```

The installable `citrees` package has a small runtime dependency set. The
benchmark and manuscript pipeline under `paper/` uses additional optional
dependencies and is installed separately with the `paper` extra or dependency
group.

## Experiment CLI (`citrees-exp`)

This repository includes a Typer-based CLI for running the paper experiments,
managing the API-server/worker experiment pipeline, managing AWS infrastructure,
and monitoring progress.

```bash
# Install experiment CLI deps
uv sync --group paper
# or
pip install -e '.[paper]'

citrees-exp --help
```

## Quick Start

```python
from citrees import (
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
    MaxValuesMethod,
)

# Classification
clf = ConditionalInferenceTreeClassifier(
    selector="mc",           # Multiple correlation for feature selection
    splitter="gini",         # Gini impurity for split quality
    alpha_selector=0.05,     # Screening threshold for feature selection
    alpha_splitter=0.05,     # Screening threshold for split selection
)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# Regression
reg = ConditionalInferenceTreeRegressor(
    selector="pc",           # Pearson correlation
    splitter="mse",          # Mean squared error
)
reg.fit(X_train, y_train)

# Forest ensemble (parallel training)
forest = ConditionalInferenceForestClassifier(
    n_estimators=100,
    max_features=MaxValuesMethod.SQRT,  # Random feature subset per split
    n_jobs=-1,                          # Use all cores
)
forest.fit(X_train, y_train)
forest_predictions = forest.predict(X_test)
forest_probabilities = forest.predict_proba(X_test)

# Feature importance (impurity decrease; subject to known caveats)
importances = forest.feature_importances_
```

## Key Features

### Statistical Feature Selection

At each node, features are tested for association with the target using
permutation tests. A feature must pass the configured Stage A screening rule
before citrees searches over thresholds for that feature. With fixed-$B$
permutation tests, the root-node p-values have the usual exchangeability-based
interpretation; with the default adaptive stopping, the values are best read as
algorithmic screening and stopping scores.

### Multiple Selector Methods

| Selector | Task           | Description                        | Complexity |
| -------- | -------------- | ---------------------------------- | ---------- |
| `mc`     | Classification | Multiple correlation (ANOVA-based) | O(n)       |
| `mi`     | Classification | Mutual information                 | O(n log n) |
| `rdc`    | Both           | Randomized Dependence Coefficient  | O(n log n) |
| `pc`     | Regression     | Pearson correlation                | O(n)       |
| `dc`     | Regression     | Distance correlation               | O(n²)      |

### Advanced Capabilities

- **Bonferroni Correction**: Controls nodewise fixed-$B$ Stage A rejection
  probability under the complete permutation null when testing multiple features
- **Feature Muting**: Removes Stage A non-rejecting tested features from
  descendant feature pools
- **Honest Estimation**: Optional sample splitting to reduce adaptive bias in
  leaf estimation (Wager & Athey, 2018)

## Algorithm Overview

The conditional inference algorithm (Hothorn et al., 2006) proceeds as follows:

```
Algorithm: Conditional Inference Tree
Input: Data (X, y), screening thresholds α_select, α_split

function BuildTree(X, y, depth):
    # Step 1: Test global null hypothesis
    for each feature j in {1, ..., p}:
        H₀: X_j ⊥ Y (feature j independent of target)
        p_j ← PermutationTest(X_j, y)

    # Apply Bonferroni correction
    α_adjusted ← α_select / p

    # Select feature with strongest association (with feature_scanning=True,
    # the default, features are tested in order of their statistic and the
    # first rejection is taken)
    j* ← argmin(p_j)

    if p_j* ≥ α_adjusted:
        return LeafNode(y)  # No feature passes Stage A screening

    # Step 2: Find an algorithmic split point for the selected feature
    for each threshold c in X_j*:
        H₀: Fixed-feature threshold score under label exchangeability
        p_c ← PermutationTest(X_j*, y, c)

    c* ← argmin(p_c)

    # Per-threshold Bonferroni over the K candidates (adjust_alpha_splitter=True,
    # the default); threshold_test='maxt' tests the candidates jointly instead
    if p_c* ≥ α_split / K:
        return LeafNode(y)  # No threshold passes the split rule

    # Step 3: Recurse
    left ← {i : X_ij* ≤ c*}
    right ← {i : X_ij* > c*}

    return InternalNode(
        feature=j*, threshold=c*,
        left=BuildTree(X[left], y[left], depth+1),
        right=BuildTree(X[right], y[right], depth+1)
    )
```

## Documentation

| Document                                       | Description                                     |
| ---------------------------------------------- | ----------------------------------------------- |
| [Algorithm Details](docs/algorithm.md)         | Deep dive into conditional inference            |
| [Selectors](docs/selectors.md)                 | Feature selection methods (mc, mi, rdc, pc, dc) |
| [Splitters](docs/splitters.md)                 | Split criteria (gini, entropy, mse, mae)        |
| [Permutation Tests](docs/permutation-tests.md) | Nodewise permutation tests and scope caveats    |
| [Honest Estimation](docs/honest-estimation.md) | Sample splitting for leaf estimation            |
| [Max-type split test](docs/maxt.md)            | Joint test over candidate thresholds            |
| [Parameters](docs/parameters.md)               | Complete parameter reference                    |
| [Basic Usage Example](examples/basic_usage.py) | Runnable classifier, regressor, and forest demo |

## Parameters Reference

The complete parameter reference with defaults and tuning guidance is
[docs/parameters.md](docs/parameters.md). The settings most users touch:

| Parameter                                      | Default                     | Description                                                           |
| ---------------------------------------------- | --------------------------- | --------------------------------------------------------------------- |
| `selector`                                     | `'mc'` (clf) / `'pc'` (reg) | Stage A statistic: `mc`, `mi`, `rdc` (clf); `pc`, `dc`, `rdc` (reg)   |
| `alpha_selector`, `alpha_splitter`             | 0.05                        | Significance levels of the two stages                                 |
| `n_resamples_selector`, `n_resamples_splitter` | `'auto'`                    | Permutation budget: `'minimum'`, `'auto'`, `'maximum'`, or an int     |
| `threshold_method`, `max_thresholds`           | `'exact'`, None             | Candidate thresholds; `'histogram'` with 256 is the benchmark setting |
| `threshold_test`                               | `'bonferroni'`              | Per-threshold Bonferroni or joint `'maxt'` Stage B test               |

Every other argument (early stopping, scanning, muting, honesty, forest
controls, `scale_resamples_with_tests`) is documented in
[docs/parameters.md](docs/parameters.md). Its Configurations section defines the
default, benchmark, and recommended configurations that the documentation and
papers refer to.

## Use Cases

citrees is intended for scenarios requiring:

- **Feature screening with reduced split-selection bias** for interpretability
- **High-dimensional data** where many features are noise
- **Permutation-based nodewise screening** for scientific applications
- **Sample-split leaf estimation** with honest estimation, without standalone
  confidence-interval or coverage guarantees

## Development

```bash
# Install with dev dependencies
uv sync

# Run tests
uv run pytest tests/unit tests/integration -v

# Run linters
uv run pre-commit run --all-files
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and
[SUPPORT.md](SUPPORT.md) for issue-reporting and support expectations.

## Releases

Release notes are maintained in [CHANGELOG.md](CHANGELOG.md).

## Citation

```bibtex
@software{citrees,
  title = {citrees: Conditional Inference Trees and Forests for Python},
  author = {Milletich, Robert and Downes, Justin and Goley, Steve and Hirst, Newel},
  year = {2026},
  url = {https://github.com/rmill040/citrees}
}
```

## References

### Core Papers

1. Hothorn, T., Hornik, K., & Zeileis, A. (2006).
   [Unbiased Recursive Partitioning: A Conditional Inference Framework](https://www.tandfonline.com/doi/abs/10.1198/106186006X133933).
   _Journal of Computational and Graphical Statistics_, 15(3), 651-674.

2. Strobl, C., Boulesteix, A. L., Zeileis, A., & Hothorn, T. (2007).
   [Bias in Random Forest Variable Importance Measures](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25).
   _BMC Bioinformatics_, 8(1), 25.

3. Strobl, C., Boulesteix, A. L., Kneib, T., Augustin, T., & Zeileis, A. (2008).
   [Conditional Variable Importance for Random Forests](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-307).
   _BMC Bioinformatics_, 9(1), 307.

### Additional Methods

4. Wager, S., & Athey, S. (2018).
   [Estimation and Inference of Heterogeneous Treatment Effects using Random Forests](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1319839).
   _JASA_, 113(523), 1228-1242.

5. Lopez-Paz, D., Hennig, P., & Schölkopf, B. (2013).
   [The Randomized Dependence Coefficient](https://arxiv.org/abs/1304.7717).
   _NeurIPS_.

6. Székely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007).
   [Measuring and Testing Dependence by Correlation of Distances](https://projecteuclid.org/journals/annals-of-statistics/volume-35/issue-6/Measuring-and-testing-dependence-by-correlation-of-distances/10.1214/009053607000000505.full).
   _Annals of Statistics_, 35(6), 2769-2794.

## License

MIT License - see [LICENSE](LICENSE) for details.
