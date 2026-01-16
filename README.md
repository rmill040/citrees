# citrees

**Conditional Inference Trees and Forests for Python**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

citrees implements statistically principled decision trees and random forests
using permutation-based hypothesis testing for variable selection. Unlike
traditional CART-style trees that greedily optimize split criteria, citrees
separates variable selection from split point selection using statistical tests
to determine significance.

**Note**: citrees is _inspired by_ the conditional inference framework (Hothorn
et al., 2006) but is not a direct port of R's `partykit::ctree`. We implement
the core principles—permutation-based variable selection, statistical stopping
rules, and unbiased splitting—while adding our own extensions like RDC
selectors and feature muting.

## Why citrees?

Traditional decision trees (CART, ID3, C4.5) suffer from **variable selection
bias**:

| Problem                | CART Behavior                           | citrees Solution                                   |
| ---------------------- | --------------------------------------- | -------------------------------------------------- |
| **Selection bias**     | Favors high-cardinality features        | Permutation tests control for multiple comparisons |
| **Spurious splits**    | Finds "good" splits by chance           | Statistical significance required to split         |
| **Overfitting**        | Requires pruning/cross-validation       | Principled stopping via hypothesis tests           |
| **Feature importance** | Biased toward frequently-split features | Importance based on statistical contribution       |

## Installation

```bash
# From source
git clone https://github.com/rmill040/citrees.git
cd citrees
pip install -e .

# With uv (recommended)
uv sync
```

## Quick Start

```python
from citrees import (
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
)

# Classification
clf = ConditionalInferenceTreeClassifier(
    selector="mc",           # Multiple correlation for feature selection
    splitter="gini",         # Gini impurity for split quality
    alpha_selector=0.05,     # Significance level for feature selection
    alpha_splitter=0.05,     # Significance level for split selection
)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

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

# Feature importance (statistically grounded)
importances = forest.feature_importances_
```

## Key Features

### Statistical Feature Selection

At each node, features are tested for association with the target using
permutation tests. Only statistically significant features (p < alpha) are
considered for splitting.

### Multiple Selector Methods

| Selector | Task           | Description                        | Complexity |
| -------- | -------------- | ---------------------------------- | ---------- |
| `mc`     | Classification | Multiple correlation (ANOVA-based) | O(n)       |
| `mi`     | Classification | Mutual information                 | O(n log n) |
| `rdc`    | Both           | Randomized Dependence Coefficient  | O(n log n) |
| `pc`     | Regression     | Pearson correlation                | O(n)       |
| `dc`     | Regression     | Distance correlation               | O(n²)      |

### Advanced Capabilities

- **Bonferroni Correction**: Controls family-wise error rate when testing
  multiple features
- **Feature Muting**: Automatically removes clearly uninformative features
- **Honest Estimation**: Sample splitting for unbiased leaf predictions (Wager &
  Athey, 2018)

## Algorithm Overview

The conditional inference algorithm (Hothorn et al., 2006) proceeds as follows:

```
Algorithm: Conditional Inference Tree
Input: Data (X, y), significance levels α_select, α_split

function BuildTree(X, y, depth):
    # Step 1: Test global null hypothesis
    for each feature j in {1, ..., p}:
        H₀: X_j ⊥ Y (feature j independent of target)
        p_j ← PermutationTest(X_j, y)

    # Apply Bonferroni correction
    α_adjusted ← α_select / p

    # Select feature with strongest association
    j* ← argmin(p_j)

    if p_j* ≥ α_adjusted:
        return LeafNode(y)  # No significant feature found

    # Step 2: Find optimal split point
    for each threshold c in X_j*:
        H₀: Split at c provides no improvement
        p_c ← PermutationTest(X_j*, y, c)

    c* ← argmin(p_c)

    if p_c* ≥ α_split:
        return LeafNode(y)  # No significant split found

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
| [Permutation Tests](docs/permutation-tests.md) | Statistical testing framework                   |
| [Honest Estimation](docs/honest-estimation.md) | Sample splitting for causal inference           |

## Parameters Reference

### Core Parameters

| Parameter              | Type        | Default          | Description                             |
| ---------------------- | ----------- | ---------------- | --------------------------------------- |
| `selector`             | str or list | `'mc'`/`'pc'`    | Feature selection method                |
| `splitter`             | str         | `'gini'`/`'mse'` | Split criterion                         |
| `alpha_selector`       | float       | 0.05             | P-value threshold for feature selection |
| `alpha_splitter`       | float       | 0.05             | P-value threshold for split selection   |
| `n_resamples_selector` | NResamples/int/None | `NResamples.AUTO` | Permutation resamples for selector (`None` disables permutation tests) |
| `n_resamples_splitter` | NResamples/int/None | `NResamples.AUTO` | Permutation resamples for splitter (`None` disables permutation tests) |

### Optimization Parameters

| Parameter                 | Type | Default | Description                          |
| ------------------------- | ---- | ------- | ------------------------------------ |
| `adjust_alpha_selector`   | bool | True    | Bonferroni correction for features   |
| `adjust_alpha_splitter`   | bool | True    | Bonferroni correction for thresholds |
| `early_stopping_selector` | EarlyStopping/None | `EarlyStopping.ADAPTIVE` | Sequential stopping rule for selector permutation tests |
| `early_stopping_splitter` | EarlyStopping/None | `EarlyStopping.ADAPTIVE` | Sequential stopping rule for splitter permutation tests |
| `early_stopping_confidence_selector` | float | 0.95 | Posterior-confidence threshold γ for adaptive stopping (selectors) |
| `early_stopping_confidence_splitter` | float | 0.95 | Posterior-confidence threshold γ for adaptive stopping (splitters) |
| `feature_muting`          | bool | True    | Remove uninformative features        |
| `feature_scanning`        | bool | True    | Test promising features first        |

### Tree Structure Parameters

| Parameter           | Type          | Default   | Description                      |
| ------------------- | ------------- | --------- | -------------------------------- |
| `max_depth`         | int           | None      | Maximum tree depth               |
| `min_samples_split` | int           | 2         | Minimum samples to split         |
| `min_samples_leaf`  | int           | 1         | Minimum samples in leaf          |
| `max_features`      | str/int/float | None      | Features per split               |
| `threshold_method`  | ThresholdMethod | `ThresholdMethod.EXACT` | How to generate split candidates |

### Forest Parameters

| Parameter          | Type  | Default      | Description                      |
| ------------------ | ----- | ------------ | -------------------------------- |
| `n_estimators`     | int   | 100          | Number of trees                  |
| `max_samples`      | float | None         | Bootstrap sample size            |
| `bootstrap_method` | BootstrapMethod | `BootstrapMethod.BAYESIAN` | Sampling method                  |
| `n_jobs`           | int   | 1            | Parallel jobs (-1 for all cores) |

## Comparison with Other Methods

| Feature                 | citrees | sklearn RF | XGBoost | R partykit |
| ----------------------- | ------- | ---------- | ------- | ---------- |
| Variable selection bias | No      | Yes        | Yes     | No         |
| Statistical stopping    | Yes     | No         | No      | Yes        |
| Permutation tests       | Yes     | No         | No      | Yes        |
| Python native           | Yes     | Yes        | Yes     | No         |
| GPU support             | No      | No         | Yes     | No         |
| Honest estimation       | Yes     | No         | No      | Yes        |

## Benchmarks

citrees excels in scenarios requiring:

- **Unbiased feature selection** for interpretability
- **High-dimensional data** where many features are noise
- **Statistical rigor** for scientific applications
- **Causal inference** with honest estimation

For pure prediction performance on tabular data, gradient boosting methods
(XGBoost, LightGBM) typically achieve higher accuracy.

## Development

```bash
# Install with dev dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Run linters
uv run pre-commit run --all-files
```

## Citation

```bibtex
@software{citrees,
  title = {citrees: Conditional Inference Trees and Forests for Python},
  author = {Milletich, Robert J.},
  year = {2024},
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
