# citrees Documentation

**Conditional Inference Trees and Forests for Python**

citrees implements statistically principled decision trees and random forests that use permutation-based hypothesis testing for variable selection. Unlike traditional CART-style trees that greedily optimize split criteria, conditional inference trees separate the variable selection step from the split point selection step, using statistical tests to determine significance.

## Why citrees?

Traditional decision trees (CART, ID3, C4.5) and their ensemble variants (Random Forest, Gradient Boosting) suffer from **variable selection bias** - they tend to favor variables with many possible split points or high cardinality. This leads to:

1. **Biased feature importance** - Variables are ranked by how often they're used, not their true predictive power
2. **Overfitting to noise** - Trees can find spurious splits on uninformative features by chance
3. **Unreliable variable selection** - Important features may be masked by correlated noise features

citrees addresses these issues by using **permutation tests** at each node to:
- Test whether ANY feature is significantly associated with the target (feature selection)
- Test whether a split actually improves prediction (split selection)

## Documentation

### Core Concepts

| Document | Description |
|----------|-------------|
| [Algorithm](algorithm.md) | Deep dive into the conditional inference algorithm |
| [Selectors](selectors.md) | Feature selection methods (mc, mi, rdc, pc, dc) |
| [Splitters](splitters.md) | Split criteria (gini, entropy, mse, mae) |
| [Permutation Tests](permutation-tests.md) | Statistical testing framework |

### Advanced Features

| Document | Description |
|----------|-------------|
| [Honest Estimation](honest-estimation.md) | Sample splitting for unbiased predictions |
| [Conformal Prediction](conformal-prediction.md) | Uncertainty quantification with coverage guarantees |
| [SHAP Integration](shap.md) | TreeSHAP feature attributions |

### Reference

| Document | Description |
|----------|-------------|
| [Parameters](parameters.md) | Complete parameter reference with tuning guidance |
| [API Reference](api.md) | Class and method documentation |

## Quick Start

```python
from citrees import (
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceForestClassifier,
)

# Single tree
tree = ConditionalInferenceTreeClassifier(
    selector="mc",           # Multiple correlation for feature selection
    splitter="gini",         # Gini impurity for split quality
    alpha_selector=0.05,     # Significance level for feature selection
    alpha_splitter=0.05,     # Significance level for split selection
)
tree.fit(X_train, y_train)

# Forest ensemble
forest = ConditionalInferenceForestClassifier(
    n_estimators=100,
    max_features="sqrt",     # Random feature subset per split
    n_jobs=-1,               # Parallel training
)
forest.fit(X_train, y_train)

# Feature importance (statistically grounded)
print(forest.feature_importances_)
```

## Key Features

### Statistical Foundation
- **Permutation Tests**: Non-parametric hypothesis tests at each node
- **Bonferroni Correction**: Controls family-wise error rate
- **Early Stopping**: Principled stopping via statistical tests

### Feature Selection Methods
- **Multiple Correlation (mc)**: ANOVA-based for classification
- **Mutual Information (mi)**: Information-theoretic measure
- **Randomized Dependence Coefficient (rdc)**: Non-linear dependence
- **Pearson Correlation (pc)**: Linear correlation for regression
- **Distance Correlation (dc)**: Detects non-linear relationships

### Advanced Capabilities
- **Honest Estimation**: Sample splitting for unbiased leaf predictions
- **Conformal Prediction**: Distribution-free prediction intervals
- **SHAP Support**: TreeSHAP-compatible explanations
- **Feature Muting**: Automatically removes uninformative features

## Installation

```bash
# From source
git clone https://github.com/rmill040/citrees.git
cd citrees
pip install -e .

# With uv (recommended)
uv sync
```

## Viewing Documentation

The documentation is built with [MkDocs](https://www.mkdocs.org/) using the [Material theme](https://squidfunk.github.io/mkdocs-material/).

### Local Development Server

```bash
# Install documentation dependencies
uv sync --group docs

# Start local server (auto-reloads on changes)
uv run mkdocs serve

# View at http://127.0.0.1:8000
```

### Build Static Site

```bash
# Build HTML documentation
uv run mkdocs build

# Output in site/ directory
```

## Citation

If you use citrees in your research, please cite:

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

1. Hothorn, T., Hornik, K., & Zeileis, A. (2006). **Unbiased recursive partitioning: A conditional inference framework.** *Journal of Computational and Graphical Statistics*, 15(3), 651-674.

2. Strobl, C., Boulesteix, A. L., Zeileis, A., & Hothorn, T. (2007). **Bias in random forest variable importance measures: Illustrations, sources and a solution.** *BMC Bioinformatics*, 8(1), 25.

3. Strobl, C., Boulesteix, A. L., Kneib, T., Augustin, T., & Zeileis, A. (2008). **Conditional variable importance for random forests.** *BMC Bioinformatics*, 9(1), 307.

### Additional Methods

4. Wager, S., & Athey, S. (2018). **Estimation and Inference of Heterogeneous Treatment Effects using Random Forests.** *JASA*, 113(523), 1228-1242.

5. Lopez-Paz, D., Hennig, P., & Schölkopf, B. (2013). **The Randomized Dependence Coefficient.** *NeurIPS*.

6. Székely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007). **Measuring and Testing Dependence by Correlation of Distances.** *Annals of Statistics*, 35(6), 2769-2794.
