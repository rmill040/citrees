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

| Document | Description |
|----------|-------------|
| [Algorithm](algorithm.md) | Deep dive into the conditional inference algorithm |
| [Parameters](parameters.md) | Complete parameter reference with tuning guidance |
| [Comparison](comparison.md) | How citrees differs from Random Forest, XGBoost, etc. |
| [Tutorial](tutorial.md) | Getting started guide with examples |
| [API Reference](api.md) | Class and method documentation |

## Quick Start

```python
from citrees import (
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceForestClassifier,
)

# Single tree
tree = ConditionalInferenceTreeClassifier(
    alpha_selector=0.05,    # Significance level for feature selection
    alpha_splitter=0.05,    # Significance level for split selection
)
tree.fit(X_train, y_train)

# Forest ensemble
forest = ConditionalInferenceForestClassifier(
    n_estimators=100,
    max_features="sqrt",    # Random feature subset per split
    n_jobs=-1,              # Parallel training
)
forest.fit(X_train, y_train)

# Feature importance (statistically grounded)
print(forest.feature_importances_)
```

## Key Features

- **Statistical Feature Selection**: Permutation tests ensure only significant features are used
- **Unbiased Variable Importance**: Importance based on impurity decrease, not split counts
- **Feature Muting**: Automatically removes clearly uninformative features during training
- **Honest Estimation**: Optional sample splitting for unbiased leaf predictions
- **Conformal Prediction**: Built-in uncertainty quantification
- **SHAP Support**: TreeSHAP-compatible explanations

## Installation

```bash
# From PyPI (when published)
pip install citrees

# From source
git clone https://github.com/rmill040/citrees.git
cd citrees
pip install -e .
```

## Citation

If you use citrees in your research, please cite:

```bibtex
@software{citrees,
  title = {citrees: Conditional Inference Trees and Forests for Python},
  author = {Miller, Ryan},
  year = {2024},
  url = {https://github.com/rmill040/citrees}
}
```

## References

The implementation is based on:

1. Hothorn, T., Hornik, K., & Zeileis, A. (2006). **Unbiased recursive partitioning: A conditional inference framework.** *Journal of Computational and Graphical Statistics*, 15(3), 651-674.

2. Strobl, C., Boulesteix, A. L., Zeileis, A., & Hothorn, T. (2007). **Bias in random forest variable importance measures: Illustrations, sources and a solution.** *BMC Bioinformatics*, 8(1), 25.

3. Strobl, C., Boulesteix, A. L., Kneib, T., Augustin, T., & Zeileis, A. (2008). **Conditional variable importance for random forests.** *BMC Bioinformatics*, 9(1), 307.
