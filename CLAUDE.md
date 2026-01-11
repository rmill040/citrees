# CLAUDE.md - AI Assistant Guide for citrees

## Project Overview

**citrees** is a Python library implementing conditional inference trees and
forests. It uses permutation-based hypothesis testing for variable selection at
each node split, providing statistically principled alternatives to traditional
CART-style decision trees.

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/

# With pre-commit hooks
uv run pre-commit install
```

## Repository Structure

```
citrees/
├── pyproject.toml          # uv/pip config, dependencies, tool settings
├── uv.lock                 # Locked dependencies
├── tox.ini                 # Tox test configuration
├── .pre-commit-config.yaml # Pre-commit hooks (black, ruff, mypy)
├── citrees/                # Main package
│   ├── __init__.py         # Exports main classes
│   ├── _tree.py            # Tree classifiers/regressors
│   ├── _forest.py          # Forest ensembles
│   ├── _selector.py        # Feature selection methods (mc, mi, rdc, pc, dc)
│   ├── _splitter.py        # Split criteria (gini, entropy, mse, mae)
│   ├── _threshold_method.py # Threshold calculation methods
│   ├── _registry.py        # Registry pattern for selectors/splitters
│   ├── _conformal.py       # Conformal prediction wrappers
│   ├── _utils.py           # Utility functions
│   └── py.typed            # PEP 561 marker
├── tests/                  # Pytest test suite
│   ├── data/               # Test datasets (parquet format)
│   ├── integration/        # Integration tests
│   └── unit/               # Unit tests
└── paper/                  # Research paper experiments
    ├── data/               # Experiment datasets
    ├── results/            # Experiment outputs (parquet, figures, tables)
    └── scripts/
        ├── configs.py              # Experiment config dataclasses
        ├── synthetic_experiments.py # Synthetic data feature selection
        ├── analysis.py             # Statistical tests and visualizations
        ├── clf_feature_selection_server.py  # Classification experiments
        ├── reg_feature_selection_server.py  # Regression experiments
        └── generate_figures.py     # Paper figure generation
```

## Core Classes

```python
from citrees import (
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
)
```

## Architecture

### Registry Pattern

Selectors and splitters are registered via decorators:

```python
from ._registry import ClassifierSelectors

@ClassifierSelectors.register("mc")
@njit(nogil=True, fastmath=True)
def mc(x, y, n_classes, random_state=None):
    ...
```

Available registries:

- `ClassifierSelectors` / `ClassifierSelectorTests` - mc, mi, rdc
- `RegressorSelectors` / `RegressorSelectorTests` - pc, dc, rdc
- `ClassifierSplitters` / `ClassifierSplitterTests` - gini, entropy
- `RegressorSplitters` / `RegressorSplitterTests` - mse, mae
- `ThresholdMethods` - exact, random, percentile, histogram

### Key Parameters

| Parameter              | Description                                                 | Default                    |
| ---------------------- | ----------------------------------------------------------- | -------------------------- |
| `selector`             | Feature selection method: str or list[str]                  | 'mc' (clf) / 'pc' (reg)    |
| `splitter`             | Split criterion                                             | 'gini' (clf) / 'mse' (reg) |
| `alpha_selector`       | P-value threshold for feature selection                     | 0.05                       |
| `alpha_splitter`       | P-value threshold for split selection                       | 0.05                       |
| `n_resamples_selector` | Permutation resamples: 'auto', 'minimum', 'maximum', or int | 'auto'                     |
| `adjust_alpha_*`       | Bonferroni correction                                       | True                       |
| `early_stopping_*`     | Stop on first significant result                            | True                       |
| `feature_muting`       | Remove uninformative features                               | True                       |
| `feature_scanning`     | Sort features by promise before testing                     | True                       |
| `threshold_method`     | How to generate split candidates                            | 'exact'                    |
| `max_features`         | Features per split: 'sqrt', 'log2', float, int              | None (all)                 |
| `max_thresholds`       | Thresholds per feature                                      | None (all)                 |

### Selector Parameter

The `selector` parameter accepts either a single string or a list of strings:

**Classification selectors:**

- `'mc'` - Multiple correlation (ANOVA-based, [0,1] scale)
- `'mi'` - Mutual information (unbounded scale, cannot be combined with others)
- `'rdc'` - Randomized Dependence Coefficient (O(n log n), [0,1] scale)

**Regression selectors:**

- `'pc'` - Pearson correlation ([0,1] scale after abs)
- `'dc'` - Distance correlation (O(n²), [0,1] scale)
- `'rdc'` - Randomized Dependence Coefficient (O(n log n), [0,1] scale)

**List-based selector (multi-selector mode):** When a list is provided, all
selectors compute their scores and the maximum is used. The permutation test is
run on the selector that produced the highest score.

```python
# Classification: only mc and rdc can be combined (both [0,1] scale)
clf = ConditionalInferenceTreeClassifier(selector=['mc', 'rdc'])

# Regression: all three can be combined
reg = ConditionalInferenceTreeRegressor(selector=['pc', 'dc', 'rdc'])
```

**Note:** For classification, `'mi'` cannot be in a list because mutual
information is unbounded [0, ∞) while `'mc'` and `'rdc'` are on [0,1] scale.

### Early Stopping and Statistical Inference

When `early_stopping_selector=True` or `early_stopping_splitter=True`, the
permutation test stops as soon as the p-value falls below alpha. This is
computationally efficient but affects p-value precision.

**For valid statistical inference (e.g., research papers):**

- Set `early_stopping_selector=False` and `early_stopping_splitter=False`
- Use `n_resamples_selector='maximum'` and `n_resamples_splitter='maximum'`

```python
# Rigorous mode for statistical inference
clf = ConditionalInferenceTreeClassifier(
    early_stopping_selector=False,
    early_stopping_splitter=False,
    n_resamples_selector='maximum',
    n_resamples_splitter='maximum',
)
```

**For faster training (acceptable for most applications):**

- Keep defaults (`early_stopping=True`, `n_resamples='auto'`)
- Results are valid but p-values are approximate upper bounds

**P-value correction:** The permutation test uses the Phipson & Smyth (2010) +1
correction: `p = (b+1)/(m+1)` instead of `p = b/m`. This ensures p-values are
never exactly zero, which is critical for multiple testing correction.

Reference: Phipson & Smyth (2010). "Permutation P-values Should Never Be Zero."
SAGMB 9(1):39. https://pubmed.ncbi.nlm.nih.gov/21044043/

### Pydantic Validation

All parameters are validated via `BaseConditionalInferenceTreeParameters`:

```python
from typing import Annotated

from pydantic import BaseModel, Field

ProbabilityFloat = Annotated[float, Field(gt=0.0, le=1.0)]

class BaseConditionalInferenceTreeParameters(BaseModel):
    alpha_selector: ProbabilityFloat
    min_samples_split: Annotated[int, Field(ge=2)]
    ...
```

## Development

### Dependencies

```toml
python = ">=3.12"
numpy = ">=1.26"
numba = ">=0.60"
scikit-learn = ">=1.5"
scipy = ">=1.14"
dcor = ">=0.6"           # Distance correlation
pydantic = ">=2.0"       # Validation
```

### Code Style

- **Formatter**: black (line-length=120)
- **Linter**: ruff
- **Type checker**: mypy (strict)
- **Import sorter**: isort

### Testing

```bash
uv run pytest tests/ -v                           # Run all tests
uv run pytest tests/integration/ -v               # Run integration tests
uv run pytest tests/unit/ -v                      # Run unit tests
uv run pytest -k "classifier" -v                  # Run by keyword
uv run pytest --cov=citrees --cov-report=term-missing  # With coverage
```

#### Testing Numba Functions

Numba `@njit` decorated functions compile to machine code, so **pytest-cov cannot
track line coverage inside them**. To get coverage of these functions, use the
`.py_func` attribute which returns the original Python function:

```python
from citrees._splitter import gini

def test_gini_py_func():
    """Test gini via .py_func for coverage tracking."""
    y = np.array([0, 0, 1, 1], dtype=np.int64)

    # Test JIT version (ensures compiled code works)
    result_jit = gini(y)

    # Test py_func version (enables coverage tracking)
    result_py = gini.py_func(y)

    # Verify both produce identical results
    assert result_jit == result_py == pytest.approx(0.5)
```

**Best practices for Numba testing:**

1. **Test both versions**: JIT-compiled code can behave differently across OS/platforms
2. **Consistency checks**: Always verify `func(x) == func.py_func(x)` for the same input
3. **Use `.py_func` in test class**: Group all `.py_func` tests in a `TestModulePyFunc` class
4. **Document the pattern**: Include a docstring explaining why `.py_func` is used

**Alternative**: Set `NUMBA_DISABLE_JIT=1` before importing Numba to disable JIT
globally, but this must be done before any Numba imports and is all-or-nothing.

### Scratch Directory

Use `scratch/` for experimental code and proving concepts. Add to .gitignore.

### Bug-Fix Pipeline (Repro → Fix → Verify → Tests)

When fixing a bug, do not jump straight to a code change. Follow this pipeline
and **do not claim results you did not run**.

1. **Prove the bug exists (repro script in `scratch/`)**
   - Write a minimal, deterministic repro script in `scratch/` (e.g.,
     `scratch/repro_<issue>.py`).
   - The script must use **real code paths** and, when needed, **real models +
     real data** (prefer existing datasets under `tests/data/`).
   - Hard-require the expected failure (e.g., `assert`, explicit exception) so
     it’s unambiguous.
   - Print enough context to debug: library versions, random seeds, key params.
   - Run it and confirm it fails:
     ```bash
     uv run python scratch/repro_<issue>.py
     ```

2. **Implement the fix (minimal + targeted)**
   - Fix the root cause in the library (avoid unrelated refactors).
   - Keep the repro script unchanged unless the bug definition changes.

3. **Verify the fix (re-run the same repro)**
   - Re-run the exact script and confirm it now passes:
     ```bash
     uv run python scratch/repro_<issue>.py
     ```

4. **Backstop with tests (unit/integration/regression)**
   - Convert the repro into a **small, deterministic** test:
     - **Unit test** for the smallest failing component.
     - **Integration/regression test** if the failure spans estimators,
       fit/predict, parallelism, or I/O.
   - Avoid network and large datasets in tests; prefer `tests/data/` or small
     generated fixtures with fixed `random_state`.
   - Run the relevant test(s):
     ```bash
     uv run pytest -k "<relevant_keyword>" -v
     ```

---

# Research Roadmap

## Current Limitations vs State of the Art

| Area               | citrees Now  | SOTA (2024-2025)         | Priority |
| ------------------ | ------------ | ------------------------ | -------- |
| Feature Importance | MDI only     | SHAP/TreeSHAP            | HIGH     |
| Uncertainty        | None         | Conformal Prediction     | HIGH     |
| Causal Inference   | None         | Honest Estimation, GRF   | MEDIUM   |
| Speed              | CPU/Numba    | GPU (cuML 20-45x faster) | MEDIUM   |
| Tree Structure     | Axis-aligned | Oblique, Neural Trees    | LOW      |

---

## HIGH PRIORITY

### 1. SHAP/TreeSHAP Integration

**Why**: Current `feature_importances_` (MDI) has known biases with correlated
features. SHAP provides theoretically grounded, consistent feature attributions.

**Research**:

- [TreeSHAP](https://shap.readthedocs.io/en/latest/) - O(TLD²) complexity for
  tree ensembles
- [FastTreeSHAP](https://engineering.linkedin.com/blog/2022/fasttreeshap--accelerating-shap-value-computation-for-trees) -
  2.5x faster (NeurIPS 2021)
- [2024 Study](https://link.springer.com/article/10.1186/s40537-024-00905-w) -
  Compared SHAP vs importance-based selection

**Implementation**:

```python
# Add to _tree.py or new _explainer.py
def shap_values(self, X):
    """Compute SHAP values using TreeSHAP algorithm."""
    import shap
    explainer = shap.TreeExplainer(self)
    return explainer.shap_values(X)
```

**Effort**: ~2-3 days (can wrap existing `shap` library)

---

### 2. Conformal Prediction for Uncertainty Quantification

**Why**: No current way to get prediction intervals or confidence sets. Critical
for high-stakes applications.

**Research**:

- [Conformal Prediction Survey](https://dl.acm.org/doi/10.1145/3736575) - ACM
  Computing Surveys 2024
- [Mondrian Conformal Prediction](https://www.sciencedirect.com/science/article/abs/pii/S0031320325009999) -
  Handles heteroscedasticity
- [Regression Trees for Prediction Intervals](https://www.sciencedirect.com/science/article/abs/pii/S0020025524012830) -
  2024

**Implementation**:

```python
class ConformalForestClassifier(ConditionalInferenceForestClassifier):
    def predict_set(self, X, alpha=0.1):
        """Return prediction sets with 1-alpha coverage guarantee."""
        # Split-conformal or CV+ approach
        ...

class ConformalForestRegressor(ConditionalInferenceForestRegressor):
    def predict_interval(self, X, alpha=0.1):
        """Return prediction intervals with 1-alpha coverage."""
        ...
```

**Effort**: ~1 week (algorithm implementation + calibration)

---

## MEDIUM PRIORITY

### 3. Honest Estimation (Sample Splitting)

**Why**: Current trees use same data for splitting and estimation, causing bias.
Honest trees use separate samples, enabling valid inference.

**Research**:

- [Wager & Athey 2018](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1319839) -
  Causal Forests paper
- [GRF Package](https://grf-labs.github.io/grf/) - Gold standard implementation
- [Honesty Trade-offs](https://arxiv.org/html/2506.13107v2) - When it helps vs
  hurts (2025)

**Implementation**:

```python
class HonestConditionalInferenceTree(ConditionalInferenceTreeClassifier):
    def __init__(self, ..., honesty=True, honesty_fraction=0.5):
        self.honesty = honesty
        self.honesty_fraction = honesty_fraction

    def fit(self, X, y):
        if self.honesty:
            X_split, X_est, y_split, y_est = train_test_split(...)
            # Use X_split, y_split for tree structure
            # Use X_est, y_est for leaf estimates
```

**Effort**: ~1 week (requires careful handling of OOB samples)

---

### 4. GPU Acceleration

**Why**: Training on large datasets is slow. cuML achieves 20-45x speedups.

**Research**:

- [cuML Random Forests](https://developer.nvidia.com/blog/accelerating-random-forests-up-to-45x-using-cuml/) -
  NVIDIA RAPIDS
- [RFX](https://arxiv.org/html/2511.19493) - GPU + QLORA (Nov 2025)

**Options**:

1. **Wrap cuML**: Use cuML for splitting, keep citrees API
2. **CUDA kernels**: Custom kernels for permutation tests
3. **JAX/Triton**: Python-native GPU compilation

**Effort**: ~2-4 weeks depending on approach

---

## LOW PRIORITY (Future Research)

### 5. Oblique Decision Trees

**Why**: Axis-aligned splits are limiting. Oblique trees use linear combinations
of features.

**Research**:

- [TAO Algorithm](https://faculty.ucmerced.edu/mcarreira-perpinan/research/TAO.html) -
  Tree Alternating Optimization
- [DTSemNet](https://arxiv.org/abs/2408.09135) - Vanilla gradient descent for
  oblique trees (2024)
- [Statistical Advantages](https://arxiv.org/abs/2407.02458) - Oblique Mondrian
  trees (2024)

### 6. Neural/Differentiable Trees

**Why**: End-to-end training with neural networks.

**Research**:

- [NCART](https://www.sciencedirect.com/science/article/abs/pii/S0031320324003297) -
  Neural CART (2024)
- [Deep Neural Decision Trees](https://arxiv.org/pdf/1806.06988)
- [Neural-Backed Decision Trees](https://research.alvinwan.com/neural-backed-decision-trees/)

### 7. Gradient Boosting Integration

**Why**: Combine conditional inference with boosting.

**Research**:

- [LightGBM techniques](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree) -
  GOSS, EFB
- [Piecewise Linear Trees](https://nimasarang.com/blog/2025-12-14-gbt-algorithms/) -
  PL-Trees in boosting

---

## Benchmarking

Compare against:

- R's `partykit::ctree` and `party::cforest` (original implementations)
- `grf` package (causal forests)
- scikit-learn `RandomForest`, `ExtraTrees`
- XGBoost, LightGBM, CatBoost

Use [TabZilla Benchmark Suite](https://arxiv.org/abs/2305.02997) for
standardized evaluation.

---

## References

### Core Papers

- Hothorn et al. (2006) - "Unbiased Recursive Partitioning" (original ctree)
- Strobl et al. (2007) - "Bias in Random Forest Variable Importance"
- Strobl et al. (2008) - "Conditional Variable Importance for Random Forests"

### Recent Advances

- [Causal Forests](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1319839) -
  Wager & Athey (2018)
- [GRF](https://grf-labs.github.io/grf/) - Athey, Tibshirani, Wager (2019)
- [TabZilla](https://arxiv.org/abs/2305.02997) - When NNs beat trees (2023)
- [Conditional Permutation Importance](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03622-2)
  (2020)
- [Distance Correlation Feature Selection](https://www.mdpi.com/1099-4300/25/9/1250)
  (2023)

---

## Notes for AI Assistants

1. **Pydantic v2**: Uses Pydantic v2 APIs (e.g., `Field`, `field_validator`,
   `model_validator`, `model_config`)
2. **Numba JIT**: Performance-critical functions use `@njit`
3. **Registry pattern**: Add new selectors/splitters via
   `@Registry.register("name")`
4. **Type hints**: All functions should have type annotations
5. **Tests**: Use pytest markers (`@pytest.mark.tree`, etc.)
6. **Data format**: Test data uses parquet (pyarrow)
