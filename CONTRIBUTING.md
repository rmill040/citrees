# Contributing to citrees

Thank you for your interest in contributing to citrees! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/rmill040/citrees.git
cd citrees

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"

# Install pre-commit hooks
uv run pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test files
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v

# Run with coverage
uv run pytest tests/ --cov=citrees --cov-report=html
```

### Code Style

We use the following tools for code quality:

- **black**: Code formatting (line-length=120)
- **ruff**: Linting
- **mypy**: Type checking (strict mode)
- **isort**: Import sorting

Run all checks:

```bash
uv run pre-commit run --all-files
```

## Project Structure

```
citrees/
├── citrees/              # Main package
│   ├── _tree.py          # Tree classifiers/regressors
│   ├── _forest.py        # Forest ensembles
│   ├── _selector.py      # Feature selection methods
│   ├── _splitter.py      # Split criteria
│   ├── _threshold_method.py  # Threshold generation
│   ├── _conformal.py     # Conformal prediction
│   ├── _importance.py    # Feature importance methods
│   ├── _registry.py      # Registry pattern
│   └── _utils.py         # Utilities
├── tests/                # Test suite
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
└── docs/                 # Documentation (MkDocs)
```

## Adding New Features

### Adding a New Selector

Selectors measure association between features and targets. To add a new one:

1. Add your function to `citrees/_selector.py`
2. Register it with the appropriate registry decorator
3. Add corresponding permutation test function

```python
from citrees._registry import ClassifierSelectors, ClassifierSelectorTests

@ClassifierSelectors.register("my_selector")
def my_selector(x: np.ndarray, y: np.ndarray, n_classes: int, random_state: int) -> float:
    """My custom selector.

    Parameters
    ----------
    x : np.ndarray
        Feature values.
    y : np.ndarray
        Target values.
    n_classes : int
        Number of classes.
    random_state : int
        Random seed.

    Returns
    -------
    float
        Association score in [0, 1].
    """
    # Your implementation
    return score

@ClassifierSelectorTests.register("my_selector")
def ptest_my_selector(
    *,
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_resamples: int,
    early_stopping: bool,
    alpha: float,
    random_state: int,
) -> float:
    """Permutation test for my_selector."""
    return _permutation_test(
        func=my_selector,
        func_arg=n_classes,
        x=x,
        y=y,
        n_resamples=n_resamples,
        early_stopping=early_stopping,
        alpha=alpha,
        random_state=random_state,
    )
```

### Adding a New Splitter

Splitters compute impurity for split selection:

```python
from citrees._registry import ClassifierSplitters, ClassifierSplitterTests

@ClassifierSplitters.register("my_splitter")
@njit(cache=True, fastmath=True, nogil=True)
def my_splitter(y: np.ndarray) -> float:
    """My custom impurity measure."""
    # Your implementation
    return impurity
```

### Performance Considerations

- Use `@njit` decorator for performance-critical functions
- Prefer in-place operations to reduce memory allocations
- Add parallel versions (`parallel=True`) for CPU-bound loops

## Pull Request Process

1. **Create a branch** from `master` for your changes
2. **Write tests** for new functionality
3. **Update documentation** if needed
4. **Run the test suite** and ensure all tests pass
5. **Run pre-commit hooks** to check code style
6. **Submit a PR** with a clear description of changes

### PR Guidelines

- Keep PRs focused on a single feature or fix
- Include tests for new code
- Update CHANGELOG.md for user-facing changes
- Reference related issues in the PR description

## Reporting Issues

When reporting bugs, please include:

- Python version and OS
- citrees version (`pip show citrees`)
- Minimal reproducible example
- Full error traceback

## Questions?

Open an issue on GitHub for questions about contributing.
