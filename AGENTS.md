# AI Assistant Guide for citrees

## Project Overview

**citrees** is a Python library implementing conditional inference trees and
forests. It uses permutation-based hypothesis testing for variable selection at
each node split, providing statistically principled alternatives to traditional
CART-style decision trees.

## Quick Start

```bash
# Install dependencies
UV_CACHE_DIR=./scratch/.uv_cache uv sync

# Run tests
UV_CACHE_DIR=./scratch/.uv_cache uv run pytest tests/

# With pre-commit hooks
UV_CACHE_DIR=./scratch/.uv_cache uv run pre-commit install
```

## Repository Structure

```
citrees/
├── pyproject.toml          # uv/pip config, dependencies, tool settings
├── uv.lock                 # Locked dependencies
├── mkdocs.yml              # Documentation config
├── .pre-commit-config.yaml # Pre-commit hooks (ruff, mypy)
├── citrees/                # Main package
│   ├── __init__.py         # Exports main classes
│   ├── _types.py           # Centralized StrEnums and type aliases
│   ├── _tree.py            # Tree classifiers/regressors
│   ├── _forest.py          # Forest ensembles
│   ├── _selector.py        # Feature selection methods (mc, mi, rdc, pc, dc)
│   ├── _splitter.py        # Split criteria (gini, entropy, mse, mae)
│   ├── _sequential.py      # Sequential stopping logic
│   ├── _threshold_method.py # Threshold calculation methods
│   ├── _registry.py        # Registry pattern for selectors/splitters
│   ├── _utils.py           # Utility functions
│   └── py.typed            # PEP 561 marker
├── tests/                  # Pytest test suite
│   ├── conftest.py         # Pytest fixtures and JIT control
│   ├── data/               # Test datasets (parquet format)
│   ├── unit/               # Unit tests for citrees/* modules
│   ├── integration/        # Integration tests for citrees/* (tree, forest, parameters, edge_cases)
│   └── paper/              # Tests for paper/scripts/* (use -m "not paper" to skip)
├── docs/                   # MkDocs documentation source
├── tools/                  # Development tools
│   └── hooks/              # Pre-commit hook scripts
└── paper/                  # Research paper experiments
    ├── arxiv/              # LaTeX manuscript (arXiv source)
    ├── docs/               # Experiment runbook, infra guide, claims tracker
    ├── data/               # Experiment datasets (parquet)
    ├── results/            # Experiment outputs (parquet, figures, tables)
    └── scripts/
        ├── adapters/       # External system adapters (S3, data loading, runner)
        ├── api/            # FastAPI queue server and pull-based worker
        ├── analysis/       # Statistical tests, visualizations, figures
        ├── cli/            # Typer CLI (citrees-exp entry point)
        ├── config/         # Configuration (settings, constants)
        ├── data_generation/# Synthetic dataset generation
        ├── infra/          # AWS setup (IAM, S3, ECR, EC2, Docker)
        ├── pipeline/       # Core experiment pipeline (stage1, stage2, grid, methods)
        ├── theory/         # Sequential stopping analysis scripts
        └── utils/          # Shared utilities (env, metrics)
```

## Core Classes

```python
from citrees import (
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
    # Enums for parameter options
    EarlyStopping,
    NResamples,
    ThresholdMethod,
    MaxValuesMethod,
    SamplingMethod,
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

| Parameter              | Description                             | Default                    |
| ---------------------- | --------------------------------------- | -------------------------- |
| `selector`             | Feature selection method: str or list   | 'mc' (clf) / 'pc' (reg)    |
| `splitter`             | Split criterion                         | 'gini' (clf) / 'mse' (reg) |
| `alpha_selector`       | P-value threshold for feature selection | 0.05                       |
| `alpha_splitter`       | P-value threshold for split selection   | 0.05                       |
| `n_resamples_selector` | NResamples enum or int                  | NResamples.AUTO            |
| `adjust_alpha_*`       | Bonferroni correction                   | True                       |
| `early_stopping_*`     | EarlyStopping enum or None              | EarlyStopping.ADAPTIVE     |
| `feature_muting`       | Remove uninformative features           | True                       |
| `feature_scanning`     | Sort features by promise before testing | True                       |
| `threshold_method`     | ThresholdMethod enum                    | ThresholdMethod.EXACT      |
| `max_features`         | MaxValuesMethod enum, float, or int     | None (all)                 |
| `max_thresholds`       | MaxValuesMethod enum, float, or int     | None (all)                 |

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

**List-based selector (multi-selector mode):** When a list is provided, citrees
uses the max-T method (Westfall & Young, 1993) for valid Type I error control.
The permutation test computes max(selector_scores) inside each permutation. Each
selector in the list must be unique (no duplicates allowed).

```python
# Classification: only mc and rdc can be combined (both [0,1] scale)
clf = ConditionalInferenceTreeClassifier(selector=['mc', 'rdc'])

# Regression: all three can be combined
reg = ConditionalInferenceTreeRegressor(selector=['pc', 'dc', 'rdc'])
```

**Note:** For classification, `'mi'` cannot be in a list because mutual
information is unbounded [0, ∞) while `'mc'` and `'rdc'` are on [0,1] scale.

### Early Stopping and Statistical Inference

The `early_stopping_selector` and `early_stopping_splitter` parameters control
how permutation tests terminate:

- `EarlyStopping.ADAPTIVE` (default): Bayesian Beta CDF stopping - valid Type I
  error (~5%), 95% faster
- `EarlyStopping.SIMPLE`: Futility + significance stopping - inflates Type I
  error to ~9%
- `None`: Full permutation test - no early stopping

**Default mode (recommended for most applications):**

```python
# Default: adaptive sequential testing with valid p-values
clf = ConditionalInferenceTreeClassifier(
    early_stopping_selector=EarlyStopping.ADAPTIVE,  # default
    early_stopping_splitter=EarlyStopping.ADAPTIVE,  # default
)
```

**Rigorous mode (maximum precision):**

```python
# Disable early stopping for maximum p-value precision
clf = ConditionalInferenceTreeClassifier(
    early_stopping_selector=None,
    early_stopping_splitter=None,
    n_resamples_selector=NResamples.MAXIMUM,
    n_resamples_splitter=NResamples.MAXIMUM,
)
```

**P-value correction:** The permutation test uses the Phipson & Smyth (2010) +1
correction: `p = (b+1)/(m+1)` instead of `p = b/m`. This ensures p-values are
never exactly zero, which is critical for multiple testing correction.

Reference: Phipson & Smyth (2010). "Permutation P-values Should Never Be Zero."
SAGMB 9(1):39. https://pubmed.ncbi.nlm.nih.gov/21044043/

### Pydantic Validation

All parameters are validated via `BaseConditionalInferenceTreeParameters`. Type
aliases and enums are centralized in `_types.py`:

```python
# citrees/_types.py
from enum import StrEnum
from typing import Annotated, TypeAlias
from pydantic import Field

ProbabilityFloat: TypeAlias = Annotated[float, Field(gt=0.0, le=1.0)]

class EarlyStopping(StrEnum):
    ADAPTIVE = "adaptive"
    SIMPLE = "simple"

class NResamples(StrEnum):
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    AUTO = "auto"
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
UV_CACHE_DIR=./scratch/.uv_cache uv run pytest tests/ -v                           # Run all tests (JIT enabled, fast)
UV_CACHE_DIR=./scratch/.uv_cache uv run pytest tests/ -m "not slow" -v             # Skip slow tests
UV_CACHE_DIR=./scratch/.uv_cache uv run pytest tests/integration/ -v               # Run integration tests
UV_CACHE_DIR=./scratch/.uv_cache uv run pytest tests/unit/ -v                      # Run unit tests
UV_CACHE_DIR=./scratch/.uv_cache uv run pytest -k "classifier" -v                  # Run by keyword
UV_CACHE_DIR=./scratch/.uv_cache uv run pytest --cov=citrees --cov-report=term-missing  # With coverage (JIT disabled)
```

#### Two Test Modes: JIT On vs Off

The test suite supports two modes controlled by `tests/conftest.py`:

| Mode     | JIT      | Speed  | Coverage | Use Case                           |
| -------- | -------- | ------ | -------- | ---------------------------------- |
| Default  | Enabled  | Fast   | No       | Validate compiled Numba code works |
| Coverage | Disabled | Slower | Yes      | Track line coverage for CI         |

**How it works:**

- JIT is **enabled by default** for fast test execution
- JIT is **automatically disabled** when running with `--cov` flag
- You can explicitly control JIT via `NUMBA_DISABLE_JIT` environment variable

```bash
# Fast tests (JIT enabled, validates compiled code)
UV_CACHE_DIR=./scratch/.uv_cache uv run pytest tests/

# Coverage tests (JIT disabled, slower but tracks coverage)
UV_CACHE_DIR=./scratch/.uv_cache uv run pytest tests/ --cov=citrees

# Explicitly control JIT
NUMBA_DISABLE_JIT=1 UV_CACHE_DIR=./scratch/.uv_cache uv run pytest tests/  # Force JIT off
NUMBA_DISABLE_JIT=0 UV_CACHE_DIR=./scratch/.uv_cache uv run pytest tests/  # Force JIT on
```

#### Slow Test Marker

Tests that are computationally expensive are marked with `@pytest.mark.slow`:

- `TestStatisticalCorrectness.test_full_ptest` - uses "auto" resamples
- `TestResamplesConfiguration.test_n_resamples_maximum` - uses "maximum"
  resamples

Skip slow tests for faster iteration:

```bash
UV_CACHE_DIR=./scratch/.uv_cache uv run pytest tests/ -m "not slow" -v
```

#### Testing Numba Functions

Numba `@njit` decorated functions compile to machine code, so **pytest-cov
cannot track line coverage inside them**. When running with coverage (`--cov`),
JIT is automatically disabled, allowing coverage tracking.

When JIT is disabled:

- All `@njit` functions run as plain Python
- No `.py_func` attribute exists (functions are already Python)
- Coverage tracking works automatically
- Tests run the same code paths as production, just without compilation

When JIT is enabled (default):

- Tests validate that compiled Numba code actually works
- Use `.py_func` attribute to test both versions if needed:

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

#### RNG Usage Pattern

The codebase uses two different RNG approaches based on Numba constraints:

**Pure Python functions** use `np.random.default_rng()` for isolated RNG
streams:

```python
def _ptest(..., random_state: int):
    rng = np.random.default_rng(random_state)
    rng.shuffle(y_)  # Doesn't contaminate global state
```

**Numba @njit functions** must use `np.random.seed()` because Numba's Generator
support is not thread-safe (see
[GitHub #7686](https://github.com/numba/numba/issues/7686)):

```python
@njit(parallel=True)
def _ptest_parallel(..., random_state: int):
    for i in prange(n_resamples):
        np.random.seed(random_state + i)  # Per-iteration seeding
        np.random.shuffle(y_perm)
```

**Key points:**

- Never use `np.random.seed()` in pure Python code (contaminates global state)
- Per-iteration seeding `(random_state + i)` in `prange` is the recommended
  Numba pattern
- All Numba functions using legacy RNG have documentation comments explaining
  why

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
     it's unambiguous.
   - Print enough context to debug: library versions, random seeds, key params.
   - Run it and confirm it fails:
     ```bash
     UV_CACHE_DIR=./scratch/.uv_cache uv run python scratch/repro_<issue>.py
     ```

2. **Implement the fix (minimal + targeted)**
   - Fix the root cause in the library (avoid unrelated refactors).
   - Keep the repro script unchanged unless the bug definition changes.

3. **Verify the fix (re-run the same repro)**
   - Re-run the exact script and confirm it now passes:
     ```bash
     UV_CACHE_DIR=./scratch/.uv_cache uv run python scratch/repro_<issue>.py
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
     UV_CACHE_DIR=./scratch/.uv_cache uv run pytest -k "<relevant_keyword>" -v
     ```

---

# Experiment CLI (`citrees-exp`)

The `citrees-exp` CLI manages the research paper's experiment infrastructure.
Defined in `pyproject.toml` as:

```toml
[project.scripts]
citrees-exp = "paper.scripts.cli.entrypoint:main"
```

Install with the `paper` dependency group:

```bash
UV_CACHE_DIR=./scratch/.uv_cache uv sync --group paper
```

## CLI Command Reference

### Top-Level Commands

| Command             | Description                             |
| ------------------- | --------------------------------------- |
| `citrees-exp run`   | Poll API server for live queue progress |
| `citrees-exp smoke` | Quick local smoke test (no API needed)  |
| `citrees-exp check` | Check S3 experiment progress            |
| `citrees-exp watch` | Interactive Rich dashboard with key nav |

### `config` Subgroup

| Command                       | Description                     |
| ----------------------------- | ------------------------------- |
| `citrees-exp config show`     | Display current config          |
| `citrees-exp config init`     | Initialize config from template |
| `citrees-exp config validate` | Validate config schema          |
| `citrees-exp config path`     | Show config file paths          |

### `list` Subgroup

| Command                     | Description                    |
| --------------------------- | ------------------------------ |
| `citrees-exp list datasets` | List available datasets        |
| `citrees-exp list methods`  | List feature selection methods |

### `infra` Subgroup (AWS)

| Command                               | Description                        |
| ------------------------------------- | ---------------------------------- |
| `citrees-exp infra setup`             | Full setup (IAM + Docker)          |
| `citrees-exp infra iam`               | Create IAM role + instance profile |
| `citrees-exp infra s3`                | Create S3 bucket                   |
| `citrees-exp infra upload-data`       | Upload datasets to S3              |
| `citrees-exp infra ecr create`        | Create ECR repository              |
| `citrees-exp infra ecr build`         | Build + push Docker image to ECR   |
| `citrees-exp infra ecr clean`         | Delete all ECR images              |
| `citrees-exp infra launch-api`        | Launch API server on EC2           |
| `citrees-exp infra api-url`           | Print running API server URL       |
| `citrees-exp infra terminate-api`     | Terminate API server instance      |
| `citrees-exp infra launch-workers`    | Launch EC2 worker instances        |
| `citrees-exp infra list-workers`      | List running worker instances      |
| `citrees-exp infra terminate-workers` | Terminate all workers              |
| `citrees-exp infra logs`              | Fetch CloudWatch logs (api/worker) |

### `cluster` Subgroup (Local Processes)

| Command                            | Description                    |
| ---------------------------------- | ------------------------------ |
| `citrees-exp cluster api-start`    | Start API queue server locally |
| `citrees-exp cluster api-status`   | Show API queue status          |
| `citrees-exp cluster worker-start` | Start worker process locally   |

## Two-Stage Pipeline

```
Stage 1: Feature Selection (pipeline/stage1.py)
  Input:  dataset + method config
  Output: s3://bucket/rankings/{task}/{dataset}/{method_id}_seed{s}.parquet

Stage 2: Downstream Evaluation (pipeline/stage2.py)
  Input:  rankings from Stage 1
  Output: s3://bucket/metrics/{task}/{dataset}/{method_id}_seed{s}.parquet
  Evaluates at k = [5, 10, 25, 50, 100, all]
  Downstream models: LR, SVM, KNN (clf) / Ridge, SVR, KNN (reg)
```

## Distributed Architecture (API Server + EC2 Workers)

The experiment infrastructure uses a pull-based API server model (not Ray):

```
┌─────────────────────┐      ┌───────────────────────────────┐
│   API Server (EC2)  │◄────►│      S3 Bucket                │
│   FastAPI + queues  │      │  rankings/ + metrics/         │
│   POST /next        │      └───────────────────────────────┘
│   GET  /status      │
└──────────┬──────────┘
           │  HTTP
    ┌──────┴──────┐
    ▼             ▼
┌────────┐  ┌────────┐
│Worker 1│  │Worker N│   EC2 instances (m5.8xlarge)
│ Docker │  │ Docker │   Pull config → execute → save to S3
└────────┘  └────────┘
```

**API server** (`paper/scripts/api/server.py`): FastAPI app with 4 lazy queues
(rankings/classification, rankings/regression, metrics/classification,
metrics/regression). On startup it builds the full experiment grid and subtracts
completed S3 artifacts. Workers call `POST /next` to get work.

**Worker** (`paper/scripts/api/worker.py`): Pull-based loop. Gets config from
API, runs `_run_selection()` or `_run_evaluation()`, saves result to S3, repeats
until queues drain or idle timeout.

## Adapters

| Module               | Purpose                                             |
| -------------------- | --------------------------------------------------- |
| `adapters/data.py`   | Dataset loading (local filesystem, S3 fallback)     |
| `adapters/runner.py` | Execution interface (`LocalRunner` for smoke tests) |
| `adapters/store.py`  | S3 artifact storage (save/load/exists/list)         |

## Pipeline Types

| Type               | Location            | Description                                      |
| ------------------ | ------------------- | ------------------------------------------------ |
| `MethodConfig`     | `pipeline/types.py` | Frozen dataclass: method + params                |
| `ExperimentConfig` | `pipeline/types.py` | Frozen dataclass: method + dataset + seed + task |
| `Result`           | `pipeline/types.py` | Mutable dataclass: config + status + data        |
| `ExperimentGrid`   | `pipeline/grid.py`  | Grid builder from CLI args                       |

## Method Categories

Methods are defined in `paper/scripts/pipeline/methods.py`:

| Category   | Classification                                                         | Regression                                                             |
| ---------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Perm. test | `ptest_mc`, `ptest_rdc`                                                | `ptest_pc`, `ptest_dc`, `ptest_rdc`                                    |
| Embedding  | `cit`, `cif`, `rf`, `et`, `xgb`, `lgbm`, `cat`, `r_ctree`, `r_cforest` | `cit`, `cif`, `rf`, `et`, `xgb`, `lgbm`, `cat`, `r_ctree`, `r_cforest` |
| Wrapper    | `boruta`, `pi`, `cpi`, `rfe`                                           | `boruta`, `pi`, `cpi`, `rfe`                                           |

## Configuration

**Config file**: `paper/scripts/infra/config.yaml` (created via
`citrees-exp config init` from `config.example.yaml`)

**Key settings** (`paper/scripts/config/settings.py`):

- `aws_region`: Default `us-east-1`
- `s3_bucket`: Auto-derived as `citrees-{account_id}`
- `experiment.n_seeds`: Default 5
- `experiment.s3_validate_uploads`: Default True

**Constants** (`paper/scripts/config/constants.py`):

- `RANDOM_STATE`: 1718
- `N_SEEDS`: 5, `N_SPLITS`: 5
- `CLF_DOWNSTREAM_MODELS`: `["lr", "svm", "knn"]`
- `REG_DOWNSTREAM_MODELS`: `["ridge", "svr", "knn"]`
- `EVALUATION_K_VALUES`: `[5, 10, 25, 50, 100]`

## Typical Workflow

```bash
# 1. Setup infrastructure (one-time)
citrees-exp config init
citrees-exp infra setup           # IAM + Docker
citrees-exp infra s3              # Create S3 bucket
citrees-exp infra upload-data     # Upload datasets

# 2. Launch API server + workers on EC2
citrees-exp infra launch-api
citrees-exp infra launch-workers --count 5   # auto-discovers API private IP

# 3. Monitor progress
citrees-exp run                       # poll queue status
citrees-exp watch                     # interactive dashboard
citrees-exp check --by-method         # S3 progress check

# 4. Tear down
citrees-exp infra terminate-workers
citrees-exp infra terminate-api
```

---

# Research Roadmap

## Current Limitations vs State of the Art

| Area             | citrees Now  | SOTA (2024-2025)         | Priority |
| ---------------- | ------------ | ------------------------ | -------- |
| Causal Inference | Honesty      | Honest Estimation, GRF   | DONE     |
| Speed            | CPU/Numba    | GPU (cuML 20-45x faster) | MEDIUM   |
| Tree Structure   | Axis-aligned | Oblique, Neural Trees    | LOW      |

---

## MEDIUM PRIORITY

### GPU Acceleration

Training on large datasets is slow. cuML achieves 20-45x speedups.

**Options**:

1. **Wrap cuML**: Use cuML for splitting, keep citrees API
2. **CUDA kernels**: Custom kernels for permutation tests
3. **JAX/Triton**: Python-native GPU compilation

**Research**:

- [cuML Random Forests](https://developer.nvidia.com/blog/accelerating-random-forests-up-to-45x-using-cuml/)
- [RFX](https://arxiv.org/html/2511.19493) - GPU + QLORA (Nov 2025)

---

## LOW PRIORITY (Future Research)

### Oblique Decision Trees

Axis-aligned splits are limiting. Oblique trees use linear combinations of
features.

- [TAO Algorithm](https://faculty.ucmerced.edu/mcarreira-perpinan/research/TAO.html)
- [DTSemNet](https://arxiv.org/abs/2408.09135) - Vanilla gradient descent (2024)

### Neural/Differentiable Trees

End-to-end training with neural networks.

- [NCART](https://www.sciencedirect.com/science/article/abs/pii/S0031320324003297) -
  Neural CART (2024)
- [Neural-Backed Decision Trees](https://research.alvinwan.com/neural-backed-decision-trees/)

### Gradient Boosting Integration

Combine conditional inference with boosting.

- [LightGBM techniques](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
- [Piecewise Linear Trees](https://nimasarang.com/blog/2025-12-14-gbt-algorithms/)

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
