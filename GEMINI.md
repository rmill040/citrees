# GEMINI.md - Context for citrees

## Project Overview

**citrees** is a Python library that implements **Conditional Inference Trees
and Forests**. It provides a statistically principled alternative to traditional
CART-style decision trees by separating variable selection from split point
selection.

- **Core Principle:** Uses permutation-based hypothesis testing to select
  variables and determine stopping criteria, mitigating selection bias towards
  high-cardinality features.
- **Key Capabilities:**
  - Classification and Regression trees/forests.
  - Multiple feature selection methods (Multiple Correlation, Mutual
    Information, Randomized Dependence Coefficient, etc.).
  - Honest estimation (sample splitting) for causal inference.
  - Feature muting to remove uninformative features.
- **Performance:** Uses **Numba** (`@njit`) for high-performance JIT compilation
  of critical paths.

## Architecture & Key Files

- **Source:** `citrees/`
  - `_tree.py`: Core tree implementations (`ConditionalInferenceTreeClassifier`,
    `ConditionalInferenceTreeRegressor`).
  - `_forest.py`: Forest ensembles (`ConditionalInferenceForestClassifier`,
    etc.).
  - `_selector.py`: Feature selection logic (Mc, Mi, Rdc, Pc, Dc).
  - `_splitter.py`: Split criteria (Gini, Entropy, Mse, Mae).
  - `_registry.py`: Registry pattern for dynamically registering selectors and
    splitters.
  - `_types.py`: Centralized type definitions and Enums.
- **Tests:** `tests/` (uses `pytest`)
  - `data/`: Test datasets in Parquet format.
  - `unit/`: Unit tests.
  - `integration/`: Integration tests.
- **Paper:** `paper/`
  - Contains code, data, and LaTeX files for the accompanying research paper.

## Development & Usage

### 1. Environment & Dependencies

The project uses **uv** for dependency management.

```bash
# Install dependencies
uv sync
```

### 2. Running Tests

Tests are built with **pytest**. Note the specific interaction with Numba JIT:

- **Fast Tests (JIT Enabled):** Validates compiled code.
  ```bash
  uv run pytest tests/
  ```
- **Coverage/Debug (JIT Disabled):** Allows line coverage tracking.
  ```bash
  uv run pytest tests/ --cov=citrees
  ```
- **Skip Slow Tests:**
  ```bash
  uv run pytest tests/ -m "not slow"
  ```

### 3. Coding Conventions

- **Style:** Adheres to `black` (line-length 120), `ruff`, and strict `mypy`.
- **Numba:** Performance-critical functions are decorated with `@njit`.
  - **RNG:** In Numba functions, use `np.random.seed(seed + i)` inside parallel
    loops. In pure Python, use `np.random.default_rng(seed)`.
- **Pydantic:** Uses Pydantic v2 for parameter validation (`citrees._types`).

### 4. Bug Fix Workflow

1.  **Reproduce:** Create a minimal reproduction script in
    `scratch/repro_<issue>.py`.
2.  **Fix:** Implement the fix in the library.
3.  **Verify:** Run the repro script again to confirm the fix.
4.  **Test:** Convert the repro into a formal test case in `tests/`.

## Key Commands Cheat Sheet

| Task            | Command                              |
| :-------------- | :----------------------------------- |
| **Install**     | `uv sync`                            |
| **Test (Fast)** | `uv run pytest tests/`               |
| **Test (Cov)**  | `uv run pytest tests/ --cov=citrees` |
| **Lint/Format** | `uv run pre-commit run --all-files`  |
| **Docs**        | `uv run mkdocs serve`                |
