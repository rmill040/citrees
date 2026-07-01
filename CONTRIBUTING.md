# Contributing to citrees

Thanks for considering a contribution. This project welcomes bug reports,
documentation fixes, tests, and focused implementation improvements for the
`citrees` Python package.

## Development Setup

```bash
git clone https://github.com/rmill040/citrees.git
cd citrees
uv sync
uv run pre-commit install
```

Use `scratch/` for local experiments and temporary repro scripts. Do not commit
large generated outputs, private credentials, or local infrastructure
configuration.

## Before Opening a Pull Request

Run the checks that match the change:

```bash
uv run pytest tests/unit tests/integration -v
uv run ruff check citrees tests/unit tests/integration
uv run ruff format --check citrees tests/unit tests/integration
```

For documentation changes:

```bash
uv sync --group docs
uv run --group docs mkdocs build --strict
```

For paper-pipeline changes:

```bash
uv sync --group paper
uv run pytest tests/paper -v
```

## Bug Reports

When reporting a bug, include:

- the installed `citrees` version or commit hash
- Python, NumPy, Numba, scikit-learn, and SciPy versions
- a minimal reproducible example
- the full traceback or incorrect output
- whether JIT was enabled or disabled through `NUMBA_DISABLE_JIT`

Small deterministic repro scripts are preferred. Use public or generated data
unless the issue is specifically about a provided test dataset.

## Pull Request Scope

Keep changes focused. A good pull request has one behavioral purpose, relevant
tests, and only the documentation needed to explain the change. Avoid unrelated
formatting churn.

When changing statistical behavior, document the inference scope precisely. In
particular, distinguish fixed-node permutation-test behavior from adaptive tree-
or forest-level model outputs.

## Review and Conduct

Project discussion should stay technical, respectful, and evidence based. See
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for the conduct expectations used in
this repository.
