# JOSS Submission Notes

Current date: 1 July 2026

## Submission Form Values

Title:

```text
citrees: Conditional Inference Trees and Forests for Python
```

Software's Git repository URL:

```text
https://github.com/rmill040/citrees
```

Name of git branch containing the paper, not the path:

```text

```

Leave blank if `paper.md` is submitted from the default branch. The current
default branch is `master`.

Software version:

```text
v0.1.0
```

This matches `pyproject.toml`. Before submission, either create and push a
`v0.1.0` tag/release or update this field to the release tag that actually
exists.

Type of submission:

```text
New submission
```

Main subject of the paper:

```text
Machine Learning
```

If the form offers
`Data Science, Artificial Intelligence, and Machine Learning`, select that.

Message to editors:

```text
citrees is a Python package implementing conditional inference trees and random forests with permutation-test-based feature selection through scikit-learn-style classifier and regressor APIs.

This is a new JOSS submission, not a resubmission or second JOSS paper about this software. No portion of this JOSS paper has been published or submitted to another peer-reviewed venue. A related methods and benchmark manuscript has been submitted as a non-peer-reviewed arXiv preprint, with identifier pending, and may be submitted separately; that manuscript focuses on theory, algorithm details, and empirical evaluation, while this JOSS submission focuses on the reusable Python package, documentation, tests, and software design. All authors are affiliated with Amazon Web Services, which supported the work; we are not aware of any other conflicts of interest.
```

## Pre-Submit Checks

- The submitting author should certify primary authorship only if true.
- Check the compile box only after the pushed GitHub Action runs successfully,
  or after confirming that the local Inara/Docker build is acceptable evidence.
- Check the code-of-conduct box only after reading and agreeing to the JOSS code
  of conduct.
- Do not submit until `paper/joss/paper.md`, `paper/joss/paper.bib`, and the
  JOSS build workflow are committed and pushed to GitHub.
- The repository currently has exactly one `paper.md` at `paper/joss/paper.md`.
  JOSS EditorialBot recursively searches the submitted branch for a single
  `paper.md`/`paper.tex`, so this nested path should be acceptable as long as no
  second paper file is added.

## Repository Readiness Audit

Reviewer-facing improvements made during the JOSS readiness pass:

- Added `CONTRIBUTING.md`, `SUPPORT.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`,
  and `CHANGELOG.md`.
- Added `.github/workflows/tests.yml` for package linting, unit/integration
  tests, and strict MkDocs builds.
- Kept `.github/workflows/joss-draft-pdf.yml` for JOSS paper builds.
- Split package runtime dependencies from paper/benchmark dependencies. A core
  `citrees` install no longer requires AWS, R, boosting libraries, Boruta, or
  benchmark-analysis packages.
- Added package metadata: keywords, classifiers, and project URLs.
- Updated README installation, experiment-CLI wording, development commands,
  support/contribution links, release-note pointer, and citation metadata.
- Fixed a stale paper-package test that still expected generated `main.bbl` in
  the arXiv source bundle. The current arXiv policy is to submit
  `references.bib` and exclude generated `.bbl` files.
- Pinned docs builds to `mkdocs>=1.6,<2` because current Material for MkDocs is
  not compatible with MkDocs 2.
- Registered existing pytest markers to remove unknown-marker warnings.
- Ignored local `tmp/` render artifacts.

Local verification completed:

```bash
uv run ruff format --check citrees tests/unit tests/integration tests/paper/test_paper_package.py
uv run ruff check citrees tests/unit tests/integration tests/paper/test_paper_package.py
uv run --group docs mkdocs build --strict
uv run pytest tests/unit tests/integration -m "not slow" -q
uv run pytest tests/paper -q
docker run --rm --volume "$PWD/paper/joss:/data" --user "$(id -u):$(id -g)" --env JOURNAL=joss openjournals/inara
```

Results:

- Ruff format/check passed.
- MkDocs strict build passed.
- Unit/integration suite passed: 539 passed, 2 slow tests deselected, 2 expected
  OOB warnings.
- Paper tests passed: 96 passed.
- JOSS Inara build passed.
- Minimal core import confirmed `citrees==0.1.0` imports without `boto3`,
  `rpy2`, `lightgbm`, `xgboost`, `boruta`, or `mrmr`.

Paper/result cleanup audit:

- The arXiv manuscript already reports the benchmark headline used by the JOSS
  impact statement: CIF ranks 4th of 17 classification methods and 3rd of 18
  regression methods. No extra benchmark-results patch is needed before the
  arXiv identifier is issued.
- Keep `paper/joss/paper.md` free of placeholder arXiv IDs, DOIs, repository
  archives, or review metadata. Add real identifiers only after they exist.
- Removed tracked unused arXiv story figures generated by archived scripts. The
  deterministic source bundle already includes only figures referenced by the
  current TeX source.
- Removed the tracked generated selection-bias demo cache and ignored
  `paper/results/cache/` for future local cache files.
- Updated local project guidance and ignore rules to describe the current
  FastAPI queue plus pull-based worker design.
- Removed one unused import found by a focused Ruff dead-code pass over
  `paper/scripts`.
- Improved core-package type and docstring hygiene: `mypy citrees` now passes,
  and explicit docstring lint over `citrees` passes.
- Coverage audit:
  - full compiled-path unit/integration suite with coverage passed
    (`539 passed`, 2 slow tests deselected, 2 expected OOB warnings) with
    reported aggregate coverage of 63%;
  - that aggregate is artificially low because Numba-compiled functions are not
    line-tracked when JIT is enabled;
  - targeted disabled-JIT coverage over numerical unit tests passed
    (`239 passed`) and shows strong coverage for the numerical internals:
    `_selector.py` 91%, `_splitter.py` 89%, `_threshold_method.py` 91%,
    `_sequential.py` 92%, and `_utils.py` 94%;
  - a full disabled-JIT coverage run is currently too slow for routine use and
    was interrupted after 301 passing tests in 8:19 while running an RDC tree
    test.
- Kept `paper/scripts/archive/` and `paper/scripts/maintenance/` in place. They
  are intentionally archived paper/provenance scripts, not imported package
  code, and deleting them before submission would remove useful experiment
  history without improving the install surface.

Remaining before clicking submit:

- Commit and push all JOSS paper, workflow, package, docs, and community-file
  changes.
- Confirm the GitHub Actions runs are green on the pushed branch.
- Create and push a `v0.1.0` tag/release, or update the JOSS form version to the
  actual release tag.
- Fill in exact AI tool/model/version details if available; the paper currently
  discloses OpenAI Codex and Claude Code using Anthropic Claude models through
  Amazon Bedrock, but not exact model versions.
- Add the real arXiv identifier only after arXiv issues it. Do not submit fake
  or placeholder arXiv metadata.

## Sources Checked

- JOSS submission requirements:
  <https://joss.readthedocs.io/en/latest/submitting.html>
- JOSS paper format: <https://joss.readthedocs.io/en/latest/paper.html>
- JOSS EditorialBot behavior:
  <https://joss.readthedocs.io/en/latest/editorial_bot.html>
