# Paper TODO

No active manuscript, package, or arXiv-submission TODO items remain.

The April 2026 readiness passes were closed after fixes and verification:

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` from
  `paper/arxiv`
- `UV_CACHE_DIR=./scratch/.uv_cache uv run pytest tests/paper/test_analysis.py tests/paper/test_paper_package.py -q`
- `UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/analysis/build_arxiv_source_bundle.py --skip-build --check`
- deterministic source-bundle hash check under changed source mtimes
- fresh reviewer re-review: no remaining required manuscript, evidence,
  citation, theory, package, or arXiv-layout blockers after the follow-up fixes
- follow-up display cleanup: tie columns removed where all tie counts were zero,
  regression trajectory added beside the classification trajectory, and the
  synthetic top-`k` figure regeneration path aligned to the paper's LaTeX/serif
  figure style

Add new items here only when they are active. Delete each item once it is
completed and verified.
