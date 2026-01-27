# arXiv manuscript (LaTeX)

Source of truth for the manuscript lives in this directory:

- `main.tex` is the entrypoint.
- `sections/` contains the main-paper content.
- `appendices/` contains proofs and implementation details.
- `references.bib` is the BibTeX database for the manuscript.

## Workflow

- Long-form scratch/theory notes live in `paper/docs/drafts.md`.
- The arXiv manuscript lives in `paper/arxiv/` and is written in LaTeX.
- As sections stabilize, migrate content from `paper/docs/drafts.md` into:
  - `paper/arxiv/sections/` (main paper), and
  - `paper/arxiv/appendices/` (proofs/technical details).

Proof organization rule:

- Each paper-facing claim has its **own appendix file** under `appendices/` (one
  appendix per claim).

## Build

If you have `latexmk`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Live preview while editing:

```bash
latexmk -pdf -pvc -interaction=nonstopmode -halt-on-error main.tex
```

Otherwise:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Open the compiled PDF:

```bash
open main.pdf
```

## Notes

- The old Markdown theory draft was renamed to `paper/docs/drafts.md` and is
  being migrated into this LaTeX manuscript.
