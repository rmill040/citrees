# Paper Directory

This directory contains the **arXiv manuscript** and **experiment pipeline** for
citrees.

## Quick Navigation

**Working on the manuscript?**

- Manuscript source: `paper/arxiv/` (LaTeX, build with `latexmk`)
- Claims tracker: `paper/docs/claims-index.md`
- Proof QA checklist: `paper/docs/writing-checklist.md`
- Reviewer watchouts: `paper/docs/reviewer-watchouts.md`
- Immediate actions: `paper/docs/next-steps.md`
- Writing notes/drafts: `paper/docs/drafts.md`

**Running experiments?**

- Experiment runbook: `paper/docs/experiments.md`
- Infrastructure (AWS/Ray): `paper/docs/infrastructure.md`
- Figures/tables map: `paper/docs/figures-plan.md`

## Directory Structure

```
paper/
├── README.md              # you are here
├── docs/                  # all documentation
│   ├── experiments.md     # experiment pipeline runbook
│   ├── infrastructure.md  # AWS/Ray infra
│   ├── claims-index.md    # theory claims tracker
│   ├── next-steps.md      # immediate actions
│   ├── writing-checklist.md # proof QA checklist
│   ├── drafts.md          # writing notes (staging)
│   └── figures-plan.md    # scripts → figures → claims
├── arxiv/                 # LaTeX manuscript (arXiv source)
│   ├── main.tex           # entrypoint
│   ├── sections/          # main paper content
│   ├── appendices/        # proofs + technical details
│   └── references.bib     # BibTeX database
├── joss/                  # JOSS submission (future)
├── scripts/               # experiments + analysis + theory
├── data/                  # datasets (parquet)
└── results/               # generated outputs (figures/tables/cache)
```

## Build Manuscript

```bash
cd paper/arxiv && latexmk -pdf main.tex && open main.pdf
```

## Run Experiments

```bash
uv sync --group paper        # install dependencies
citrees-exp smoke classification   # local smoke test
citrees-exp run classification   # full pipeline (skips existing)
citrees-exp check            # check progress
citrees-exp watch            # live dashboard
```

> **Note:** AWS operations require credentials. Set `AWS_PROFILE` if not using
> default profile. See `paper/docs/experiments.md` for full setup.
