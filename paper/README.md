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

## Inferential scope (p-values)

To avoid accidental overclaiming in figures, captions, and analysis scripts, the
arXiv manuscript follows a strict “scope contract”:

- **Calibrated p-values:** only fixed-node/root Stage~A permutation p-values
  computed in **fixed-`B`** mode under the **nodewise complete (global)
  permutation null** (exchangeability target of the permutation scheme).
- **Algorithmic statistics:** Stage~B threshold tests, internal-node tests, and
  early-stopped permutation outputs (unless additional selective-inference
  machinery or sample splitting is used).
- **Caption rule:** any calibration plot must state the simulated null (e.g.,
  “complete global null”) and avoid featurewise/partial-null language unless a
  restricted permutation scheme is implemented.

See the manuscript scope table (`paper/arxiv/sections/03_method.tex`,
Table~`tab:pvalue-scope`) and the figure map (`paper/docs/figures-plan.md`) for
the “do/don’t” wording.

## Directory Structure

```
paper/
├── README.md              # you are here
├── docs/                  # all documentation
│   ├── experiments.md     # experiment pipeline runbook
│   ├── infrastructure.md  # AWS/EC2 infra
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
│   ├── adapters/          # S3, data loading, runner
│   ├── api/               # FastAPI queue server + worker
│   ├── analysis/          # stats, figures, synthetic analysis
│   ├── cli/               # citrees-exp CLI commands
│   ├── config/            # settings + constants
│   ├── data_generation/   # synthetic dataset generation
│   ├── infra/             # AWS (IAM, S3, ECR, EC2, Docker)
│   ├── pipeline/          # stage1, stage2, grid, methods
│   ├── theory/            # sequential stopping analysis
│   └── utils/             # env, metrics
├── data/                  # datasets (parquet)
└── results/               # generated outputs (figures/tables/cache)
```

## Build Manuscript

```bash
cd paper/arxiv && latexmk -pdf main.tex && open main.pdf
```

## Run Experiments

```bash
uv sync --group paper              # install dependencies
citrees-exp smoke classification   # local smoke test (no API needed)
citrees-exp cluster api-start      # start API server locally
citrees-exp cluster worker-start   # start worker locally
citrees-exp run                    # poll queue progress
citrees-exp check                  # check S3 progress
citrees-exp watch                  # live dashboard
```

> **Note:** AWS operations require credentials. Set `AWS_PROFILE` if not using
> default profile. See `paper/docs/experiments.md` for full setup and
> `paper/docs/infrastructure.md` for distributed EC2 deployment.
