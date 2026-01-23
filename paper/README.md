# Paper directory (citrees): manuscript + experiments

This directory contains **two related but distinct tracks**:

1. The **arXiv preprint manuscript** (LaTeX, full details).
2. The **benchmark/experiment pipeline** used to generate empirical results.

The goal is to keep the manuscript “airtight” (claims ↔ assumptions ↔ proofs)
while keeping experiments reproducible.

## Resolution order (what is authoritative when docs disagree)

When something conflicts, resolve in this order:

1. `paper/arxiv/` — **Manuscript source of truth** (what we are actually
   claiming).
2. `paper/CLAIMS_INDEX.md` — **Index of all formal claims** and where they are
   proved.
3. `paper/WRITING_CHECKLIST.md` — **Process checklist** (proof QA + migration
   rules).
4. `paper/notes/notes.md` — **Scratchpad / staging notes** (may be messy;
   migrate into LaTeX).
5. `paper/docs/README.md` and `paper/TODO.md` — **Experiment pipeline docs**
   (Ray/S3/local analysis).
6. `paper/joss/` — **Future JOSS submission track** (derived from the arXiv
   preprint; not authoritative today).
7. `paper/results/` — **Generated artifacts** (never the source of truth; can be
   regenerated).

## Directory structure (current)

```
paper/
├── README.md                  # you are here (high-level map)
├── WRITING_CHECKLIST.md       # internal paper workplan + proof QA checklist
├── CLAIMS_INDEX.md            # internal index of every formal claim
├── TODO.md                    # experiment pipeline TODOs / open issues
├── arxiv/                     # LaTeX manuscript (arXiv preprint source)
│   ├── main.tex               # entrypoint
│   ├── sections/              # main paper content
│   ├── appendices/            # proofs + technical details
│   ├── references.bib         # BibTeX database for the manuscript
│   └── README.md              # build/preview commands (latexmk)
├── joss/                      # JOSS submission (later; derived from arXiv)
│   ├── paper.md.txt            # JOSS author instructions (saved verbatim)
│   └── README.md               # JOSS carve-out notes / checklist (short)
├── notes/                     # internal writing notes (staging area)
│   ├── notes.md               # long-form notes to migrate into LaTeX
│   ├── figures_plan.md        # map: paper-facing figures/tables -> scripts/artifacts
│   └── README.md              # staging-area notes about notes/ (not the manuscript)
├── docs/                      # experiment pipeline / infra docs
│   ├── README.md              # experiment pipeline overview (Ray/S3/local)
│   └── infrastructure.md      # AWS/Ray infra notes
├── scripts/                   # experiments + analysis + theory calibration scripts
├── data/                      # datasets (parquet)
└── results/                   # generated outputs (parquet/figures/tables)
    └── README.md              # explains generated artifact layout
```

## Manuscript workflow (arXiv / LaTeX)

Start here:

- `paper/arxiv/main.tex`
- `paper/CLAIMS_INDEX.md` (keep in sync with the theory section)
- `paper/WRITING_CHECKLIST.md` (proof QA for each claim)

Build/preview:

```bash
cd paper/arxiv
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
open main.pdf
```

Live rebuild:

```bash
cd paper/arxiv
latexmk -pdf -pvc -interaction=nonstopmode -halt-on-error main.tex
```

### Rule: one appendix per claim

Before adding new math claims, update:

- `paper/CLAIMS_INDEX.md` (new claim row),
- the manuscript statement (with `\\label{...}`),
- and create the proof appendix location (one appendix per claim).

## Experiments workflow (high level)

The experiments are a two-stage pipeline (feature-ranking → downstream eval) run
locally or on Ray/AWS.

Start here:

- `paper/docs/README.md` (full runbook)
- `paper/TODO.md` (open issues)

Artifacts:

- S3 is the source of truth for distributed runs; `paper/results/` is a local
  cache.

## JOSS (Journal of Open Source Software) carve-out (later)

We plan to publish a **full arXiv preprint first** (LaTeX, all theory/details),
then carve out a short **JOSS** software paper afterwards.

- JOSS author instructions (saved verbatim): `paper/joss/paper.md.txt`

## Directory refactor check (before we go deeper)

Current structure is workable for drafting, but there are a few **optional**
refactors we may want _before_ migrating lots of content:

1. Proof appendices: split `paper/arxiv/appendices/` into “one file per claim”
   (hard requirement from our workflow). This is the next structural change.
2. Notes vs experiments docs: notes live under `paper/notes/` and experiment
   runbooks live under `paper/docs/`. Keep it that way to avoid mixing concerns.
3. Results outputs: `paper/results/` has multiple subtrees (`analysis/`,
   `figures/`, `tables/`, etc.). We should avoid relying on paths in the
   manuscript; the manuscript should only consume final, versioned figures that
   live under `paper/arxiv/` when we get close to submission.
