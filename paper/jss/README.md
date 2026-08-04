# Journal of Statistical Software Article

This directory contains the manuscript and replication materials for the
`citrees` Journal of Statistical Software submission. The article documents the
statistical procedures, implementation, software interface, comparisons, and
applications. The published arXiv manuscript remains the broad feature-selection
benchmark; this article is a self-contained software treatment.

## Article Scope

The manuscript uses the following structure:

1. Statistical motivation and relationship to conditional inference trees,
   conditional inference forests, CART, and existing open-source software.
2. Nodewise association testing, threshold selection, multiplicity adjustment,
   sequential stopping, multi-selector tests, and forest construction.
3. Python implementation, estimator interface, validation, introspection,
   extensibility, and parallel execution.
4. Statistical calibration, split-variable selection bias, behavioral comparison
   with `partykit`, and computational scaling.
5. A complete analysis workflow using public biomedical data.
6. A concise summary of the existing broad benchmark, with results traced to the
   published arXiv manuscript and canonical repository artifacts.
7. Limitations, reproducibility, and software availability.

The authors are Robert Milletich, Justin Downes, and Newel Hirst. The
affiliation is Amazon Web Services.

## Evidence

| Analysis                | Purpose                                                                                        | Primary comparisons                                             |
| ----------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Null calibration        | Measure feature-test and root-split rejection under independence                               | Fixed-budget p-values; exhaustive, adaptive, and simple fitting |
| Split-variable bias     | Measure selection frequency when noise variables differ only in cardinality                    | `citrees`, `partykit::ctree`, CART                              |
| Power and recovery      | Show linear, nonlinear, correlated, and interaction boundaries                                 | All compatible `citrees` selectors and multi-selector tests     |
| Reference behavior      | Compare split decisions, conditional root agreement, native feature summaries, and predictions | `citrees`, `partykit::ctree`, `partykit::cforest`               |
| Scaling                 | Measure runtime and peak memory across controlled problem dimensions                           | `citrees`, `partykit`, scikit-learn                             |
| DGRP application        | Demonstrate leakage-safe screening and linkage-disequilibrium-aware stability analysis         | Tree, forest, linear, and marginal baselines                    |
| Tutorial                | Demonstrate the estimator interface in an executable workflow                                  | Breast Cancer Wisconsin Diagnostic data                         |
| Broad benchmark context | Summarize the corrected benchmark without duplicating it                                       | Final arXiv v2 artifacts                                        |

The primary biomedical application is the public Drosophila Genetic Reference
Panel cardiac dataset associated with DOI `10.7554/eLife.82459` and Zenodo DOI
`10.5281/zenodo.5582846`. DGRP lines are the independent units. The analysis
uses one prespecified primary cardiac trait, genotype filtering and confounder
handling defined before model fitting, and linkage-disequilibrium-aware
stability summaries. All outcome-dependent operations remain inside the training
folds. The Breast Cancer Wisconsin Diagnostic data provide the executable
end-to-end tutorial.

## Claim Boundaries

- Simulation results estimate calibration, power, and known failure boundaries
  across the specified designs.
- Reference comparisons quantify split decisions, conditional root agreement,
  native feature-summary concordance, and held-out prediction behavior under
  identical folds and aligned structural controls.
- Cardiac results are an adapted predictive reanalysis of screening and
  stability among DGRP lines.
- DGRP lines are the independent units, and repeated resampling quantifies
  uncertainty over these lines.
- Benchmark context comes only from final corrected arXiv v2 artifacts.

## Replication

The final submission provides a single entry point with two profiles:

- `quick` retains the datasets, metric definitions, output schemas, and
  validation rules with reduced folds and computational budgets for a regular
  computer.
- `full` rebuilds the reported simulation estimates and computational
  measurements and may require parallel hardware.

Generated outputs remain below `paper/jss/results/` as ignored build artifacts.
A full-profile receipt records the clean source revision, dependency versions,
inputs, and output hashes before manuscript tables and figures enter the
submission source.

Run the complete replication suite from the repository root:

```bash
uv run python -m paper.jss.replication --profile quick
uv run python -m paper.jss.replication --profile full
```

The command dispatches calibration, matched behavior, controlled performance,
tutorial, and DGRP preparation analyses. It verifies each child receipt and
artifact hash before atomically publishing the combined output directory. Use
`--output-dir` for a new destination; an existing destination is rejected to
prevent results from different executions from being mixed.

Calibration and matched behavior support deterministic distributed execution.
Each worker runs one indexed shard into the corresponding analysis directory:

```bash
uv run python -m paper.jss.replication.shards calibration-shard \
  --component selector --profile full --shard-index 0 --num-shards 100 \
  --output-dir <shard-root>/calibration/selector/00000

uv run python -m paper.jss.replication.shards behavior-shard \
  --profile full --shard-index 0 --num-shards 100 \
  --output-dir <shard-root>/behavior/00000
```

The calibration components are `selector`, `root`, `cardinality_tree`, and
`cardinality_forest`. Every index from zero through `num-shards - 1` is
required for each component. The canonical suite merges complete shard trees:

```bash
uv run python -m paper.jss.replication --profile full \
  --shard-root <shard-root> --output-dir <result-root>
```

The merge validates assignment coverage, source and input hashes, package and R
versions, artifact hashes, schemas, row inventories, and execution contexts.
The combined output includes the shard receipts used to construct each final
analysis.

## Manuscript

The source uses version 3.6 of the official JSS LaTeX style downloaded from:

<https://www.jstatsoft.org/public/journals/1/jss-article-tex.zip>

Build the current draft from the repository root:

```bash
cd paper/jss
latexmk -pdf -interaction=nonstopmode -halt-on-error article.tex
```

The source follows the JSS markup conventions for programming languages,
packages, and code. Publication metadata such as volume, issue, publication
date, and acceptance date is omitted until supplied by the journal.
