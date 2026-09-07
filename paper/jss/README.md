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
| Reference behavior      | Compare split decisions, conditional root agreement, native feature summaries, and predictions | `citrees`, `partykit::ctree`, `partykit::cforest`               |
| Scaling                 | Measure runtime and peak memory across controlled problem dimensions                           | `citrees`, `partykit`, scikit-learn                             |
| TCGA-BRCA application   | Demonstrate leakage-safe screening, held-out prediction, and selection stability               | Tree, forest, linear, and marginal baselines                    |
| Tutorial                | Demonstrate the estimator interface in an executable workflow                                  | Breast Cancer Wisconsin Diagnostic data                         |
| RDC sensitivity         | Measure the accuracy, ranking stability, and runtime effect of random projection count         | 5, 10, 20, and 40 projections on four small datasets            |
| Broad benchmark context | Summarize the corrected benchmark without duplicating it                                       | Final arXiv v2 artifacts                                        |

The primary biomedical application is public TCGA-BRCA breast tumour gene
expression, with predictors and outcomes drawn from independent assays: the
predictors are protein-coding transcript abundances, and the outcomes are
receptor statuses determined by immunohistochemistry and FISH. Tumour samples
are the independent units. The design is frozen in
`replication/tcga_brca-specification.json`: oestrogen-receptor status is the
single primary outcome, progesterone-receptor and HER2 status are secondary
under a Holm adjustment, and every outcome-dependent step -- standardization,
the candidate screen, and model fitting -- stays inside the training fold. PAM50
subtype was considered and rejected because it is called from the same RNA-seq
that supplies the predictors. The Breast Cancer Wisconsin Diagnostic data
provide the executable end-to-end tutorial.

## Claim Boundaries

- Simulation results estimate null calibration and split-variable selection bias
  across the specified designs.
- Reference comparisons quantify split decisions, conditional root agreement,
  native feature-summary concordance, and held-out prediction behavior under
  identical folds and aligned structural controls.
- TCGA-BRCA results are a predictive screening and stability analysis, not a
  claim about clinical utility or about causal transcriptional mechanism.
- Tumour samples are the independent units, and repeated stratified
  cross-validation quantifies uncertainty over these samples.
- Benchmark context comes only from final corrected arXiv v2 artifacts.

## Replication Setup

The replication suite requires Python 3.12 or later, R, and the R package
`partykit`. Install the locked Python environment from the repository root:

```bash
uv sync --group paper
```

The exact Linux reference environment is defined in
`paper/benchmark/infra/docker/Dockerfile`. It pins R 4.5.2, `Formula` 1.2-5,
`inum` 1.0-5, `libcoin` 1.0-10, `mvtnorm` 1.3-3, and `partykit` 1.2-24.
Replication receipts record the R and Python package versions actually used.

The suite provides three profiles:

- `smoke` exercises every analysis and validation path with the smallest
  workload.
- `quick` retains the datasets, metric definitions, output schemas, and
  validation rules with reduced folds and computational budgets.
- `full` rebuilds the manuscript analyses and computational measurements and may
  require parallel hardware.

The RDC sensitivity child uses all four projection counts and five folds in
every profile. `smoke` uses Wine with seed 0, `quick` uses Wine, Glass, Heart
Statlog, and Parkinsons with seed 0, and `full` uses all four datasets with
seeds 0 through 4.

Generated outputs remain below `paper/jss/results/` as ignored build artifacts.
A full-profile receipt records the clean source revision, dependency versions,
inputs, and output hashes before manuscript tables and figures enter the
submission source.

Run the complete replication suite from the repository root. Each output
directory must be new:

```bash
uv run python -m paper.jss.replication --profile smoke \
  --output-dir paper/jss/results/replication-smoke
uv run python -m paper.jss.replication --profile quick \
  --output-dir paper/jss/results/replication-quick
uv run python -m paper.jss.replication --profile full \
  --output-dir paper/jss/results/replication-full
```

The command dispatches calibration, matched behavior, controlled performance,
tutorial, TCGA-BRCA application, and RDC projection-sensitivity analyses. It
verifies each child receipt and artifact hash before atomically publishing the
combined output directory. Use `--output-dir` for a new destination; an existing
destination is rejected to prevent results from different executions from being
mixed. The full profile also requires a clean Git worktree and rejects source
changes during execution.

Run the RDC child independently with the same new-directory rule:

```bash
uv run python -m paper.jss.replication.rdc_sensitivity --profile smoke \
  --output-dir paper/jss/results/rdc-sensitivity-smoke
```

The TCGA-BRCA inputs are derived once from four pinned public sources, and
`paper/jss/data/tcga_brca/manifest.json` records every source URL, its sha256,
the unsupervised filter chain, and the digests of the three derived files. Use
`--tcga-data-dir` to read them from another location:

```bash
uv run python -m paper.jss.replication --profile quick \
  --tcga-data-dir /path/to/tcga_brca \
  --output-dir paper/jss/results/replication-quick
```

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
`cardinality_forest`. Every index from zero through `num-shards - 1` is required
for each component. The canonical suite merges complete shard trees:

```bash
uv run python -m paper.jss.replication --profile full \
  --shard-root <shard-root> --output-dir <result-root>
```

The merge validates assignment coverage, source and input hashes, package and R
versions, artifact hashes, schemas, row inventories, and execution contexts. The
combined output includes the shard receipts used to construct each final
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
