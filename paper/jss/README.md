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
| NHANES application      | Show variable-ranking behavior under mixed predictor cardinality with shuffled noise controls  | citrees forest, partykit::cforest, scikit-learn random forest   |
| Tutorial                | Demonstrate the estimator interface in an executable workflow                                  | Breast Cancer Wisconsin Diagnostic data                         |
| RDC sensitivity         | Measure the accuracy, ranking stability, and runtime effect of random projection count         | 5, 10, 20, and 40 projections on four small datasets            |
| Threshold test          | Compare Bonferroni and max-type split tests: null calibration, scaling, six real datasets      | mc and pc selectors, adaptive and exhaustive stopping           |
| Cost decomposition      | Attribute the benchmark forest's fitting time at the performance reference condition           | Serial, parallel, threshold test off, adjustment off, max-type  |
| Head-to-head timing     | Time 100-tree forests end to end on synthetic and real classification data                     | `citrees`, `partykit::cforest`, scikit-learn                    |
| Broad benchmark context | Summarize the corrected benchmark without duplicating it                                       | Final arXiv v2 artifacts                                        |

The primary biomedical application is the public NHANES 2021-2023 survey. The
outcome is diabetes, defined from glycohemoglobin and self-reported diagnosis,
and the 27 predictors are demographics, examination measurements, laboratory
values, and questionnaire items whose number of distinct values runs from 2 to
roughly the sample size. Three shuffled copies of real columns are noise by
construction at binary, medium, and high cardinality. The design is frozen in
`replication/nhanes_diabetes-specification.json`: the primary hypothesis is that
impurity-based random-forest importance ranks the high-cardinality noise column
as more important than citrees does, with the partykit reference fitted on
identical input. The Breast Cancer Wisconsin Diagnostic data provide the
executable end-to-end tutorial.

## Claim Boundaries

- Simulation results estimate null calibration and split-variable selection bias
  across the specified designs.
- Reference comparisons quantify split decisions, conditional root agreement,
  native feature-summary concordance, and held-out prediction behavior under
  identical folds and aligned structural controls.
- NHANES results describe variable-ranking behavior under mixed predictor
  cardinality; they are not a clinical prediction model or an aetiological
  claim.
- Survey participants are the independent units, and repeated stratified
  cross-validation quantifies uncertainty over these participants.
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
- `full` rebuilds the manuscript analyses and computational measurements. The
  cost decomposition and head-to-head timing children refuse to run their `full`
  profile on a host with fewer than 32 logical CPUs, and the head-to-head child
  also needs R and `partykit`. `smoke` and `quick` make no timing claim.

The wall-clock time and hardware of each profile come from the execution
receipts under `paper/jss/results/`. The `full` controlled-performance analysis
ran as 88 shards on EC2 c6a.8xlarge hosts (32 logical CPUs, AMD EPYC 7R13,
x86_64 Linux): 62,663 seconds (about 17.4 hours) from first launch to last
receipt, and 637,756 seconds (about 177 hours) of summed shard time, with single
shards taking from 47 to 16,807 seconds. The `full` NHANES and threshold-test
analyses ran on x86_64 Linux EC2 hosts; their wall-clock time is not recorded.
For the `full` calibration, matched-behavior, tutorial, RDC sensitivity, cost
decomposition, and head-to-head timing analyses, and for every `smoke` and
`quick` run, wall-clock time and hardware are not recorded.

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
tutorial, NHANES application, RDC projection-sensitivity, threshold-test, cost
decomposition, and head-to-head timing analyses. It verifies each child receipt
and artifact hash before atomically publishing the combined output directory.
Use `--output-dir` for a new destination; an existing destination is rejected to
prevent results from different executions from being mixed. The full profile
also requires a clean Git worktree and rejects source changes during execution.

Run any child independently with the same new-directory rule, for example:

```bash
uv run python -m paper.jss.replication.rdc_sensitivity --profile smoke \
  --output-dir paper/jss/results/rdc-sensitivity-smoke
uv run python -m paper.jss.replication.head_to_head_timing --profile smoke \
  --output-dir paper/jss/results/head-to-head-smoke
```

The NHANES inputs are 15 public SAS transport files (14 MB) downloaded from the
CDC on first use and verified against the digests pinned in the specification.
Use `--nhanes-data-dir` to keep them in another location:

```bash
uv run python -m paper.jss.replication --profile quick \
  --nhanes-data-dir /path/to/nhanes_2021_2023 \
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

## Head-to-head timing and cost decomposition

`paper/jss/replication/head_to_head_timing.py` produces the head-to-head timing
tables: fit-only wall-clock time of 100-tree classification forests, one fresh
child process per fit, for six configurations (the benchmark `citrees`
configuration, the same without the Bonferroni adjustment over candidate
thresholds, the same with the max-type test, `partykit::cforest` on 1 and 32
cores, and a scikit-learn random forest). The `full` profile covers 30 cells (20
synthetic, 10 real) with two repeats per cell and one on letter, isolet, and
gisette, and censors a fit after 1,200 seconds. It runs the cells serially on
one host; `--shard-index` and `--num-shards` split the fits round-robin across
hosts outside the suite. The child writes one row per fit
(`head_to_head_raw.csv`) and the median per cell and configuration
(`head_to_head_summary.csv`), the schema of the tracked
`paper/results/tables/paper_h2h_onepass_summary.csv` behind the tables. The
tracked summary has no scikit-learn column.

`paper/jss/replication/performance_decomposition.py` produces the cost
decomposition table: the benchmark tree and forest at the controlled-performance
reference condition (1,000 observations, 50 predictors, 100 trees, three timed
repeats) under variants that isolate the cost of parallel execution, the
threshold test, its Bonferroni adjustment, and the max-type test, next to the
exhaustive tree and forest. Its summary has the schema of the tracked
`paper/results/tables/paper_performance_decomposition.csv`.

## Tables the suite does not regenerate

The following tracked CSVs under `paper/results/tables/` come from EC2 runs of
suite children with an analysis switch set, outside the suite command. The
builder `uv run python paper/analysis/build_review3_tables.py` aggregates them
from the raw outputs under `../data/review3/results/` and `../data/review4/`,
which are archived in the project's reference release under
`reference/v3/ablations/review3/` and `reference/v3/ablations/review4/`. The
suite receipts do not record the switches.

- The false-split and Stage-B-only tables of the per-threshold rule without the
  adjustment over candidates (`paper_noadjust_calibration.csv`,
  `paper_noadjust_stageb.csv`):
  `threshold_test --profile full --tests bonferroni_unadjusted` with
  `--studies calibration` and `--studies stageb`.
- The matched-behavior positive control with feature and threshold scanning on
  (`paper_behavior_scanning_control.csv`): the behavior shards with
  `CITREES_BEHAVIOR_SCANNING=1`.
- The NHANES shuffled-column controls (`paper_nhanes_controls.csv`):
  `nhanes_diabetes --profile full` with `CITREES_NHANES_CONTROLS=1`, and again
  with `CITREES_NHANES_REDRAW=1` for the per-repeat redraw.
- The equal-work exhaustive rows of the controlled-performance table
  (`paper_performance_equal_work.csv`): the 88 performance shards with
  `CITREES_PERF_EQUAL_WORK=1`, plus the second-fit reruns of the equal-work
  citrees arm and of `partykit` at its own budget. The controlled-performance
  table is therefore regenerated by the suite except for these rows.
- The cost decomposition (`paper_performance_decomposition.csv`): the tracked
  copy is the builder's aggregation of an EC2 run of the decomposition child;
  the suite regenerates it directly.

The head-to-head timing tables are regenerated by the suite's head-to-head
child; the tracked `paper_h2h_onepass_summary.csv` has no builder in the
repository.

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
