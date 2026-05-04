# Analysis Contract

This document defines the standard analysis contract for the paper.
It is the rule set that all core empirical claims must satisfy.

## 1. Standardization Rules

### 1.0 Narrative Guardrail

The paper is **not** a Python-vs-R library bakeoff.

That means:

- we do not frame the contribution as "our Python package beats the R package";
- we do not let implementation language or library ecosystem become the main
  object of comparison;
- we treat `r_ctree` and `r_cforest` as scientifically relevant reference
  implementations, not as the paper's sole target;
- we lead with the statistical testing rule, feature-selection behavior, and
  downstream scientific pattern, not with engineering convenience.

If engineering differences matter, they should be presented as secondary
practical context after the scientific claim is already clear.

### 1.1 Mirror Classification And Regression

If an analysis is part of the core story for classification, we should attempt
to mirror the same analysis for regression.

If the regression version is too weak or too small to support a parallel
claim, that must be stated explicitly. We should not silently build a much more
complete analysis stack for classification and then let regression drift into an
incompatible format.

### 1.2 Consider All Downstream Models

Headline claims must not be based on a cherry-picked downstream model such as
LR or SVM alone.

For the core benchmark story:

- Classification downstreams: `lr`, `svm`, `knn`
- Regression downstreams: `ridge`, `svr`, `knn`

Single-downstream views are still allowed, but only as supporting diagnostics.
The default interpretation must come from the all-downstream picture.

### 1.3 Trends Over `k`, Not Single-`k`

We do not center single-`k` findings.

Core reporting uses the standard values of `k`:

- `k = 5, 10, 25, 50, 100`

Every trend must either:

- show the full `k` trajectory, or
- summarize performance over all available `k` values for each dataset.

### 1.4 Dataset And Cell Accounting

Every analysis must state what dataset and cell set it uses:

- all-method complete-case,
- pairwise-available,
- per-task full available set,
- or another explicitly defined subset.

Aggregate claims must report the relevant dataset counts. We should never let
changing dataset or cell availability across `k`, downstreams, or tasks stay
implicit.

### 1.5 One Config Contract Per Analysis Family

Each family of analyses must use one explicit config-selection contract.

Examples of acceptable contracts:

- best global config per method family within task;
- fixed deployable config per family;
- descriptive all-config sensitivity analysis.

What is not acceptable is silently switching between:

- best global config,
- per-dataset best config,
- per-`k` best config,
- family-average across all configs.

If a script uses a different contract, that has to be called out.

### 1.6 Benchmark-Internal Selection Requires Disclosure

If configs are selected on the same surface later summarized, the resulting
claims are descriptive post-selection summaries, not held-out or nested
selection estimates.

That means:

- benchmark-internal best-config summaries are allowed,
- but the manuscript or support doc must say that the selection was internal to
  the benchmark surface,
- and we should not let wording imply external tuning, held-out meta-selection,
  or deployment-ready generalization guarantees that were never evaluated.

### 1.7 Supporting Proxy Studies Are Not Canonical Reruns

Some analyses are useful even when they do not rerun the full canonical
benchmark pipeline. Examples include implementation ablations, small mechanism
checks, and proxy runtime slices.

Those studies are allowed only if:

- they are labeled as supporting or proxy analyses,
- their dataset/support surface is stated explicitly,
- and they are not described as if they were reruns of the locked canonical
  Stage 1 / Stage 2 benchmark.

If a supporting analysis uses a different protocol, different datasets, or a
different scoring surface, that must be stated directly.

## 2. Reporting Layers

The paper should use two reporting layers, in this order.

### 2.1 Stratified Layer

Show the fully stratified results:

- by task,
- by downstream model,
- by `k`,
- with support counts.

This is the audit layer. It shows where patterns are consistent and where they
depend on the slice.

### 2.2 Aggregate Layer

Only after the stratified layer is understood do we make aggregate claims.

Acceptable aggregate objects include:

- mean delta over all downstream models and all available `k` values;
- mean rank over all downstream models and all available `k` values;
- consistency counts across downstream models;
- dataset-level summaries over all available `k`.

The aggregate layer must not hide whether the pattern is driven by a single
downstream model.

## 3. What Counts As A Core Claim

A core claim should satisfy all of the following:

1. It is visible across the full `k` trajectory or an available-cell aggregate
   over `k`.
2. It is checked across all downstream models for that task.
3. It is mirrored across classification and regression when appropriate, or the
   asymmetry is explicitly justified.
4. Its dataset and cell set is stated.
5. It uses the same config contract as the rest of the core analysis family.

If any one of these is false, the claim is not ready for the main text yet.
