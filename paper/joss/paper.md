---
title: "citrees: Conditional Inference Trees and Forests for Python"
tags:
  - Python
  - machine learning
  - feature selection
  - decision trees
  - permutation tests
authors:
  - name: Robert Milletich
    affiliation: 1
    corresponding: true
  - name: Justin Downes
    affiliation: 1
  - name: Steve Goley
    affiliation: 1
  - name: Newel Hirst
    affiliation: 1
affiliations:
  - name: Amazon Web Services, United States
    index: 1
date: 1 July 2026
bibliography: paper.bib
---

# Summary

`citrees` is a Python package for conditional inference decision trees and
random forests. It builds trees in two stages: permutation tests first identify
features associated with the response, and threshold search is then applied to
the selected feature. Separating these steps helps reduce the high-cardinality
selection bias that can affect conventional CART-style trees
[@Breiman1984ClassificationAndRegressionTrees;
@Strobl2007BiasRFVariableImportance].

The package provides scikit-learn-style classifiers and regressors with
prediction, classifier probabilities, decision paths, impurity-decrease
feature-importance outputs for split-derived rankings, and parallel forest
support. Its selectors include multiple correlation for classification, Pearson
and distance correlation for regression [@Szekely2007DistanceCorrelation], and
randomized dependence coefficient selectors [@LopezPaz2013RDC]. `citrees` also
supports Bonferroni correction and fixed-budget Phipson-Smyth-corrected Monte
Carlo p-values [@PhipsonSmyth2010PermutationPValues], plus optional sequential
stopping for faster model-building, threshold subsampling, bootstrapping,
out-of-bag scoring, and sample-split leaf re-estimation.

# Statement of need

Tree-based models are widely used for prediction, feature ranking, and
exploratory analysis, but standard CART-style training can conflate variable
selection with threshold search [@Breiman1984ClassificationAndRegressionTrees].
A continuous or high-cardinality noise feature has many opportunities to produce
an apparently good split, which can bias both the tree structure and the
resulting feature importance. The conditional inference framework addresses this
by testing whether each feature is associated with the response before selecting
a split threshold [@Hothorn2006UnbiasedRecursivePartitioning].

`citrees` brings this workflow into a modern Python package with estimator
interfaces that fit naturally into Python machine-learning workflows. The target
audience includes researchers comparing feature-selection methods, applied
scientists who need interpretable feature rankings, and method developers who
want a configurable implementation of conditional-inference-style trees and
forests. The package supports encoded binary, ordinal, continuous, and
high-cardinality features, and is especially useful when many candidate
thresholds raise selection-bias concerns.

# State of the field

The original conditional inference tree and forest methods are implemented in R,
most notably through the `partykit` ecosystem [@HothornZeileis2015Partykit;
@RPartykitPackage]. These packages are mature and remain the reference
implementation for many statistical workflows. In Python, the most commonly used
tree and forest estimators are provided by scikit-learn
[@Pedregosa2011ScikitLearn]. They provide robust CART-style decision trees and
random forests, but they do not implement conditional-inference feature testing
as the tree-growing rule.

`citrees` fills this gap as a standalone Python estimator family: it implements
conditional-inference tree-growing, Numba-accelerated permutation tests,
configurable association selectors, and forest ranking utilities. These
statistical tree-growing rules are outside scikit-learn's CART estimators, so
`citrees` packages them with dedicated validation and documentation.

Feature-selection workflows in Python are often built from filter methods,
wrapper methods, or embedded importance scores from tree and boosting libraries.
Those tools can rank features [@Li2018FeatureSelectionDataPerspective], but they
do not provide conditional inference trees and forests as reusable Python
estimators. `citrees` is complementary to that ecosystem: it provides the
estimator implementation, while the companion methods manuscript evaluates the
resulting rankings against those alternatives.

# Software design

At the public API, `citrees` follows scikit-learn conventions. Internally, tree
and forest estimators, parameter validation, selector tests, split scores, and
threshold generation live in separate modules. This keeps the estimator API
stable while allowing users to change the statistical test or split strategy.

Computationally expensive association and split statistics are implemented with
Numba so the estimators remain usable for repeated permutation testing. The
implementation separates pure tree construction from forest aggregation, which
allows forests to clone tree estimators and train them in parallel. The same
separation also supports bootstrap sampling, stratified sampling for
classification forests, out-of-bag scoring, and optional sample-split leaf
re-estimation.

The benchmark scripts under `paper/` support the separate methods and benchmark
manuscript; the JOSS submission focuses on the reusable Python package and its
estimator design.

The main trade-off is speed versus the amount of permutation testing. Fixed
permutation budgets make node-level p-values easiest to interpret. Adaptive
stopping, feature scanning, and threshold subsampling reduce runtime for model
building, and `citrees` exposes these settings so users can choose the right
level of testing for their study.

# Research impact statement

`citrees` is accompanied by a reproducible benchmark study comparing conditional
inference tree and forest rankings with tree-based, filter, and wrapper
feature-selection baselines. The repository includes experiment configuration,
analysis scripts, generated tables and figures, and the manuscript describing
the protocol and results. In the benchmark's complete-case summary, CIF ranks
fourth among 17 classification methods on 22 real-data datasets and third among
18 regression methods on 8 datasets, with additional synthetic feature-recovery
experiments. Because configurations are selected on the benchmark itself, these
results are descriptive rather than confirmatory.

The companion manuscript is the venue for the theory, algorithm details, and
benchmark interpretation. This JOSS paper uses those materials as evidence that
the software supports a complete research workflow, while keeping the focus on
the package that reviewers and users can install, test, and reuse.

The software is packaged under an MIT license, has public development history
spanning multiple years, and includes unit, integration, and paper-pipeline
tests [@citrees]. Documentation covers installation, estimator usage, selector
and splitter options, permutation testing, honest estimation, and configuration
options. These materials make the package reusable for researchers who want to
reproduce the benchmarks, compare conditional-inference-style rankings with
other selectors, or use the estimators in their own Python workflows.

# AI usage disclosure

Generative AI tools were used, including OpenAI Codex, a GPT-5-based coding
agent, and Claude Code using Anthropic Claude models through Amazon Bedrock;
exact Bedrock model IDs were not retained in the repository. These tools helped
with code review, refactoring, test scaffolding, documentation edits, and
drafting and revising manuscript text. Human authors reviewed, edited, and
validated AI-assisted outputs, ran the tests and manuscript-build checks, and
made the core scientific, software-design, and submission decisions. AI tools
are not listed as authors and were not used to make independent evaluative
decisions about the software's correctness or scholarly significance.

# Acknowledgements

The authors acknowledge Amazon Web Services for supporting this work. The
authors are responsible for the design, implementation, analysis, and contents
of this submission.

# References
