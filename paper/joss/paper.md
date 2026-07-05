---
title: "citrees: Conditional Inference Trees and Forests for Python"
tags:
  - Python
  - machine learning
  - feature selection
  - decision trees
  - random forests
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

`citrees` is a Python package for conditional inference trees and forests for
classification and regression. The package provides scikit-learn-style
estimators for tabular data, including predictions, class-probability estimates
for classifiers, out-of-bag scoring, decision paths, and split-derived feature
importances.

The package supports prediction and feature importances for conditional
inference trees and forests. Users can fit a single tree for a readable
partition, or a forest for more stable predictions and feature importances. The
same estimator interface supports train/test evaluation and studies that compare
association tests, stopping rules, or feature-ranking behavior.

The core tree-growing procedure separates feature selection from threshold
selection. At each node, `citrees` first tests which features are associated
with the response, then searches for a threshold on the selected feature. This
reduces the advantage a feature gets simply because it offers many thresholds to
try
[@Breiman1984ClassificationAndRegressionTrees;
@Strobl2007BiasRFVariableImportance].
`citrees` includes association statistics for linear and nonlinear dependence:
multiple correlation and mutual information for classification, Pearson and
distance correlation for regression, and the randomized dependence coefficient
for both tasks [@Szekely2007DistanceCorrelation; @LopezPaz2013RDC]. It also
provides corrected permutation p-values [@PhipsonSmyth2010PermutationPValues],
sequential stopping, threshold subsampling, bootstrapping, parallel forest
fitting, feature muting, and optional leaf estimates after sample splitting.

# Statement of need

Decision trees and forests are used for prediction and to compute feature
importances. Standard CART training chooses the feature and threshold
jointly [@Breiman1984ClassificationAndRegressionTrees]. This can favor features
with many possible thresholds, because they have more chances to produce an
apparently good split even under noise. The effect matters for the fitted model
itself: it can change tree structure, predictions, and split-derived summaries
such as feature importances. Conditional inference trees address this issue by
testing feature association before threshold search
[@Hothorn2006UnbiasedRecursivePartitioning].

`citrees` provides this approach as a native Python estimator library. It is
designed for users who want to fit conditional inference trees and forests for
classification or regression, use them in scikit-learn-style workflows, compute
feature importances, and choose practical fitting controls such as permutation
budgets, sequential stopping, feature scanning, feature muting, and threshold
search settings.
The fitted estimators can produce predictions and feature importances.

Python users can train CART-style trees, run a separate feature selector before
fitting, or call R packages from Python. These options do not put
conditional-inference feature tests inside a scikit-learn-style estimator.
`citrees` does: fitted trees make predictions, provide decision paths, and
compute split-derived feature importances.

# State of the field

The original conditional inference tree and forest methods are implemented in R,
most notably through the `partykit` ecosystem [@HothornZeileis2015Partykit;
@RPartykitPackage]. These packages remain the reference implementation for many
statistical workflows. In Python, scikit-learn provides the standard tree and
forest estimators [@Pedregosa2011ScikitLearn], but these estimators use
CART-style split selection rather than conditional inference feature tests.

`citrees` provides a native Python implementation of conditional inference trees
and forests. It implements conditional inference tree growth directly, rather
than wrapping R or adding importance scores after CART training. The package
provides classifier and regressor trees and forests, Numba-accelerated
permutation-test computations [@Lam2015Numba], configurable association tests,
decision paths, and split-derived feature importances. Python users can
therefore use conditional inference trees and forests in workflows similar to
those used for scikit-learn estimators: prediction, comparison studies, and
feature analysis.

Other Python tools cover adjacent tasks. Pre-fit feature selectors can rank
variables before training, and permutation-importance tools can compute
importances after a predictor is fitted. `citrees` addresses a different point
in the modeling process: feature association testing is part of recursive
partitioning itself, alongside threshold search, prediction, and feature
importances.

# Software design

The public API follows scikit-learn conventions. `citrees` provides conditional
inference tree and forest estimators for classification and regression,
including `fit`, `predict`, classifier `predict_proba`, decision paths, fitted
tree structure, forest out-of-bag scores, and split-derived feature
importances.

The implementation separates the estimator interface from the statistical
components used during fitting. Association tests, split criteria, threshold
search methods, parameter validation, and forest aggregation live in separate
modules. Classification selectors include multiple correlation, mutual
information, and the randomized dependence coefficient. Regression selectors
include Pearson correlation, distance correlation, and the randomized dependence
coefficient. Split criteria include Gini and entropy for classification and mean
squared or absolute error for regression. This lets users vary association
tests or threshold-search controls without changing the surrounding workflow.

These components are configurable through estimator parameters. Users can switch
association tests, split criteria, and threshold-search methods without changing
the estimator class or downstream evaluation code. Trees and forests share the
same selector and splitter implementations, so the package presents one
consistent interface across single-tree and ensemble models.

Permutation testing is the main computational cost. `citrees` uses Numba for
association and split statistics, fits forest trees in parallel, and includes
the main runtime controls directly: permutation budgets, adaptive stopping,
feature scanning, feature muting, threshold scanning, and threshold subsampling.
Fixed permutation budgets support the most direct interpretation of nodewise
p-values, while adaptive settings provide faster fitting options.

These controls let users trade runtime for permutation-test precision. Full
permutation budgets give the most direct nodewise p-values but can dominate
fitting time in forests and high-dimensional data. Adaptive stopping and
threshold subsampling reduce that cost for exploratory or predictive fitting,
while fixed budgets remain available when users want the direct permutation
calculation.

# Research impact statement

The main research impact of `citrees` is that it provides a Python estimator
family for conditional inference trees and forests. Researchers can use the
package to fit predictive models, compute feature importances, and compare
association tests and stopping rules.

Milletich et al. [-@Milletich2026FeatureSelection] demonstrate this use in real
and synthetic classification and regression benchmarks. They use `citrees`
forests as fitted predictive models, compute feature importances from the fitted
splits, and evaluate selected features with downstream balanced accuracy and
$R^2$. In their real-data aggregate, the selected CIF configuration ranks fourth
among 17 classification methods and third among 18 regression methods.

Milletich et al. also report runtime ablations for practical fitting controls.
In the CIF ablations, disabling adaptive stopping makes fits 4.0--8.4 times
slower across task groups, with absolute downstream score changes no larger than
0.006. These are descriptive results, but they show that `citrees` can be used
to study prediction, feature importances, and runtime behavior in the same Python
workflow.

Outside the benchmark setting, users can fit a `citrees` tree or forest, make
predictions, compute feature importances, and vary association tests, stopping
rules, feature muting, or threshold search without changing the estimator API.

# AI usage disclosure

During development and manuscript preparation, the authors used OpenAI Codex
(GPT-5.5) and Claude Code (Claude Opus 4.8), both through Amazon Bedrock. These
tools helped with code edits, tests, package documentation, and manuscript
revision. Human authors reviewed and edited AI-assisted work, ran automated
tests and manuscript builds, and made the scientific, software design, and
submission decisions. The authors remain responsible for the accuracy, design,
and contents of the submission.

# Acknowledgements

The authors acknowledge Amazon Web Services for supporting this work. The
authors are responsible for the design, implementation, analysis, and contents
of this submission.

# References
