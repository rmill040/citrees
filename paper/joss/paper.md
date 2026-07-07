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
classification and regression. It provides tree and forest estimators for
tabular data while keeping the statistical split-selection procedure explicit
and configurable.

The core tree-growing procedure follows the conditional-inference separation of
feature selection from threshold selection
[@Hothorn2006UnbiasedRecursivePartitioning]. At each node, `citrees` first tests
which features are associated with the response, then searches for a threshold on
the selected feature. This reduces the advantage a feature gets simply because it
offers many thresholds to try
[@Breiman1984ClassificationAndRegressionTrees;
@Strobl2007BiasRFVariableImportance].
`citrees` includes association statistics for linear and nonlinear dependence:
multiple correlation and mutual information for classification, Pearson and
distance correlation for regression, and an RDC-style nonlinear dependence
statistic for both tasks, based on the empirical-CDF random feature map of
Lopez-Paz et al. [@Szekely2007DistanceCorrelation; @LopezPaz2013RDC]. It also
provides corrected permutation p-values [@PhipsonSmyth2010PermutationPValues],
sequential stopping, threshold subsampling, bootstrapping, parallel forest
fitting, feature muting, and optional leaf estimates after sample splitting.

# Statement of need

Conditional inference trees address a model-fitting problem in decision trees.
In CART training, the split variable and threshold are selected together, so a
feature with many possible thresholds has more opportunities to produce a
favorable split under noise [@Breiman1984ClassificationAndRegressionTrees].
Conditional inference trees separate these decisions: they test feature-response
association at a node, use that test to choose a split variable, and then search
thresholds for that feature [@Hothorn2006UnbiasedRecursivePartitioning].

For Python users, the practical need is a trainable conditional-inference model
that fits familiar estimator workflows. Users can train CART trees and forests
with scikit-learn [@Pedregosa2011ScikitLearn], run feature selectors before
training, or compute permutation importances after training. Similar
conditional-inference tree and forest capabilities are available in R, most
notably through the `partykit` ecosystem
[@HothornZeileis2015Partykit; @RPartykitPackage]. To our knowledge, Python does
not provide these models through the common scikit-learn estimator API used in
many data science workflows.

`citrees` provides these estimators for classification and regression. Users can
fit a single tree for inspection or a forest for prediction, evaluate the model
with train/test code, inspect decision paths and learned tree structure, and
compute feature summaries from the splits used by the model. The association
test, stopping rule, permutation budget, and threshold search are fitting
choices, so they belong in the estimator rather than in a separate post-training
summary.

# State of the field

Conditional inference trees and forests are established in the R statistical
software ecosystem through `ctree`, `cforest`, and the `partykit` framework
[@Hothorn2006UnbiasedRecursivePartitioning; @HothornZeileis2015Partykit;
@RPartykitPackage]. Related work on random forest variable importance and
conditional variable importance also shows why split selection and feature
ranking are linked questions in tree ensembles
[@Strobl2007BiasRFVariableImportance; @Strobl2008ConditionalVIM].

`citrees` brings this class of models into Python while exposing choices that
are often fixed inside a tree implementation or handled outside the model. It
includes a configurable set of linear and nonlinear association tests, including
an RDC-style nonlinear dependence statistic for classification and regression
[@LopezPaz2013RDC]. It supports max-T permutation tests that combine compatible
bounded selectors [@WestfallYoung1993ResamplingBasedMultipleTesting], corrected
permutation p-values [@PhipsonSmyth2010PermutationPValues], optional sample
splitting for leaf estimation, and permutation test computations accelerated
with Numba [@Lam2015Numba].

# Software design

The public API provides conditional inference tree and forest estimators for
classification and regression, including `fit`, `predict`, classifier
`predict_proba`, decision paths, learned tree structure, forest out-of-bag scores,
and feature importances computed from splits.

The implementation separates the estimator interface from the statistical
components used during fitting. Association tests, split criteria, and threshold
search methods are registered independently from the tree and forest estimators,
while Pydantic validation checks parameter combinations before fitting.
Classification selectors include multiple correlation, mutual information, and
an RDC-style nonlinear dependence statistic. Regression selectors include
Pearson correlation, distance correlation, and the same RDC-style statistic.
Split criteria include Gini and entropy for classification and mean squared or
absolute error for regression.

This structure keeps fitting controls explicit while reusing the same selector
and splitter implementations across tree and forest estimators. Users can change
association tests, split criteria, and threshold search methods through estimator
parameters.

Permutation testing is the main computational cost. `citrees` uses Numba for
association and split statistics, fits forest trees in parallel, and exposes
runtime controls for permutation budgets, adaptive stopping, feature scanning,
feature muting, and threshold search. Full permutation budgets give direct
nodewise permutation p-values but can dominate fitting time in forests and
high-dimensional data; adaptive stopping and threshold subsampling reduce that
cost for exploratory or predictive fitting.

# Research impact statement

Conditional inference trees and forests are used as models in applied research,
with recent examples in epistasis detection, PM10 air pollution modeling,
National Park visitor surveys, and conversation analysis of gaze and speaker
selection [@Saha2022EpiMEIF; @Sohrab2024PM10; @Fristrup2024Lighting;
@Ruhleman2024GazeAlternation]. Conditional variable-importance work also
connects this model family to feature-ranking questions
[@Strobl2008ConditionalVIM]. `citrees` makes this model class available in
Python for fitting, prediction, evaluation, and inspection. Researchers can
compare association tests, stopping rules, and threshold search choices as part
of model training rather than as steps before or after it.

Milletich et al. [-@Milletich2026FeatureSelection] demonstrate this use in real
and synthetic classification and regression benchmarks. They use `citrees`
forests as predictive models, compute feature importances from the learned
splits, and evaluate selected features with downstream balanced accuracy and
$R^2$. Across the real data aggregate, the selected CIF configuration ranks
fourth among 17 classification methods and third among 18 regression methods.

The companion benchmark also reports dataset-level ranks and runtime ablations
for the selected CIF configuration. The configuration is in the top half on 21
of 22 classification datasets and 6 of 8 regression datasets. In CIF runtime
ablations, disabling adaptive stopping makes fits 4.0--8.4 times slower across
task groups, with absolute downstream score changes no larger than 0.006
[@Milletich2026FeatureSelection]. These descriptive results document predictive
rank, dataset coverage, and runtime tradeoffs for the evaluated `citrees` forest.

# AI usage disclosure

During development and manuscript preparation, the authors used OpenAI Codex
(GPT-5.5) and Claude Code (Claude Opus 4.8), both through Amazon Bedrock. These
tools helped with code edits, tests, package documentation, and manuscript
revision. Human authors reviewed and edited AI-assisted work, ran automated
tests and manuscript builds, and made the scientific, software design, and
submission decisions. The authors remain responsible for the accuracy, design,
and contents of the submission.

# Acknowledgements

The authors acknowledge Amazon Web Services for supporting this work.

# References
