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

`citrees` is a Python package for conditional inference trees and forests. It
separates two decisions that standard CART training usually makes at the same
time: which feature to split on and where to place the split. At each node,
`citrees` first tests which features are associated with the response, then
searches for a threshold on the selected feature. This reduces the advantage a
feature gets simply because it offers many thresholds to try
[@Breiman1984ClassificationAndRegressionTrees;
@Strobl2007BiasRFVariableImportance].

The package provides classifier and regressor estimators that follow
scikit-learn conventions for tabular data. Fitted estimators can make
predictions, estimate class probabilities for classifiers, return decision
paths, and compute feature importances from their splits for feature ranking.
Association tests include multiple correlation and mutual information for
classification, Pearson and distance correlation for regression
[@Szekely2007DistanceCorrelation], and the randomized dependence coefficient for
both tasks [@LopezPaz2013RDC]. `citrees` also provides corrected permutation
p-values [@PhipsonSmyth2010PermutationPValues], sequential stopping, threshold
subsampling, bootstrapping, out-of-bag (OOB) scoring, and optional leaf
estimates after sample splitting.

# Statement of need

Decision trees and forests are widely used for prediction, feature ranking, and
exploratory analysis. Standard CART training chooses the feature and threshold
jointly [@Breiman1984ClassificationAndRegressionTrees]. When one feature has
many possible thresholds, it gets more opportunities to look useful even when it
is noise. This can change both the fitted tree and the feature ranking derived
from it. Conditional inference trees address the problem by testing feature
association before threshold search [@Hothorn2006UnbiasedRecursivePartitioning].

This distinction matters when the ranking itself will be interpreted. In a
simulation study, the question may be whether a method recovers known signal
features. In an applied tabular analysis, the ranking may determine which
variables are inspected, validated, or passed to a smaller downstream model. For
these uses, an accurate predictor is not enough: the feature selection mechanism
also has to be explicit and inspectable.

`citrees` is intended for researchers comparing feature selection methods,
applied scientists who need interpretable rankings, and method developers who
want a configurable implementation of conditional inference trees and forests.
It supports encoded binary and ordinal features, continuous features, and
features with many possible values. It also exposes the permutation budgets,
stopping rules, and threshold search options used during fitting.

# State of the field

The original conditional inference tree and forest methods are implemented in R,
most notably through the `partykit` ecosystem [@HothornZeileis2015Partykit;
@RPartykitPackage]. These packages remain the reference implementation for many
statistical workflows. In Python, the most common tree and forest estimators are
provided by scikit-learn [@Pedregosa2011ScikitLearn]. They provide robust CART
decision trees and random forests, but they do not use conditional inference
feature tests as the tree growth rule.

`citrees` gives Python users native estimators for conditional inference trees
and forests. It is not a wrapper around the R implementation, and it is not a
post-hoc importance method layered on top of CART. Extending scikit-learn's CART
estimators would not provide conditional-inference growth without changing their
splitting rule. `citrees` implements conditional inference tree growth, uses
Numba to accelerate permutation tests
[@Lam2015Numba], and provides association tests and forest feature importances
directly in Python. Researchers can use conditional inference trees and forests
in ordinary Python experiments and pipelines that already follow scikit-learn
conventions.

Python users can already rank features with filter methods, wrapper methods, and
importance scores from fitted tree or boosting models
[@Li2018FeatureSelectionDataPerspective]. What is missing is a native
conditional inference tree and forest implementation that uses feature
association tests as the tree growth rule. `citrees` provides that estimator
interface directly in Python.

# Software design

The public API follows scikit-learn conventions. Internally, `citrees` separates
association tests, split criteria, and threshold search strategies from the
estimator interface. Researchers can change the association test while keeping
the rest of a tree or forest fixed, and classification and regression share the
same user-facing API with task-specific statistics.

Permutation testing is computationally expensive, so association and split
statistics use Numba. Forest estimators fit trees in parallel and support
bootstrap sampling, stratified sampling for classification, out-of-bag (OOB)
scoring, and optional leaf estimates after sample splitting. Fitted estimators
return decision paths and tree structure, and compute feature importances from
their splits so users can inspect which features drove the fitted tree or
forest.

The package makes the cost of permutation testing explicit. Fixed permutation
budgets support the most direct interpretation of nodewise p-values. Adaptive
stopping, feature scanning, and threshold subsampling reduce fitting time, but
should be treated as algorithmic fitting choices rather than exact equivalents
to full permutation tests.

# Research impact statement

`citrees` supports research workflows where feature rankings are not just
byproducts of prediction, but objects that researchers inspect and compare. The
package is designed for method comparisons, synthetic signal-recovery studies,
and applied tabular analyses that use rankings to choose variables for later
modeling.

A companion benchmark in the repository covers real and synthetic classification
and regression tasks [@Milletich2026FeatureSelection]. It compares the
conditional inference forest implementation with common tree, filter, wrapper,
and embedded baselines and evaluates downstream performance at selected feature
counts. In the real-data aggregate reported by the companion manuscript, the
selected conditional inference forest configuration ranks fourth among 17
classification methods and third among 18 regression methods. Those aggregate
ranks come from `paper/results/tables/paper_benchmark_method_aggregate.csv`.
These rankings are descriptive because the benchmark also selects configurations
on the reported surface, but they show that the implementation is a usable
baseline rather than only a reference translation.

This gives researchers a Python baseline for conditional inference trees and
forests, reusable code for fitting models and inspecting rankings, and a
reproducible comparison scaffold for studying feature-selection behavior in
tabular data.

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
