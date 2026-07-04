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
estimators for tabular data, including prediction, classifier probability
estimation, out-of-bag scoring, decision paths, fitted tree and forest
inspection, and split-derived feature importances.

The core tree-growing procedure separates feature selection from threshold
selection. At each node, `citrees` first tests which features are associated
with the response, then searches for a threshold on the selected feature. This
reduces the advantage a feature gets simply because it offers many thresholds to
try
[@Breiman1984ClassificationAndRegressionTrees;
@Strobl2007BiasRFVariableImportance].
`citrees` includes configurable association tests, corrected permutation
p-values [@PhipsonSmyth2010PermutationPValues], sequential stopping, threshold
subsampling, bootstrapping, parallel forest fitting, and optional leaf estimates
after sample splitting.

# Statement of need

Decision trees and forests are used for prediction, model inspection, and
exploratory analysis. Standard CART training chooses the feature and threshold
jointly [@Breiman1984ClassificationAndRegressionTrees]. This can favor features
with many possible thresholds, because they have more chances to produce an
apparently good split even under noise. The effect matters for the fitted model
itself: it can change tree structure, predictions, and split-derived summaries
such as feature importances. Conditional inference trees address this issue by
testing feature association before threshold search
[@Hothorn2006UnbiasedRecursivePartitioning].

`citrees` provides this approach as a native Python estimator library. It is
designed for users who want to fit conditional inference trees and forests for
classification or regression, use them in scikit-learn-style workflows, inspect
the fitted structure, and choose practical fitting controls such as permutation
budgets, sequential stopping, feature scanning, and threshold search settings.
The same fitted estimators can also be used for prediction, model inspection,
and feature analysis.

# State of the field

The original conditional inference tree and forest methods are implemented in R,
most notably through the `partykit` ecosystem [@HothornZeileis2015Partykit;
@RPartykitPackage]. These packages remain the reference implementation for many
statistical workflows. In Python, scikit-learn provides the standard tree and
forest estimators [@Pedregosa2011ScikitLearn], but these estimators use
CART-style split selection rather than conditional inference feature tests.

`citrees` provides a native Python implementation of conditional inference trees
and forests. It implements conditional inference tree growth directly, rather
than wrapping R or adding post-hoc importance scores to CART models. The package
provides classifier and regressor trees and forests, Numba-accelerated
permutation-test computations [@Lam2015Numba], configurable association tests,
fitted-model inspection, and split-derived feature importances. Python users can
therefore use conditional inference trees and forests in workflows similar to
those used for scikit-learn estimators: prediction, model inspection, comparison
studies, and feature analysis.

# Software design

The public API follows scikit-learn conventions. `citrees` provides conditional
inference tree and forest estimators for classification and regression,
including `fit`, `predict`, classifier `predict_proba`, decision paths, fitted
tree structure, forest out-of-bag scores, and split-derived feature
importances.

The implementation separates the estimator interface from the statistical
components used during fitting. Association tests, split criteria, threshold
search methods, parameter validation, and forest aggregation live in separate
modules. This allows classification and regression estimators to share the same
user-facing design while using task-specific statistics, and it lets users vary
association tests or threshold-search controls without changing the surrounding
workflow.

Permutation testing is the main computational cost. `citrees` uses Numba for
association and split statistics, fits forest trees in parallel, and exposes the
main runtime controls directly: permutation budgets, adaptive stopping, feature
scanning, threshold scanning, and threshold subsampling. Fixed permutation
budgets support the most direct interpretation of nodewise p-values, while the
faster settings are exposed as model-building choices for practical fitting.

# Research impact statement

The main research impact of `citrees` is that it provides a Python estimator
family for conditional inference trees and forests. Researchers can use the
package to fit predictive models, inspect tree and forest structure, compare
association tests and stopping rules, and compute split-derived feature
importances when feature analysis is part of the study.

The companion benchmark [@Milletich2026FeatureSelection] demonstrates one such
workflow. It studies real and synthetic classification and regression tasks,
uses fitted split importances as feature rankings, and evaluates selected
features with downstream balanced accuracy and $R^2$. In the real-data
aggregate reported by the benchmark, the selected CIF configuration ranks
fourth among 17 classification methods and third among 18 regression methods.
Those aggregate ranks come from
`paper/results/tables/paper_benchmark_method_aggregate.csv`.

The benchmark also reports runtime ablations for practical fitting controls. In
the CIF ablations, disabling adaptive stopping makes fits 4.0--8.4 times slower
across task groups, with absolute downstream score changes no larger than
0.006. These results should be read as descriptive benchmark evidence, but they
show that `citrees` can be used to study prediction, feature analysis, and
runtime behavior in a single reproducible Python workflow.

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
