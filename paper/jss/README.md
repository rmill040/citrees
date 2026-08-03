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

| Analysis                | Purpose                                                                                | Primary comparisons                                         |
| ----------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Null calibration        | Measure feature-test and root-split rejection under independence                       | Fixed permutation tests, adaptive stopping, simple stopping |
| Split-variable bias     | Measure selection frequency when noise variables differ only in cardinality            | `citrees`, `partykit::ctree`, CART                          |
| Power and recovery      | Show linear, nonlinear, correlated, and interaction boundaries                         | All compatible `citrees` selectors and multi-selector tests |
| Reference behavior      | Compare fitted structure, rankings, predictions, and documented differences            | `citrees`, `partykit::ctree`, `partykit::cforest`           |
| Scaling                 | Measure runtime and peak memory across controlled problem dimensions                   | `citrees`, `partykit`, scikit-learn                         |
| DGRP application        | Demonstrate leakage-safe screening and linkage-disequilibrium-aware stability analysis | Tree, forest, linear, and marginal baselines                |
| Tutorial                | Demonstrate the estimator interface in an executable workflow                          | Breast Cancer Wisconsin Diagnostic data                     |
| Broad benchmark context | Summarize the corrected benchmark without duplicating it                               | Final arXiv v2 artifacts                                    |

The primary biomedical application is the public Drosophila Genetic Reference
Panel cardiac dataset associated with DOI `10.7554/eLife.82459` and Zenodo DOI
`10.5281/zenodo.5582846`. DGRP lines are the independent units. The analysis
uses one prespecified primary cardiac trait, genotype filtering and confounder
handling defined before model fitting, and linkage-disequilibrium-aware
stability summaries. All outcome-dependent operations remain inside the training
folds. The Breast Cancer Wisconsin Diagnostic data provide the executable
end-to-end tutorial.

## Claim Boundaries

- Simulation results evaluate calibration, power, and known failure boundaries.
  They do not establish universal superiority.
- Reference comparisons test behavioral agreement under matched controls. They
  do not assume identical trees from different statistics and implementations.
- Cardiac results are an adapted predictive reanalysis. They do not replicate
  the mixed-effects Epi-MEIF model, identify causal variants, or support
  clinical conclusions.
- DGRP lines are the independent units. Repeated resampling quantifies
  uncertainty but does not create additional independent observations.
- Benchmark context comes only from final corrected arXiv v2 artifacts.

## Replication

The final submission provides a single entry point with two profiles:

- `quick` rebuilds analogous manuscript results within the JSS reviewer-time
  target on a regular computer.
- `full` rebuilds the reported simulation estimates and computational
  measurements and may require parallel hardware.

Generated outputs are written below `paper/jss/results/` and are not tracked.
Tables and figures cited by the manuscript are copied into the submission source
only after a full-profile receipt records the clean source revision, dependency
versions, inputs, and output hashes.

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
