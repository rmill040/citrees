# Relevant Research for citrees Paper

This document compiles relevant research papers organized by topic area for the
citrees paper on conditional inference trees.

## Scope note (keep reviewer-safe)

- This is an internal bibliography backlog, not a source of truth for claims.
- Only add citations to the manuscript when they directly support text/claims we
  actually make.
- The arXiv bibliography source of truth is `paper/arxiv/references.bib` (avoid
  duplicating BibTeX here to prevent drift).

---

## 1. Conditional Inference Trees (Core References)

### Cited in manuscript

| Reference                       | Venue                              | Relevance                                                                        |
| ------------------------------- | ---------------------------------- | -------------------------------------------------------------------------------- |
| Hothorn, Hornik, Zeileis (2006) | JCGS                               | **MUST-CITE**: Original ctree paper                                              |
| Hothorn & Zeileis (2015)        | JMLR                               | partykit R implementation                                                        |
| Strasser \& Weber (1999)        | Mathematical Methods of Statistics | Permutation-statistics foundation for linear-statistic tests (asymptotic theory) |

---

## 2. Permutation Tests & P-value Calculation

### Already Cited

| Reference               | Venue        | Relevance                                        |
| ----------------------- | ------------ | ------------------------------------------------ |
| Phipson & Smyth (2010)  | SAGMB        | +1 correction for permutation p-values           |
| Besag & Clifford (1991) | Biometrika   | Sequential Monte Carlo p-values                  |
| Gandy (2009)            | JASA         | Sequential MC tests with bounded resampling risk |
| Westfall & Young (1993) | Wiley (book) | maxT resampling-based multiple testing           |

### Mentioned in manuscript (now cited; out-of-scope for guarantees)

| Reference                                                                              | Venue                    | DOI/Link                    | Relevance                           |
| -------------------------------------------------------------------------------------- | ------------------------ | --------------------------- | ----------------------------------- |
| O'Brien & Fleming (1979) "A multiple testing procedure for clinical trials"            | Biometrics 35(3):549-556 | DOI:10.2307/2530245         | Classic group sequential boundaries |
| Pocock (1977) "Group sequential methods in the design and analysis of clinical trials" | Biometrika 64(2):191-199 | DOI:10.1093/biomet/64.2.191 | Alternative sequential boundaries   |
| Lan & DeMets (1983) "Discrete sequential boundaries for clinical trials"               | Biometrika 70(3):659-663 | DOI:10.1093/biomet/70.3.659 | Alpha spending function approach    |

---

## 3. Multiple Testing & FDR Control

### Already Cited

| Reference               | Venue | Relevance                       |
| ----------------------- | ----- | ------------------------------- |
| Westfall & Young (1993) | Wiley | maxT procedure for FWER control |

### Mentioned in manuscript (now cited; out-of-scope for guarantees)

| Reference                                                          | Venue                | DOI/Link                                                                            | Relevance                                                         |
| ------------------------------------------------------------------ | -------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Benjamini & Hochberg (1995) "Controlling the False Discovery Rate" | JRSS-B 57(1):289-300 | [Wiley](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1995.tb02031.x) | Foundational FDR paper (use if contrasting FDR vs.\ FWER control) |

### Mentioned in manuscript (now cited; out-of-scope for guarantees)

| Reference                                                                   | Venue                         | DOI/Link                                           | Relevance                                              |
| --------------------------------------------------------------------------- | ----------------------------- | -------------------------------------------------- | ------------------------------------------------------ |
| Barber & Candes (2015) "Controlling the False Discovery Rate via Knockoffs" | Ann. Statist. 43(5):2055-2085 | [arXiv:1404.5609](https://arxiv.org/abs/1404.5609) | Knockoff filter for FDR control without p-values       |
| Meinshausen & Buhlmann (2010) "Stability Selection"                         | JRSS-B 72(4):417-473          | [arXiv:0809.2932](https://arxiv.org/abs/0809.2932) | Subsampling-based feature selection with error control |

---

## 4. Random Forest Theory & Consistency

### Already Cited

| Reference            | Venue            | Relevance         |
| -------------------- | ---------------- | ----------------- |
| Breiman (2001)       | Machine Learning | Original RF paper |
| Geurts et al. (2006) | Machine Learning | Extra-Trees       |

### Added to manuscript (now cited)

| Reference                                                      | Venue                         | DOI/Link                                                                | Relevance                                                            |
| -------------------------------------------------------------- | ----------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Scornet, Biau, Vert (2015)** "Consistency of Random Forests" | Ann. Statist. 43(4):1716-1741 | [arXiv:1405.2881](https://arxiv.org/abs/1405.2881)                      | **HIGH PRIORITY**: First consistency proof for Breiman's original RF |
| **Biau & Scornet (2016)** "A Random Forest Guided Tour"        | TEST 25(2):197-227            | [Springer](https://link.springer.com/article/10.1007/s11749-016-0481-7) | **HIGH PRIORITY**: Comprehensive review of RF theory                 |

---

## 5. Variable Importance & Feature Selection Bias

### Already Cited

| Reference               | Venue              | Relevance                                              |
| ----------------------- | ------------------ | ------------------------------------------------------ |
| Strobl et al. (2007)    | BMC Bioinformatics | Bias in RF variable importance                         |
| Strobl et al. (2008)    | BMC Bioinformatics | Conditional permutation importance (correlation-aware) |
| Kursa & Rudnicki (2010) | JSS                | Boruta feature selection                               |
| Peng et al. (2005)      | IEEE TPAMI         | mRMR feature selection                                 |
| Guyon et al. (2002)     | Machine Learning   | RFE for gene selection                                 |
| Lundberg & Lee (2017)   | NeurIPS            | SHAP values                                            |

### Nice-to-Have

| Reference                                                                                   | Venue                           | DOI/Link                                                                    | Relevance                                  |
| ------------------------------------------------------------------------------------------- | ------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------ |
| Altmann et al. (2010) "Permutation importance: a corrected feature importance measure"      | Bioinformatics 26(10):1340-1347 | [Oxford](https://academic.oup.com/bioinformatics/article/26/10/1340/193348) | PIMP correction for permutation importance |
| Nicodemus et al. (2010) "Behaviour of RF permutation-based VIM under predictor correlation" | BMC Bioinformatics 11:110       | DOI:10.1186/1471-2105-11-110                                                | Empirical study of VIM behavior            |

---

## 6. Dependence Measures (Selectors)

### Already Cited

| Reference                      | Venue         | Relevance                     |
| ------------------------------ | ------------- | ----------------------------- |
| Szekely, Rizzo, Bakirov (2007) | Ann. Statist. | Distance correlation          |
| Lopez-Paz et al. (2013)        | arXiv         | RDC                           |
| Kraskov et al. (2004)          | Phys. Rev. E  | Mutual information estimation |

### Nice-to-Have

| Reference                                             | Venue                              | DOI/Link               | Relevance                                |
| ----------------------------------------------------- | ---------------------------------- | ---------------------- | ---------------------------------------- |
| Szekely & Rizzo (2009) "Brownian Distance Covariance" | Ann. Appl. Statist. 3(4):1236-1265 | DOI:10.1214/09-AOAS312 | Extension of distance correlation theory |

---

## 7. Post-Selection Inference

### Already Cited

| Reference                   | Venue               | Relevance                                  |
| --------------------------- | ------------------- | ------------------------------------------ |
| Berk et al. (2013)          | Ann. Statist.       | Valid post-selection inference (PoSI)      |
| Lee et al. (2016)           | Ann. Statist.       | Exact post-selection inference for lasso   |
| Leeb & Potscher (2015)      | Statistical Science | Confidence intervals after model selection |
| Fithian, Sun, Taylor (2014) | arXiv               | Optimal inference after model selection    |
| Dwork et al. (2015)         | Science             | Reusable holdout                           |

This section is well-covered.

---

## 8. BART & Bayesian Decision Trees

### Added to manuscript (now cited)

| Reference                                                                        | Venue                            | DOI/Link                                                                                                                                                           | Relevance                                                              |
| -------------------------------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| **Chipman, George, McCulloch (2010)** "BART: Bayesian Additive Regression Trees" | Ann. Appl. Statist. 4(1):266-298 | [Project Euclid](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-4/issue-1/BART-Bayesian-additive-regression-trees/10.1214/09-AOAS285.full) | **HIGH PRIORITY**: Foundational BART paper, major alternative to ctree |

### Nice-to-Have

| Reference                                                                            | Venue                            | DOI/Link                                                                                                                                                                                               | Relevance                     |
| ------------------------------------------------------------------------------------ | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------- |
| Hill (2011) "Bayesian nonparametric modeling for causal inference"                   | JCGS 20(1):217-240               | DOI:10.1198/jcgs.2010.08162                                                                                                                                                                            | BART for causal inference     |
| Hahn, Murray, Carvalho (2020) "Bayesian Regression Tree Models for Causal Inference" | Bayesian Analysis 15(3):965-1056 | [DOI:10.1214/19-BA1195](https://projecteuclid.org/journals/bayesian-analysis/volume-15/issue-3/Bayesian-Regression-Tree-Models-for-Causal-Inference--Regularization-Confounding/10.1214/19-BA1195.pdf) | Bayesian Causal Forests (BCF) |

---

## 9. Causal Forests & Honest Estimation

### Added to manuscript (now cited)

| Reference                                                                                                   | Venue                         | DOI/Link                                                                               | Relevance                                                |
| ----------------------------------------------------------------------------------------------------------- | ----------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **Wager & Athey (2018)** "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests" | JASA 113(523):1228-1242       | [Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1319839) | **HIGH PRIORITY**: Causal forests with asymptotic theory |
| **Athey, Tibshirani, Wager (2019)** "Generalized Random Forests"                                            | Ann. Statist. 47(2):1148-1178 | [arXiv:1610.01271](https://arxiv.org/abs/1610.01271)                                   | **HIGH PRIORITY**: GRF framework, honest estimation      |

### Nice-to-Have

| Reference                                                                                                           | Venue                  | DOI/Link                                                           | Relevance                      |
| ------------------------------------------------------------------------------------------------------------------- | ---------------------- | ------------------------------------------------------------------ | ------------------------------ |
| Athey & Imbens (2016) "Recursive partitioning for heterogeneous causal effects"                                     | PNAS 113(27):7353-7360 | DOI:10.1073/pnas.1510489113                                        | Causal trees, honest splitting |
| Cui et al. (2023) "Estimating heterogeneous treatment effects with right-censored data via causal survival forests" | JRSS-B 85(2):179-211   | [Oxford](https://academic.oup.com/jrsssb/article/85/2/179/7058918) | Extension to survival outcomes |

---

## 10. Survival Trees & Forests

### Added to manuscript (now cited)

| Reference                                            | Venue                            | DOI/Link                                                                                                                                           | Relevance                                                      |
| ---------------------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Ishwaran et al. (2008)** "Random Survival Forests" | Ann. Appl. Statist. 2(3):841-860 | [Project Euclid](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-2/issue-3/Random-survival-forests/10.1214/08-AOAS169.full) | **HIGH PRIORITY**: RSF, extension of RF to right-censored data |

---

## 11. Alternative Tree Algorithms

### Already Cited

| Reference             | Venue             | Relevance                                                 |
| --------------------- | ----------------- | --------------------------------------------------------- |
| Breiman et al. (1984) | Wadsworth         | CART                                                      |
| Geurts et al. (2006)  | Machine Learning  | Extra-Trees                                               |
| Loh & Shih (1997)     | Statistica Sinica | Unbiased split selection for classification trees (QUEST) |

### Nice-to-Have

| Reference                                                                                     | Venue                            | DOI/Link            | Relevance                    |
| --------------------------------------------------------------------------------------------- | -------------------------------- | ------------------- | ---------------------------- |
| Quinlan (1993) "C4.5: Programs for Machine Learning"                                          | Morgan Kaufmann                  | ISBN:1558602380     | C4.5 decision tree algorithm |
| Kass (1980) "An exploratory technique for investigating large quantities of categorical data" | Applied Statistics 29(2):119-127 | DOI:10.2307/2986296 | CHAID algorithm              |

---

## 12. Boosting Methods

### Already Cited

| Reference                   | Venue         | Relevance                  |
| --------------------------- | ------------- | -------------------------- |
| Friedman (2001)             | Ann. Statist. | Gradient boosting machines |
| Chen & Guestrin (2016)      | KDD           | XGBoost                    |
| Ke et al. (2017)            | NeurIPS       | LightGBM                   |
| Prokhorenkova et al. (2018) | NeurIPS       | CatBoost                   |

This section is well-covered.

---

## 13. Bootstrap Methods

### Already Cited

| Reference    | Venue         | Relevance          |
| ------------ | ------------- | ------------------ |
| Efron (1979) | Ann. Statist. | Original bootstrap |
| Rubin (1981) | Ann. Statist. | Bayesian bootstrap |

This section is well-covered.

---

## Summary (actionable)

- Already in `paper/arxiv/references.bib` and cited in the manuscript: Hothorn
  et al. (2006); Hothorn \& Zeileis (2015); Strasser \& Weber (1999); Phipson \&
  Smyth (2010); Besag \& Clifford (1991); Gandy (2009); Westfall \& Young
  (1993); Breiman (2001); Geurts et al. (2006); Scornet et al. (2015); Biau \&
  Scornet (2016); Chipman et al. (2010); Athey \& Imbens (2016); Wager \& Athey
  (2018); Athey et al. (2019); Ishwaran et al. (2008); Benjamini \& Hochberg
  (1995); Barber \& Cand{\`e}s (2015); Meinshausen \& B{\"u}hlmann (2010);
  O'Brien \& Fleming (1979); Pocock (1977); Lan \& DeMets (1983); and the
  feature-selection baselines used in experiments (e.g., Strobl et al. (2007,
  2008), Guyon et al. (2002), Peng et al. (2005), Kursa \& Rudnicki (2010),
  Lundberg \& Lee (2017)).
- Keep the remaining entries above as a backlog; only promote them into the
  manuscript when they directly support text/claims we actually make (avoid
  expanding scope unintentionally).

_Last edited: 2026-01-26. Verify bibliographic metadata before adding new
entries to `paper/arxiv/references.bib`._
