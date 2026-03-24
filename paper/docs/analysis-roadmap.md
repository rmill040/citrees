# Analysis Roadmap

Comprehensive list of every analysis to explore for the citrees paper. Data is
clean and aggregated (March 2026). CLF: 30 configs × 32 datasets, REG: 31
configs × 16 datasets, 5 seeds × 5 folds each.

Status legend: [ ] not started, [~] in progress, [x] done

---

## 1. Overall Method Comparison (Friedman / Nemenyi)

- [ ] 1.1 CD diagram: all methods, balanced_accuracy, averaged across k and downstream models
- [ ] 1.2 CD diagram: all methods, f1_macro
- [ ] 1.3 CD diagram: all methods, accuracy
- [ ] 1.4 CD diagram: regression R²
- [ ] 1.5 CD diagram: regression MAE
- [ ] 1.6 Best-config-per-method CD diagram (removes config averaging penalty — CIF-MC vs rf vs cat etc.)
- [ ] 1.7 Friedman test statistics table with p-values, Kendall's W effect size
- [ ] 1.8 Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction (significance matrix)
- [ ] 1.9 Cohen's d effect size matrix for key pairwise comparisons

---

## 2. K-Value Analysis (Feature Budget Sensitivity)

- [ ] 2.1 **Trend plots**: balanced_accuracy vs k for each method (line chart, k on x-axis, one line per method)
- [ ] 2.2 Same trend plots per downstream model (LR, SVM, KNN separate panels)
- [ ] 2.3 Same trend plots per dataset (or grouped by dataset characteristics)
- [ ] 2.4 CD diagrams at EACH k value separately (k=5, 10, 25, 50, 100) — does CIF rank change with k?
- [ ] 2.5 K-curve shape table: Δ(5→25), Δ(25→100), Δ(5→100) per method — who benefits most from more features?
- [ ] 2.6 K-sensitivity metric: variance of rank across k values — who is k-robust vs k-sensitive?
- [ ] 2.7 Regression: R² vs k trend plots
- [ ] 2.8 Interaction: k × dimensionality — does optimal k depend on p?
- [ ] 2.9 Interaction: k × dataset size — does optimal k depend on n?

---

## 3. Downstream Model Analysis

- [ ] 3.1 CLF × LR: CD diagram + ranking table
- [ ] 3.2 CLF × SVM: CD diagram + ranking table
- [ ] 3.3 CLF × KNN: CD diagram + ranking table
- [ ] 3.4 REG × Ridge: CD diagram
- [ ] 3.5 REG × SVR: CD diagram (CIF claims #1)
- [ ] 3.6 REG × KNN: CD diagram (CIF claims #2)
- [ ] 3.7 Per-model consistency: does method ranking change across downstream models? (rank correlation)
- [ ] 3.8 Linear vs nonlinear downstream: do CIF rankings favor nonlinear downstream models?
- [ ] 3.9 Interaction: downstream model × k — best k varies by model?

---

## 4. Dimensionality Analysis

- [ ] 4.1 Low-dim (p ≤ 100) subset: CD diagram, rankings (CIF claims parity)
- [ ] 4.2 Mid-dim (100 < p ≤ 1000) subset: CD diagram
- [ ] 4.3 High-dim (p > 1000) subset: CD diagram (CIF struggles here)
- [ ] 4.4 Scatter: p vs CIF relative advantage (Spearman correlation)
- [ ] 4.5 Performance vs n/p ratio — methods behave differently when n >> p vs n << p?
- [ ] 4.6 Per-method performance curves as p increases (line chart, p on x-axis)
- [ ] 4.7 Heatmap: method × dataset ordered by p — visual pattern of who degrades with dimensionality

---

## 5. Dataset Characteristic Analysis

- [ ] 5.1 Real vs synthetic: do rankings change?
- [ ] 5.2 Binary vs multiclass classification: CIF might shine on multiclass
- [ ] 5.3 Balanced vs imbalanced classes
- [ ] 5.4 Small n (< 500) vs large n (> 5000)
- [ ] 5.5 Sparse binary features (dexter, dorothea) vs dense continuous
- [ ] 5.6 Per-dataset performance table: method × dataset grid (heatmap)
- [ ] 5.7 Dataset characteristics table for paper: n, p, classes, source, type
- [ ] 5.8 Dataset clustering: which datasets produce similar method rankings? (dendrogram or PCA)

---

## 6. CIF Configuration Sensitivity

- [ ] 6.1 MC vs RDC selector: head-to-head per dataset, CD diagram, effect sizes
- [ ] 6.2 Honesty vs no honesty: effect size per dataset
- [ ] 6.3 4-config spread per dataset: heatmap showing sensitivity (best − worst CIF config)
- [ ] 6.4 Best CIF config (CIF-MC) vs all methods: dedicated ranking
- [ ] 6.5 CIF oracle (best config per dataset) vs best config: config selection ceiling
- [ ] 6.6 Config sensitivity vs dimensionality: does config matter more on high-dim?
- [ ] 6.7 Config sensitivity vs dataset type: which data properties make config choice critical?
- [ ] 6.8 CIT config analysis: same MC vs RDC, honesty breakdown for single tree

---

## 7. CIT vs CIF (Ensembling Value)

- [ ] 7.1 CIT vs CIF head-to-head: per k, per dataset, effect sizes
- [ ] 7.2 CIF − CIT gap vs k: trend plot showing ensembling value grows with k
- [ ] 7.3 CIT vs CIF on small datasets: does ensembling help when n is small?
- [ ] 7.4 CIT vs CIF stability comparison
- [ ] 7.5 CIT vs CIF per downstream model

---

## 8. CIF vs R Implementations

- [ ] 8.1 CIF vs r_cforest: head-to-head per dataset, all k values
- [ ] 8.2 CIF vs r_ctree: same
- [ ] 8.3 Win rate tables: CIF wins X/Y datasets at each k
- [ ] 8.4 Effect size (Cohen's d): how large are improvements?
- [ ] 8.5 Scatter plot: CIF accuracy vs r_cforest accuracy per dataset
- [ ] 8.6 Regression: CIF vs r_cforest R² comparison

---

## 9. Embedding Value (CIF vs Standalone Permutation Tests)

- [ ] 9.1 CIF vs ptest_mc: per dataset, all k — when does tree embedding help vs hurt?
- [ ] 9.2 CIF vs ptest_rdc: same
- [ ] 9.3 Per-dataset pattern analysis: structured data (face images) vs sparse data
- [ ] 9.4 Interaction: embedding value × dimensionality
- [ ] 9.5 Interaction: embedding value × dataset type (synthetic)
- [ ] 9.6 CIT vs ptest_mc: does even a single tree add value over standalone test?

---

## 10. Synthetic Ground Truth: Feature Selection Quality

- [ ] 10.1 Precision@k by method: k=5, 10, 20 (bar chart or table)
- [ ] 10.2 Recall@k by method
- [ ] 10.3 F1@k by method
- [ ] 10.4 **Precision vs k trend plot**: precision@k as k increases, one line per method
- [ ] 10.5 **Recall vs k trend plot**: same
- [ ] 10.6 Precision@k by dataset type (7 types): heatmap or grouped bar
- [ ] 10.7 CIF toeplitz strength deep dive: why does CIF get ~0.94?
- [ ] 10.8 CIF weak signal weakness deep dive: why 0.165 vs 0.792 (rfe)?
- [ ] 10.9 Confounder resistance: confounder_rate@k by method (lower = better)
- [ ] 10.10 Redundant feature handling: precision_ir vs precision comparison
- [ ] 10.11 Synthetic → real correlation: Spearman between precision@10 and real balanced_accuracy
- [ ] 10.12 By method category: do ptest methods beat embedding methods on any synthetic type?
- [ ] 10.13 Regression synthetic: same analyses as above for reg task
- [ ] 10.14 CLF vs REG synthetic comparison: do methods rank the same across tasks?

---

## 11. Ranking Stability

- [ ] 11.1 Nogueira stability index @k: all methods, k=5, 10, 25
- [ ] 11.2 Jaccard@k across seeds/folds: per method
- [ ] 11.3 Stability vs accuracy scatterplot: Pareto frontier analysis
- [ ] 11.4 CIF stability by config: is MC more stable than RDC?
- [ ] 11.5 Stability by dimensionality: does stability degrade with p?
- [ ] 11.6 Stability by dataset: which datasets produce unstable rankings?
- [ ] 11.7 Feature overlap between methods: Jaccard between top-k of different methods on same data
- [ ] 11.8 Regression stability: same analyses

---

## 12. P-Value Calibration

- [ ] 12.1 Type I error rate under null (rejection rate at α=0.05)
- [ ] 12.2 Adaptive vs fixed-B: does early stopping bias p-values?
- [ ] 12.3 ECDF plot: empirical CDF should be uniform under null
- [ ] 12.4 Calibration by B (n_resamples): how many permutations needed?
- [ ] 12.5 P-value distribution under alternative: power analysis

---

## 13. Pairwise Head-to-Head Comparisons

- [ ] 13.1 Win/loss/tie tables: method A vs B counts across all datasets
- [ ] 13.2 CIF vs RF detailed scatter: per-dataset accuracy comparison
- [ ] 13.3 CIF vs CatBoost detailed scatter
- [ ] 13.4 CIF vs LightGBM detailed scatter
- [ ] 13.5 CIF vs RFE detailed scatter
- [ ] 13.6 CIF vs Boruta detailed scatter
- [ ] 13.7 Pairwise dominance matrix: for each pair, % datasets where A > B

---

## 14. Cross-Task Analysis

- [ ] 14.1 Rank correlation CLF ↔ REG: Spearman between method ranks across tasks
- [ ] 14.2 Method strengths by task: who's a specialist vs generalist?
- [ ] 14.3 Per-method CLF vs REG performance comparison

---

## 15. Ranking-Level Analysis (exploit the feature_ranking column directly)

### 15.1 Rank agreement between methods
- [ ] 15.1.1 Kendall tau between every pair of methods' rankings on same dataset/seed/fold
- [ ] 15.1.2 Method similarity matrix: average rank correlation across all datasets (heatmap)
- [ ] 15.1.3 MDS or t-SNE of methods based on ranking similarity — do method categories cluster?
- [ ] 15.1.4 Do methods agree more on top features than bottom features? (top-k rank correlation vs full)
- [ ] 15.1.5 Consensus features: features ranked top-10 by ≥80% of methods on each dataset
- [ ] 15.1.6 Controversial features: high variance in rank across methods — what makes a feature divisive?

### 15.2 Ranking shape and decisiveness
- [ ] 15.2.1 Rank entropy per method: is the ranking "sharp" (clear winners) or "flat" (ambiguous)?
- [ ] 15.2.2 Effective number of features: at what rank position does ranking quality drop off?
- [ ] 15.2.3 Cumulative accuracy curve: accuracy as features are added one-by-one in ranked order
- [ ] 15.2.4 Diminishing returns point: k where adding more features stops helping (per method)
- [ ] 15.2.5 Rank position vs downstream contribution: NDCG-style weighted evaluation

### 15.3 Rank reversal and disagreement
- [ ] 15.3.1 Features that CIF ranks top-5 but RF ranks bottom-50% (and vice versa)
- [ ] 15.3.2 Rank reversal frequency: how often do methods put the same feature on opposite ends?
- [ ] 15.3.3 Which dataset types produce the most method disagreement?

### 15.4 Synthetic ranking quality (beyond precision@k)
- [ ] 15.4.1 Kendall tau between predicted ranking and true importance ordering (full ranking quality)
- [ ] 15.4.2 Spearman between predicted rank and true feature importance scores
- [ ] 15.4.3 NDCG@k: position-weighted precision (penalizes ranking a good feature at position 8 vs 2)
- [ ] 15.4.4 Mean reciprocal rank of the most important feature — who finds the #1 feature fastest?
- [ ] 15.4.5 Rank of each true informative feature: distribution plot per method

### 15.5 Continuous precision/recall curves on synthetic (compute from raw rankings)
Current synthetic analysis only has k=5,10,20. We have raw feature_ranking
lists in clf_rankings.parquet — recompute precision/recall at EVERY k from 1
to p using ground truth. This requires joining rankings with synthetic metadata.

- [ ] 15.5.1 **Precision@k curve**: k=1..p on x-axis, precision on y-axis, one line per method
  - Per dataset type (7 types × curve per method)
  - Averaged across all synthetic datasets
  - Shows: at what k do methods find ALL informative features?
- [ ] 15.5.2 **Recall@k curve**: same structure — when does recall hit 1.0?
  - Critical question: at what k does CIF achieve 100% recall vs RF vs ET?
- [ ] 15.5.3 **F1@k curve**: combined precision/recall tradeoff
- [ ] 15.5.4 **Per-dataset-type curves**: separate panels for toeplitz, weak_signal, etc.
- [ ] 15.5.5 Area under the precision@k curve (AUPC): single summary metric for full ranking quality
- [ ] 15.5.6 k* (recovery point): smallest k where recall = 1.0 per method per dataset
  - Table: method × dataset type → median k* (lower = better ranking)
  - How many features do you need to select to capture everything?
- [ ] 15.5.7 Precision@1: who puts the single most important feature first?
- [ ] 15.5.8 Precision@n_informative: at k = exact number of informative features, what's precision?
  - This is the "oracle k" — if you knew how many features to pick
- [ ] 15.5.9 Same curves for regression synthetic datasets

### 15.6 Feature split count analysis (from rankings on synthetic)
- [ ] 15.6.1 **Informative vs noise feature selection counts**: for tree methods (CIT, CIF, RF, ET),
  count how many times each feature appears in top-k — are informative features selected
  disproportionately more?
- [ ] 15.6.2 Bar chart: average rank of informative features vs noise features per method
- [ ] 15.6.3 Separation plot: distribution of informative feature ranks vs noise feature ranks (overlapping histograms)
- [ ] 15.6.4 Per-method "signal-to-noise ratio" in rankings: mean_rank(informative) / mean_rank(noise)
- [ ] 15.6.5 CIF vs RF vs ET vs XGB comparison: who separates signal from noise best visually?
- [ ] 15.6.6 Same analysis by dataset type: on toeplitz, CIF should sharply separate; on weak_signal, less so

---

## 16. Failure Mode and Success Mode Analysis

### 16.1 When does CIF WIN?
- [ ] 16.1.1 Dataset characteristics when CIF beats RF/cat/lgbm: n, p, n_classes, class_balance
- [ ] 16.1.2 Logistic regression: P(CIF wins) ~ f(n, p, n_classes, ...) — predictive model
- [ ] 16.1.3 Decision rules: simple heuristic for when to use CIF vs alternatives
- [ ] 16.1.4 CIF win rate by dataset characteristic bin (small/medium/large n, low/mid/high p)

### 16.2 When does CIF FAIL?
- [ ] 16.2.1 What do the worst CIF datasets have in common? (feature correlation? sparsity? noise?)
- [ ] 16.2.2 Gap analysis: magnitude of CIF loss vs best method on each dataset
- [ ] 16.2.3 Is CIF failure consistent across downstream models and k values, or just specific combos?
- [ ] 16.2.4 Does CIF fail gracefully (close to best) or catastrophically (far from best)?

### 16.3 Method category analysis
- [ ] 16.3.1 Group: embedding (cit/cif/rf/et/xgb/lgbm/cat) vs filter (ptest_*) vs wrapper (boruta/rfe/pi/cpi)
- [ ] 16.3.2 Category-level CD diagrams: which category dominates?
- [ ] 16.3.3 Category × dataset type interaction: which category works best on each data type?
- [ ] 16.3.4 Category × dimensionality interaction
- [ ] 16.3.5 Conditional inference (cit/cif/r_ctree/r_cforest/ptest_*) vs heuristic (rf/et/xgb/lgbm/cat) vs wrapper

---

## 17. Fold-Level and Variance Analysis

- [ ] 17.1 Variance across folds per method: who is most sensitive to train/test split?
- [ ] 17.2 Variance across seeds per method: who is most sensitive to random seed?
- [ ] 17.3 Fold-level outlier detection: are some folds consistently bad for certain methods?
- [ ] 17.4 Coefficient of variation of accuracy by method: normalized variance comparison
- [ ] 17.5 Variance decomposition: how much variance comes from fold vs seed vs dataset vs method?
- [ ] 17.6 Bootstrap confidence intervals on ALL main gap claims (CIF−RF, CIF−r_cforest, etc.)
- [ ] 17.7 Bayesian analysis: posterior probability that CIF > RF on a random dataset

---

## 18. Metric Sensitivity

- [ ] 18.1 Do accuracy and balanced_accuracy give different rankings? (rank correlation between metrics)
- [ ] 18.2 Does f1_macro favor different methods than accuracy?
- [ ] 18.3 Is AUC more favorable to CIF than accuracy?
- [ ] 18.4 Metric agreement matrix: Kendall tau between method rankings under each metric
- [ ] 18.5 Which conclusions change depending on metric choice? (robustness check)
- [ ] 18.6 Regression: R² vs MAE vs MSE — do they agree?

---

## 19. Effective Feature Selection (n_features_selected)

- [ ] 19.1 At k=100 on small datasets (p < 100): actual features selected vs k
- [ ] 19.2 Does CIF tend to select fewer features than requested? (alpha threshold filtering)
- [ ] 19.3 Effective k vs requested k by method: who under-selects?
- [ ] 19.4 "No selection" baseline: accuracy with ALL features vs best k
- [ ] 19.5 Optimal k per dataset: which k gives best accuracy? (varies by dataset)
- [ ] 19.6 k=p performance: using all features, do rankings still matter? (ordering effect)

---

## 20. Regression Deep Dives

- [ ] 20.1 Negative R² analysis: which methods produce rankings that HURT performance?
- [ ] 20.2 COEPRA datasets deep dive: p >> n, high noise — why does everything struggle?
- [ ] 20.3 Are there datasets where feature selection never helps? (all methods R² < full model)
- [ ] 20.4 CIF #1 with SVR claim: deep dive, per-dataset breakdown, effect size
- [ ] 20.5 Linear vs nonlinear downstream: Ridge (linear) vs SVR/KNN (nonlinear) gap by method
- [ ] 20.6 Feature selection benefit: best-k accuracy minus all-features accuracy by method

---

## 21. Practical Recommendation Analysis

- [ ] 21.1 Expected regret analysis: if you pick one method, what's the expected loss vs oracle?
- [ ] 21.2 Portfolio analysis: what set of 2-3 methods covers the most datasets?
- [ ] 21.3 Decision flowchart: when to use CIF vs RF vs ptest_mc (based on dataset properties)
- [ ] 21.4 "Safe default" analysis: which method has the best worst-case performance?
- [ ] 21.5 Risk analysis: which methods have the highest variance in relative performance?

---

## 22. Timing + Ablation (OFFLINE — separate workstream)

NOT from EC2 elapsed_seconds. Run controlled local benchmarks.

- [ ] 22.1 Ablation table: CIF all optimizations → remove one at a time → none
  - Feature scanning on/off
  - Feature muting on/off
  - Adaptive early stopping on/off
  - Multi-selector max-T (mc+rdc vs mc alone)
- [ ] 22.2 Scaling curves: time vs n (500, 1K, 5K, 10K)
- [ ] 22.3 Scaling curves: time vs p (10, 100, 500, 1K)
- [ ] 22.4 CIF vs RF/ET wall time on same data
- [ ] 22.5 Speed-accuracy Pareto: does removing optimizations hurt accuracy?
- [ ] 22.6 Early stopping p-value calibration under ablation

**Design:** timing = 1 seed, 3 repeats, median. Accuracy = 5 seeds × 5 folds.
Datasets: 2-3 representative + synthetic for scaling curves.

---

## 23. Publication Figures (Final Selection)

- [ ] 23.1 Main CD diagram (all methods, main metric)
- [ ] 23.2 K-curve trend plot (top methods, balanced_accuracy vs k)
- [ ] 23.3 Synthetic precision@k heatmap by dataset type
- [ ] 23.4 CIF vs R implementations scatter
- [ ] 23.5 Stability vs accuracy scatterplot
- [ ] 23.6 Dimensionality effect plot
- [ ] 23.7 P-value calibration ECDF
- [ ] 23.8 Ablation table (from offline timing)
- [ ] 23.9 Dataset characteristics table
- [ ] 23.10 Method overview table (category, key params)
- [ ] 23.11 Method similarity dendrogram/heatmap (from ranking agreement)
- [ ] 23.12 Cumulative accuracy curve (features added in ranked order)
- [ ] 23.13 Failure mode scatter (CIF gap vs dataset property)
- [ ] 23.14 Practical decision flowchart

---

## 24. Statistical Power and Type I/II Error (Permutation Test Properties)

### 24.1 Type I error control
- [ ] 24.1.1 Rejection rate under null at α = 0.01, 0.05, 0.10 — is it calibrated at all levels?
- [ ] 24.1.2 Type I error by number of features tested (multiple testing burden)
- [ ] 24.1.3 Type I error with vs without Bonferroni correction — how conservative?
- [ ] 24.1.4 Conditional Type I error: given that feature scanning reorders features, is the test still valid?

### 24.2 Statistical power
- [ ] 24.2.1 Power curves: P(reject null | true effect size) at varying signal strengths
- [ ] 24.2.2 Power vs n_resamples: how many permutations needed for adequate power?
- [ ] 24.2.3 Power vs sample size: at what n does the test reliably detect signal?
- [ ] 24.2.4 Power comparison: adaptive early stopping vs full permutation — does stopping lose power?
- [ ] 24.2.5 Power vs feature dimensionality: does power degrade as p grows (Bonferroni penalty)?

### 24.3 P-value behavior
- [ ] 24.3.1 P-value distributions under null (should be uniform) — stratified by selector type
- [ ] 24.3.2 P-value distributions under alternative — peaked near zero?
- [ ] 24.3.3 P-value granularity: with n_resamples=minimum (199), smallest non-zero p = 1/200 — is this too coarse?
- [ ] 24.3.4 Phipson-Smyth +1 correction impact: how much does it shift p-values?

---

## 25. Sensitivity to α Threshold

- [ ] 25.1 CIF performance at α_selector = 0.01, 0.05, 0.10, 0.20 (offline local experiment)
- [ ] 25.2 Number of features passing the test at each α level
- [ ] 25.3 Does a permissive α (0.20) fix the weak signal problem?
- [ ] 25.4 Does a strict α (0.01) improve precision on synthetic at the cost of recall?
- [ ] 25.5 Optimal α per dataset type: is there a universally good α, or must it be tuned?
- [ ] 25.6 α_selector vs α_splitter: independent effects or coupled?

---

## 26. Feature Correlation Structure and Its Effect

- [ ] 26.1 Eigenvalue spectrum of feature correlation matrix per dataset
- [ ] 26.2 Intrinsic dimensionality (participation ratio) per dataset
- [ ] 26.3 CIF relative advantage vs intrinsic dimensionality — better on low intrinsic dim?
- [ ] 26.4 Feature correlation strength vs CIF's confounder resistance
- [ ] 26.5 Toeplitz analysis: varying ρ from 0 to 0.99 — at what correlation does CIF break?
- [ ] 26.6 Conditional independence structure: can CIF detect features that are marginally independent but conditionally dependent (and vice versa)?
- [ ] 26.7 Feature groups: when informative features are correlated with each other, who handles it best?

---

## 27. Noise Robustness

- [ ] 27.1 Performance as noise features are ADDED: fix informative features, vary p (10, 50, 100, 500, 1000)
- [ ] 27.2 At what noise ratio (p_noise / p_informative) does each method break down?
- [ ] 27.3 Performance with label noise (flip_y): how robust is each method to mislabeled examples?
- [ ] 27.4 Feature noise: what happens when informative features have measurement noise added?
- [ ] 27.5 CIF's hypothesis testing vs RF's impurity: which degrades more gracefully with noise?

---

## 28. Sample Size Effects (Learning Curves)

- [ ] 28.1 Feature selection quality vs n: subsample datasets to n=50, 100, 200, 500, 1000
- [ ] 28.2 At what n does CIF become competitive? (minimum viable sample size)
- [ ] 28.3 Small n regime (n < 100): does CIF's statistical test help or hurt vs heuristic methods?
- [ ] 28.4 Large n regime (n > 10000): does the advantage of statistical testing diminish?
- [ ] 28.5 n/p ratio thresholds: at what n/p does each method reliably outperform random selection?

---

## 29. Feature Importance Concentration and Distribution

- [ ] 29.1 Gini coefficient of feature importance scores: how concentrated is each method's ranking?
- [ ] 29.2 Importance entropy: flat rankings (all features similar) vs peaked (clear winners)
- [ ] 29.3 CIF vs RF importance distribution shape: does CIF produce sharper rankings?
- [ ] 29.4 Importance gap between top-k and rest: how decisive is the cutoff?
- [ ] 29.5 Does importance concentration predict downstream accuracy? (concentrated = better?)
- [ ] 29.6 "Importance cliff" detection: is there a natural cutoff point in each method's ranking?

---

## 30. Robustness and Sensitivity Checks

### 30.1 Leave-one-dataset-out
- [ ] 30.1.1 Remove each dataset: do Friedman rankings change? Which datasets are influential?
- [ ] 30.1.2 Identify datasets that disproportionately favor/penalize CIF

### 30.2 Leave-one-method-out
- [ ] 30.2.1 Remove each method: does CIF's rank change? (Is CIF's rank influenced by weak methods?)

### 30.3 Subsetting analysis
- [ ] 30.3.1 Real-only analysis: remove synthetic datasets, do conclusions hold?
- [ ] 30.3.2 Synthetic-only analysis: same
- [ ] 30.3.3 "Easy" datasets (mean accuracy > 0.9) vs "hard" datasets (< 0.7): different rankings?

### 30.4 Conclusion stability
- [ ] 30.4.1 Accumulation curves: as datasets are added one-by-one, when do rankings stabilize?
- [ ] 30.4.2 Bootstrap Friedman: resample datasets with replacement, how often does CIF's rank change?
- [ ] 30.4.3 How many datasets are needed for statistically stable conclusions?

---

## 31. Transfer and Generalization

- [ ] 31.1 Cross-seed transfer: rankings from seed 0 applied to seed 1's test data — how much accuracy is lost?
- [ ] 31.2 Cross-fold transfer: ranking from fold 0 applied to fold 1 — ranking generalization
- [ ] 31.3 Cross-dataset transfer: ranking learned on dataset A, features selected on dataset B — any signal?
- [ ] 31.4 Which methods produce the most transferable rankings?

---

## 32. Tree/Forest Structure Analysis (CIT/CIF specific)

- [ ] 32.1 Tree depth distribution: how deep do CIT trees grow on each dataset?
- [ ] 32.2 Number of leaves: CIT tree complexity vs dataset complexity
- [ ] 32.3 Features used per tree: how many of p features actually appear in splits?
- [ ] 32.4 Feature depth: where in the tree does each feature first appear? (root features = most important)
- [ ] 32.5 CIF: number of trees that select each feature — agreement across forest
- [ ] 32.6 CIF: feature importance correlation between individual trees — diversity measure
- [ ] 32.7 Does tree depth correlate with ranking quality? (deeper = more features tested = better ranking?)
- [ ] 32.8 Muted features: how many features get muted (rejected) at each node on average?
- [ ] 32.9 Early stopping frequency: what fraction of permutation tests stop early?

---

## 33. Dataset Meta-Features and Meta-Learning

- [ ] 33.1 Compute dataset meta-features: n, p, n/p ratio, n_classes, class_entropy, feature_correlation_mean, eigenvalue_decay, noise_ratio
- [ ] 33.2 Meta-learning model: predict best method from meta-features
- [ ] 33.3 CIF advantage landscape: in which region of meta-feature space does CIF dominate?
- [ ] 33.4 Similarity between datasets: which datasets cluster together based on method performance profiles?
- [ ] 33.5 Dataset difficulty: define difficulty as (1 - best_method_accuracy), correlate with CIF rank

---

## 34. Multi-Objective Analysis

- [ ] 34.1 Accuracy × stability Pareto frontier (all methods plotted)
- [ ] 34.2 Accuracy × runtime Pareto frontier (from offline timing)
- [ ] 34.3 Precision × recall Pareto on synthetic
- [ ] 34.4 Accuracy × interpretability proxy (tree depth or number of selected features)
- [ ] 34.5 Multi-criteria ranking: Pareto dominance — which methods are never dominated?
- [ ] 34.6 User preference weighting: if you weight accuracy 70% stability 30%, who wins?

---

## 35. Per-Synthetic-Dataset Deep Dives

Each synthetic type tests a specific hypothesis. Deep dive into each:

### 35.1 Standard (baseline)
- [ ] 35.1.1 All methods should perform reasonably — who's the baseline champion?
- [ ] 35.1.2 Is there a ceiling effect? Are all methods saturated?

### 35.2 Bias (high-cardinality noise)
- [ ] 35.2.1 The core conditional inference claim: CIF should resist selection bias
- [ ] 35.2.2 How much bias does RF/ET show? (false positive rate on high-cardinality noise features)
- [ ] 35.2.3 Does CIF's test correctly identify high-cardinality features as noise?
- [ ] 35.2.4 Rank of the high-cardinality noise features by method — who puts them highest?

### 35.3 Confounder (correlated noise)
- [ ] 35.3.1 confounder_rate@k by method: who selects confounders?
- [ ] 35.3.2 Does conditioning (the "C" in CIT) actually help? CIF vs RF confounder rates
- [ ] 35.3.3 How does confounder rate change with k? (more features = more confounders selected?)

### 35.4 Toeplitz (highly correlated features)
- [ ] 35.4.1 CIF near-perfect here — decompose: which folds/seeds miss?
- [ ] 35.4.2 ptest_mc/pc hits 1.0 — what's different about the marginal test vs embedded test?
- [ ] 35.4.3 Which methods are WORST on toeplitz? Why?

### 35.5 Weak signal
- [ ] 35.5.1 CIF's biggest failure — 0.165 vs rfe 0.792
- [ ] 35.5.2 Is this the α threshold rejecting real features? What p-values do informative features get?
- [ ] 35.5.3 Would a less strict α (0.10, 0.20) fix this? (offline experiment)
- [ ] 35.5.4 Power analysis: at this signal strength, what's P(reject null) for informative features?
- [ ] 35.5.5 Is ptest_mc also bad here? (0.068) — fundamental limitation of hypothesis testing approach

### 35.6 Nonlinear
- [ ] 35.6.1 RDC should shine here (captures nonlinear dependencies) — does it?
- [ ] 35.6.2 MC vs RDC gap on nonlinear: is RDC's nonlinear detection worth the cost?
- [ ] 35.6.3 Tree-based methods should naturally capture nonlinear splits — do they outperform filter methods?

### 35.7 Redundant
- [ ] 35.7.1 precision_ir (informative + redundant) vs precision (informative only): who recovers the full signal group?
- [ ] 35.7.2 Do methods spread selections across redundant copies, or concentrate on originals?
- [ ] 35.7.3 Feature muting effect: does muting remove redundant copies after the first is found?

### 35.8 Regression-specific types
- [ ] 35.8.1 Friedman function: a known nonlinear target — who recovers the 5 informative features?
- [ ] 35.8.2 Heteroscedastic: variance depends on features — does this confuse the permutation test?
- [ ] 35.8.3 High-dim regression (p=500, k=5): the hardest setting — who survives?

---

## 36. Unexploited Metrics (roc_auc, auc, f1 binary)

All three columns are fully populated but never analyzed.

- [ ] 36.1 ROC-AUC rankings: do AUC-based rankings favor different methods than accuracy?
- [ ] 36.2 AUC vs balanced_accuracy rank correlation: how different are they?
- [ ] 36.3 F1 binary vs F1 macro: do they disagree on multiclass datasets?
- [ ] 36.4 AUC CD diagrams: separate Friedman/Nemenyi using AUC
- [ ] 36.5 On which datasets does metric choice change the winner? (metric-sensitive datasets)
- [ ] 36.6 CIF rank under every metric: does CIF look better/worse under AUC than accuracy?

---

## 37. Multi-Config Method Analysis (XGB 5 configs, LGBM 2, r_cforest 4)

### 37.1 XGBoost importance type comparison
- [ ] 37.1.1 5 XGB configs use different importance_type (gain, weight, cover, total_gain, total_cover)
- [ ] 37.1.2 Which importance type produces the best feature ranking? (downstream accuracy)
- [ ] 37.1.3 Which importance type agrees most with ground truth? (synthetic precision)
- [ ] 37.1.4 XGB config sensitivity: spread across 5 configs per dataset (like CIF 6.3)
- [ ] 37.1.5 Best XGB config vs best CIF config: head-to-head

### 37.2 LightGBM importance type comparison
- [ ] 37.2.1 2 LGBM configs: which importance type is better?
- [ ] 37.2.2 LGBM config sensitivity per dataset

### 37.3 r_cforest configuration comparison
- [ ] 37.3.1 4 r_cforest configs: testtype × replace — which combo is best?
- [ ] 37.3.2 Bonferroni vs MonteCarlo for r_cforest: same as r_ctree pattern?
- [ ] 37.3.3 replace=TRUE vs FALSE: does subsampling help R's cforest?
- [ ] 37.3.4 Best r_cforest config vs worst: how sensitive is it?

### 37.4 Fair comparison: best-config-per-method
- [ ] 37.4.1 For EVERY method that has multiple configs, select the best one
- [ ] 37.4.2 Re-run Friedman/Nemenyi with only best configs — fairer comparison
- [ ] 37.4.3 Oracle comparison: best config per method per dataset — ceiling analysis

---

## 38. Continuous DGP Parameter Correlations (Synthetic)

We have continuous parameters from the data generating process. Correlate
method performance against each:

- [ ] 38.1 Performance vs class_sep (0.1, 0.5, 1.0, 2.0): who benefits most from stronger signal?
- [ ] 38.2 Performance vs toeplitz_rho (0.0, 0.95): correlation strength effect
- [ ] 38.3 Performance vs flip_y (0.0, 0.1): label noise effect
- [ ] 38.4 Performance vs n_correlated_noise (0, 20): confounder count effect
- [ ] 38.5 Performance vs n_high_cardinality_noise (0, 50): bias feature count effect
- [ ] 38.6 Performance vs n_redundant (0, 20): redundant feature count effect
- [ ] 38.7 Interaction: class_sep × method — who's most sensitive to signal strength?
- [ ] 38.8 Regression: performance vs noise level (1.0, 10.0, 50.0)
- [ ] 38.9 Scatter matrix: all DGP params vs precision@10, colored by method

---

## 39. Individual Feature-Level Analysis (Synthetic Ground Truth)

We know WHICH features are informative. Analyze at the individual feature level.

- [ ] 39.1 Per-informative-feature rank distribution: boxplot of rank assigned to feature j across seeds/folds
- [ ] 39.2 "Easy" vs "hard" informative features: which features are consistently found vs consistently missed?
- [ ] 39.3 Is there a specific informative feature that ALL methods miss? Why?
- [ ] 39.4 Feature difficulty vs method: some features only CIF finds, some only RF finds
- [ ] 39.5 Per-noise-feature rank: do some noise features consistently fool methods?
- [ ] 39.6 "Trojan" noise features: noise features that correlate with target by chance — who falls for them?
- [ ] 39.7 Rank assigned to the SINGLE most important feature by each method (per dataset)
- [ ] 39.8 Rank assigned to the LEAST important informative feature — the hardest signal to detect

---

## 40. Downstream Model Tells Us About Ranking Quality

The downstream model choice reveals properties of the selected feature set.

- [ ] 40.1 If LR works well with method X's features but KNN doesn't → features are linearly separable but redundant
- [ ] 40.2 If KNN works well but LR doesn't → features capture nonlinear structure
- [ ] 40.3 Linear vs nonlinear downstream gap by method: (SVM accuracy - LR accuracy) per method
- [ ] 40.4 Which methods produce "linear-friendly" feature sets? (LR outperforms KNN)
- [ ] 40.5 Which methods produce "interaction-dependent" feature sets? (KNN outperforms LR)
- [ ] 40.6 CIF linear vs nonlinear gap compared to RF, ET: does CIF capture more nonlinear structure?
- [ ] 40.7 Regression: Ridge vs SVR gap by method — who selects features with nonlinear signal?

---

## 41. Feature Selection Benefit Analysis

Does feature selection actually help vs using all features?

- [ ] 41.1 Best-k accuracy minus k=p accuracy (all features) per method per dataset
- [ ] 41.2 Which datasets benefit MOST from feature selection? (largest improvement over all-features)
- [ ] 41.3 Which datasets benefit LEAST? (feature selection hurts — should just use everything)
- [ ] 41.4 Per-method: how often does the best k beat k=p? (% of datasets where selection helps)
- [ ] 41.5 Optimal k as fraction of p: is there a universal rule (e.g., "select p/10")?
- [ ] 41.6 CIF's optimal k vs RF's optimal k: do different methods peak at different k?
- [ ] 41.7 Feature selection benefit vs dimensionality: does selection help more on high-dim?
- [ ] 41.8 Feature selection benefit vs n: does selection help more on small samples?

---

## 42. Visualization Techniques Not Yet Considered

- [ ] 42.1 Bump charts: method rank evolution across k values (k on x-axis, rank on y-axis, lines cross)
- [ ] 42.2 Parallel coordinates: each axis is a dataset, lines show method performance
- [ ] 42.3 Radar/spider charts: per method, axes are different evaluation criteria
- [ ] 42.4 Small multiples: one accuracy-vs-k panel per dataset (faceted by dataset)
- [ ] 42.5 Ridgeline plots: distribution of accuracy per method (overlapping density plots)
- [ ] 42.6 Forest plot: effect sizes with CIs for CIF vs each competitor (meta-analysis style)
- [ ] 42.7 Upset plot: feature set overlap between methods' top-k selections
- [ ] 42.8 Waterfall chart: contribution of each CIF optimization to overall performance
- [ ] 42.9 Sankey diagram: feature flow from ranking to downstream accuracy contribution
- [ ] 42.10 Tile plot / calendar heatmap: method × dataset × k 3D view of performance

---

## 43. Comparison Fairness and Baselines

- [ ] 43.1 Random feature selection baseline: accuracy at each k with random features (expected value)
- [ ] 43.2 How much does each method beat random selection? (lift over random)
- [ ] 43.3 "Always select first k" baseline: accuracy using features 0..k-1 (no ranking)
- [ ] 43.4 Hyperparameter fairness: XGB has 5 configs, RF has 1 — is this fair?
- [ ] 43.5 What if we gave CIF 5 configs too? (e.g., vary alpha, n_resamples)
- [ ] 43.6 Per-method hyperparameter sensitivity: how much does the best config beat the worst?
- [ ] 43.7 Normalized performance: (method accuracy - random baseline) / (oracle - random baseline)

---

## 44. Offline Experiments Needed (NOT from existing data)

Beyond the timing ablation (Section 22), these require new local runs:

### 44.1 α sensitivity experiment
- [ ] 44.1.1 Run CIT/CIF at α = {0.001, 0.01, 0.05, 0.10, 0.20, 0.50} on synthetic data
- [ ] 44.1.2 Precision/recall/F1 at each α
- [ ] 44.1.3 Does permissive α fix weak signal failure?
- [ ] 44.1.4 Does strict α improve precision on toeplitz?
- [ ] 44.1.5 Optimal α per dataset type

### 44.2 n_resamples experiment
- [ ] 44.2.1 Run at B = {49, 99, 199, 499, 999, 4999} on synthetic data
- [ ] 44.2.2 P-value precision vs B
- [ ] 44.2.3 Ranking quality vs B — at what B do rankings stabilize?
- [ ] 44.2.4 Runtime vs B (linear scaling?)

### 44.3 n_estimators experiment (CIF)
- [ ] 44.3.1 Run CIF at n_estimators = {10, 25, 50, 100, 200, 500}
- [ ] 44.3.2 Ranking quality vs n_estimators — when do rankings saturate?
- [ ] 44.3.3 Stability vs n_estimators — more trees = more stable?
- [ ] 44.3.4 Runtime vs n_estimators

### 44.4 Bootstrap effect
- [ ] 44.4.1 CIF with bootstrap=True vs bootstrap=False
- [ ] 44.4.2 Does bootstrap help or hurt on biased datasets? (Strobl 2007 claim)
- [ ] 44.4.3 Subsampling rate (0.5, 0.632, 0.8, 1.0): effect on ranking quality

### 44.5 Multi-selector experiment
- [ ] 44.5.1 CIF with selector=['mc'] vs selector=['rdc'] vs selector=['mc','rdc']
- [ ] 44.5.2 Does the max-T combination beat the individual selectors?
- [ ] 44.5.3 On which dataset types does multi-selector help?

### 44.6 Noise injection experiment
- [ ] 44.6.1 Take a real dataset, progressively add noise features (p_noise = 10, 50, 100, 500)
- [ ] 44.6.2 Track each method's accuracy degradation curve
- [ ] 44.6.3 Who is most robust to noise injection?
- [ ] 44.6.4 At what p_noise / p_real ratio does each method fall below random?

### 44.7 Sample size subsampling experiment
- [ ] 44.7.1 Take a large real dataset, subsample to n = {50, 100, 200, 500, 1000, 2000}
- [ ] 44.7.2 Feature selection quality vs n curve per method
- [ ] 44.7.3 At what n does CIF become competitive?
- [ ] 44.7.4 Small-n regime: does the permutation test help or hurt?

### 44.8 Feature removal (ablation) experiment
- [ ] 44.8.1 Remove top-1, top-2, ..., top-k features from the selected set
- [ ] 44.8.2 How quickly does accuracy drop? (steeper = ranking captured real signal)
- [ ] 44.8.3 Compare drop curves across methods — who captures non-redundant signal?
- [ ] 44.8.4 This directly tests ranking QUALITY, not just top-k composition

---

## 45. Reproducibility and Reporting

- [ ] 45.1 Verify all results are from artifact_version v2 (no v1 contamination)
- [ ] 45.2 Document exact software versions (from library_versions column)
- [ ] 45.3 Git SHA consistency: verify all results from same codebase version
- [ ] 45.4 Data availability: document how to reproduce from raw data
- [ ] 45.5 Computational cost summary: total EC2 hours, cost, instance types used
- [ ] 45.6 Limitations section content: what we can and cannot conclude

---

## Notes

- All analyses use data from `paper/results/` (aggregated, grid-validated)
- Synthetic analyses use ground truth from parquet metadata
- Method grouping: use `method_base` for aggregating configs, `method_id` for config-level analysis
- For CIF best-config analysis, use CIF-MC (cif__6556bbc3e830b25a for CLF)
- k values in evaluation: 5, 10, 25, 50, 100 (standard), plus dataset-specific k=p
- Dataset-specific k=p values should be handled carefully (not all methods have them)
