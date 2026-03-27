# Ablation Experiments

Controlled experiments that isolate the effect of individual CIF design choices
on feature selection quality. These complement the main benchmark (which compares
CIF against external methods on real datasets) by answering *why* CIF works and
*when* it breaks.

## Architecture

```
paper/scripts/
├── experiments/                        # Ablation experiments (produce CSVs)
│   ├── _common.py                      # Shared: datasets, builders, eval, I/O
│   ├── alpha_sweep.py                  # Alpha threshold sensitivity
│   ├── bootstrap_vs_subsampling.py     # Bootstrap vs feature subsampling
│   ├── max_t_selector.py              # Max-T vs Bonferroni correction
│   ├── n_estimators_sweep.py          # Ensemble size effect
│   ├── noise_robustness.py            # Noise-feature tolerance
│   ├── optimization_ablation.py       # Full optimization ablation (largest)
│   ├── power_analysis.py              # Type I/II error of permutation test
│   ├── real_dataset_ablation.py       # Ablation on real-world datasets
│   ├── resamples_and_honesty.py       # Permutation budget + honesty
│   ├── sample_size_curves.py          # Minimum viable sample size
│   ├── scaling_curves.py              # Wall-clock time vs n, p
│   └── strictness_continuum.py        # Conservative → RF-like spectrum
│
├── analysis/                           # Downstream analysis (consume data → figures/tables)
│   ├── aggregate.py                    # S3 download + parquet aggregation
│   ├── analyze_synthetic.py            # Synthetic precision/recall
│   ├── figures_ablation.py             # Ablation figures from CSVs
│   ├── figures_benchmark.py            # Main benchmark figures/tables
│   ├── figures_cd.py                   # Critical difference diagrams
│   └── stats.py                        # Statistical tests + ranking tables
```

## Running experiments

```bash
# Run a single experiment (each produces one CSV)
uv run python -m paper.scripts.experiments.alpha_sweep
uv run python -m paper.scripts.experiments.noise_robustness
# etc.
```

## Configuration

All experiments use the same constants, hardcoded in `_common.py`:

| Constant | Value | Purpose |
|---|---|---|
| `N_SEEDS` | 5 | Independent random seeds per setting |
| `N_ESTIMATORS` | 100 | CIF/RF/ET trees |
| `N_JOBS` | -1 | Use all CPU cores |
| `RANDOM_STATE` | 1718 | Base seed (seeds = 1718..1722) |
| `K` | 10 | Top-k for precision/recall/F1 |

## Output

Each experiment saves a CSV to two locations:
- `paper/results/tables/<experiment_name>.csv` (repo)
- `../data/ablation/<experiment_name>.csv` (local data mirror)

All CSVs share a common schema prefix:

| Column | Type | Description |
|---|---|---|
| `experiment` | str | Experiment name (matches filename) |
| `variant` | str | Method/config being tested |
| `dataset_type` | str | Dataset name |
| `task` | str | `"clf"` or `"reg"` (when applicable) |
| `n_features` | int | Number of features |
| `n_samples` | int | Number of samples |

Followed by aggregated metrics (mean/std over seeds):

| Metric | Description |
|---|---|
| `precision_at_10_mean/std` | Fraction of top-10 that are truly informative |
| `recall_at_10_mean/std` | Fraction of informative features in top-10 |
| `f1_at_10_mean/std` | Harmonic mean of P@10 and R@10 |
| `spread_at_10_mean/std` | Max-min index spread in top-10 / (p-1) |
| `lr_ba_mean/std` | Downstream LR balanced accuracy (clf only) |
| `knn_ba_mean/std` | Downstream KNN balanced accuracy (clf only) |
| `ridge_r2_mean/std` | Downstream Ridge R² (reg only) |
| `knn_r2_mean/std` | Downstream KNN R² (reg only) |
| `elapsed_seconds_mean/std` | Wall-clock training time |
| `confounder_rate_at_10_mean/std` | Fraction of confounders in top-10 |
| `mean_depth_mean/std` | Average tree depth (structure experiments) |
| `mean_features_used_mean/std` | Average features with nonzero importance |

## Experiment details

### alpha_sweep

**Question:** How does `alpha_selector` affect ranking quality?

Tests 6 alpha levels (0.001, 0.01, 0.05, 0.10, 0.20, 0.50) on all 14
synthetic datasets (8 clf + 6 reg). Key finding: alpha=0.05 with Bonferroni
is too conservative for weak-signal datasets; alpha=0.20 without Bonferroni
is a better tradeoff.

### n_estimators_sweep

**Question:** How many trees are needed for stable rankings?

Tests 6 levels (1, 5, 10, 25, 50, 100) on challenging datasets. Shows
diminishing returns past ~25 trees for precision@10, but confounder rejection
keeps improving through 100.

### noise_robustness

**Question:** At what noise-feature ratio does CIF break vs RF/ET?

Adds 0-1000 pure noise features to three base datasets (easy/moderate/many_info).
CIF maintains precision longer than RF/ET due to hypothesis testing that can
reject uninformative features outright.

### sample_size_curves

**Question:** What is the minimum viable sample size for CIF?

Tests n = 50-2000 on three configs (easy/moderate/highdim). CIF needs ~200
samples to match RF on easy problems; on high-dimensional problems, both
methods struggle below n=500.

### bootstrap_vs_subsampling

**Question:** Can feature subsampling replace bootstrap for tree diversity?

Tests 8 combinations of bootstrap on/off × max_features (all/sqrt/log2/half)
on 6 clf datasets. Shows bootstrap and feature subsampling are complementary:
bootstrap improves stability, feature subsampling improves diversity.

### resamples_and_honesty

**Question:** Does more permutation precision or honesty help ranking quality?

Tests 5 B levels (49-999) and honesty on/off. Shows diminishing returns past
B=199 (the "minimum" default) and modest honesty effect on ranking quality.

### scaling_curves

**Question:** How does CIF wall-clock time scale vs RF/ET?

Measures time vs n (200-5000, p=100) and time vs p (20-1000, n=1000). Shows
CIF has ~3-5x overhead vs RF due to permutation testing, scaling roughly
linearly with both n and p.

### real_dataset_ablation

**Question:** Which CIF optimizations matter on real-world datasets?

Tests 6 CIF variants + 3 baselines on 7 clf and 2 reg datasets from sklearn
and OpenML. No ground truth — uses downstream accuracy (LR/KNN balanced
accuracy or Ridge/KNN R²).

### optimization_ablation

**Question:** Which individual optimization contributes most to CIF quality?

The largest experiment: 9 CIF variants (scanning, muting, adaptive stopping,
bootstrap, subsampling, combinations) + 3 baselines on all synthetic and real
datasets. Key finding: adaptive stopping saves ~95% of compute with no quality
loss; feature muting has the largest impact on precision.

### strictness_continuum

**Question:** What happens as CIF transitions from conservative to RF-like?

Tests 9 CIF configs along the strictness spectrum (strict default →
alpha=0.99 wide-open) + RF/ET baselines + optional R baselines (r_ctree,
r_cforest). Captures tree structure (depth, n_features_used) to show the
causal chain: alpha/Bonferroni → depth → features used → ranking quality.

### max_t_selector

**Question:** Does max-T (Westfall-Young) beat Bonferroni for multi-selector?

Tests MC alone (with/without Bonferroni), RDC alone, and max-T combining
MC+RDC. Shows max-T provides a principled middle ground: more power than
Bonferroni, better control than no correction.

### power_analysis

**Question:** Does adaptive early stopping inflate Type I error?

500 simulations each under null (no signal) and alternative (varying signal
strength) hypotheses. Tests adaptive vs fixed-B stopping across 5 B levels
and 3 alpha levels. Validates the theoretical guarantee from the paper.

## Shared infrastructure (`_common.py`)

### Synthetic datasets

14 factories (8 clf + 6 reg), each returning `(X, y, informative_indices, name)`:

| Dataset | n | p | k | Challenge |
|---|---|---|---|---|
| `clf_standard_easy` | 1000 | 100 | 10 | Well-separated, baseline |
| `clf_standard_hard` | 200 | 1000 | 5 | High-dimensional, few samples |
| `clf_weak_signal` | 1000 | 100 | 10 | Low separation + label noise |
| `clf_nonlinear` | 1000 | 100 | 5 | Binarized Friedman1 |
| `clf_toeplitz` | 1000 | 100 | 10 | Toeplitz correlation (ρ=0.95) |
| `clf_confounder` | 1000 | 120 | 10 | +20 correlated confounders |
| `clf_bias` | 1000 | 100 | 10 | +50 high-cardinality integer noise |
| `clf_redundant` | 1000 | 70 | 10 | +20 linear combos of informative |
| `reg_friedman` | 1000 | 100 | 5 | Friedman1 function |
| `reg_linear` | 1000 | 100 | 10 | Linear, moderate noise |
| `reg_highdim` | 200 | 500 | 5 | High-dimensional |
| `reg_toeplitz` | 1000 | 100 | 10 | Toeplitz correlation |
| `reg_weak_signal` | 1000 | 100 | 10 | Very high noise (σ=100) |
| `reg_confounder` | 1000 | 120 | 10 | +20 correlated confounders |

All datasets use `shuffle_columns()` to randomize informative feature positions,
preventing positional bias.

### Model builders

- `build_cif(task, seed, **overrides)` — CIF with default paper config
- `build_cit(task, seed, **overrides)` — single conditional inference tree
- `build_baseline(method, task, seed)` — RF, ET, or CIT

### Evaluation

- `fit_and_evaluate(X, y, info, model, seed, task)` — fit + ranking + all metrics
- `fit_and_evaluate_with_structure(...)` — also captures tree depth/features used
- `aggregate_seeds(seed_results, base_row)` — mean/std across seeds

### Shared variant definitions

Two reusable variant dictionaries for consistency across experiments:

- `OPTIMIZATION_VARIANTS` — 9 CIF configs (used by `optimization_ablation`)
- `REAL_ABLATION_VARIANTS` — 6 CIF configs (used by `real_dataset_ablation`)
- `BASELINES` — `["rf", "et", "cit"]`

## Relationship to the paper

| Paper section | Experiments used |
|---|---|
| Table: Optimization ablation | `optimization_ablation` |
| Figure: Alpha sensitivity | `alpha_sweep` |
| Figure: Noise robustness | `noise_robustness` |
| Figure: Sample size curves | `sample_size_curves` |
| Figure: Scaling curves | `scaling_curves` |
| Table: Strictness continuum | `strictness_continuum` |
| Table: Real dataset ablation | `real_dataset_ablation` |
| Appendix: Type I error | `power_analysis` |
| Appendix: Max-T comparison | `max_t_selector` |
| Appendix: Bootstrap analysis | `bootstrap_vs_subsampling` |
| Appendix: n_resamples/honesty | `resamples_and_honesty` |
| Appendix: n_estimators effect | `n_estimators_sweep` |

## Implementation

The experiment scripts in `experiments/` are the canonical implementation.
All ablation CSVs use semantic names (e.g., `alpha_sweep.csv`) with an
`experiment` column.
