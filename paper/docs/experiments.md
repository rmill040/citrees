# Paper Experiments

Scripts and data for reproducing the citrees paper experiments.

> **AWS Note:** Commands that access S3 or Ray clusters require AWS credentials.
> Set `AWS_PROFILE` if not using your default profile.

## Directory Structure

```
paper/
├── data/                              # Datasets (parquet format)
│   ├── classification/
│   │   ├── real/
│   │   │   └── clf_*.parquet
│   │   └── synthetic/
│   │       └── clf_synthetic_*.parquet
│   └── regression/
│       ├── real/
│       │   └── reg_*.parquet
│       └── synthetic/
│           └── reg_synthetic_*.parquet
├── scripts/
│   ├── adapters/                     # External system adapters
│   │   ├── data.py                   # Dataset loading and S3 caching
│   │   ├── runner.py                 # Ray runner abstraction
│   │   └── store.py                  # S3 artifact storage
│   ├── cli/                          # CLI commands
│   │   ├── app.py                    # Main CLI application
│   │   ├── cluster.py                # Ray cluster operations
│   │   └── ...                       # Other CLI modules
│   ├── config/                       # Configuration
│   │   ├── settings.py               # Config dataclasses
│   │   └── constants.py              # Static constants
│   ├── pipeline/                     # Core experiment pipeline
│   │   ├── stage1.py                 # Stage 1: Feature selection
│   │   ├── stage2.py                 # Stage 2: Downstream evaluation
│   │   ├── methods.py                # Method definitions
│   │   ├── grid.py                   # Experiment grid builder
│   │   └── types.py                  # Type definitions
│   ├── analysis/                     # Analysis and visualization
│   │   ├── stats.py                  # Statistical tests + tables
│   │   ├── synthetic_analysis.py     # Precision/recall@k analysis
│   │   └── generate_figures.py       # Paper figure generation
│   ├── data_generation/              # Dataset generation
│   │   └── generate_synthetic_datasets.py
│   ├── utils/                        # Shared utilities
│   │   ├── env.py                    # Environment helpers
│   │   └── metrics.py                # Evaluation metrics
│   └── infra/                        # Infrastructure
│       └── ray/
│           ├── cluster.yaml          # Ray cluster config
│           └── setup_cluster.py      # Cluster config generator
└── results/                          # Local cache (S3 is source of truth)
```

## Quick Start

### Local Testing

```bash
# Install dependencies
uv sync --group paper

# Test feature selection locally
uv run python -c "
from paper.scripts.pipeline.stage1 import filter_selector
import numpy as np
X = np.random.randn(100, 20)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
ranking = filter_selector(X, y, method='mc', task='classification', random_state=0)
print(f'Top 5 features: {ranking[:5]}')
"
```

### Distributed (AWS)

See [infrastructure.md](infrastructure.md) for full AWS setup.

```bash
# One-time setup (IAM + S3 + ECR + Docker + cluster.yaml)
citrees-exp infra setup

# Start Ray cluster
citrees-exp cluster up --yes

# Run experiments (skips existing results by default)
citrees-exp run classification

# Check progress
citrees-exp check

# Tear down
citrees-exp cluster down --yes
```

Note: The Ray cluster runs inside Docker on EC2. To allow containers to fetch
instance profile credentials via IMDSv2, the cluster config sets hop limit 2:

```yaml
MetadataOptions:
  HttpTokens: required
  HttpPutResponseHopLimit: 2
```

## Two-Stage Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Feature Selection (pipeline/stage1.py)                         │
│                                                                          │
│ Ray Workers ──→ S3 (rankings)                                           │
│                                                                          │
│ N configs = methods × datasets × seeds                                  │
│ Output: s3://bucket/rankings/{task}/{dataset}/{method_id}_seed{s}.parquet  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Downstream Evaluation (pipeline/stage2.py)                     │
│                                                                          │
│ Ray Workers ──→ S3 (metrics)                                            │
│                                                                          │
│ Evaluates at k = [5, 10, 25, 50, 100, all]                             │
│ Output: s3://bucket/metrics/{task}/{dataset}/{method_id}_seed{s}.parquet   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Benefits:**

- Stage 1 (slow) runs independently from Stage 2 (fast)
- Can re-run Stage 2 with different downstream models
- Skips existing results by default (submit only configs without artifacts)
- Spot instance fault tolerance via Ray

**Default workflow:**

```bash
# Preview what would run (skips existing by default)
citrees-exp run classification --dry-run

# Run only configs missing in S3 (default behavior)
citrees-exp run classification
```

**Reruns:** use `--force` to re-run everything, or delete specific S3 objects
then re-run.

## End-to-End Analysis Sequence (Ray → S3 → Local)

This is the **full** analysis flow for real‑data benchmarks.

### 1) Run experiments

```bash
citrees-exp run classification                   # Stage 1 + 2 (skips existing)
citrees-exp run classification --stage stage1   # Stage 1 only
citrees-exp run classification --stage stage2   # Stage 2 only
citrees-exp run classification --force          # Re-run everything
```

### 2) Download + aggregate S3 artifacts to local parquet

This produces: `paper/results/clf_evaluation.parquet` and
`paper/results/reg_evaluation.parquet`.

```bash
S3_BUCKET=your-bucket-name \
uv run python paper/scripts/analysis/download_and_aggregate.py --stage all --task all
```

### 3) Run statistical analysis (tables/figures)

```bash
uv run python paper/scripts/analysis/stats.py
```

### 4) Synthetic‑only figures (separate pipeline)

`generate_figures.py` **does not** use Ray or S3; it generates synthetic figures
locally.

```bash
uv run python paper/scripts/analysis/generate_figures.py --profile paper
```

---

## Methods

### Classification (21 methods)

| Method      | Type        | Description                               |
| ----------- | ----------- | ----------------------------------------- |
| `mc`        | filter      | Multiple correlation (ANOVA-based)        |
| `mi`        | filter      | Mutual information                        |
| `rdc`       | filter      | Randomized dependence coefficient         |
| `mrmr`      | filter      | Minimum Redundancy Maximum Relevance      |
| `ptest_mc`  | permutation | MC with permutation test                  |
| `ptest_mi`  | permutation | MI with permutation test                  |
| `ptest_rdc` | permutation | RDC with permutation test                 |
| `cit`       | embedding   | Conditional Inference Tree (citrees)      |
| `cif`       | embedding   | Conditional Inference Forest (citrees)    |
| `r_ctree`   | embedding   | R partykit ctree (Hothorn et al., 2006)   |
| `r_cforest` | embedding   | R partykit cforest (Hothorn et al., 2006) |
| `rf`        | embedding   | Random Forest                             |
| `et`        | embedding   | Extra Trees                               |
| `xgb`       | embedding   | XGBoost                                   |
| `lgbm`      | embedding   | LightGBM                                  |
| `cat`       | embedding   | CatBoost                                  |
| `boruta`    | wrapper     | Boruta feature selection                  |
| `pi`        | wrapper     | Permutation importance                    |
| `cpi`       | wrapper     | Conditional permutation importance        |
| `shap`      | wrapper     | SHAP importance                           |
| `rfe`       | wrapper     | Recursive Feature Elimination             |

**Downstream models:** LogisticRegression, SVM, kNN

### Regression (21 methods)

| Method      | Type        | Description                               |
| ----------- | ----------- | ----------------------------------------- |
| `pc`        | filter      | Pearson correlation                       |
| `dc`        | filter      | Distance correlation                      |
| `rdc`       | filter      | Randomized dependence coefficient         |
| `mrmr`      | filter      | Minimum Redundancy Maximum Relevance      |
| `ptest_pc`  | permutation | PC with permutation test                  |
| `ptest_dc`  | permutation | DC with permutation test                  |
| `ptest_rdc` | permutation | RDC with permutation test                 |
| `cit`       | embedding   | Conditional Inference Tree (citrees)      |
| `cif`       | embedding   | Conditional Inference Forest (citrees)    |
| `r_ctree`   | embedding   | R partykit ctree (Hothorn et al., 2006)   |
| `r_cforest` | embedding   | R partykit cforest (Hothorn et al., 2006) |
| `rf`        | embedding   | Random Forest                             |
| `et`        | embedding   | Extra Trees                               |
| `xgb`       | embedding   | XGBoost                                   |
| `lgbm`      | embedding   | LightGBM                                  |
| `cat`       | embedding   | CatBoost                                  |
| `boruta`    | wrapper     | Boruta feature selection                  |
| `pi`        | wrapper     | Permutation importance                    |
| `cpi`       | wrapper     | Conditional permutation importance        |
| `shap`      | wrapper     | SHAP importance                           |
| `rfe`       | wrapper     | Recursive Feature Elimination             |

**Downstream models:** Ridge, SVR, kNN

### R partykit Methods (r_ctree, r_cforest)

The R methods use `rpy2` to call the original ctree/cforest implementation from
Hothorn et al. (2006) via R's `partykit` package. This enables direct comparison
between citrees and the original R implementation.

**r_ctree parameters:**

| Parameter   | Values                             | Description                |
| ----------- | ---------------------------------- | -------------------------- |
| `teststat`  | quadratic, maximum                 | Test statistic type        |
| `testtype`  | Bonferroni, MonteCarlo, Univariate | P-value computation method |
| `alpha`     | 0.05, 0.01                         | Significance level         |
| `nresample` | 1000, 9999                         | Monte Carlo permutations   |
| `minsplit`  | 20, 10                             | Min samples to split       |
| `minbucket` | 7, 5                               | Min samples in leaf        |

**r_cforest parameters:**

| Parameter            | Values                             | Description                     |
| -------------------- | ---------------------------------- | ------------------------------- |
| `teststat`           | quadratic, maximum                 | Test statistic type             |
| `testtype`           | Bonferroni, MonteCarlo, Univariate | P-value computation method      |
| `mincriterion`       | 0.95, 0.99, 0                      | Split threshold (0 = no stop)   |
| `nresample`          | 1000, 9999                         | Monte Carlo permutations        |
| `ntree`              | 100                                | Number of trees                 |
| `mtry`               | sqrt, all                          | Features sampled per split      |
| `replace`            | False, True                        | Bootstrap vs subsampling        |
| `fraction`           | 0.632, 0.8                         | Subsample fraction              |
| `varimp_conditional` | False, True                        | Conditional variable importance |
| `varimp_nperm`       | 1, 5                               | Permutations for varimp         |

**Grid sizes after filtering:** r_ctree = 64 configs, r_cforest = 576 configs

## Synthetic Datasets

Generate datasets with known ground truth for precision/recall@k:

```bash
uv run python paper/scripts/data_generation/generate_synthetic_datasets.py
```

**Dataset types (169 total):**

| Type        | Count | Description                                        |
| ----------- | ----- | -------------------------------------------------- |
| STANDARD    | 108   | Varying features, informative, samples, separation |
| BIAS        | 9     | High-cardinality noise (selection bias test)       |
| NONLINEAR   | 6     | Friedman #1 (tests nonlinear methods)              |
| CORRELATED  | 6     | Correlated feature blocks                          |
| REDUNDANT   | 3     | Linear combinations of informative features        |
| CORR_NOISE  | 4     | Correlated noise features (confounders)            |
| TOEPLITZ    | 24    | Toeplitz covariance structure                      |
| WEAK_SIGNAL | 9     | Low class separation + label noise                 |

Ground truth stored in parquet schema metadata.

### Flat File Naming (Synthetic)

All synthetic **classification** datasets are saved under
`paper/data/classification/synthetic/` with the prefix `clf_synthetic_`:

```
clf_synthetic_{name}.parquet
```

Name patterns by dataset type:

```
synthetic_p{p}_k{k}_n{n}_sep{sep}
synthetic_bias_noise{n_noise}_levels{n_levels}
synthetic_nonlinear_p{p}_n{n}
synthetic_corr_blocks{n_corr}_r{rho}
synthetic_redundant{n_redundant}
synthetic_corr_noise_p{p}_k{k}_n{n}_noise{n_corr}_r{rho}
synthetic_toeplitz_p{p}_k{k}_n{n}_r{rho}
synthetic_weak_p{p}_k{k}_n{n}_sep{sep}_flip{flip}
```

Metadata fields stored in parquet schema include `config`,
`informative_indices`, `redundant_indices`, `correlated_indices`,
`correlated_noise_indices`, and `noise_indices`.

## Check Progress

```bash
citrees-exp check                     # Stage 1 (rankings)
citrees-exp check --stage metrics     # Stage 2 (metrics)
citrees-exp check --by-method         # Grouped by method
citrees-exp check --by-dataset        # Grouped by dataset
citrees-exp watch                     # Live dashboard
```

## Result Structure

### Stage 1 Output (rankings)

```
s3://bucket/rankings/{task}/{dataset}/{method_id}_seed{seed}.parquet

Columns:
- fold_idx: int
- feature_ranking: list[int]      # Full ranking [best → worst]
- dataset, task, seed
- method_id, method, method_base
- artifact_version
- n_samples, n_features
- selection_cpus
- elapsed_seconds
- created_at_utc
- git_sha

```

### Stage 2 Output (metrics)

```
s3://bucket/metrics/{task}/{dataset}/{method_id}_seed{seed}.parquet

Columns:
- fold_idx: int
- k: int                          # Number of features used
- downstream_model: str           # lr, svm, knn / ridge, svr, knn
- dataset, task, seed
- method_id, method, method_base
- artifact_version
- n_samples, n_features
- n_features_selected
- evaluation_cpus
- elapsed_seconds
- created_at_utc
- git_sha
- accuracy, f1, f1_macro, balanced_accuracy, roc_auc, auc: float  # Classification metrics
- r2, rmse, mae: float            # Regression metrics
```

## Aggregation Policy

Stage 2 generates results per **fold × k × downstream_model × seed × dataset**.
The current statistical pipeline in `paper/scripts/analysis/stats.py` aggregates
by **dataset × method** and **averages across all rows** present (folds, seeds,
k values, and downstream models). If you want a fixed `k` or a specific
downstream model, filter the evaluation parquet files before running `stats.py`
or adjust the aggregation logic.

**Granular outputs (per-model / per-k).** `stats.py` now also emits
per‑downstream‑model outputs (prefixes like `clf_lr_*` or `reg_ridge_*`) and
per‑model‑per‑k outputs (prefixes like `clf_lr_k10_*`), in addition to the
overall aggregated `clf_*` / `reg_*` summaries.

**Optional pre-filtering examples (before `stats.py`).**

```python
# Example: keep only k=10 and downstream_model="lr" for classification
import pandas as pd
df = pd.read_parquet("paper/results/clf_evaluation.parquet")
df = df[(df["k"] == 10) & (df["downstream_model"] == "lr")]
df.to_parquet("paper/results/clf_evaluation_k10_lr.parquet")
```

```python
# Example: keep only k=10 for regression (all downstream models)
import pandas as pd
df = pd.read_parquet("paper/results/reg_evaluation.parquet")
df = df[df["k"] == 10]
df.to_parquet("paper/results/reg_evaluation_k10.parquet")
```

Then point `stats.py` at the filtered parquet files.

## Analysis

After experiments complete:

```bash
# Download and aggregate from S3 (recommended)
S3_BUCKET=my-bucket uv run python paper/scripts/analysis/download_and_aggregate.py --task all

# Or manually download from S3
aws s3 sync s3://bucket/rankings/ paper/results/rankings/
aws s3 sync s3://bucket/metrics/ paper/results/metrics/

# Synthetic analysis (precision/recall@k)
uv run python paper/scripts/analysis/synthetic_analysis.py \
    --results-dir paper/results/rankings/classification \
    --data-dir paper/data/classification/synthetic \
    --output paper/results/synthetic_analysis.parquet

# Statistical analysis (includes runtime tables/figures)
uv run python paper/scripts/analysis/stats.py

# Generate figures
uv run python paper/scripts/analysis/generate_figures.py
```

`stats.py` emits overall aggregates plus per‑downstream‑model and
per‑model‑per‑k tables/figures in:

- `paper/results/tables/` (CSV/LaTeX)
- `paper/results/figures/` (PNG)

See prefixes like `clf_lr_*`, `clf_lr_k10_*`, `reg_ridge_*`.

**OOB scoring note (forests).** If you enable `oob_score=True` in the forest
models, OOB predictions are computed only for samples that are out-of-bag for at
least one tree; the reported OOB score uses those samples only. OOB scoring
requires bootstrap to be enabled.

## Config Calculation

**Classification:** 21 methods × N datasets × 10 seeds **Regression:** 21
methods × N datasets × 10 seeds

Example with 169 synthetic + 7 real datasets = 176 datasets:

- Classification: 21 × 176 × 10 = **36,960 configs**
- Regression: 21 × 176 × 10 = **36,960 configs**

Note: Each method has its own hyperparameter grid (see
`paper/scripts/pipeline/methods.py`). Total unique configurations including
hyperparameters:

- Classification: ~214,000 configs
- Regression: ~120,000 configs
