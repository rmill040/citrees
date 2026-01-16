# Paper Experiments

Scripts and data for reproducing the citrees paper experiments.

## Directory Structure

```
paper/
├── data/                              # Datasets (parquet format)
│   ├── clf_*.parquet                 # Classification datasets
│   ├── clf_synthetic_*.parquet       # Synthetic datasets (with ground truth)
│   └── reg_*.parquet                 # Regression datasets
├── scripts/
│   ├── experiments/                  # Core experiment runners
│   │   ├── ray_feature_selection.py  # Stage 1: Distributed feature selection
│   │   ├── ray_eval.py               # Stage 2: Distributed downstream eval
│   │   └── check_progress.py         # Progress monitoring via S3
│   ├── analysis/                     # Analysis and visualization
│   │   ├── analysis.py               # Statistical tests
│   │   ├── synthetic_analysis.py     # Precision/recall@k analysis
│   │   └── generate_figures.py       # Paper figure generation
│   ├── data_generation/              # Dataset generation
│   │   └── generate_synthetic_datasets.py
│   ├── utils/                        # Shared utilities
│   │   ├── config.py                 # Hyperparameter grids
│   │   ├── constants.py              # Method lists, defaults
│   │   ├── eval_models.py            # Downstream model definitions
│   │   └── metrics.py                # Evaluation metrics
│   └── infra/                        # Infrastructure
│       ├── config.py                 # Configuration dataclasses
│       ├── config.yaml               # Experiment settings
│       └── ray/
│           ├── cluster.yaml          # Ray cluster config
│           └── setup_cluster.py      # AMI update helper
└── results/                          # Local cache (S3 is source of truth)
```

## Quick Start

### Local Testing

```bash
# Install dependencies
uv sync --group paper

# Test feature selection locally
uv run python -c "
from paper.scripts.experiments.ray_feature_selection import run_selection
import numpy as np
X = np.random.randn(100, 20)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
results = run_selection(X, y, 'mc', 'classification', seed=0)
print(f'Got {len(results)} fold results')
print(f'Top 5 features: {results[0][\"feature_ranking\"][:5]}')
"
```

### Distributed (AWS)

See [infrastructure.md](infrastructure.md) for full AWS setup.

```bash
# Generate config first
AWS_PROFILE=personal uv run python paper/scripts/infra/ray/setup_cluster.py --generate

# Start Ray cluster
AWS_PROFILE=personal uv run ray up paper/scripts/infra/ray/cluster.yaml --yes

# Run feature selection
AWS_PROFILE=personal uv run ray submit paper/scripts/infra/ray/cluster.yaml \
    paper/scripts/experiments/ray_feature_selection.py

# Run evaluation
AWS_PROFILE=personal uv run ray submit paper/scripts/infra/ray/cluster.yaml \
    paper/scripts/experiments/ray_eval.py

# Tear down
AWS_PROFILE=personal uv run ray down paper/scripts/infra/ray/cluster.yaml --yes
```

## Two-Stage Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Feature Selection (ray_feature_selection.py)                   │
│                                                                          │
│ Ray Workers ──→ S3 (rankings)                                           │
│                                                                          │
│ N configs = methods × datasets × seeds                                  │
│ Output: s3://bucket/rankings/{task}/{dataset}/{method}_seed{s}.parquet  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Downstream Evaluation (ray_eval.py)                            │
│                                                                          │
│ Ray Workers ──→ S3 (metrics)                                            │
│                                                                          │
│ Evaluates at k = [5, 10, 25, 50, 100, all]                             │
│ Output: s3://bucket/metrics/{task}/{dataset}/{method}_seed{s}.parquet   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Benefits:**
- Stage 1 (slow) runs independently from Stage 2 (fast)
- Can re-run Stage 2 with different downstream models
- Full resume via S3 file existence checks
- Spot instance fault tolerance via Ray

## Methods

### Classification (19 methods)

| Method | Type | Description |
|--------|------|-------------|
| `mc` | filter | Multiple correlation (ANOVA-based) |
| `mi` | filter | Mutual information |
| `rdc` | filter | Randomized dependence coefficient |
| `mrmr` | filter | Minimum Redundancy Maximum Relevance |
| `ptest_mc` | permutation | MC with permutation test |
| `ptest_mi` | permutation | MI with permutation test |
| `ptest_rdc` | permutation | RDC with permutation test |
| `cit` | embedding | Conditional Inference Tree |
| `cif` | embedding | Conditional Inference Forest |
| `rf` | embedding | Random Forest |
| `et` | embedding | Extra Trees |
| `xgb` | embedding | XGBoost |
| `lgbm` | embedding | LightGBM |
| `cat` | embedding | CatBoost |
| `boruta` | wrapper | Boruta feature selection |
| `pi` | wrapper | Permutation importance |
| `cpi` | wrapper | Conditional permutation importance |
| `shap` | wrapper | SHAP importance |
| `rfe` | wrapper | Recursive Feature Elimination |

**Downstream models:** LogisticRegression, SVM, kNN

### Regression (19 methods)

| Method | Type | Description |
|--------|------|-------------|
| `pc` | filter | Pearson correlation |
| `dc` | filter | Distance correlation |
| `rdc` | filter | Randomized dependence coefficient |
| `mrmr` | filter | Minimum Redundancy Maximum Relevance |
| `ptest_pc` | permutation | PC with permutation test |
| `ptest_dc` | permutation | DC with permutation test |
| `ptest_rdc` | permutation | RDC with permutation test |
| `cit` | embedding | Conditional Inference Tree |
| `cif` | embedding | Conditional Inference Forest |
| `rf` | embedding | Random Forest |
| `et` | embedding | Extra Trees |
| `xgb` | embedding | XGBoost |
| `lgbm` | embedding | LightGBM |
| `cat` | embedding | CatBoost |
| `boruta` | wrapper | Boruta feature selection |
| `pi` | wrapper | Permutation importance |
| `cpi` | wrapper | Conditional permutation importance |
| `shap` | wrapper | SHAP importance |
| `rfe` | wrapper | Recursive Feature Elimination |

**Downstream models:** Ridge, SVR, kNN

## Synthetic Datasets

Generate datasets with known ground truth for precision/recall@k:

```bash
uv run python paper/scripts/data_generation/generate_synthetic_datasets.py
```

**Dataset types (169 total):**

| Type | Count | Description |
|------|-------|-------------|
| STANDARD | 108 | Varying features, informative, samples, separation |
| BIAS | 9 | High-cardinality noise (selection bias test) |
| NONLINEAR | 6 | Friedman #1 (tests nonlinear methods) |
| CORRELATED | 6 | Correlated feature blocks |
| REDUNDANT | 3 | Linear combinations of informative features |
| CORR_NOISE | 4 | Correlated noise features (confounders) |
| TOEPLITZ | 24 | Toeplitz covariance structure |
| WEAK_SIGNAL | 9 | Low class separation + label noise |

Ground truth stored in parquet schema metadata.

### Flat File Naming (Synthetic)

All synthetic datasets are saved as flat files under `paper/data/` with the prefix
`clf_synthetic_`:

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

Metadata fields stored in parquet schema include `config`, `informative_indices`,
`redundant_indices`, `correlated_indices`, `correlated_noise_indices`, and
`noise_indices`.

## Check Progress

```bash
# Stage 1 (feature selection)
AWS_PROFILE=personal uv run python paper/scripts/experiments/check_progress.py --stage rankings

# Stage 2 (evaluation)
AWS_PROFILE=personal uv run python paper/scripts/experiments/check_progress.py --stage metrics

# By method
AWS_PROFILE=personal uv run python paper/scripts/experiments/check_progress.py --stage rankings --by-method

# By dataset
AWS_PROFILE=personal uv run python paper/scripts/experiments/check_progress.py --stage rankings --by-dataset
```

## Result Structure

### Stage 1 Output (rankings)

```
s3://bucket/rankings/{task}/{dataset}/{method}_seed{seed}.parquet

Columns:
- fold_idx: int
- train_indices: list[int]
- test_indices: list[int]
- feature_ranking: list[int]      # Full ranking [best → worst]
- embedding_train_preds: list     # For embedding methods
- embedding_test_preds: list
- embedding_train_proba: list     # For classifiers with predict_proba
- embedding_test_proba: list
```

### Stage 2 Output (metrics)

```
s3://bucket/metrics/{task}/{dataset}/{method}_seed{seed}.parquet

Columns:
- fold_idx: int
- k: int                          # Number of features used
- downstream_model: str           # lr, svm, knn / ridge, svr, knn
- accuracy, f1, roc_auc: float    # Classification metrics
- r2, rmse, mae: float            # Regression metrics
```

## Analysis

After experiments complete:

```bash
# Download from S3
aws s3 sync s3://bucket/rankings/ paper/results/rankings/
aws s3 sync s3://bucket/metrics/ paper/results/metrics/

# Synthetic analysis (precision/recall@k)
uv run python paper/scripts/analysis/synthetic_analysis.py \
    --results-dir paper/results/rankings/classification \
    --data-dir paper/data \
    --output paper/results/synthetic_analysis.parquet

# Statistical analysis
uv run python paper/scripts/analysis/analysis.py

# Generate figures
uv run python paper/scripts/analysis/generate_figures.py
```

## Config Calculation

**Classification:** 19 methods × N datasets × 10 seeds
**Regression:** 19 methods × N datasets × 10 seeds

Example with 169 synthetic + 7 real datasets = 176 datasets:
- Classification: 19 × 176 × 10 = **33,440 configs**
- Regression: 19 × 176 × 10 = **33,440 configs**
