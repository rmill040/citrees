# Paper Experiments

Scripts and data for reproducing the citrees paper experiments.

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
│   ├── experiments/                  # Core experiment runners
│   │   ├── run_pipeline.py           # Stage 1 + Stage 2 sequential runner
│   │   ├── ray_feature_selection.py  # Stage 1: Distributed feature selection
│   │   ├── ray_eval.py               # Stage 2: Distributed downstream eval
│   │   ├── smoke_run.py              # Small end-to-end smoke test
│   │   └── check_progress.py         # Progress monitoring via S3
│   ├── analysis/                     # Analysis and visualization
│   │   ├── stats.py                  # Statistical tests + tables
│   │   ├── synthetic_analysis.py     # Precision/recall@k analysis
│   │   └── generate_figures.py       # Paper figure generation
│   ├── data_generation/              # Dataset generation
│   │   └── generate_synthetic_datasets.py
│   ├── utils/                        # Shared utilities
│   │   ├── config.py                 # Hyperparameter grids
│   │   ├── constants.py              # Method lists, defaults
│   │   ├── eval_models.py            # Downstream model definitions
│   │   ├── experiment_configs.py     # Method variants + labeling
│   │   └── metrics.py                # Evaluation metrics
│   └── infra/                        # Infrastructure
│       ├── config.py                 # Configuration dataclasses
│       ├── config.yaml               # Experiment settings
│       ├── cli.py                    # Cluster + experiment CLI helpers
│       ├── compute.py                # Instance sizing helpers
│       ├── resources.py              # Resource calculators
│       └── ray/
│           ├── cluster.yaml          # Ray cluster config
│           └── setup_cluster.py      # Cluster config generator / AMI helper
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
# One-time setup:
# - creates IAM role + instance profile for Ray autoscaling + S3/ECR access
# - creates S3 bucket (citrees-{account_id})
# - creates ECR repo (citrees-{account_id})
# - builds + pushes Docker image to ECR
# - generates paper/scripts/infra/ray/cluster.yaml with your IP + provenance
AWS_PROFILE=personal uv run python paper/scripts/infra/ray/setup_cluster.py --setup

# If you already have the IAM role + Docker image, you can regenerate cluster.yaml only:
# AWS_PROFILE=personal uv run python paper/scripts/infra/ray/setup_cluster.py --generate

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
│ STAGE 1: Feature Selection (ray_feature_selection.py)                   │
│                                                                          │
│ Ray Workers ──→ S3 (rankings)                                           │
│                                                                          │
│ N configs = methods × datasets × seeds                                  │
│ Output: s3://bucket/rankings/{task_type}/{dataset}/{method_id}_seed{s}.parquet  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Downstream Evaluation (ray_eval.py)                            │
│                                                                          │
│ Ray Workers ──→ S3 (metrics)                                            │
│                                                                          │
│ Evaluates at k = [5, 10, 25, 50, 100, all]                             │
│ Output: s3://bucket/metrics/{task_type}/{dataset}/{method_id}_seed{s}.parquet   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Benefits:**

- Stage 1 (slow) runs independently from Stage 2 (fast)
- Can re-run Stage 2 with different downstream models
- Only-missing runs via S3 listings (submit only configs without artifacts)
- Spot instance fault tolerance via Ray

**Only-missing workflow (recommended):**

```bash
# Preview exact configs that will run (print full grid)
AWS_PROFILE=personal uv run python paper/scripts/experiments/run_pipeline.py \
    --stage all --only-missing --dry-run --dry-run-limit 100000

# Run only configs missing in S3 (rankings + metrics)
AWS_PROFILE=personal uv run python paper/scripts/experiments/run_pipeline.py \
    --stage all --only-missing
```

**Reruns:** delete the specific S3 objects you want to recompute, then re-run
with `--only-missing`.

## End-to-End Analysis Sequence (Ray → S3 → Local)

This is the **full** analysis flow for real‑data benchmarks. The steps below are
ordered and explicit.

### 0) Prereqs (once)

```bash
uv sync --group paper
export AWS_PROFILE=personal
export S3_BUCKET=your-bucket-name
# Optional but recommended for provenance on remote workers:
export GIT_SHA=$(git rev-parse HEAD)
```

### 1) Run Stage 1 (feature selection on Ray)

Option A (recommended, handles missing-only logic):

```bash
AWS_PROFILE=personal S3_BUCKET=your-bucket-name \
uv run python paper/scripts/experiments/run_pipeline.py \
    --stage stage1 --only-missing
```

Option B (direct Ray submit):

```bash
AWS_PROFILE=personal S3_BUCKET=your-bucket-name \
uv run ray submit paper/scripts/infra/ray/cluster.yaml \
    paper/scripts/experiments/ray_feature_selection.py
```

### 2) Run Stage 2 (downstream evaluation on Ray)

Stage 2 **requires Stage 1 rankings** in S3 for each (method, dataset, seed).

```bash
AWS_PROFILE=personal S3_BUCKET=your-bucket-name \
uv run python paper/scripts/experiments/run_pipeline.py \
    --stage stage2 --only-missing
```

### 3) Download + aggregate S3 artifacts to local parquet

This produces: `paper/results/clf_evaluation.parquet` and
`paper/results/reg_evaluation.parquet` plus the rankings parquets (if
requested).

```bash
S3_BUCKET=your-bucket-name \
uv run python paper/scripts/analysis/download_and_aggregate.py --stage all --task-type all
```

### 4) Run statistical analysis (tables/figures)

```bash
uv run python paper/scripts/analysis/stats.py
```

### 5) Synthetic‑only figures (separate pipeline)

`generate_figures.py` **does not** use Ray or S3; it generates synthetic figures
locally.

```bash
uv run python paper/scripts/analysis/generate_figures.py --profile paper
```

---

## Methods

### Classification (19 methods)

| Method      | Type        | Description                          |
| ----------- | ----------- | ------------------------------------ |
| `mc`        | filter      | Multiple correlation (ANOVA-based)   |
| `mi`        | filter      | Mutual information                   |
| `rdc`       | filter      | Randomized dependence coefficient    |
| `mrmr`      | filter      | Minimum Redundancy Maximum Relevance |
| `ptest_mc`  | permutation | MC with permutation test             |
| `ptest_mi`  | permutation | MI with permutation test             |
| `ptest_rdc` | permutation | RDC with permutation test            |
| `cit`       | embedding   | Conditional Inference Tree           |
| `cif`       | embedding   | Conditional Inference Forest         |
| `rf`        | embedding   | Random Forest                        |
| `et`        | embedding   | Extra Trees                          |
| `xgb`       | embedding   | XGBoost                              |
| `lgbm`      | embedding   | LightGBM                             |
| `cat`       | embedding   | CatBoost                             |
| `boruta`    | wrapper     | Boruta feature selection             |
| `pi`        | wrapper     | Permutation importance               |
| `cpi`       | wrapper     | Conditional permutation importance   |
| `shap`      | wrapper     | SHAP importance                      |
| `rfe`       | wrapper     | Recursive Feature Elimination        |

**Downstream models:** LogisticRegression, SVM, kNN

### Regression (19 methods)

| Method      | Type        | Description                          |
| ----------- | ----------- | ------------------------------------ |
| `pc`        | filter      | Pearson correlation                  |
| `dc`        | filter      | Distance correlation                 |
| `rdc`       | filter      | Randomized dependence coefficient    |
| `mrmr`      | filter      | Minimum Redundancy Maximum Relevance |
| `ptest_pc`  | permutation | PC with permutation test             |
| `ptest_dc`  | permutation | DC with permutation test             |
| `ptest_rdc` | permutation | RDC with permutation test            |
| `cit`       | embedding   | Conditional Inference Tree           |
| `cif`       | embedding   | Conditional Inference Forest         |
| `rf`        | embedding   | Random Forest                        |
| `et`        | embedding   | Extra Trees                          |
| `xgb`       | embedding   | XGBoost                              |
| `lgbm`      | embedding   | LightGBM                             |
| `cat`       | embedding   | CatBoost                             |
| `boruta`    | wrapper     | Boruta feature selection             |
| `pi`        | wrapper     | Permutation importance               |
| `cpi`       | wrapper     | Conditional permutation importance   |
| `shap`      | wrapper     | SHAP importance                      |
| `rfe`       | wrapper     | Recursive Feature Elimination        |

**Downstream models:** Ridge, SVR, kNN

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
s3://bucket/rankings/{task_type}/{dataset}/{method_id}_seed{seed}.parquet

Columns:
- fold_idx: int
- feature_ranking: list[int]      # Full ranking [best → worst]
- dataset, task_type, seed
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
s3://bucket/metrics/{task_type}/{dataset}/{method_id}_seed{seed}.parquet

Columns:
- fold_idx: int
- k: int                          # Number of features used
- downstream_model: str           # lr, svm, knn / ridge, svr, knn
- dataset, task_type, seed
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
S3_BUCKET=my-bucket uv run python paper/scripts/analysis/download_and_aggregate.py --task-type all

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

**Classification:** 19 methods × N datasets × 10 seeds **Regression:** 19
methods × N datasets × 10 seeds

Example with 169 synthetic + 7 real datasets = 176 datasets:

- Classification: 19 × 176 × 10 = **33,440 configs**
- Regression: 19 × 176 × 10 = **33,440 configs**
