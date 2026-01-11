# Paper Experiments

This directory contains scripts and data for reproducing the experiments in the
citrees paper.

## Directory Structure

```
paper/
├── data/                    # Datasets (parquet format)
│   ├── clf_*.parquet       # Classification datasets
│   └── reg_*.parquet       # Regression datasets
├── scripts/
│   ├── nested_cv_server.py # FastAPI server for experiment configs
│   ├── nested_cv_worker.py # Worker that runs nested CV experiments
│   ├── configs.py          # Experiment config dataclasses
│   ├── analysis.py         # Statistical tests and visualizations
│   ├── ec2_launch.py       # EC2 server/worker launcher
│   └── generate_figures.py # Paper figure generation
└── results/
    └── *.parquet           # Experiment outputs
```

## Quick Start (Local Validation)

```bash
# Install dependencies
uv sync

# Test nested CV worker locally
LOCAL_TEST=1 uv run python paper/scripts/nested_cv_worker.py

# Or use Docker
docker-compose run --rm sanity
```

---

## Experiment Design: Nested Cross-Validation

We use **proper nested CV** where feature selection happens INSIDE each fold:

```
For each outer fold:
  1. Split data into train/test
  2. Run feature selection on TRAIN ONLY
  3. Evaluate selected features with downstream models on TEST
  4. For embedding methods: also capture model's own predictions
```

This avoids data leakage that plagues traditional feature selection benchmarks.

---

## Methods (29 Total)

### Classification (16 methods)

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
| `boruta` | wrapper | Boruta feature selection |
| `pi` | wrapper | sklearn permutation importance |
| `shap` | wrapper | SHAP importance (RF base) |

**Downstream models:** LR, SVM, kNN

### Regression (13 methods)

| Method | Type | Description |
|--------|------|-------------|
| `pc` | filter | Pearson correlation |
| `dc` | filter | Distance correlation |
| `rdc` | filter | Randomized dependence coefficient |
| `ptest_pc` | permutation | PC with permutation test |
| `ptest_dc` | permutation | DC with permutation test |
| `ptest_rdc` | permutation | RDC with permutation test |
| `cit` | embedding | Conditional Inference Tree |
| `cif` | embedding | Conditional Inference Forest |
| `rf` | embedding | Random Forest |
| `et` | embedding | Extra Trees |
| `xgb` | embedding | XGBoost |
| `lgbm` | embedding | LightGBM |
| `boruta` | wrapper | Boruta feature selection |
| `pi` | wrapper | sklearn permutation importance |

**Downstream models:** Ridge, SVR, kNN

---

## Distributed Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     DISTRIBUTED ARCHITECTURE                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   ┌─────────────────┐         ┌─────────────────────────────────┐   │
│   │ nested_cv_      │  HTTP   │         EC2 Fleet               │   │
│   │ server.py       │◄────────│   ┌─────────┐  ┌─────────┐     │   │
│   │ (FastAPI)       │         │   │ Worker  │  │ Worker  │ ... │   │
│   └────────┬────────┘         │   │   1     │  │   2     │     │   │
│            │                  │   └────┬────┘  └────┬────┘     │   │
│            │                  │        │            │           │   │
│            ▼                  └────────┼────────────┼───────────┘   │
│   ┌─────────────────┐                  │            │               │
│   │   DynamoDB      │◄─────────────────┴────────────┘               │
│   │   (results)     │              Results stored                   │
│   └─────────────────┘                                               │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Running Experiments

### 1. Create DynamoDB Tables

```bash
# Classification tables
AWS_PROFILE=personal aws dynamodb create-table \
    --table-name ClfNestedCV \
    --attribute-definitions AttributeName=config_idx,AttributeType=N \
    --key-schema AttributeName=config_idx,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST --region us-east-1

AWS_PROFILE=personal aws dynamodb create-table \
    --table-name ClfNestedCVFail \
    --attribute-definitions AttributeName=config_idx,AttributeType=N \
    --key-schema AttributeName=config_idx,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST --region us-east-1

# Regression tables
AWS_PROFILE=personal aws dynamodb create-table \
    --table-name RegNestedCV \
    --attribute-definitions AttributeName=config_idx,AttributeType=N \
    --key-schema AttributeName=config_idx,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST --region us-east-1

AWS_PROFILE=personal aws dynamodb create-table \
    --table-name RegNestedCVFail \
    --attribute-definitions AttributeName=config_idx,AttributeType=N \
    --key-schema AttributeName=config_idx,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST --region us-east-1
```

### 2. Start Server

```bash
# Classification experiments
TABLE_NAME=Clf AWS_PROFILE=personal \
    uv run uvicorn paper.scripts.nested_cv_server:app --host 0.0.0.0 --port 8000

# Check status
curl http://localhost:8000/status/
# {"n_configs_remaining": 14400, "hosts": {}}
```

### 3. Run Workers

```bash
# Local worker (for validation)
URL=http://localhost:8000 TABLE_NAME=Clf AWS_PROFILE=personal N_JOBS=1 \
    uv run python paper/scripts/nested_cv_worker.py

# EC2 workers (production)
# See ec2_launch.py for automated deployment
```

---

## Result Structure

Each experiment stores:

```python
{
    "config_idx": 123,
    "task_type": "classification",
    "dataset": "iris",
    "method": "rf",
    "n_features_list": [1, 2, ..., 20],
    "results": {
        "folds": [
            {
                "fold": 0,
                "selected_features": [3, 1, 4, 0, 2, ...],
                "selection_time": 0.5,

                # Downstream metrics (selected features evaluated with LR/SVM/kNN)
                "downstream_metrics": {
                    5: {
                        "lr": {"acc": 0.9, "f1": 0.9, "auc": 0.95},
                        "svm": {"acc": 0.88, "f1": 0.87, "auc": 0.92},
                        "knn": {"acc": 0.85, "f1": 0.84, "auc": 0.90}
                    },
                    10: {...},
                },

                # Embedding model metrics (for rf/xgb/cit/etc only)
                "embedding_metrics": {"acc": 0.92, "f1": 0.91, "auc": 0.96}
            },
            ...
        ],
        "aggregated": {...}  # Mean/std across folds
    }
}
```

---

## Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `TABLE_NAME` | DynamoDB table prefix | Required | `Clf` or `Reg` |
| `URL` | FastAPI server URL | Required | `http://10.0.0.1:8000` |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` | `us-east-1` |
| `AWS_PROFILE` | AWS credentials profile | None | `personal` |
| `N_JOBS` | Parallel jobs | `-1` | `4` |
| `LOCAL_TEST` | Run local test mode | None | `1` |

---

## Docker Usage

```bash
# Sanity check
docker-compose run --rm sanity

# Run pytest
docker-compose run --rm test

# Interactive shell
docker-compose run --rm shell

# Local validation
docker-compose run --rm validate
```

---

## Config Calculation

**Classification:** 16 methods × N datasets × 30 seeds = N × 480 configs
**Regression:** 13 methods × N datasets × 30 seeds = N × 390 configs

---

## Analysis

After experiments complete:

```bash
# Run statistical analysis
uv run python paper/scripts/analysis.py

# Generate figures
uv run python paper/scripts/generate_figures.py
```

### Generated Outputs

| Output | Description |
|--------|-------------|
| Critical difference diagrams | Nemenyi post-hoc test visualization |
| Box plots | Method comparison distributions |
| Heatmaps | Performance by experimental factors |
| Friedman test tables | Overall significance tests |
| Rankings | Method rankings with confidence intervals |
