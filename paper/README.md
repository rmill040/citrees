# Paper Experiments

This directory contains scripts and data for reproducing the experiments in the
citrees paper.

## Directory Structure

```
paper/
├── data/                              # Datasets (parquet format)
│   ├── clf_*.parquet                 # Classification datasets
│   └── reg_*.parquet                 # Regression datasets
├── scripts/
│   ├── feature_selection_server.py   # Stage 1: FastAPI server
│   ├── feature_selection_worker.py   # Stage 1: Feature selection workers
│   ├── eval_server.py                # Stage 2: FastAPI server
│   ├── eval_worker.py                # Stage 2: Model evaluation workers
│   ├── analysis.py                   # Statistical tests and visualizations
│   ├── ec2_launch.py                 # EC2 server/worker launcher
│   └── generate_figures.py           # Paper figure generation
└── results/                          # Local cache (S3 is source of truth)
```

## Quick Start (Local Validation)

```bash
# Install dependencies
uv sync

# Test Stage 1 (feature selection) locally
LOCAL_TEST=1 S3_BUCKET=citrees-results AWS_PROFILE=personal \
    uv run python paper/scripts/feature_selection_worker.py

# Test Stage 2 (model evaluation) locally
LOCAL_TEST=1 S3_BUCKET=citrees-results AWS_PROFILE=personal \
    uv run python paper/scripts/eval_worker.py

# Or use Docker
docker-compose run --rm sanity
```

---

## Two-Stage Architecture

We split experiments into two independent stages for maximum parallelization:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Feature Selection (EC2 fleet)                                  │
│                                                                          │
│ Server ──→ Workers ──→ S3 (rankings) + DynamoDB (tracking)              │
│                                                                          │
│ 11,520 configs (16 methods × 24 datasets × 30 seeds)                    │
│ Output: s3://citrees-results/rankings/{task}/{dataset}/{method}_seed{s} │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Downstream Evaluation (EC2 fleet)                              │
│                                                                          │
│ Server ──→ Workers ──→ S3 (metrics) + DynamoDB (tracking)               │
│                                                                          │
│ Same 11,520 configs, each evaluates k=1..n_features                     │
│ Output: s3://citrees-results/metrics/{task}/{dataset}/{method}_seed{s}  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Benefits:**
- Stage 1 (slow feature selection) runs independently from Stage 2 (fast evaluation)
- Can re-run Stage 2 with different models without repeating Stage 1
- Evaluate at ALL k values (1 to n_features) for complete accuracy curves
- Full resume capability via DynamoDB tracking

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

## AWS Resources

### S3 Bucket

```bash
AWS_PROFILE=personal aws s3 mb s3://citrees-results --region us-east-1
```

### DynamoDB Tables

```bash
# Stage 1 tracking
for table in ClfFeatureSelection RegFeatureSelection; do
    AWS_PROFILE=personal aws dynamodb create-table \
        --table-name $table \
        --attribute-definitions AttributeName=config_idx,AttributeType=N \
        --key-schema AttributeName=config_idx,KeyType=HASH \
        --billing-mode PAY_PER_REQUEST --region us-east-1
done

# Stage 2 tracking
for table in ClfDownstreamEval RegDownstreamEval; do
    AWS_PROFILE=personal aws dynamodb create-table \
        --table-name $table \
        --attribute-definitions AttributeName=config_idx,AttributeType=N \
        --key-schema AttributeName=config_idx,KeyType=HASH \
        --billing-mode PAY_PER_REQUEST --region us-east-1
done
```

---

## Running Experiments

### Stage 1: Feature Selection

```bash
# Start server
S3_BUCKET=citrees-results TABLE_NAME=ClfFeatureSelection AWS_PROFILE=personal \
    uv run uvicorn paper.scripts.feature_selection_server:app --host 0.0.0.0 --port 8000

# Check status
curl http://localhost:8000/status/
# {"n_configs_remaining": 11520, "hosts": {}, ...}

# Start workers (can run many in parallel on EC2)
URL=http://localhost:8000 S3_BUCKET=citrees-results TABLE_NAME=ClfFeatureSelection \
    AWS_PROFILE=personal uv run python paper/scripts/feature_selection_worker.py
```

### Stage 2: Downstream Evaluation

```bash
# Start server (only serves configs with Stage 1 rankings in S3)
S3_BUCKET=citrees-results TABLE_NAME=ClfDownstreamEval AWS_PROFILE=personal \
    uv run uvicorn paper.scripts.eval_server:app --host 0.0.0.0 --port 8000

# Check status
curl http://localhost:8000/status/

# Start workers
URL=http://localhost:8000 S3_BUCKET=citrees-results TABLE_NAME=ClfDownstreamEval \
    AWS_PROFILE=personal uv run python paper/scripts/eval_worker.py
```

---

## Result Structure

### Stage 1 Output (S3 rankings)

```
s3://citrees-results/rankings/{task}/{dataset}/{method}_seed{seed}.parquet

Columns:
- fold_idx: int
- train_indices: list[int]
- test_indices: list[int]
- feature_ranking: list[int]      # Full ranking [best → worst]
- selection_time_seconds: float
- embedding_train_preds: list     # For embedding methods
- embedding_test_preds: list      # For embedding methods
```

### Stage 2 Output (S3 metrics)

```
s3://citrees-results/metrics/{task}/{dataset}/{method}_seed{seed}.parquet

Columns:
- fold_idx: int
- n_features: int                 # k value (1 to total_features)
- lr_acc, lr_f1, lr_roc_auc, lr_pr_auc: float      # Classification
- svm_acc, svm_f1, svm_roc_auc, svm_pr_auc: float
- knn_acc, knn_f1, knn_roc_auc, knn_pr_auc: float
- ridge_mse, ridge_mae, ridge_r2: float  # Regression
- svr_mse, svr_mae, svr_r2: float
- knn_mse, knn_mae, knn_r2: float
- embedding_acc, embedding_f1: float     # For embedding methods
```

---

## Resume Logic

Both stages support full resume:

1. **Server startup**: Queries DynamoDB for completed configs, only serves remaining
2. **Worker**: Checks S3 before processing (defensive skip if file exists)
3. **Stale timeout**: Pending configs older than 30-60 min are reset (dead worker recovery)
4. **Crash recovery**: Just restart server and workers - they pick up where they left off

---

## Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `S3_BUCKET` | S3 bucket for results | `citrees-results` | `citrees-results` |
| `TABLE_NAME` | DynamoDB table name | Required | `ClfFeatureSelection` |
| `URL` | FastAPI server URL | Required | `http://10.0.0.1:8000` |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` | `us-east-1` |
| `AWS_PROFILE` | AWS credentials profile | None | `personal` |
| `N_JOBS` | Parallel jobs (Stage 2) | `-1` | `4` |
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

**Classification:** 16 methods × 24 datasets × 30 seeds = 11,520 configs
**Regression:** 13 methods × N datasets × 30 seeds = N × 390 configs

---

## Analysis

After experiments complete:

```bash
# Download results from S3
aws s3 sync s3://citrees-results/metrics/ paper/results/metrics/

# Run statistical analysis
uv run python paper/scripts/analysis.py

# Generate figures
uv run python paper/scripts/generate_figures.py
```

### Generated Outputs

| Output | Description |
|--------|-------------|
| Accuracy vs k curves | Performance at each feature subset size |
| Critical difference diagrams | Nemenyi post-hoc test visualization |
| Box plots | Method comparison distributions |
| Heatmaps | Performance by experimental factors |
| Friedman test tables | Overall significance tests |
| Rankings | Method rankings with confidence intervals |
