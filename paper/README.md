# Paper Experiments

This directory contains scripts and data for reproducing the experiments in the citrees paper.

## Directory Structure

```
paper/
├── data/                    # Datasets (parquet format)
│   ├── clf_*.parquet       # Classification datasets (24)
│   └── reg_*.parquet       # Regression datasets (8)
├── scripts/                 # Experiment scripts
│   ├── clf_*               # Classification experiments
│   ├── reg_*               # Regression experiments
│   ├── synthetic_*.py      # Synthetic data experiments
│   └── timing.py           # Performance benchmarks
└── results/                 # Output directory (created by scripts)
```

## Quick Start

```bash
# Install dependencies
./scripts/install_dependencies.sh

# Run synthetic experiments locally
uv run python scripts/synthetic_experiments.py
```

## Experiment Overview

### 1. Feature Selection Experiments

Compare citrees feature ranking against baselines (RF, XGBoost, LightGBM, etc.) on real datasets.

**Architecture**: Distributed server-worker pattern using FastAPI + DynamoDB

| Script | Description |
|--------|-------------|
| `clf_feature_selection_server.py` | FastAPI server that serves experiment configurations |
| `clf_feature_selection_worker.py` | Worker that runs feature selection methods |
| `clf_cv_server.py` | Serves configurations for downstream evaluation |
| `clf_cv_worker.py` | Evaluates feature rankings with SVM downstream |
| `clf_cv_worker_v2.py` | Multi-model downstream (SVM, LR, kNN) |

**Running distributed experiments:**

```bash
# Terminal 1: Start server
export TABLE_NAME=ClfFeatureSelection
uvicorn clf_feature_selection_server:app --host 0.0.0.0 --port 8000

# Terminal 2+: Start workers
export URL=http://localhost:8000
export TABLE_NAME=ClfFeatureSelection
export N_JOBS_OUTER=1
export N_JOBS_INNER=-1
uv run python clf_feature_selection_worker.py
```

### 2. Synthetic Experiments

Controlled experiments with known ground truth to evaluate feature selection quality.

```bash
uv run python scripts/synthetic_experiments.py
```

**Varies:**
- Total features: 50, 100, 500, 1000
- Informative features: 5, 10, 20
- Sample sizes: 200, 500, 1000
- Signal strength: 0.5, 1.0, 2.0

**Metrics:**
- Precision@k / Recall@k for true feature recovery
- Downstream classification accuracy

### 3. Timing Experiments

Benchmark different citrees configurations.

```bash
uv run python scripts/timing.py
```

## Datasets

### Classification (24 datasets)

| Dataset | Samples | Features | Classes | Domain |
|---------|---------|----------|---------|--------|
| ALLAML | 72 | 7,129 | 2 | Genomics |
| CLL_SUB_111 | 111 | 11,340 | 3 | Genomics |
| arcene | 100 | 10,000 | 2 | Mass spectrometry |
| dexter | 300 | 20,000 | 2 | Text |
| dorothea | 800 | 100,000 | 2 | Drug discovery |
| gisette | 6,000 | 5,000 | 2 | Digit recognition |
| isolet | 7,797 | 616 | 26 | Speech |
| madelon | 2,000 | 500 | 2 | Synthetic |
| ... | | | | |

### Regression (8 datasets)

| Dataset | Samples | Features | Domain |
|---------|---------|----------|--------|
| coepra1-3 | varies | varies | Chemical |
| comm_violence | varies | varies | Social |
| community_crime | varies | varies | Social |
| ... | | | |

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `TABLE_NAME` | DynamoDB table prefix | `ClfFeatureSelection` |
| `URL` | FastAPI server URL | `http://localhost:8000` |
| `N_JOBS_OUTER` | Parallel workers | `1` |
| `N_JOBS_INNER` | Parallelism within worker | `-1` |
| `DATA_DIR` | Data directory (for analysis) | `/path/to/data` |
| `SKIP` | Methods to skip (comma-separated) | `xgb,lightgbm` |

## Methods Compared

### Feature Selection Methods

**Filter methods:**
- `mc` - Multiple correlation (classification)
- `mi` - Mutual information (classification)
- `pc` - Pearson correlation (regression)
- `dc` - Distance correlation (regression)
- `ptest_*` - Permutation test variants

**Embedded methods:**
- `cit` - Conditional Inference Tree
- `cif` - Conditional Inference Forest
- `rf` - Random Forest
- `et` - Extra Trees
- `dt` - Decision Tree
- `xgb` - XGBoost
- `lightgbm` - LightGBM
- `catboost` - CatBoost
- `lr` / `lr_l1` / `lr_l2` - Logistic/Linear Regression

### Downstream Models (for evaluation)

- SVM (original)
- Logistic Regression (v2)
- k-NN (v2)

## Reproducing Results

1. **Setup AWS resources** (for distributed runs):
   - Create DynamoDB tables: `{TABLE_NAME}`, `{TABLE_NAME}Fail`, `{TABLE_NAME}Metrics`
   - Launch EC2 instances with IAM role for DynamoDB access

2. **Run feature selection**:
   ```bash
   # Generate configurations and run
   ./scripts/run.sh scripts/clf_feature_selection_worker.py
   ```

3. **Run downstream evaluation**:
   ```bash
   ./scripts/run.sh scripts/clf_cv_worker_v2.py
   ```

4. **Analyze results**:
   ```bash
   export DATA_DIR=/path/to/exported/data
   export GET_DATA=1
   uv run python scripts/clf_cv_analysis.py
   ```
