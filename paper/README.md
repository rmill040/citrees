# Paper Experiments

This directory contains scripts and data for reproducing the experiments in the
citrees paper.

## Directory Structure

```
paper/
├── data/                    # Datasets (parquet format)
│   ├── clf_*.parquet       # Classification datasets (24 real + synthetic)
│   ├── reg_*.parquet       # Regression datasets (8)
│   └── synthetic_ground_truth.json  # Ground truth for synthetic datasets
├── scripts/
│   ├── configs.py          # Experiment config dataclasses
│   ├── generate_synthetic_datasets.py  # Generate synthetic datasets
│   ├── comprehensive_analysis.py  # Full analysis with ground truth
│   ├── analysis.py         # Statistical tests and visualizations
│   ├── timing.py           # Timing benchmarks
│   ├── ec2_launch.py       # EC2 server/worker launcher
│   ├── clf_*               # Classification experiments
│   └── reg_*               # Regression experiments
└── results/
    ├── comprehensive_analysis/  # Full analysis output
    │   ├── figures/
    │   └── tables/
    └── *.parquet           # Experiment outputs
```

## Quick Start (Local)

```bash
# Install paper dependencies
uv sync --group paper

# Generate synthetic datasets (creates clf_syn_*.parquet files)
uv run python paper/scripts/generate_synthetic_datasets.py

# Run comprehensive analysis (after experiments complete)
uv run python paper/scripts/comprehensive_analysis.py
```

---

## Experiment Overview

### NEW: Synthetic Datasets with Ground Truth

Synthetic datasets are now integrated into the main pipeline. Generate them
before running experiments:

```bash
uv run python paper/scripts/generate_synthetic_datasets.py
```

**Dataset Types:**

| Type               | Purpose                                                 | Count |
| ------------------ | ------------------------------------------------------- | ----- |
| **Selection Bias** | High-cardinality noise features to test selection bias  | 27    |
| **Standard**       | Varying n_features, n_informative, class_sep            | 162   |
| **Nonlinear**      | Friedman #1 function, tests RDC vs MC                   | 27    |
| **Correlated**     | Correlated feature blocks, tests conditional importance | 18    |

**Total: ~234 synthetic datasets** with ground truth stored in
`synthetic_ground_truth.json`

---

### Key Experiments

#### 1. Selection Bias Demonstration (NEW)

**Goal**: Prove citrees avoids selection bias that plagues CART/RF

**Design**:

- Synthetic data with **uninformative high-cardinality features** (50-500 unique
  values)
- Truly informative low-cardinality features
- RF/CART will spuriously select high-cardinality noise
- citrees should correctly ignore them

**Metrics**:

- Noise selection rate (% of top-k that are noise features)
- Precision@k for true informative features

#### 2. Selector Comparison (NEW)

**Goal**: Compare MC vs RDC vs MI on different relationship types

**Design**:

- Linear datasets → MC should excel
- Nonlinear datasets (Friedman #1) → RDC should help
- Permutation test variants (ptest_mc, ptest_rdc, ptest_mi)

#### 3. Timing Analysis (NEW)

**Goal**: Quantify computational cost

**Metrics**:

- Wall-clock time per experiment (captured by workers)
- Scaling with n_samples, n_features

---

### Feature Selection Experiments (Distributed)

Compare citrees feature ranking against baselines on real and synthetic
datasets.

**Architecture:** Distributed server-worker pattern using FastAPI + DynamoDB

```
┌──────────────────────────────────────────────────────────────────────┐
│                     DISTRIBUTED ARCHITECTURE                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   ┌─────────────────┐         ┌─────────────────────────────────┐   │
│   │   FastAPI       │  HTTP   │         EC2 Fleet               │   │
│   │   Server        │◄────────│   ┌─────────┐  ┌─────────┐     │   │
│   │   (configs)     │         │   │ Worker  │  │ Worker  │ ... │   │
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

**Scripts:**

| Script                            | Type     | Description                                      |
| --------------------------------- | -------- | ------------------------------------------------ |
| `clf_feature_selection_server.py` | Server   | Serves experiment configurations                 |
| `clf_feature_selection_worker.py` | Worker   | Runs feature selection methods                   |
| `clf_cv_server.py`                | Server   | Serves downstream evaluation configs             |
| `clf_cv_worker.py`                | Worker   | Multi-model evaluation (SVM, LR, kNN, XGB, LGBM) |
| `clf_cv_analysis.py`              | Analysis | Generate rankings and figures                    |
| `reg_*`                           | Same     | Regression equivalents                           |

---

## Distributed Computing Setup

### Prerequisites

1. **AWS Account** with permissions for:
   - DynamoDB (create tables, read/write)
   - EC2 (launch instances)
   - IAM (create roles)

2. **DynamoDB Tables** (create these first):

   ```
   ClfFeatureSelection        # Stores feature selection results
   ClfFeatureSelectionFail    # Stores failed experiments
   ClfFeatureSelectionMetrics # Stores CV evaluation results
   ClfFeatureSelectionMetricsFail

   # Same for regression:
   RegFeatureSelection
   RegFeatureSelectionFail
   RegFeatureSelectionMetrics
   RegFeatureSelectionMetricsFail
   ```

3. **IAM Role** for EC2 instances with DynamoDB access:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "dynamodb:PutItem",
           "dynamodb:GetItem",
           "dynamodb:Scan",
           "dynamodb:Query"
         ],
         "Resource": "arn:aws:dynamodb:us-east-1:*:table/*"
       }
     ]
   }
   ```

### Running the Server

The server generates experiment configurations and serves them to workers.

```bash
# Classification feature selection
export TABLE_NAME=ClfFeatureSelection
export AWS_DEFAULT_REGION=us-east-1
uvicorn clf_feature_selection_server:app --host 0.0.0.0 --port 8000

# Monitor status
curl http://localhost:8000/status
# {"n_configs_remaining": 234567, "hosts": {"10.0.0.5": 100, ...}}
```

### Running Workers (EC2 Fleet)

Launch EC2 instances with the IAM role and run workers:

```bash
# Environment setup
export URL=http://<server-ip>:8000
export TABLE_NAME=ClfFeatureSelection
export AWS_DEFAULT_REGION=us-east-1
export N_JOBS_OUTER=1       # Number of parallel requests to server
export N_JOBS_INNER=-1      # Parallelism within each experiment
export SKIP=""              # Optional: comma-separated methods to skip

# Install dependencies
pip install -e ".[bench]"  # or use uv

# Run worker (runs until no configs remaining)
python paper/scripts/clf_feature_selection_worker.py
```

### Worker Scaling

Scale horizontally by launching more EC2 instances. General guidelines:

- **Instance type**: Use compute-optimized instances (c5, c6i, c7i families) for
  CPU-bound workloads
- **N_JOBS_OUTER**: Set to 1-2 per 4 vCPUs to avoid overwhelming the server
- **N_JOBS_INNER**: Set to `-1` to use all cores for each experiment
- **Fleet size**: More instances = faster completion; each worker is stateless

The server tracks progress via DynamoDB, so workers can be added/removed at any
time without losing work.

### Automated EC2 Deployment

Use the provided launch scripts for automated deployment:

```bash
# Launch server instance
python paper/scripts/ec2_launch.py server \
    --table-name ClfFeatureSelection \
    --instance-type c5.xlarge \
    --key-name your-key-pair \
    --iam-role citrees-worker-role \
    --security-group sg-xxx

# Launch worker fleet
python paper/scripts/ec2_launch.py worker \
    --server-url http://<server-private-ip>:8000 \
    --table-name ClfFeatureSelection \
    --instance-type c5.xlarge \
    --count 10 \
    --key-name your-key-pair \
    --iam-role citrees-worker-role \
    --security-group sg-xxx
```

See `paper/scripts/ec2_launch.py --help` for all options.

### Nested CV Evaluation

Given the limitations of evaluating feature selection with standard CV (where
test fold data informs feature selection), we use nested cross-validation:

```bash
export URL=http://<server-ip>:8000
export TABLE_NAME=ClfFeatureSelection
uv run python paper/scripts/nested_cv_worker.py
```

Feature selection occurs within each CV fold on training data only.

---

### Running CV Evaluation (Phase 2 - Legacy)

After feature selection completes, run downstream model evaluation:

```bash
# Server (serves configs from DynamoDB feature selection results)
export TABLE_NAME=ClfFeatureSelection
uvicorn clf_cv_server:app --host 0.0.0.0 --port 8000

# Workers
export URL=http://<server-ip>:8000
export TABLE_NAME=ClfFeatureSelection
python paper/scripts/clf_cv_worker.py
```

---

## Analysis and Figures

### Running Analysis

```bash
uv run python paper/scripts/analysis.py
```

### Generated Figures

| Figure                          | File                              | Description                         |
| ------------------------------- | --------------------------------- | ----------------------------------- |
| **Critical Difference Diagram** | `cd_precision@10.png`             | Nemenyi post-hoc test visualization |
| **Critical Difference Diagram** | `cd_downstream_acc_mean.png`      | For downstream accuracy             |
| **Box Plots**                   | `boxplot_precision@10.png`        | Method comparison distributions     |
| **Box Plots**                   | `boxplot_downstream_acc_mean.png` | Accuracy distributions              |
| **Heatmaps**                    | `heatmap_precision@10_*.png`      | Performance by experimental factors |

### Generated Tables

| Table             | File                                 | Description                |
| ----------------- | ------------------------------------ | -------------------------- |
| **Friedman Test** | `friedman_synthetic.csv/.tex`        | Overall significance test  |
| **Rankings**      | `ranks_precision@10.csv/.tex`        | Method rankings with CD    |
| **Rankings**      | `ranks_downstream_acc_mean.csv/.tex` | Accuracy rankings          |
| **Summary Stats** | `summary_synthetic.csv`              | Mean ± std for all metrics |

### Critical Difference Diagrams

These diagrams show:

- **Horizontal axis**: Average rank across datasets (lower = better)
- **Black bars**: Connect methods not significantly different (Nemenyi test)
- **CD bar**: Critical difference threshold

Example interpretation:

```
    1    2    3    4    5    6    7
    |----ciforest
    |----citree
         |----rf
              |----et
                   |----xgb
                        |----lgbm
                             |----dt

    CD = 1.23

Methods connected by bars are NOT significantly different.
```

---

## Methods Compared

### Feature Selection Methods

**Filter methods (score-based):** | Method | Task | Description |
|--------|------|-------------| | `mc` | Classification | Multiple correlation
(ANOVA-based) | | `mi` | Classification | Mutual information | | `rdc` | Both |
Randomized dependence coefficient | | `pc` | Regression | Pearson correlation |
| `dc` | Regression | Distance correlation |

**Embedded methods (model-based):** | Method | Description | Importance Type |
|--------|-------------|-----------------| | `cit` | Conditional Inference Tree
| Impurity decrease | | `cif` | Conditional Inference Forest | Averaged impurity
decrease | | `rf` | Random Forest | Impurity decrease | | `et` | Extra Trees |
Impurity decrease | | `dt` | Decision Tree | Impurity decrease | | `xgb` |
XGBoost | Gain / Weight / Cover | | `lightgbm` | LightGBM | Gain / Split | |
`catboost` | CatBoost | Permutation importance | | `lr_l1` | Lasso | Coefficient
magnitude | | `lr_l2` | Ridge | Coefficient magnitude |

### Downstream Models (for CV evaluation)

**Classification:** | Model | Description | |-------|-------------| | `svm` |
Support Vector Machine (balanced) | | `lr` | Logistic Regression (balanced) | |
`knn` | k-Nearest Neighbors (k=5, distance-weighted) | | `xgb` | XGBoost
Classifier | | `lgbm` | LightGBM Classifier |

**Regression:** | Model | Description | |-------|-------------| | `svr` |
Support Vector Regression | | `ridge` | Ridge Regression (α=1.0) | | `knn` |
k-Nearest Neighbors Regressor | | `xgb` | XGBoost Regressor | | `lgbm` |
LightGBM Regressor |

---

## Datasets

### Classification (24 datasets)

| Dataset      | Samples | Features | Classes | Domain            |
| ------------ | ------- | -------- | ------- | ----------------- |
| ALLAML       | 72      | 7,129    | 2       | Genomics          |
| CALL_SUB_111 | 111     | 11,340   | 3       | Genomics          |
| arcene       | 100     | 10,000   | 2       | Mass spectrometry |
| dexter       | 300     | 20,000   | 2       | Text              |
| dorothea     | 800     | 100,000  | 2       | Drug discovery    |
| gisette      | 6,000   | 5,000    | 2       | Digit recognition |
| isolet       | 7,797   | 616      | 26      | Speech            |
| madelon      | 2,000   | 500      | 2       | Synthetic         |
| ...          |         |          |         |                   |

### Regression (8 datasets)

| Dataset         | Samples | Features | Domain   |
| --------------- | ------- | -------- | -------- |
| coepra1-3       | varies  | varies   | Chemical |
| comm_violence   | varies  | varies   | Social   |
| community_crime | varies  | varies   | Social   |
| ...             |         |          |          |

---

## Environment Variables Reference

| Variable             | Description                   | Default     | Example                 |
| -------------------- | ----------------------------- | ----------- | ----------------------- |
| `TABLE_NAME`         | DynamoDB table prefix         | Required    | `ClfFeatureSelection`   |
| `URL`                | FastAPI server URL            | Required    | `http://10.0.0.1:8000`  |
| `AWS_DEFAULT_REGION` | AWS region                    | `us-east-1` | `us-east-1`             |
| `N_JOBS_OUTER`       | Parallel server requests      | `1`         | `4`                     |
| `N_JOBS_INNER`       | Parallelism within experiment | `-1`        | `-1` (all cores)        |
| `SKIP`               | Methods to skip               | None        | `xgb,lightgbm,catboost` |
| `DATA_DIR`           | Data directory (analysis)     | None        | `/path/to/data`         |
| `GET_DATA`           | Export from DynamoDB          | `0`         | `1`                     |

---

## Troubleshooting

### Common Issues

**Server shows 0 configs remaining:**

- Configs already processed are stored in DynamoDB
- Clear tables or use different `TABLE_NAME` for fresh run

**Worker connection refused:**

- Check server is running and accessible
- Verify security groups allow port 8000

**Out of memory on large datasets:**

- Reduce `N_JOBS_INNER` to `1`
- Use larger instance type
- Add `max_depth` limit for trees

**DynamoDB throughput exceeded:**

- Increase provisioned capacity or use on-demand
- Reduce `N_JOBS_OUTER` on workers

### Logging

Workers use `loguru` for structured logging:

```python
from loguru import logger
logger.info("Processing config {config_idx}")
```

Check logs for experiment progress and errors.
