# Distributed Experiment Infrastructure

Run citrees feature selection experiments at scale on AWS using Ray.

> **AWS Note:** All commands require AWS credentials. Set `AWS_PROFILE` if not
> using your default profile.

## Quick Start

```bash
# 1. One-time setup (IAM + S3 + ECR + Docker + cluster.yaml)
citrees-exp infra setup

# 2. Deploy cluster
citrees-exp cluster up --yes

# 3. Run experiments (skips existing results by default)
citrees-exp run classification

# 4. Monitor progress
citrees-exp check
citrees-exp watch  # live dashboard

# 5. Tear down when done
citrees-exp cluster down --yes
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Ray Cluster                              │
│                                                                 │
│  ┌────────────────┐                                             │
│  │   Head Node    │                                             │
│  │  c6i.4xlarge   │──────────────────────────────────┐          │
│  │                │                                   │          │
│  │  - Scheduler   │                                   │          │
│  │  - Dashboard   │                                   │          │
│  └────────────────┘                                   │          │
│          │                                            │          │
│          ▼                                            ▼          │
│  ┌────────────────────────┐    ┌────────────────────────┐       │
│  │  Selection Workers     │    │  Eval Workers          │       │
│  │  c6i.8xlarge (spot)    │    │  c6i.4xlarge (spot)    │       │
│  │  32 vCPUs, 64GB        │    │  c6i.xlarge (spot)     │       │
│  │                        │    │  4 vCPUs, 8GB          │       │
│  │  max: 250              │    │  max: 250              │       │
│  │                        │    │                        │       │
│  │  Stage 1:              │    │  Stage 2:              │       │
│  │  Feature selection     │    │  Downstream eval       │       │
│  └────────────────────────┘    └────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │      S3      │
                       │  - rankings  │
                       │  - metrics   │
                       └──────────────┘
```

## Configuration

### Setup Script

```bash
# Full setup (recommended): IAM + S3 + ECR + Docker image + cluster.yaml
citrees-exp infra setup

# Regenerate cluster.yaml only
citrees-exp infra generate

# What setup does:
# - Creates IAM role + instance profile for Ray autoscaling + S3/ECR access
# - Creates S3 bucket (citrees-{account_id})
# - Creates ECR repo (citrees-{account_id})
# - Builds + pushes Docker image to ECR
# - Generates cluster.yaml with your public IP + git provenance
```

### Cluster Config (`paper/scripts/infra/ray/cluster.yaml`)

Generated from `cluster.example.yaml`. Key settings:

```yaml
cluster_name: citrees

provider:
  type: aws
  region: us-east-1

available_node_types:
  head:
    node_config:
      InstanceType: c6i.4xlarge
      ImageId: ami-xxx # Ubuntu 22.04

  selection_worker:
    node_config:
      InstanceType: c6i.8xlarge # 32 vCPUs, 64GB
      InstanceMarketOptions:
        MarketType: spot
    resources:
      selection: 100
    max_workers: 250

  eval_worker:
    node_config:
      InstanceType: c6i.xlarge # 4 vCPUs, 8GB
      InstanceMarketOptions:
        MarketType: spot
    resources:
      evaluation: 100
    max_workers: 250
```

### Docker + Instance Profile Credentials

When using Docker on EC2, containers must reach IMDS to fetch instance profile
credentials. Ensure the EC2 metadata hop limit is at least 2 (IMDSv2), which is
set in the cluster config via `MetadataOptions`:

```yaml
MetadataOptions:
  HttpTokens: required
  HttpPutResponseHopLimit: 2
```

### Experiment Config (`paper/scripts/infra/ray/cluster.yaml`)

```yaml
aws_region: us-east-1
# Note: S3 bucket name is derived automatically as citrees-{account_id} by setup_cluster.py

experiment:
  type: classification # or "regression"
  n_seeds: 5
```

## CLI Commands

### Cluster Management

```bash
citrees-exp cluster up --yes      # Start cluster
citrees-exp cluster status        # Check cluster status
citrees-exp cluster dashboard     # View dashboard (opens browser)
citrees-exp cluster ssh           # SSH to head node
citrees-exp cluster logs          # View logs
citrees-exp cluster down --yes    # Tear down cluster
```

### Running Experiments

```bash
citrees-exp run classification                   # Full pipeline (skips existing)
citrees-exp run classification --stage stage1   # Stage 1 only
citrees-exp run classification --stage stage2   # Stage 2 only
citrees-exp run classification -m cit,rf        # Specific methods
citrees-exp run classification --dry-run        # Preview what would run
citrees-exp run classification --force          # Re-run everything
```

## Worker Pools

The cluster uses separate worker pools for each stage:

| Pool               | Instance    | vCPUs | RAM  | Resource          | Purpose                   |
| ------------------ | ----------- | ----- | ---- | ----------------- | ------------------------- |
| `head`             | c6i.4xlarge | 16    | 32GB | -                 | Scheduler, dashboard      |
| `selection_worker` | c6i.8xlarge | 32    | 64GB | `selection: 100`  | Feature selection (heavy) |
| `eval_worker`      | c6i.xlarge  | 4     | 8GB  | `evaluation: 100` | Downstream eval (light)   |

Tasks are routed via custom resources:

- `@ray.remote(resources={"selection": 1})` → runs on selection_worker
- `@ray.remote(resources={"evaluation": 1})` → runs on eval_worker

## S3 Structure

```
s3://citrees-{account_id}/
├── data/                           # Datasets (workers download on-demand)
│   ├── classification/
│   │   ├── real/
│   │   │   └── clf_{name}.parquet
│   │   └── synthetic/
│   │       └── clf_{name}.parquet
│   └── regression/
│       ├── real/
│       │   └── reg_{name}.parquet
│       └── synthetic/
│           └── reg_{name}.parquet
├── rankings/                       # Stage 1 outputs
│   └── classification/
│       └── {dataset}/
│           ├── {method_id}_seed0.parquet
│           ├── {method_id}_seed1.parquet
│           └── ...
└── metrics/                        # Stage 2 outputs
    └── classification/
        └── {dataset}/
            ├── {method_id}_seed0.parquet
            └── ...
```

### Dataset Sync

Datasets are **not** baked into the Docker image (saves ~643MB). Workers
download from S3 on-demand to `/tmp/citrees-data/` cache.

```bash
# Upload datasets to S3 (one-time, before running experiments)
aws s3 sync paper/data/ s3://$S3_BUCKET/data/ --exclude "*.DS_Store"

# Download datasets locally (for development)
aws s3 sync s3://$S3_BUCKET/data/ paper/data/
```

Workers automatically fall back to S3 when local files don't exist:

1. Check local path (`paper/data/...`)
2. If not found, download from S3 to `/tmp/citrees-data/` cache
3. Load from cache

## Monitoring

### Check Progress

```bash
citrees-exp check                     # Stage 1 (rankings) progress
citrees-exp check --stage metrics     # Stage 2 progress
citrees-exp check --by-method         # Grouped by method
citrees-exp check --by-dataset        # Grouped by dataset
citrees-exp watch                     # Live dashboard
```

### Ray Dashboard

Access at `http://<head-ip>:8265` or via:

```bash
citrees-exp cluster dashboard
```

Features:

- Real-time task progress
- Worker utilization
- Error logs
- Resource usage

## Missing-only Runs (Recommended)

The pipeline filters the grid **before** submission using S3 listings, giving
you a deterministic, auditable list of configs that will run.

```bash
# Preview what would run (skips existing by default)
citrees-exp run classification --dry-run

# Run only configs missing in S3 (default behavior)
citrees-exp run classification
```

**Reruns:** use `--force` to re-run everything, or delete specific S3 objects
then re-run normally.

## Fault Tolerance

- **Spot interruption**: Ray reschedules tasks from terminated workers.
- **Crash recovery**: just re-run `citrees-exp run` - it skips completed
  configs.

## Cost Estimates

| Configuration                      | Spot Price | Est. Daily Cost |
| ---------------------------------- | ---------- | --------------- |
| 1 head (c6i.4xlarge)               | varies     | varies          |
| 10 selection workers (c6i.8xlarge) | varies     | varies          |
| 10 eval workers (c6i.4xlarge)      | varies     | varies          |

**Tips:**

- Spot prices fluctuate; check AWS pricing for current rates
- Workers auto-scale based on pending tasks
- Tear down cluster when not in use

## Troubleshooting

### Workers not starting

```bash
citrees-exp cluster status    # Check cluster status
citrees-exp cluster logs      # View logs
citrees-exp cluster ssh       # SSH to head node and inspect
```

### Tasks failing

```bash
citrees-exp cluster dashboard   # View task errors in dashboard
citrees-exp cluster logs -f     # Stream logs continuously
```

### AMI issues

```bash
citrees-exp infra update-ami    # Fetch current Ubuntu 22.04 AMI and update cluster.yaml
```
