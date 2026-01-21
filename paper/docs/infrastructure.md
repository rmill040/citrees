# Distributed Experiment Infrastructure

Run citrees feature selection experiments at scale on AWS using Ray.

## Quick Start

```bash
# 1. Generate config (creates S3 bucket, fills placeholders)
AWS_PROFILE=personal uv run python paper/scripts/infra/ray/setup_cluster.py --generate

# 2. Deploy cluster
AWS_PROFILE=personal uv run ray up paper/scripts/infra/ray/cluster.yaml --yes

# 3. Run feature selection (Stage 1)
AWS_PROFILE=personal uv run ray submit paper/scripts/infra/ray/cluster.yaml \
    paper/scripts/experiments/ray_feature_selection.py

# 4. Run downstream evaluation (Stage 2)
AWS_PROFILE=personal uv run ray submit paper/scripts/infra/ray/cluster.yaml \
    paper/scripts/experiments/ray_eval.py

# 5. Monitor progress
AWS_PROFILE=personal uv run python paper/scripts/experiments/check_progress.py --stage rankings

# 6. Tear down when done
AWS_PROFILE=personal uv run ray down paper/scripts/infra/ray/cluster.yaml --yes
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
│  │  32 vCPUs, 64GB        │    │  16 vCPUs, 32GB        │       │
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

### Setup Script (`paper/scripts/infra/ray/setup_cluster.py`)

```bash
# Generate cluster.yaml from template
AWS_PROFILE=personal uv run python paper/scripts/infra/ray/setup_cluster.py --generate

# What it does:
# - Fetches your public IP for security group rules
# - Creates S3 bucket if needed
# - Fills in __MY_IP__, __BRANCH__, __S3_BUCKET__ placeholders
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
      InstanceType: c6i.4xlarge # 16 vCPUs, 32GB
      InstanceMarketOptions:
        MarketType: spot
    resources:
      evaluation: 100
    max_workers: 250
```

### Experiment Config (`paper/scripts/infra/config.yaml`)

```yaml
region: us-east-1

s3:
  bucket_name: null # Auto: citrees-results-{account_id}

experiment:
  type: classification # or "regression"
  n_seeds: 10
```

## Ray Commands

### Cluster Management

```bash
# Start cluster
AWS_PROFILE=personal uv run ray up paper/scripts/infra/ray/cluster.yaml --yes

# Check cluster status
AWS_PROFILE=personal uv run ray exec paper/scripts/infra/ray/cluster.yaml \
    '$HOME/citrees/.venv/bin/ray status'

# View dashboard (opens browser)
AWS_PROFILE=personal uv run ray dashboard paper/scripts/infra/ray/cluster.yaml

# SSH to head node
AWS_PROFILE=personal uv run ray attach paper/scripts/infra/ray/cluster.yaml

# Tear down cluster
AWS_PROFILE=personal uv run ray down paper/scripts/infra/ray/cluster.yaml --yes
```

### Running Jobs

```bash
# Submit feature selection job
AWS_PROFILE=personal uv run ray submit paper/scripts/infra/ray/cluster.yaml \
    paper/scripts/experiments/ray_feature_selection.py

# Submit evaluation job
AWS_PROFILE=personal uv run ray submit paper/scripts/infra/ray/cluster.yaml \
    paper/scripts/experiments/ray_eval.py
```

## Worker Pools

The cluster uses separate worker pools for each stage:

| Pool               | Instance    | vCPUs | RAM  | Resource          | Purpose                   |
| ------------------ | ----------- | ----- | ---- | ----------------- | ------------------------- |
| `head`             | c6i.4xlarge | 16    | 32GB | -                 | Scheduler, dashboard      |
| `selection_worker` | c6i.8xlarge | 32    | 64GB | `selection: 100`  | Feature selection (heavy) |
| `eval_worker`      | c6i.4xlarge | 16    | 32GB | `evaluation: 100` | Downstream eval (light)   |

Tasks are routed via custom resources:

- `@ray.remote(resources={"selection": 1})` → runs on selection_worker
- `@ray.remote(resources={"evaluation": 1})` → runs on eval_worker

## S3 Structure

```
s3://citrees-results-{account_id}/
├── rankings/
│   └── classification/
│       └── {dataset}/
│           ├── {method_id}_seed0.parquet
│           ├── {method_id}_seed1.parquet
│           └── ...
└── metrics/
    └── classification/
        └── {dataset}/
            ├── {method_id}_seed0.parquet
            └── ...
```

## Monitoring

### Check Progress

```bash
# Stage 1 progress
AWS_PROFILE=personal uv run python paper/scripts/experiments/check_progress.py --stage rankings

# Stage 2 progress
AWS_PROFILE=personal uv run python paper/scripts/experiments/check_progress.py --stage metrics

# By method
AWS_PROFILE=personal uv run python paper/scripts/experiments/check_progress.py --stage rankings --by-method

# By dataset
AWS_PROFILE=personal uv run python paper/scripts/experiments/check_progress.py --stage rankings --by-dataset
```

### Ray Dashboard

Access at `http://<head-ip>:8265` or via:

```bash
AWS_PROFILE=personal uv run ray dashboard paper/scripts/infra/ray/cluster.yaml
```

Features:

- Real-time task progress
- Worker utilization
- Error logs
- Resource usage

## Missing-only Runs (Recommended)

The pipeline does **not** auto-skip inside each task. Instead, you filter the
grid **before** submission using S3 listings. This gives you a deterministic,
auditable list of configs that will run.

```bash
# Preview the exact configs that will run (print full list)
AWS_PROFILE=personal uv run python paper/scripts/experiments/run_pipeline.py \
    --stage all --only-missing --dry-run --dry-run-limit 100000

# Run only configs missing in S3
AWS_PROFILE=personal uv run python paper/scripts/experiments/run_pipeline.py \
    --stage all --only-missing
```

**Reruns:** delete the relevant S3 objects (rankings/metrics) and re-run with
`--only-missing`.

## Fault Tolerance

- **Spot interruption**: Ray reschedules tasks from terminated workers.
- **Crash recovery**: re-run with `--only-missing` to fill gaps.

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
# Check cluster status
AWS_PROFILE=personal uv run ray exec paper/scripts/infra/ray/cluster.yaml \
    '$HOME/citrees/.venv/bin/ray status'

# Check autoscaler logs
AWS_PROFILE=personal uv run ray attach paper/scripts/infra/ray/cluster.yaml
cat /tmp/ray/session_latest/logs/monitor.log
```

### Tasks failing

```bash
# View task errors in dashboard
AWS_PROFILE=personal uv run ray dashboard paper/scripts/infra/ray/cluster.yaml

# Or check logs on head
AWS_PROFILE=personal uv run ray attach paper/scripts/infra/ray/cluster.yaml
cat /tmp/ray/session_latest/logs/raylet.out
```

### AMI issues

```bash
# Update to latest Ubuntu 22.04 AMI
AWS_PROFILE=personal uv run python paper/scripts/infra/ray/setup_cluster.py --update-ami
```
