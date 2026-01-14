# Distributed Experiment Infrastructure

Run citrees feature selection experiments at scale on AWS using Ray.

## Quick Start

```bash
# 1. Update cluster config with latest AMI
AWS_PROFILE=personal uv run python paper/scripts/infra/ray/setup_cluster.py --update-ami

# 2. Validate configuration
AWS_PROFILE=personal uv run python paper/scripts/infra/ray/setup_cluster.py --validate

# 3. Start Ray cluster (spot instances)
AWS_PROFILE=personal ray up paper/scripts/infra/ray/cluster.yaml

# 4. Run feature selection (Stage 1)
AWS_PROFILE=personal ray submit paper/scripts/infra/ray/cluster.yaml \
    paper/scripts/ray_feature_selection.py

# 5. Run downstream evaluation (Stage 2)
AWS_PROFILE=personal ray submit paper/scripts/infra/ray/cluster.yaml \
    paper/scripts/ray_eval.py

# 6. Monitor progress
AWS_PROFILE=personal uv run python paper/scripts/check_progress.py --stage rankings

# 7. Tear down when done
AWS_PROFILE=personal ray down paper/scripts/infra/ray/cluster.yaml
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Ray Cluster                               │
│                                                                   │
│  ┌────────────────┐                                              │
│  │   Head Node    │                                              │
│  │  c7i.4xlarge   │──────────────────────────────────┐          │
│  │                │                                   │          │
│  │  - Scheduler   │                                   │          │
│  │  - Dashboard   │                                   │          │
│  └────────────────┘                                   │          │
│          │                                            │          │
│          ▼                                            ▼          │
│  ┌────────────────────────┐    ┌────────────────────────┐       │
│  │  Selection Workers     │    │  Eval Workers          │       │
│  │  c7i.8xlarge (spot)    │    │  c7i.4xlarge (spot)    │       │
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

### Cluster Config (`paper/scripts/infra/ray/cluster.yaml`)

```yaml
cluster_name: citrees

provider:
  type: aws
  region: us-east-1

available_node_types:
  head:
    node_config:
      InstanceType: c7i.4xlarge
      ImageId: ami-xxx  # Ubuntu 22.04

  selection_worker:
    node_config:
      InstanceType: c7i.8xlarge  # 32 vCPUs
      InstanceMarketOptions:
        MarketType: spot
    resources:
      selection: 100  # 100 concurrent tasks per worker
    min_workers: 0
    max_workers: 250

  eval_worker:
    node_config:
      InstanceType: c7i.4xlarge  # 16 vCPUs
      InstanceMarketOptions:
        MarketType: spot
    resources:
      evaluation: 100
    min_workers: 0
    max_workers: 250
```

### Experiment Config (`paper/scripts/infra/config.yaml`)

```yaml
region: us-east-1

s3:
  bucket_name: null  # Auto: citrees-results-{account_id}

experiment:
  type: classification  # or "regression"
  n_seeds: 10
```

## Ray Commands

### Cluster Management

```bash
# Start cluster
ray up paper/scripts/infra/ray/cluster.yaml

# Check cluster status
ray status

# Scale workers manually
ray up paper/scripts/infra/ray/cluster.yaml --min-workers 50

# View dashboard (opens browser)
ray dashboard paper/scripts/infra/ray/cluster.yaml

# SSH to head node
ray attach paper/scripts/infra/ray/cluster.yaml

# Tear down cluster
ray down paper/scripts/infra/ray/cluster.yaml
```

### Running Jobs

```bash
# Submit feature selection job
ray submit paper/scripts/infra/ray/cluster.yaml paper/scripts/ray_feature_selection.py

# Submit evaluation job
ray submit paper/scripts/infra/ray/cluster.yaml paper/scripts/ray_eval.py

# Run with specific config
ray submit paper/scripts/infra/ray/cluster.yaml paper/scripts/ray_feature_selection.py \
    --runtime-env-json='{"env_vars": {"EXPERIMENT_TYPE": "regression"}}'
```

## Worker Pools

The cluster uses separate worker pools for each stage:

| Pool | Instance | vCPUs | RAM | Resource | Purpose |
|------|----------|-------|-----|----------|---------|
| `selection_worker` | c7i.8xlarge | 32 | 64GB | `selection: 100` | Feature selection (heavy) |
| `eval_worker` | c7i.4xlarge | 16 | 32GB | `evaluation: 100` | Downstream eval (light) |

Tasks are routed via custom resources:
- `@ray.remote(resources={"selection": 1})` → runs on selection_worker
- `@ray.remote(resources={"evaluation": 1})` → runs on eval_worker

## S3 Structure

```
s3://citrees-results-{account_id}/
├── rankings/
│   └── classification/
│       └── {dataset}/
│           ├── mc_seed0.parquet
│           ├── rf_seed0.parquet
│           └── ...
└── metrics/
    └── classification/
        └── {dataset}/
            ├── mc_seed0.parquet
            └── ...
```

## Monitoring

### Check Progress

```bash
# Stage 1 progress
AWS_PROFILE=personal uv run python paper/scripts/check_progress.py --stage rankings

# Stage 2 progress
AWS_PROFILE=personal uv run python paper/scripts/check_progress.py --stage metrics

# By method
AWS_PROFILE=personal uv run python paper/scripts/check_progress.py --stage rankings --by-method

# By dataset
AWS_PROFILE=personal uv run python paper/scripts/check_progress.py --stage rankings --by-dataset
```

### Ray Dashboard

Access at `http://<head-ip>:8265` or via:
```bash
ray dashboard paper/scripts/infra/ray/cluster.yaml
```

Features:
- Real-time task progress
- Worker utilization
- Error logs
- Resource usage

## Resume & Fault Tolerance

Both stages support full resume:

1. **S3 check**: Each task checks if output exists before processing
2. **Spot interruption**: Ray automatically reschedules tasks from terminated workers
3. **Crash recovery**: Just re-run the script - completed tasks are skipped

```bash
# Re-run after interruption - only processes remaining configs
ray submit paper/scripts/infra/ray/cluster.yaml paper/scripts/ray_feature_selection.py
```

## Cost Estimates

| Configuration | Spot Price | Est. Daily Cost |
|---------------|------------|-----------------|
| 1 head (c7i.4xlarge) | ~$0.20/hr | ~$4.80 |
| 10 selection workers (c7i.8xlarge) | ~$0.40/hr each | ~$96 |
| 10 eval workers (c7i.4xlarge) | ~$0.20/hr each | ~$48 |

**Tips:**
- Spot instances are 60-90% cheaper than on-demand
- Workers auto-scale based on pending tasks
- Tear down cluster when not in use
- Use `--min-workers 0` to allow scale-to-zero

## Troubleshooting

### Workers not starting

```bash
# Check cluster status
ray status

# Check autoscaler logs
ray attach paper/scripts/infra/ray/cluster.yaml
cat /tmp/ray/session_latest/logs/monitor.log
```

### Tasks failing

```bash
# View task errors in dashboard
ray dashboard paper/scripts/infra/ray/cluster.yaml

# Or check logs on head
ray attach paper/scripts/infra/ray/cluster.yaml
cat /tmp/ray/session_latest/logs/raylet.out
```

### AMI issues

```bash
# Update to latest Ubuntu 22.04 AMI
AWS_PROFILE=personal uv run python paper/scripts/infra/ray/setup_cluster.py --update-ami
```

### S3 permission errors

Ensure your AWS credentials have S3 access:
```bash
AWS_PROFILE=personal aws s3 ls s3://citrees-results-xxx/
```
