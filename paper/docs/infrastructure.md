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
      selection: 32
    max_workers: 250

  eval_worker:
    node_config:
      InstanceType: c6i.xlarge # 4 vCPUs, 8GB
      InstanceMarketOptions:
        MarketType: spot
    resources:
      evaluation: 4
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

### Experiment Config (`paper/scripts/infra/config.yaml`)

```yaml
aws_region: us-east-1

cluster:
  max_workers: 50
  upscaling_speed: 10.0
  idle_timeout_minutes: 5

experiment:
  type: classification
  n_seeds: 5
  stale_timeout_minutes: 30

  # Stage 1 resource tiers (LIGHT/STANDARD/HEAVY defined in code)
  # Per-method overrides take priority over tier defaults
  selection_cpus_overrides: {}
  selection_memory_gb_overrides: {}

  # Stage 2 (flat defaults)
  evaluation_cpus_default: 1
  evaluation_cpus_overrides: {}
  evaluation_memory_gb_default: 2.0
  evaluation_memory_gb_overrides: {}

  s3_validate_uploads: true
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

| Pool               | Instance    | vCPUs | RAM  | Resource        | Purpose              |
| ------------------ | ----------- | ----- | ---- | --------------- | -------------------- |
| `head`             | c6i.4xlarge | 16    | 32GB | -               | Scheduler, dashboard |
| `selection_worker` | c6i.8xlarge | 32    | 64GB | `selection: 32` | Feature selection    |
| `eval_worker`      | c6i.xlarge  | 4     | 8GB  | `evaluation: 4` | Downstream eval      |

Tasks are routed via **two resource dimensions**:

- **Custom resource** (`selection` / `evaluation`): routes task to the correct
  worker pool. Set to vCPU count to cap concurrency per node.
- **`num_cpus` + `memory`**: set dynamically per-task via `.options()` based on
  the method's tier. Ray uses these for bin-packing (e.g., two HEAVY tasks
  requesting 16 CPUs each fill a 32-vCPU node).

## Resource Scheduling

### Selection Tiers

Stage 1 tasks are assigned CPU and memory based on method complexity:

| Tier         | CPUs | Memory | Methods                                             |
| ------------ | ---- | ------ | --------------------------------------------------- |
| **LIGHT**    | 1    | 2 GB   | `mc, pc, rdc, mi, dc, mrmr`, all `ptest_*` variants |
| **STANDARD** | 8    | 4 GB   | `rf, et, xgb, lgbm, cat, pi, cpi, rfe, r_ctree`     |
| **HEAVY**    | 16   | 8 GB   | `cit, cif, boruta, shap, r_cforest`                 |

Stage 2 evaluation uses flat defaults: 1 CPU, 2 GB.

### Dynamic Scheduling via `.options()`

Resources are NOT hardcoded in `@ray.remote` decorators. Instead, the CLI
computes per-task resource requirements and passes them at submission time:

    app.py → _stage1_options(cfg) → {num_cpus, memory}
                                          ↓
    runner.py → task_fn.options(**opts).remote(cfg, store)

This allows Ray to pack tasks efficiently — a LIGHT method (1 CPU) and a HEAVY
method (16 CPUs) can share the same 32-vCPU worker node.

### Override Hierarchy

Resource resolution follows this priority (highest first):

1. **Config per-method override** — `selection_cpus_overrides: {cif: 32}`
2. **Tier default** — LIGHT / STANDARD / HEAVY lookup from `methods.py`

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

## Skipping and Re-running

### Default: skip completed configs

The pipeline uses **two layers** of filtering:

1. **Batch S3 listing** (before submission): `store.list_completed()` fetches
   all existing keys and removes them from the grid.
2. **Per-task check** (inside each Ray worker): `store.exists()` checks before
   running, in case another worker completed the same config concurrently.

```bash
# Preview what would run (skips existing by default)
citrees-exp run classification --dry-run

# Run only configs missing in S3 (default behavior)
citrees-exp run classification
```

### `--force`: re-run everything

When `--force` is passed, both layers are bypassed:

- Batch listing is skipped.
- Per-task checks are disabled via `IgnoreExistsStore`, a wrapper that returns
  `False` for `exists()` on the forced stages while delegating all other
  operations (save, load) to the real S3Store.

```bash
# Re-run all configs regardless of existing results
citrees-exp run classification --force
```

You can also delete specific S3 objects and re-run normally to selectively
re-execute individual configs.

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

### Local driver OOM (exit code 137) or Ray Client "large messages" warnings

If you attempt to run a very large grid (millions of configs), submitting all
tasks at once can overwhelm the local driver and/or the Ray Client control
plane. The CLI now applies backpressure, but you can further reduce load by
lowering the in-flight task cap and/or reducing the grid size.

```bash
# Reduce driver memory / Ray Client chatter
citrees-exp run classification --max-in-flight 256

# Reduce the experiment grid size (recommended for iteration)
citrees-exp run classification -m cit,rf -d arcene --seeds 0 --max-configs-per-method 10
```

### Run from the head node (avoid Ray Client)

For long-running cluster jobs, prefer submitting a driver to the head node.

```bash
# Run the CLI on the head node via Ray's SSH-based submit
citrees-exp cluster submit paper/scripts/cli/run_cli.py run classification --stage all
```

If you are seeing stale dependencies or old Docker images on the head node,
terminate the head node (not just restart it) and bring the cluster up again:

```bash
citrees-exp cluster down --yes
citrees-exp cluster up --yes
```

### AMI issues

```bash
citrees-exp infra update-ami    # Fetch current Ubuntu 22.04 AMI and update cluster.yaml
```
