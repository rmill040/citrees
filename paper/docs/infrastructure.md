# Distributed Experiment Infrastructure

Run citrees feature selection experiments at scale on AWS using an API server +
EC2 worker model.

> **AWS Note:** All commands require AWS credentials. Set `AWS_PROFILE` if not
> using your default profile.

## Quick Start

```bash
# 1. One-time setup (IAM + S3 + ECR + Docker + datasets)
citrees-exp infra setup
citrees-exp infra s3
citrees-exp infra upload-data

# 2. Launch API server + workers
citrees-exp infra launch-api
citrees-exp infra launch-workers --count 5   # auto-discovers API private IP

# 3. Monitor progress
citrees-exp run                       # poll queue status
citrees-exp check                     # S3 progress snapshot
citrees-exp watch                     # interactive Rich dashboard

# 4. Tear down when done
citrees-exp infra terminate-workers
citrees-exp infra terminate-api
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    API Server (EC2)                              │
│                    FastAPI + lazy queues                         │
│                                                                 │
│  Startup:                                                       │
│    1. Build full experiment grid                                │
│    2. Check S3 for completed work                               │
│    3. Serve remaining configs via POST /next                    │
│                                                                 │
│  Endpoints:                                                     │
│    POST /next    → returns next config to worker                │
│    GET  /status  → queue status for monitoring                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP
                ┌──────────┼──────────┐
                ▼          ▼          ▼
          ┌──────────┐ ┌──────────┐ ┌──────────┐
          │ Worker 1 │ │ Worker 2 │ │ Worker N │
          │  Docker  │ │  Docker  │ │  Docker  │
          │ m5.8xl   │ │ m5.8xl   │ │ m5.8xl   │
          └────┬─────┘ └────┬─────┘ └────┬─────┘
               │            │            │
               ▼            ▼            ▼
          ┌──────────────────────────────────────┐
          │              S3 Bucket                │
          │  rankings/{task}/{dataset}/...        │
          │  metrics/{task}/{dataset}/...         │
          │  data/{task}/{source}/...             │
          └──────────────────────────────────────┘
```

**API server** (`paper/scripts/api/server.py`): FastAPI app with 4 lazy queues
(rankings/classification, rankings/regression, metrics/classification,
metrics/regression). Queue allocation: 90% rankings, 10% metrics. Skips empty
queues and falls back to the other stage.

**Workers** (`paper/scripts/api/worker.py`): Pull-based loop with graceful
shutdown (SIGINT/SIGTERM). Each worker: calls `POST /next` → deserializes config
→ runs `_run_selection()` or `_run_evaluation()` → saves result to S3 → repeats.
Workers exit when all queues drain or idle timeout is reached.

## Configuration

### Setup Script

```bash
# Full setup (recommended): IAM role + Docker image
citrees-exp infra setup

# What setup does:
# 1. Creates IAM role + instance profile (citrees-worker)
#    Policies: S3, ECR, IAM:PassRole
# 2. Builds Docker image from paper/scripts/infra/docker/Dockerfile
# 3. Pushes to ECR with :latest and :{git_sha} tags

# Individual steps (if you prefer):
citrees-exp infra iam           # IAM role only
citrees-exp infra ecr create    # ECR repo only
citrees-exp infra ecr build     # Build + push Docker image
citrees-exp infra s3            # S3 bucket only
citrees-exp infra upload-data   # Upload datasets to S3
```

### Docker + Instance Profile Credentials

When using Docker on EC2, containers must reach IMDS to fetch instance profile
credentials. The EC2 instances are launched with hop limit 2 (IMDSv2):

```yaml
MetadataOptions:
  HttpTokens: required
  HttpPutResponseHopLimit: 2
```

### Experiment Config (`paper/scripts/infra/config.yaml`)

Create from template: `citrees-exp config init`

```yaml
aws_region: us-east-1

experiment:
  n_seeds: 5
  s3_validate_uploads: true
```

View config: `citrees-exp config show`
Validate: `citrees-exp config validate`

## CLI Commands

### Infrastructure Management

```bash
citrees-exp infra setup              # Full setup (IAM + Docker)
citrees-exp infra iam                # Create IAM role
citrees-exp infra s3                 # Create S3 bucket
citrees-exp infra upload-data        # Upload datasets to S3
citrees-exp infra ecr create         # Create ECR repository
citrees-exp infra ecr build          # Build + push Docker image
citrees-exp infra ecr clean          # Delete all ECR images
```

### API Server Management

```bash
citrees-exp infra launch-api                     # Launch on EC2 (m5.large default)
citrees-exp infra launch-api -i m5.xlarge        # Custom instance type
citrees-exp infra api-url                        # Print running API URL
citrees-exp infra terminate-api                  # Terminate API instance
citrees-exp cluster api-start                    # Start locally
citrees-exp cluster api-status                   # Check queue status
```

### Worker Management

```bash
citrees-exp infra launch-workers --count 5       # Launch 5 EC2 workers
citrees-exp infra launch-workers -n 10 --spot    # Spot instances
citrees-exp infra launch-workers -i m5.4xlarge   # Custom instance type
citrees-exp infra list-workers                   # List running workers
citrees-exp infra terminate-workers              # Terminate all workers
citrees-exp cluster worker-start                 # Start worker locally
```

### Monitoring

```bash
citrees-exp run                       # Poll API server queue progress
citrees-exp run --api-url http://...  # Custom API URL
citrees-exp check                     # S3 progress (rankings, default)
citrees-exp check --stage metrics     # S3 progress (metrics)
citrees-exp check --by-method         # Grouped by method
citrees-exp check --by-dataset        # Grouped by dataset
citrees-exp watch                     # Interactive Rich dashboard
```

### Logs

```bash
citrees-exp infra logs api                  # API server CloudWatch logs
citrees-exp infra logs worker               # Worker CloudWatch logs
citrees-exp infra logs api --tail 50        # Last 50 events
citrees-exp infra logs worker -i i-0abc123  # Specific instance
```

## EC2 Instances

### API Server

- **Default type**: `m5.large`
- **Docker**: Pulls ECR image, runs `citrees-exp cluster api-start` on port 8000
- **CloudWatch**: Log group `/citrees/api`
- **Restart policy**: `--restart unless-stopped`
- **Tagged**: `citrees-role: api`

### Workers

- **Default type**: `m5.8xlarge`
- **Docker**: Pulls ECR image, runs `citrees-exp cluster worker-start`
- **CloudWatch**: Log group `/citrees/worker`
- **Idle timeout**: Exits after 60 consecutive empty-queue polls
- **Auto-terminate**: Container exit triggers instance termination
- **Spot support**: `--spot` flag for cost savings
- **Tagged**: `citrees-role: worker`

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
│   └── {task}/
│       └── {dataset}/
│           ├── {method_id}_seed0.parquet
│           ├── {method_id}_seed1.parquet
│           └── ...
└── metrics/                        # Stage 2 outputs
    └── {task}/
        └── {dataset}/
            ├── {method_id}_seed0.parquet
            └── ...
```

### Dataset Sync

Datasets are **not** baked into the Docker image. Workers download from S3
on-demand to `/tmp/citrees-data/` cache.

```bash
# Upload datasets to S3 (recommended)
citrees-exp infra upload-data

# Or manually
aws s3 sync paper/data/ s3://$S3_BUCKET/data/ --exclude "*.DS_Store"

# Download datasets locally (for development)
aws s3 sync s3://$S3_BUCKET/data/ paper/data/
```

Workers automatically fall back to S3 when local files don't exist:

1. Check local path (`paper/data/...`)
2. If not found, download from S3 to `/tmp/citrees-data/` cache
3. Load from cache

## Skipping and Re-running

### Default: skip completed configs

The API server checks S3 for completed artifacts on startup and excludes them
from the queues. Only incomplete configs are served to workers.

### Re-running specific configs

Delete the specific S3 objects and restart the API server. It will re-discover
the missing artifacts and add them to the queue.

```bash
# Delete specific result
aws s3 rm s3://bucket/rankings/classification/arcene/cit__abc123_seed0.parquet

# Restart API server - it will pick up the gap
citrees-exp infra terminate-api
citrees-exp infra launch-api
```

## Fault Tolerance

- **Worker crash**: Work item is not marked complete in S3. On API server
  restart, it re-appears in the queue.
- **Spot interruption**: Same as crash — restart workers and the API server will
  serve any incomplete configs.
- **API server crash**: Workers will retry connection. Restart the API server;
  it re-scans S3 and rebuilds queues from scratch.

## Cost Management

**Tips:**

- Use `--spot` for workers to reduce costs
- Workers auto-terminate when queues drain (no idle charges)
- Terminate API server and workers when not in use
- Use `citrees-exp infra list-workers` to audit running instances
- Check AWS pricing for current spot rates

## Troubleshooting

### Workers not connecting

```bash
# Check API server is running
citrees-exp infra api-url               # Should print URL
citrees-exp cluster api-status          # Should show queue counts

# Check worker logs
citrees-exp infra logs worker
citrees-exp infra logs worker -i i-0abc123   # Specific instance
```

### Tasks failing

```bash
# Check API server logs for errors
citrees-exp infra logs api

# Check worker logs
citrees-exp infra logs worker --tail 200
```

### Docker image issues

```bash
# Rebuild and push
citrees-exp infra ecr build

# Clean old images
citrees-exp infra ecr clean

# Relaunch workers with new image
citrees-exp infra terminate-workers
citrees-exp infra launch-workers --count 5
```

### Stale results

If workers produced incorrect results, delete the S3 artifacts and restart:

```bash
# Delete all rankings for a specific method
aws s3 rm s3://bucket/rankings/classification/ --recursive --include "*cit__*"

# Restart API + workers
citrees-exp infra terminate-api && citrees-exp infra terminate-workers
citrees-exp infra launch-api
citrees-exp infra launch-workers --count 5
```
