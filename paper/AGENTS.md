# AI Assistant Guide for citrees — Paper & Experiments

This guide covers the research paper's experiment infrastructure under `paper/`.
For library-level guidance (the `citrees` package, tests, code style, research
roadmap), see the root `AGENTS.md`.

# Experiment CLI (`citrees-exp`)

The `citrees-exp` CLI manages the research paper's experiment infrastructure.
Defined in `pyproject.toml` as:

```toml
[project.scripts]
citrees-exp = "paper.benchmark.cli.entrypoint:main"
```

Install with the `paper` dependency group:

```bash
uv sync --group paper
```

## CLI Command Reference

### Top-Level Commands

| Command             | Description                             |
| ------------------- | --------------------------------------- |
| `citrees-exp run`   | Poll API server for live queue progress |
| `citrees-exp smoke` | Quick local smoke test (no API needed)  |
| `citrees-exp check` | Reconcile manifest artifacts in S3      |

### `config` Subgroup

| Command                       | Description                     |
| ----------------------------- | ------------------------------- |
| `citrees-exp config show`     | Display current config          |
| `citrees-exp config init`     | Initialize config from template |
| `citrees-exp config validate` | Validate config schema          |
| `citrees-exp config path`     | Show config file paths          |

### `list` Subgroup

| Command                     | Description                    |
| --------------------------- | ------------------------------ |
| `citrees-exp list datasets` | List available datasets        |
| `citrees-exp list methods`  | List feature selection methods |

### `infra` Subgroup (AWS)

| Command                               | Description                        |
| ------------------------------------- | ---------------------------------- |
| `citrees-exp infra setup`             | Create S3 and build Docker image   |
| `citrees-exp infra s3`                | Create S3 bucket                   |
| `citrees-exp infra upload-data`       | Upload datasets to S3              |
| `citrees-exp infra ecr create`        | Create ECR repository              |
| `citrees-exp infra ecr build`         | Build + push Docker image to ECR   |
| `citrees-exp infra ecr clean`         | Delete all ECR images              |
| `citrees-exp infra launch-api`        | Launch API server on EC2           |
| `citrees-exp infra api-url`           | Print exact campaign API URL       |
| `citrees-exp infra terminate-api`     | Terminate exact campaign API       |
| `citrees-exp infra launch-workers`    | Launch EC2 worker instances        |
| `citrees-exp infra list-workers`      | List running worker instances      |
| `citrees-exp infra terminate-workers` | Terminate all workers              |
| `citrees-exp infra logs`              | Fetch CloudWatch logs (api/worker) |

### `cluster` Subgroup (Local Processes)

| Command                            | Description                    |
| ---------------------------------- | ------------------------------ |
| `citrees-exp cluster api-start`    | Start API queue server locally |
| `citrees-exp cluster api-status`   | Show API queue status          |
| `citrees-exp cluster worker-start` | Start worker process locally   |

## Two-Stage Pipeline

```
Stage 1: Feature Selection (pipeline/stage1.py)
  Input:  dataset + method config
  Output: s3://bucket/rankings/{task}/{dataset}/{method_id}_seed{s}.parquet

Stage 2: Downstream Evaluation (pipeline/stage2.py)
  Input:  rankings from Stage 1
  Output: s3://bucket/metrics/{task}/{dataset}/{method_id}_seed{s}.parquet
  Evaluates at k = [5, 10, 25, 50, 100, all]
  Downstream models: LR, SVM, KNN (clf) / Ridge, SVR, KNN (reg)
```

## Distributed Architecture (API Server + EC2 Workers)

The experiment infrastructure uses a pull-based API server model:

Each distributed launch derives a deterministic IAM instance profile from the
campaign digest and exact output prefix. Runtime instances can write only below
that one prefix.

```
┌─────────────────────┐      ┌───────────────────────────────┐
│   API Server (EC2)  │◄────►│      S3 Bucket                │
│   FastAPI + queues  │      │  rankings/ + metrics/         │
│   POST /next        │      └───────────────────────────────┘
│   GET  /status      │
└──────────┬──────────┘
           │  HTTP
    ┌──────┴──────┐
    ▼             ▼
┌────────┐  ┌────────┐
│Worker 1│  │Worker N│   EC2 instances (m5.8xlarge)
│ Docker │  │ Docker │   Pull config → execute → save to S3
└────────┘  └────────┘
```

**API server** (`paper/benchmark/api/server.py`): FastAPI app with 4 lazy queues
(rankings/classification, rankings/regression, metrics/classification,
metrics/regression). On startup it builds the full experiment grid and subtracts
completed S3 artifacts. Workers call `POST /next` to get work.

**Worker** (`paper/benchmark/api/worker.py`): Pull-based loop. Gets config from
API, runs `_run_selection()` or `_run_evaluation()`, saves result to S3, repeats
until queues drain or idle timeout.

## Adapters

| Module               | Purpose                                             |
| -------------------- | --------------------------------------------------- |
| `adapters/data.py`   | Dataset loading (local filesystem, S3 fallback)     |
| `adapters/runner.py` | Execution interface (`LocalRunner` for smoke tests) |
| `adapters/store.py`  | S3 artifact storage (save/load/exists/list)         |

## Pipeline Types

| Type               | Location            | Description                                      |
| ------------------ | ------------------- | ------------------------------------------------ |
| `MethodConfig`     | `pipeline/types.py` | Frozen dataclass: method + params                |
| `ExperimentConfig` | `pipeline/types.py` | Frozen dataclass: method + dataset + seed + task |
| `Result`           | `pipeline/types.py` | Mutable dataclass: config + status + data        |
| `ExperimentGrid`   | `pipeline/grid.py`  | Grid builder from CLI args                       |

## Method Categories

Methods are defined in `paper/benchmark/pipeline/methods.py`:

| Category   | Classification                                                         | Regression                                                             |
| ---------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Perm. test | `ptest_mc`, `ptest_rdc`                                                | `ptest_pc`, `ptest_dc`, `ptest_rdc`                                    |
| Embedding  | `cit`, `cif`, `rf`, `et`, `xgb`, `lgbm`, `cat`, `r_ctree`, `r_cforest` | `cit`, `cif`, `rf`, `et`, `xgb`, `lgbm`, `cat`, `r_ctree`, `r_cforest` |
| Wrapper    | `boruta`, `pi`, `cpi`, `rfe`                                           | `boruta`, `pi`, `cpi`, `rfe`                                           |

## Configuration

**Config file**: `paper/benchmark/infra/config.yaml` (created via
`citrees-exp config init` from `config.example.yaml`)

**Key settings** (`paper/benchmark/config/settings.py`):

- `aws_region`: Default `us-east-1`
- `s3_bucket`: Auto-derived as `citrees-{account_id}`
- `experiment.n_seeds`: Default 5
- `experiment.s3_validate_uploads`: Default True

**Constants** (`paper/benchmark/config/constants.py`):

- `RANDOM_STATE`: 1718
- `N_SEEDS`: 5, `N_SPLITS`: 5
- `CLF_DOWNSTREAM_MODELS`: `["lr", "svm", "knn"]`
- `REG_DOWNSTREAM_MODELS`: `["ridge", "svr", "knn"]`
- `EVALUATION_K_VALUES`: `[5, 10, 25, 50, 100]`

## Typical Workflow

```bash
# 1. Setup infrastructure (one-time)
citrees-exp config init
citrees-exp infra setup           # S3 + immutable Docker image
citrees-exp infra upload-data     # Upload datasets

# 2. Launch API server + workers on EC2
citrees-exp infra launch-api
citrees-exp infra launch-workers --count 5   # auto-discovers API private IP

# 3. Monitor progress
citrees-exp run                                      # poll queue status
citrees-exp check --manifest scratch/manifest.csv    # exact S3 reconciliation

# 4. Tear down
citrees-exp infra terminate-workers
citrees-exp infra terminate-api \
    --artifact-prefix "$CITREES_ARTIFACT_PREFIX" \
    --campaign-sha256 "$CITREES_CAMPAIGN_SHA256" \
    --stage rankings
```
