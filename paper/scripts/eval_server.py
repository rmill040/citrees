"""
Stage 2: Downstream Evaluation Server

FastAPI server that distributes downstream evaluation configs to workers.
Only serves configs that have completed Stage 1 (rankings exist in S3).

Usage:
    S3_BUCKET=citrees-results TABLE_NAME=ClfDownstreamEval AWS_PROFILE=personal \
        uv run uvicorn paper.scripts.downstream_eval_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import boto3
from fastapi import FastAPI, Response
from loguru import logger

# ============================================================================
# Configuration
# ============================================================================

S3_BUCKET = os.environ.get("S3_BUCKET", "citrees-results")
TABLE_NAME = os.environ.get("TABLE_NAME", "ClfDownstreamEval")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
STALE_TIMEOUT_MINUTES = 60  # Longer timeout for Stage 2 (more work per config)
N_SEEDS = 30

# Classification methods
CLF_METHODS = [
    "mc", "mi", "rdc", "mrmr",
    "ptest_mc", "ptest_mi", "ptest_rdc",
    "cit", "cif", "rf", "et", "xgb", "lgbm",
    "boruta", "pi", "shap",
]

# Regression methods
REG_METHODS = [
    "pc", "dc", "rdc",
    "ptest_pc", "ptest_dc", "ptest_rdc",
    "cit", "cif", "rf", "et", "xgb", "lgbm",
    "boruta", "pi",
]

# ============================================================================
# Globals
# ============================================================================

app = FastAPI(title="Downstream Evaluation Server")
configs: list[dict[str, Any]] = []
config_lock = Lock()
host_counts: dict[str, int] = defaultdict(int)

s3 = boto3.client("s3", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(TABLE_NAME)


# ============================================================================
# Config Generation
# ============================================================================

def get_datasets(task_type: str) -> list[str]:
    """Get dataset names from paper/data directory."""
    data_dir = Path(__file__).resolve().parents[1] / "data"
    prefix = "clf_" if task_type == "classification" else "reg_"
    datasets = []
    for f in data_dir.glob(f"{prefix}*.parquet"):
        name = f.stem.replace(prefix, "").replace(".snappy", "")
        datasets.append(name)
    return sorted(datasets)


def s3_file_exists(s3_path: str) -> bool:
    """Check if S3 file exists."""
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1]
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except:
        return False


def list_s3_rankings(task_type: str) -> set[tuple[str, str, int]]:
    """List all completed rankings in S3."""
    completed = set()
    prefix = f"rankings/{task_type}/"

    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                # Parse: rankings/classification/gisette/mc_seed0.parquet
                key = obj["Key"]
                parts = key.split("/")
                if len(parts) >= 4:
                    dataset = parts[2]
                    filename = parts[3]
                    # Parse: mc_seed0.parquet
                    method_seed = filename.replace(".parquet", "")
                    if "_seed" in method_seed:
                        method, seed_str = method_seed.rsplit("_seed", 1)
                        seed = int(seed_str)
                        completed.add((dataset, method, seed))
    except Exception as e:
        logger.warning(f"Error listing S3: {e}")

    return completed


def generate_all_configs() -> list[dict[str, Any]]:
    """Generate all method × dataset × seed configurations."""
    all_configs = []
    config_idx = 0

    # Determine task type from TABLE_NAME
    task_type = "classification" if "Clf" in TABLE_NAME else "regression"
    methods = CLF_METHODS if task_type == "classification" else REG_METHODS
    datasets = get_datasets(task_type)

    logger.info(f"Task type: {task_type}")
    logger.info(f"Methods: {len(methods)}")
    logger.info(f"Datasets: {len(datasets)}")
    logger.info(f"Seeds: {N_SEEDS}")

    for method in methods:
        for dataset in datasets:
            for seed in range(N_SEEDS):
                rankings_s3_path = f"s3://{S3_BUCKET}/rankings/{task_type}/{dataset}/{method}_seed{seed}.parquet"
                metrics_s3_path = f"s3://{S3_BUCKET}/metrics/{task_type}/{dataset}/{method}_seed{seed}.parquet"
                all_configs.append({
                    "config_idx": config_idx,
                    "task_type": task_type,
                    "method": method,
                    "dataset": dataset,
                    "seed": seed,
                    "rankings_s3_path": rankings_s3_path,
                    "metrics_s3_path": metrics_s3_path,
                })
                config_idx += 1

    return all_configs


def get_completed_configs() -> set[tuple[str, str, int]]:
    """Query DynamoDB for completed configurations."""
    completed = set()

    try:
        paginator = dynamodb.meta.client.get_paginator("scan")
        for page in paginator.paginate(
            TableName=TABLE_NAME,
            FilterExpression="#s = :s",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": {"S": "completed"}},
            ProjectionExpression="dataset, method, seed",
        ):
            for item in page.get("Items", []):
                key = (
                    item["dataset"]["S"],
                    item["method"]["S"],
                    int(item["seed"]["N"]),
                )
                completed.add(key)
    except Exception as e:
        logger.warning(f"Error querying DynamoDB: {e}")

    return completed


def reset_stale_pending() -> int:
    """Reset configs that have been pending too long."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STALE_TIMEOUT_MINUTES)
    cutoff_str = cutoff.isoformat()
    reset_count = 0

    try:
        paginator = dynamodb.meta.client.get_paginator("scan")
        for page in paginator.paginate(
            TableName=TABLE_NAME,
            FilterExpression="#s = :s AND started_at < :t",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":s": {"S": "pending"},
                ":t": {"S": cutoff_str},
            },
            ProjectionExpression="config_idx",
        ):
            for item in page.get("Items", []):
                config_idx = int(item["config_idx"]["N"])
                table.delete_item(Key={"config_idx": config_idx})
                reset_count += 1
    except Exception as e:
        logger.warning(f"Error resetting stale pending: {e}")

    return reset_count


# ============================================================================
# Startup
# ============================================================================

@app.on_event("startup")
async def startup():
    """Load configs on startup."""
    global configs

    # Reset stale pending configs
    reset_count = reset_stale_pending()
    if reset_count:
        logger.info(f"Reset {reset_count} stale pending configs")

    # Determine task type
    task_type = "classification" if "Clf" in TABLE_NAME else "regression"

    # Get configs that have Stage 1 rankings in S3
    stage1_completed = list_s3_rankings(task_type)
    logger.info(f"Found {len(stage1_completed)} Stage 1 rankings in S3")

    # Get configs already completed in Stage 2
    stage2_completed = get_completed_configs()
    logger.info(f"Found {len(stage2_completed)} Stage 2 completed in DynamoDB")

    # Generate all configs
    all_configs = generate_all_configs()

    # Filter: must have Stage 1 done, must not have Stage 2 done
    configs = [
        c for c in all_configs
        if (c["dataset"], c["method"], c["seed"]) in stage1_completed
        and (c["dataset"], c["method"], c["seed"]) not in stage2_completed
    ]

    # Shuffle for load balancing
    random.shuffle(configs)

    logger.info(f"Total possible configs: {len(all_configs)}")
    logger.info(f"Ready for Stage 2: {len(configs)}")


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def get_config(response: Response, host: str = "unknown"):
    """Pop and return next config, or 204 if no more work."""
    with config_lock:
        if not configs:
            response.status_code = 204
            return None

        config = configs.pop()
        host_counts[host] += 1

    return config


@app.get("/status/")
async def get_status():
    """Return current status."""
    with config_lock:
        return {
            "n_configs_remaining": len(configs),
            "hosts": dict(host_counts),
            "table_name": TABLE_NAME,
            "s3_bucket": S3_BUCKET,
        }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
