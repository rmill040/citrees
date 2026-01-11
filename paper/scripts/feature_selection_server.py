"""
Stage 1: Feature Selection Server

FastAPI server that distributes feature selection configs to workers.
Tracks completion via DynamoDB and enables resume from crashes.

Usage:
    S3_BUCKET=citrees-results TABLE_NAME=ClfFeatureSelection AWS_PROFILE=personal \
        uv run uvicorn paper.scripts.feature_selection_server:app --host 0.0.0.0 --port 8000
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
TABLE_NAME = os.environ.get("TABLE_NAME", "ClfFeatureSelection")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
STALE_TIMEOUT_MINUTES = 30
N_SEEDS = 30

# Classification methods
CLF_METHODS = [
    # Filter methods
    "mc", "mi", "rdc", "mrmr",
    # Permutation test methods
    "ptest_mc", "ptest_mi", "ptest_rdc",
    # Embedding methods (tree-based)
    "cit", "cif", "rf", "et", "xgb", "lgbm",
    # Wrapper methods
    "boruta", "pi", "shap",
]

# Regression methods
REG_METHODS = [
    # Filter methods
    "pc", "dc", "rdc",
    # Permutation test methods
    "ptest_pc", "ptest_dc", "ptest_rdc",
    # Embedding methods (tree-based)
    "cit", "cif", "rf", "et", "xgb", "lgbm",
    # Wrapper methods
    "boruta", "pi",
]

# ============================================================================
# Globals
# ============================================================================

app = FastAPI(title="Feature Selection Server")
configs: list[dict[str, Any]] = []
config_lock = Lock()
host_counts: dict[str, int] = defaultdict(int)

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
                s3_path = f"s3://{S3_BUCKET}/rankings/{task_type}/{dataset}/{method}_seed{seed}.parquet"
                all_configs.append({
                    "config_idx": config_idx,
                    "task_type": task_type,
                    "method": method,
                    "dataset": dataset,
                    "seed": seed,
                    "s3_path": s3_path,
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
            FilterExpression="status = :s",
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
    """Reset configs that have been pending too long (worker probably died)."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STALE_TIMEOUT_MINUTES)
    cutoff_str = cutoff.isoformat()
    reset_count = 0

    try:
        paginator = dynamodb.meta.client.get_paginator("scan")
        for page in paginator.paginate(
            TableName=TABLE_NAME,
            FilterExpression="status = :s AND started_at < :t",
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
    """Load configs on startup, filtering out completed ones."""
    global configs

    # Reset stale pending configs
    reset_count = reset_stale_pending()
    if reset_count:
        logger.info(f"Reset {reset_count} stale pending configs")

    # Get completed configs
    completed = get_completed_configs()
    logger.info(f"Found {len(completed)} completed configs in DynamoDB")

    # Generate all configs and filter
    all_configs = generate_all_configs()
    configs = [
        c for c in all_configs
        if (c["dataset"], c["method"], c["seed"]) not in completed
    ]

    # Shuffle for load balancing
    random.shuffle(configs)

    logger.info(f"Total configs: {len(all_configs)}")
    logger.info(f"Remaining configs: {len(configs)}")


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
