"""Nested CV Server - generates experiment configurations.

FastAPI server that generates method × dataset × seed configurations
for nested CV experiments. Filters already-processed configs from DynamoDB.

Usage:
    TABLE_NAME=Clf AWS_PROFILE=personal uv run uvicorn paper.scripts.nested_cv_server:app --port 8000

Endpoints:
    GET /        - Pop and return next config
    GET /status/ - Return remaining count and host stats
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from loguru import logger

app = FastAPI(title="Nested CV Server")

HERE = Path(__file__).resolve()
DATA_DIR = HERE.parents[1] / "data"
RANDOM_STATE = 1718
SEEDS = list(range(30))  # 30 random seeds for reproducibility

# Classification methods (16)
CLF_METHODS = [
    # Filter (fast, univariate)
    "mc", "mi", "rdc", "mrmr",
    # Permutation tests (medium)
    "ptest_mc", "ptest_mi", "ptest_rdc",
    # Embedding - citrees
    "cit", "cif",
    # Embedding - sklearn/boosting
    "rf", "et", "xgb", "lgbm",
    # Wrapper
    "boruta", "pi", "shap",
]

# Regression methods (13)
REG_METHODS = [
    # Filter (fast, univariate)
    "pc", "dc", "rdc",
    # Permutation tests (medium)
    "ptest_pc", "ptest_dc", "ptest_rdc",
    # Embedding - citrees
    "cit", "cif",
    # Embedding - sklearn/boosting
    "rf", "et", "xgb", "lgbm",
    # Wrapper
    "boruta", "pi",
]

CONFIGS: list[dict[str, Any]] = []
HOSTS: dict[str, int] = defaultdict(int)


def scan_dynamodb_processed(table_name: str, region: str) -> set[int]:
    """Scan DynamoDB table and return set of processed config_idx values."""
    processed = set()
    try:
        dynamodb = boto3.client("dynamodb", region_name=region)
        paginator = dynamodb.get_paginator("scan")
        for page in paginator.paginate(TableName=table_name):
            for item in page.get("Items", []):
                if "config_idx" in item:
                    processed.add(int(item["config_idx"]["N"]))
        logger.info(f"Found {len(processed)} already processed configs in {table_name}")
    except Exception as e:
        logger.warning(f"Could not scan DynamoDB table {table_name}: {e}")
    return processed


@app.on_event("startup")
def create_configurations() -> None:
    """Generate all configurations on server startup."""
    global CONFIGS

    table_prefix = os.environ.get("TABLE_NAME", "Clf")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    # Determine task type from table prefix
    if table_prefix.lower().startswith("clf"):
        task_type = "classification"
        methods = CLF_METHODS
        file_prefix = "clf_"
    else:
        task_type = "regression"
        methods = REG_METHODS
        file_prefix = "reg_"

    # Load dataset metadata
    files = [f for f in os.listdir(DATA_DIR) if f.startswith(file_prefix) and f.endswith(".parquet")]
    if not files:
        logger.warning(f"No {file_prefix}*.parquet files found in {DATA_DIR}")
        return

    datasets = []
    for f in files:
        df = pd.read_parquet(DATA_DIR / f)
        dataset_name = f.replace(file_prefix, "").replace(".snappy.parquet", "").replace(".parquet", "")
        ds_info = {
            "dataset": dataset_name,
            "n_samples": df.shape[0],
            "n_features": df.shape[1] - 1,  # Exclude target column
        }
        if task_type == "classification":
            ds_info["n_classes"] = len(df["y"].unique())
        datasets.append(ds_info)
        logger.info(f"Loaded metadata for {dataset_name}: {ds_info}")

    # Generate all configs: method × dataset × seed
    config_idx = 0
    for method in methods:
        for ds in datasets:
            for seed in SEEDS:
                config = {
                    "config_idx": config_idx,
                    "task_type": task_type,
                    "method": method,
                    "random_state": seed,
                    **ds,
                }
                CONFIGS.append(config)
                config_idx += 1

    total_configs = len(CONFIGS)
    logger.info(f"Generated {total_configs} configs ({len(methods)} methods × {len(datasets)} datasets × {len(SEEDS)} seeds)")

    # Filter already processed configs
    table_name = f"{table_prefix}NestedCV"
    processed = scan_dynamodb_processed(table_name, region)
    if processed:
        CONFIGS[:] = [c for c in CONFIGS if c["config_idx"] not in processed]
        logger.info(f"Filtered to {len(CONFIGS)} remaining configs (removed {total_configs - len(CONFIGS)} already processed)")

    # Shuffle for load balancing
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(CONFIGS)

    logger.info(f"Server ready with {len(CONFIGS)} configs for {task_type}")


@app.get("/")
async def get_config(request: Request) -> dict[str, Any]:
    """Pop and return the next configuration."""
    if CONFIGS:
        HOSTS[request.client.host] += 1
        return CONFIGS.pop()
    return {}


@app.get("/status/")
async def get_status() -> dict[str, Any]:
    """Return server status."""
    return {
        "n_configs_remaining": len(CONFIGS),
        "hosts": dict(HOSTS),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
