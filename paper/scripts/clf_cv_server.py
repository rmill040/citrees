"""Classifier metrics - SERVER."""
import boto3
from boto3.dynamodb.types import TypeDeserializer
from collections import defaultdict
from copy import deepcopy
from decimal import Decimal
from fastapi import FastAPI, Request
import json
from loguru import logger
import os
import numpy as np
from typing import Any, Dict


app = FastAPI()


CONFIGS = []
HOSTS = defaultdict(lambda: 0)
CACHE = {}
RANDOM_STATE = 1718
CACHE_HITS = 0


class DecimalEncoder(json.JSONEncoder):
    """Handle decimal data."""

    def default(self, obj: Any) -> str:
        """Cast decimal types."""
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)


def create_key(config: Dict[str, Any]) -> str:
    """Create key for cache."""
    return config["dataset"] + "," + config["feature_ranks"]


@app.on_event("startup")
def create_configurations() -> None:
    """Generate configurations for feature selection."""
    global CONFIGS

    # Populate configs
    deserializer = TypeDeserializer()
    ddb_paginator = boto3.client("dynamodb", region_name="us-east-1").get_paginator("scan")
    config_idx = 0
    configs = []
    for j, page in enumerate(ddb_paginator.paginate(TableName=os.environ["TABLE_NAME"]), 1):
        if j > 1:
            break
        if len(configs):
            logger.info(f"Page {j} of DDB data: {len(configs)} configs loaded")
        for config in page["Items"]:
            # Format and update config
            config = {k: deserializer.deserialize(v) for k, v in config.items()}
            config = json.loads(json.dumps(config, cls=DecimalEncoder))

            if int(config["n_features"]) >= 100:
                n_features_to_keep = np.arange(5, 105, 5)
            else:
                n_features_to_keep = np.arange(1, int(config["n_features"]) + 1)

            feature_ranks = config["feature_ranks"].split(",")
            for n_features in n_features_to_keep:
                new_config = deepcopy(config)
                config_idx += 1
                new_config["config_idx"] = config_idx
                new_config["feature_ranks"] = ",".join(feature_ranks[:n_features])
                new_config["n_features_used"] = int(n_features)
                CONFIGS.append(new_config)

    logger.info(f"Server ready with ({len(CONFIGS)}) configurations for feature selection")

    # Random permutation
    prng = np.random.RandomState(RANDOM_STATE)
    CONFIGS = prng.permutation(CONFIGS).tolist()


@app.get("/")
async def get_config(request: Request) -> Dict[str, Any]:
    """Get configuration for feature selection."""
    global CACHE_HITS

    if len(CONFIGS):
        HOSTS[request.client.host] += 1
        config = CONFIGS.pop()
        key = create_key(config)
        if key in CACHE:
            logger.info("Cache hit, config already processed!")
            CACHE_HITS += 1
        return CACHE.get(key, config)
    else:
        return {}


@app.post("/cache")
async def cache_results(request: Request) -> None:
    """Update cache with metric results."""
    req = await request.json()
    key = req.pop("key")
    results = req.pop("results")
    if key not in CACHE:
        CACHE[key] = results


@app.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get status of feature selection."""
    return dict(n_configs_remaining=len(CONFIGS), hosts=HOSTS, cache_hits=CACHE_HITS)
