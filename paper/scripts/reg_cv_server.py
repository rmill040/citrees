"""Regression metrics - SERVER."""
import concurrent.futures
import itertools
import json
import os
from collections import defaultdict
from decimal import Decimal
from typing import Any, Dict

import boto3
import numpy as np
from boto3.dynamodb.types import TypeDeserializer
from fastapi import FastAPI, Request
from loguru import logger

app = FastAPI()

CONFIGS = []
HOSTS = defaultdict(lambda: 0)
RANDOM_STATE = 1718


def parallel_scan_table(dynamo_client: Any, *, TableName: str, **kwargs: Dict[str, Any]) -> None:
    """Generates all the items in a DynamoDB table."""
    # How many segments to divide the table into?  As long as this is >= to the
    # number of threads used by the ThreadPoolExecutor, the exact number doesn't
    # seem to matter.
    total_segments = 25

    # How many scans to run in parallel?  If you set this really high you could
    # overwhelm the table read capacity, but otherwise I don't change this much.
    max_scans_in_parallel = 5

    # Schedule an initial scan for each segment of the table.  We read each
    # segment in a separate thread, then look to see if there are more rows to
    # read -- and if so, we schedule another scan.
    tasks_to_do = [
        {
            **kwargs,
            "TableName": TableName,
            "Segment": segment,
            "TotalSegments": total_segments,
        }
        for segment in range(total_segments)
    ]

    # Make the list an iterator, so the same tasks don't get run repeatedly.
    scans_to_run = iter(tasks_to_do)

    with concurrent.futures.ThreadPoolExecutor() as executor:

        # Schedule the initial batch of futures.  Here we assume that
        # max_scans_in_parallel < total_segments, so there's no risk that
        # the queue will throw an Empty exception.
        futures = {
            executor.submit(dynamo_client.scan, **scan_params): scan_params
            for scan_params in itertools.islice(scans_to_run, max_scans_in_parallel)
        }

        while futures:
            # Wait for the first future to complete.
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

            for fut in done:
                yield from fut.result()["Items"]

                scan_params = futures.pop(fut)

                # A Scan reads up to N items, and tells you where it got to in
                # the LastEvaluatedKey.  You pass this key to the next Scan operation,
                # and it continues where it left off.
                try:
                    scan_params["ExclusiveStartKey"] = fut.result()["LastEvaluatedKey"]
                except KeyError:
                    break
                tasks_to_do.append(scan_params)

            # Schedule the next batch of futures.  At some point we might run out
            # of entries in the queue if we've finished scanning the table, so
            # we need to spot that and not throw.
            for scan_params in itertools.islice(scans_to_run, len(done)):
                futures[executor.submit(dynamo_client.scan, **scan_params)] = scan_params


class DecimalEncoder(json.JSONEncoder):
    """Handle decimal data."""

    def default(self, obj: Any) -> str:
        """Cast decimal types."""
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)


@app.on_event("startup")
def create_configurations() -> None:
    """Generate configurations for feature selection."""
    global CONFIGS

    # Populate configs
    deserializer = TypeDeserializer()
    dynamodb = boto3.client("dynamodb", region_name="us-east-1")
    for j, config in enumerate(parallel_scan_table(dynamodb, TableName=os.environ["TABLE_NAME"]), 1):
        if j % 10_000 == 0:
            logger.info(f"{j} configs processed for testing feature selection")
        config = {k: deserializer.deserialize(v) for k, v in config.items()}
        config = json.loads(json.dumps(config, cls=DecimalEncoder))
        config["config_idx"] = int(config["config_idx"])
        CONFIGS.append(config)

    # Pull all items from DynamoDB and see what has already been processed
    processed = set()
    for config in parallel_scan_table(dynamodb, TableName=os.environ["TABLE_NAME"] + "Metrics"):
        if len(processed) % 10_000 == 0:
            logger.info(f"{len(processed)} configs already processed from feature selection metrics table")
        processed.add(int(config["config_idx"]["N"]))

    if processed:
        logger.info(f"Already processed ({len(processed)}) configurations, removing from list")
        CONFIGS = list(filter(lambda config: config["config_idx"] not in processed, CONFIGS))

    logger.info(f"Server ready with ({len(CONFIGS)}) configurations for feature selection")

    # Random permutation
    prng = np.random.RandomState(RANDOM_STATE)
    CONFIGS = prng.permutation(CONFIGS).tolist()


@app.get("/")
async def get_config(request: Request) -> Dict[str, Any]:
    """Get configuration for feature selection."""
    if len(CONFIGS):
        HOSTS[request.client.host] += 1
        return CONFIGS.pop()
    else:
        return {}


@app.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get status of feature selection."""
    return {"n_configs_remaining": len(CONFIGS), "hosts": HOSTS}
