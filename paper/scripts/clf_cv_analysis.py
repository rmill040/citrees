"""Classifier metrics - ANALYSIS."""
import json
import os
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from boto3.dynamodb.types import TypeDeserializer
from loguru import logger


DATA_DIR = Path(os.environ["DATA_DIR"]).resolve()


class DecimalEncoder(json.JSONEncoder):
    """Handle decimal data."""

    def default(self, obj: Any) -> str:
        """Cast decimal types."""
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)


def format_raw_data() -> None:
    """Format raw data dump from DynamoDB and save as CSV files."""
    deserde = TypeDeserializer()
    results = defaultdict(list)
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    for file in files:
        logger.info(f"Processing file ({file})")
        for j, line in enumerate(open(DATA_DIR / file), 1):
            if j % 25_000 == 0:
                logger.info(f"Processing row ({j})")
            row = json.loads(line)["Item"]
            row.pop("feature_ranks")
            for key, value in row.items():
                row[key] = deserde.deserialize(value)
            row = json.loads(json.dumps(row, cls=DecimalEncoder))
            for key, value in row["metrics"].items():
                if key == "feature_ranks":
                    continue
                dtype = int if key == "n_features_used" else float
                row["metrics"][key] = list(map(dtype, value))
            results[row["method"]].append(row)

    total = sum([len(results[key]) for key in results.keys()])
    logger.info(f"{total} total configurations processed for feature selection")

    keys = list(results.keys())
    for key in keys:
        logger.info(f"Writing dataset ({key}) to disk")
        df = pd.json_normalize(results.pop(key)).fillna("None")
        df.to_csv(DATA_DIR / (key + ".csv"), index=False)


if __name__ == "__main__":
    if bool(os.environ.get("GET_DATA")):
        format_raw_data()

    # TODO: Add other stuff here
