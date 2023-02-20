"""Classifier metrics - ANALYSIS."""
import os
import json
from collections import defaultdict
from typing import Any, Dict

from joblib import delayed, Parallel
import pandas as pd


def cast_ddb_dtypes(node: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively cast data types for DDB data."""
    if type(node) is dict:
        for key in node.keys():
            if "N" in node[key]:
                value = float(node[key]["N"])
                if value.is_integer():
                    value = int(value)
                node[key] = value
            elif "S" in node[key]:
                node[key] = str(node[key]["S"])
            elif "NULL" in node[key]:
                node[key] = bool(node[key]["NULL"])
            elif "M" in node[key]:
                node[key] = node[key]["M"]
                cast_ddb_dtypes(node[key])
            else:
                pass
    
    return node


data = defaultdict(list)


def read(f):
    tmp = defaultdict(list)
    for line in open(f):
        item = cast_ddb_dtypes(json.loads(line)["Item"])
        tmp[item["method"] + "_" + item["dataset"]].append(item)
    return tmp
        

files = [f for f in os.listdir(".") if f.endswith(".json")]
results = Parallel(n_jobs=16, verbose=50, backend="loky")(delayed(read)(f) for f in files)

for result in results:
    for k, v in result.items():
        data[k].extend(v)


for j, key in enumerate(data.keys(), 1):
    path = "csv/" + key + ".csv"
    df = pd.json_normalize(data[key])
    df = df.fillna(value="None")
    df.to_csv(path, index=False)
    del df