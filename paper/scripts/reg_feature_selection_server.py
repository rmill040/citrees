"""Regression experiments - SERVER."""
import inspect
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import boto3
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from loguru import logger

from citrees._registry import Registry

app = FastAPI()


HERE = Path(__file__).resolve()
DATA_DIR = HERE.parents[1] / "data"
FILES = [f for f in os.listdir(DATA_DIR) if f.startswith("reg_")]

METHODS = Registry("Methods")
RANDOM_STATE = 1718
HOSTS = defaultdict(lambda: 0)
CONFIGS = []


@METHODS.register("dc")
def dc() -> List[Dict[str, Any]]:
    """Distance correlation hyperparameters."""
    method = inspect.currentframe().f_code.co_name

    params = [{"standardize": True, "random_state": RANDOM_STATE, "method": method}]

    return params


@METHODS.register("pc")
def pc() -> List[Dict[str, Any]]:
    """Pearson correlation hyperparameters."""
    method = inspect.currentframe().f_code.co_name

    params = [{"standardize": True, "random_state": RANDOM_STATE, "method": method}]

    return params


@METHODS.register("hybrid")
def hybrid() -> List[Dict[str, Any]]:
    """Hybrid selector hyperparameters."""
    method = inspect.currentframe().f_code.co_name

    params = [{"standardize": True, "random_state": RANDOM_STATE, "method": method}]

    return params


@METHODS.register("ptest_dc")
def ptest_dc() -> List[Dict[str, Any]]:
    """Permutation testing with distance correlation hyperparameters."""
    method = inspect.currentframe().f_code.co_name

    params = []
    for alpha in [0.10, 0.05]:
        for n_resamples in ["minimum", "maximum", "auto"]:
            for early_stopping in [True, False]:
                params.append(
                    {
                        "standardize": True,
                        "alpha": alpha,
                        "n_resamples": n_resamples,
                        "early_stopping": early_stopping,
                        "random_state": RANDOM_STATE,
                        "method": method,
                    }
                )

    return params


@METHODS.register("ptest_pc")
def ptest_pc() -> List[Dict[str, Any]]:
    """Permutation testing with Pearson correlation hyperparameters."""
    method = inspect.currentframe().f_code.co_name

    params = []
    for alpha in [0.10, 0.05, 0.01]:
        for n_resamples in ["minimum", "maximum", "auto"]:
            for early_stopping in [True, False]:
                params.append(
                    {
                        "standardize": True,
                        "alpha": alpha,
                        "n_resamples": n_resamples,
                        "early_stopping": early_stopping,
                        "random_state": RANDOM_STATE,
                        "method": method,
                    }
                )

    return params


@METHODS.register("ptest_hybrid")
def ptest_hybrid() -> List[Dict[str, Any]]:
    """Permutation testing with Pearson correlation or distance correlation hyperparameters."""
    method = inspect.currentframe().f_code.co_name

    params = []
    for alpha in [0.10, 0.05]:
        for n_resamples in ["minimum", "maximum", "auto"]:
            for early_stopping in [True, False]:
                params.append(
                    {
                        "standardize": True,
                        "alpha": alpha,
                        "n_resamples": n_resamples,
                        "early_stopping": early_stopping,
                        "random_state": RANDOM_STATE,
                        "method": method,
                    }
                )

    return params


####################
# EMBEDDED METHODS #
####################


@METHODS.register("lr")
def lr() -> List[Dict[str, Any]]:
    """Linear regression hyperparameters."""
    method = inspect.currentframe().f_code.co_name

    params = [{"method": method}]

    return params


@METHODS.register("lr_l1")
def lr_l1() -> List[Dict[str, Any]]:
    """Linear regression with L1 norm hyperparameters."""
    method = inspect.currentframe().f_code.co_name

    params = []
    for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        params.append(
            {
                "alpha": alpha,
                "random_state": RANDOM_STATE,
                "method": method,
            }
        )

    return params


@METHODS.register("lr_l2")
def lr_l2() -> List[Dict[str, Any]]:
    """Linear regression with L2 norm hyperparameters."""
    method = inspect.currentframe().f_code.co_name

    params = []
    for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        params.append(
            {
                "alpha": alpha,
                "random_state": RANDOM_STATE,
                "method": method,
            }
        )

    return params


@METHODS.register("xgb")
def xgb() -> List[Dict[str, Any]]:
    """XGBOOST regressor hyperparameters."""
    method = inspect.currentframe().f_code.co_name

    params = []
    for max_depth in [1, 2, 4, 6, 8]:
        for learning_rate in [0.001, 0.01, 0.1]:
            for subsample in [0.8, 0.9, 1.0]:
                for colsample_bytree in [0.8, 0.9, 1.0]:
                    for reg_alpha in [0.001, 0.01, None]:
                        for reg_lambda in [0.001, 0.01, None]:
                            for importance_type in ["gain", "weight", "cover", "total_gain", "total_cover"]:
                                params.append(
                                    {
                                        "max_depth": max_depth,
                                        "learning_rate": learning_rate,
                                        "subsample": subsample,
                                        "colsample_bytree": colsample_bytree,
                                        "reg_alpha": reg_alpha,
                                        "reg_lambda": reg_lambda,
                                        "importance_type": importance_type,
                                        "n_estimators": 100,
                                        "n_jobs": 1,
                                        "random_state": RANDOM_STATE,
                                        "method": method,
                                    }
                                )

    return params


@METHODS.register("lightgbm")
def lightgbm() -> List[Dict[str, Any]]:
    """LightGBM regressor hyperparameters."""
    method = inspect.currentframe().f_code.co_name

    params = []
    for max_depth in [1, 2, 4, 6, 8]:
        for learning_rate in [0.001, 0.01, 0.1]:
            for subsample in [0.8, 0.9, 1.0]:
                for colsample_bytree in [0.8, 0.9, 1.0]:
                    for reg_alpha in [0.001, 0.01, None]:
                        for reg_lambda in [0.001, 0.01, None]:
                            for importance_type in ["split", "gain"]:
                                params.append(
                                    {
                                        "max_depth": max_depth,
                                        "learning_rate": learning_rate,
                                        "subsample": subsample,
                                        "colsample_bytree": colsample_bytree,
                                        "reg_alpha": reg_alpha,
                                        "reg_lambda": reg_lambda,
                                        "importance_type": importance_type,
                                        "n_estimators": 100,
                                        "n_jobs": 1,
                                        "random_state": RANDOM_STATE,
                                        "method": method,
                                    }
                                )

    return params


@METHODS.register("catboost")
def catboost() -> List[Dict[str, Any]]:
    """CatBoost regressor hyperparameters."""
    method = inspect.currentframe().f_code.co_name

    params = []
    for depth in [1, 2, 4, 6, 8]:
        for learning_rate in [0.001, 0.01, 0.1]:
            for l2_leaf_reg in [1, 3, 5, 7, 9]:
                for colsample_bylevel in [0.8, 0.9, 1.0]:
                    params.append(
                        {
                            "depth": depth,
                            "learning_rate": learning_rate,
                            "l2_leaf_reg": l2_leaf_reg,
                            "colsample_bylevel": colsample_bylevel,
                            "thread_count": 1,
                            "n_estimators": 100,
                            "random_state": RANDOM_STATE,
                            "verbose": 0,
                            "method": method,
                            "allow_writing_files": False,
                        }
                    )

    return params


@METHODS.register("dt")
def dt() -> List[Dict[str, Any]]:
    """Decision tree regressor hyperparameters."""
    method = inspect.currentframe().f_code.co_name

    params = [
        {
            "splitter": "best",
            "random_state": RANDOM_STATE,
            "method": method,
        }
    ]

    return params


@METHODS.register("rt")
def rt() -> List[Dict[str, Any]]:
    """Random decision tree regressor hyperparameters."""
    method = inspect.currentframe().f_code.co_name

    params = [
        {
            "splitter": "random",
            "random_state": RANDOM_STATE,
            "method": method,
        }
    ]

    return params


@METHODS.register("rf")
def rf() -> List[Dict[str, Any]]:
    """Random forest regressor hyperparameters."""
    method = inspect.currentframe().f_code.co_name

    params = []
    for max_samples in [None, 0.8]:
        params.append(
            {
                "max_samples": max_samples,
                "n_estimators": 100,
                "n_jobs": 1,
                "random_state": RANDOM_STATE,
                "method": method,
            }
        )

    return params


@METHODS.register("et")
def et() -> List[Dict[str, Any]]:
    """Extra trees regressor hyperparameters."""
    method = inspect.currentframe().f_code.co_name

    params = [
        {
            "n_estimators": 100,
            "n_jobs": 1,
            "random_state": RANDOM_STATE,
            "method": method,
        }
    ]

    return params


def _filter_param_conflicts(params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out hyperparameters with conflicting settings."""
    # SELECTOR constraints

    # 1. No permutation tests (value == None) =>
    # - No alpha adjustment
    # - No feature muting
    # - No early stopping
    params = list(
        filter(
            lambda p: not (
                p["n_resamples_selector"] is None
                and (p["adjust_alpha_selector"] or p["feature_muting"] or p["early_stopping_selector"])
            ),
            params,
        )
    )

    # 2. No early stopping (value == False) =>
    # - No feature scanning
    params = list(filter(lambda p: not (not p["early_stopping_selector"] and p["feature_scanning"]), params))

    # SPLITTER constraints
    # 1. No permutation test (value == None) =>
    # - No alpha adjustment
    # - No early stopping
    params = list(
        filter(
            lambda p: not (
                p["n_resamples_splitter"] is None and (p["adjust_alpha_splitter"] or p["early_stopping_splitter"])
            ),
            params,
        )
    )

    # 2. No early stopping (value == False) =>
    # - No threshold scanning
    params = list(filter(lambda p: not (not p["early_stopping_splitter"] and p["threshold_scanning"]), params))

    return params


@METHODS.register("cit")
def cit() -> List[Dict[str, Any]]:
    """Conditional inference tree regressor hyperparameters."""
    method = inspect.currentframe().f_code.co_name

    params = []
    for n_resamples_selector in ["minimum", "maximum", "auto", None]:
        for n_resamples_splitter in ["minimum", "maximum", "auto", None]:
            for adjust_alpha_selector in [True, False]:
                for adjust_alpha_splitter in [True, False]:
                    for feature_scanning in [True, False]:
                        for threshold_scanning in [True, False]:
                            for threshold_method in [
                                "exact",
                                "random",
                                "percentile",
                                "histogram",
                            ]:
                                hyperparameters = {
                                    "n_resamples_selector": n_resamples_selector,
                                    "n_resamples_splitter": n_resamples_splitter,
                                    "adjust_alpha_selector": adjust_alpha_selector,
                                    "adjust_alpha_splitter": adjust_alpha_splitter,
                                    "feature_scanning": feature_scanning,
                                    "threshold_scanning": threshold_scanning,
                                    "threshold_method": threshold_method,
                                    "early_stopping_selector": True,
                                    "early_stopping_splitter": True,
                                    "feature_muting": True,
                                    "random_state": RANDOM_STATE,
                                    "verbose": 0,
                                    "method": method,
                                }

                                if threshold_method == "exact":
                                    for max_thresholds in [None]:
                                        hyperparameters["max_thresholds"] = max_thresholds
                                        params.append(hyperparameters)
                                elif threshold_method == "random":
                                    for max_thresholds in [0.5, 0.8]:
                                        hyperparameters = deepcopy(hyperparameters)
                                        hyperparameters["max_thresholds"] = max_thresholds
                                        params.append(hyperparameters)
                                elif threshold_method == "percentile":
                                    for max_thresholds in [10, 50]:
                                        hyperparameters = deepcopy(hyperparameters)
                                        hyperparameters["max_thresholds"] = max_thresholds
                                        params.append(hyperparameters)
                                else:
                                    # Histogram method
                                    for max_thresholds in [128, 256]:
                                        hyperparameters = deepcopy(hyperparameters)
                                        hyperparameters["max_thresholds"] = max_thresholds
                                        params.append(hyperparameters)

    # Filter out bad combinations of parameters
    params = _filter_param_conflicts(params)

    return params


@METHODS.register("cif")
def cif() -> List[Dict[str, Any]]:
    """Conditional inference forest regressor hyperparameters."""
    method = inspect.currentframe().f_code.co_name

    params = []
    for selector in ["pc", "dc", "hybrid"]:
        for n_resamples_selector in ["minimum", "maximum", "auto", None]:
            for n_resamples_splitter in ["minimum", "maximum", "auto", None]:
                for adjust_alpha_selector in [True, False]:
                    for adjust_alpha_splitter in [True, False]:
                        for feature_scanning in [True, False]:
                            for threshold_scanning in [True, False]:
                                for max_samples in [None, 0.8]:
                                    for bootstrap_method in ["bayesian", "classic"]:
                                        for threshold_method in [
                                            "exact",
                                            "random",
                                            "percentile",
                                            "histogram",
                                        ]:
                                            hyperparameters = {
                                                "selector": selector,
                                                "n_resamples_selector": n_resamples_selector,
                                                "n_resamples_splitter": n_resamples_splitter,
                                                "adjust_alpha_selector": adjust_alpha_selector,
                                                "adjust_alpha_splitter": adjust_alpha_splitter,
                                                "feature_scanning": feature_scanning,
                                                "threshold_scanning": threshold_scanning,
                                                "max_samples": max_samples,
                                                "bootstrap_method": bootstrap_method,
                                                "threshold_method": threshold_method,
                                                "feature_muting": True,
                                                "early_stopping_selector": True,
                                                "early_stopping_splitter": True,
                                                "n_estimators": 100,
                                                "n_jobs": 1,
                                                "random_state": RANDOM_STATE,
                                                "verbose": 0,
                                                "method": method,
                                            }

                                            if threshold_method == "exact":
                                                for max_thresholds in [None]:
                                                    hyperparameters = deepcopy(hyperparameters)
                                                    hyperparameters["max_thresholds"] = max_thresholds
                                                    params.append(hyperparameters)
                                            elif threshold_method == "random":
                                                for max_thresholds in [0.5, 0.8]:
                                                    hyperparameters = deepcopy(hyperparameters)
                                                    hyperparameters["max_thresholds"] = max_thresholds
                                                    params.append(hyperparameters)
                                            elif threshold_method == "percentile":
                                                for max_thresholds in [10, 50]:
                                                    hyperparameters = deepcopy(hyperparameters)
                                                    hyperparameters["max_thresholds"] = max_thresholds
                                                    params.append(hyperparameters)
                                            else:
                                                # Histogram method
                                                for max_thresholds in [128, 256]:
                                                    hyperparameters = deepcopy(hyperparameters)
                                                    hyperparameters["max_thresholds"] = max_thresholds
                                                    params.append(hyperparameters)

    # Filter out bad combinations of parameters
    params = _filter_param_conflicts(params)

    return params


#############
# REST API #
#############


@app.on_event("startup")
def create_configurations() -> None:
    """Generate configurations for feature selection."""
    global CONFIGS

    ds_configs = {}
    for f in FILES:
        df = pd.read_parquet(os.path.join(DATA_DIR, f))
        dataset = f.replace("reg_", "").replace(".snappy.parquet", "")
        n_samples = df.shape[0]
        n_features = df.shape[1] - 1
        ds_configs[f] = {
            "dataset": dataset,
            "n_samples": n_samples,
            "n_features": n_features,
        }
        del df

    config_idx = 0
    hp_configs = {method: METHODS[method]() for method in METHODS.keys()}
    for method in hp_configs.keys():
        for config in hp_configs[method]:
            for name in ds_configs.keys():
                CONFIGS.append({**config, **ds_configs[name], **{"config_idx": config_idx}})
                config_idx += 1

    assert config_idx == len(CONFIGS)

    # Pull all items from DynamoDB and see what has already been processed
    processed = set()
    paginator = boto3.client("dynamodb", region_name="us-east-1").get_paginator("scan")
    for page in paginator.paginate(TableName=os.environ["TABLE_NAME"]):
        for config in page["Items"]:
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
