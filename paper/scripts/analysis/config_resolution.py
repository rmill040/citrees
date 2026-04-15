"""Resolve experiment method IDs back to concrete parameter settings."""

from __future__ import annotations

from typing import Any

import pandas as pd

from paper.scripts.pipeline.config import get_method_configs
from paper.scripts.pipeline.types import MethodConfig


def _enumerate_method_configs(method_base: str, task: str) -> dict[str, dict[str, Any]]:
    """Return a method_id -> params mapping for one method family and task."""
    try:
        configs = get_method_configs(method_base, task)
    except ValueError:
        default_id = MethodConfig(name=method_base).label
        return {default_id: {}}

    mapping: dict[str, dict[str, Any]] = {}
    for cfg in configs:
        params = {k: v for k, v in cfg.items() if k not in {"method", "random_state"}}
        method_id = MethodConfig(name=method_base, params=tuple(sorted(params.items()))).label
        mapping[method_id] = params
    return mapping


def resolve_method_config_details(best_configs: pd.DataFrame) -> pd.DataFrame:
    """Expand a benchmark best-config table into explicit parameter columns."""
    records: list[dict[str, Any]] = []
    cache: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}

    for row in best_configs.to_dict(orient="records"):
        task = str(row["task"])
        method_base = str(row["method_base"])
        method_id = str(row["method_id"])
        cache_key = (task, method_base)

        if cache_key not in cache:
            cache[cache_key] = _enumerate_method_configs(method_base, task)

        params = cache[cache_key].get(method_id)
        record = dict(row)
        record["config_resolved"] = params is not None
        if params is not None:
            record.update(params)
        records.append(record)

    details = pd.DataFrame.from_records(records)
    return details.sort_values(["task", "method_base", "method_id"]).reset_index(drop=True)


def resolve_method_config(task: str, method_base: str, method_id: str) -> MethodConfig:
    """Resolve one benchmark method_id back to its MethodConfig."""
    params = _enumerate_method_configs(method_base, task).get(method_id)
    if params is None:
        raise KeyError(f"Could not resolve {method_id} for task={task}, method_base={method_base}")
    return MethodConfig(name=method_base, params=tuple(sorted(params.items())))
