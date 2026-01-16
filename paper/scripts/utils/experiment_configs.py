"""Experiment config utilities for method variants and labeling."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


METHOD_VARIANTS: dict[str, list[dict[str, Any]]] = {
    # Explicitly include both muting states for cit/cif.
    "cit": [
        {"label": "muting", "params": {"feature_muting": True}},
        {"label": "no_muting", "params": {"feature_muting": False}},
    ],
    "cif": [
        {"label": "muting", "params": {"feature_muting": True}},
        {"label": "no_muting", "params": {"feature_muting": False}},
    ],
}


def expand_method_configs(methods: list[str]) -> list[dict[str, Any]]:
    """Expand a list of base methods into per-method config dicts."""
    configs: list[dict[str, Any]] = []
    for method in methods:
        variants = METHOD_VARIANTS.get(method)
        if not variants:
            configs.append({"method": method})
            continue
        for variant in variants:
            cfg: dict[str, Any] = {"method": method}
            label = variant.get("label")
            if label:
                cfg["label"] = label
            params = variant.get("params") or {}
            if params:
                cfg["params"] = params
            configs.append(cfg)
    return configs


def extract_params(config: dict[str, Any]) -> dict[str, Any]:
    """Extract model params from a config dict."""
    if "params" in config:
        return dict(config["params"] or {})
    return {
        k: v
        for k, v in config.items()
        if k not in {"method", "label", "name", "random_state"}
    }


def config_label(config: dict[str, Any]) -> str:
    """Create a stable label for a config, used in filenames and reporting."""
    method = config["method"]
    label = config.get("label") or config.get("name")
    params = extract_params(config)
    if label:
        return f"{method}__{_slugify(label)}"
    if not params:
        return method
    payload = json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]
    return f"{method}__cfg{digest}"


def _slugify(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "config"
