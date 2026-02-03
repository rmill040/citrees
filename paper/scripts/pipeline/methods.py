"""Method registry for feature selection experiments.

Centralizes method definitions, variants, and descriptions for both
classification and regression tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

from paper.scripts.pipeline.types import MethodConfig


@dataclass
class MethodInfo:
    """Metadata about a feature selection method.

    Parameters
    ----------
    name : str
        Base method name (e.g., "cit", "rf").
    display_name : str
        Human-readable name for display.
    description : str
        Brief description of the method.
    category : str
        Method category: "filter", "ptest", "embedding", "wrapper".
    tasks : set[str]
        Supported task types: {"classification", "regression"}.
    """

    name: str
    display_name: str
    description: str
    category: str
    tasks: frozenset[str]


# Method metadata registry
METHOD_INFO: dict[str, MethodInfo] = {
    # Permutation test methods
    "ptest_mc": MethodInfo(
        name="ptest_mc",
        display_name="Permutation Test (MC)",
        description="ANOVA with permutation p-values",
        category="ptest",
        tasks=frozenset({"classification"}),
    ),
    "ptest_rdc": MethodInfo(
        name="ptest_rdc",
        display_name="Permutation Test (RDC)",
        description="RDC with permutation p-values",
        category="ptest",
        tasks=frozenset({"classification", "regression"}),
    ),
    "ptest_pc": MethodInfo(
        name="ptest_pc",
        display_name="Permutation Test (PC)",
        description="Pearson with permutation p-values",
        category="ptest",
        tasks=frozenset({"regression"}),
    ),
    "ptest_dc": MethodInfo(
        name="ptest_dc",
        display_name="Permutation Test (DC)",
        description="Distance corr with permutation p-values",
        category="ptest",
        tasks=frozenset({"regression"}),
    ),
    # Embedding methods (tree-based)
    "cit": MethodInfo(
        name="cit",
        display_name="Conditional Inference Tree",
        description="Permutation-based splits",
        category="embedding",
        tasks=frozenset({"classification", "regression"}),
    ),
    "cif": MethodInfo(
        name="cif",
        display_name="Conditional Inference Forest",
        description="Ensemble of CITs",
        category="embedding",
        tasks=frozenset({"classification", "regression"}),
    ),
    "rf": MethodInfo(
        name="rf",
        display_name="Random Forest",
        description="MDI importance",
        category="embedding",
        tasks=frozenset({"classification", "regression"}),
    ),
    "et": MethodInfo(
        name="et",
        display_name="Extra Trees",
        description="MDI importance",
        category="embedding",
        tasks=frozenset({"classification", "regression"}),
    ),
    "xgb": MethodInfo(
        name="xgb",
        display_name="XGBoost",
        description="Gradient boosting importance",
        category="embedding",
        tasks=frozenset({"classification", "regression"}),
    ),
    "lgbm": MethodInfo(
        name="lgbm",
        display_name="LightGBM",
        description="Gradient boosting importance",
        category="embedding",
        tasks=frozenset({"classification", "regression"}),
    ),
    "cat": MethodInfo(
        name="cat",
        display_name="CatBoost",
        description="Gradient boosting importance",
        category="embedding",
        tasks=frozenset({"classification", "regression"}),
    ),
    # R partykit methods
    "r_ctree": MethodInfo(
        name="r_ctree",
        display_name="R ctree",
        description="Hothorn et al. (2006) via rpy2",
        category="embedding",
        tasks=frozenset({"classification", "regression"}),
    ),
    "r_cforest": MethodInfo(
        name="r_cforest",
        display_name="R cforest",
        description="Hothorn et al. (2006) via rpy2",
        category="embedding",
        tasks=frozenset({"classification", "regression"}),
    ),
    # Wrapper methods
    "boruta": MethodInfo(
        name="boruta",
        display_name="Boruta",
        description="RF shadow feature comparison",
        category="wrapper",
        tasks=frozenset({"classification", "regression"}),
    ),
    "pi": MethodInfo(
        name="pi",
        display_name="Permutation Importance",
        description="Model-agnostic",
        category="wrapper",
        tasks=frozenset({"classification", "regression"}),
    ),
    "cpi": MethodInfo(
        name="cpi",
        display_name="Conditional Permutation Importance",
        description="Conditional on other features",
        category="wrapper",
        tasks=frozenset({"classification", "regression"}),
    ),
    "shap": MethodInfo(
        name="shap",
        display_name="SHAP",
        description="TreeSHAP values",
        category="wrapper",
        tasks=frozenset({"classification", "regression"}),
    ),
    "rfe": MethodInfo(
        name="rfe",
        display_name="Recursive Feature Elimination",
        description="Backward elimination",
        category="wrapper",
        tasks=frozenset({"classification", "regression"}),
    ),
}


# Classification methods (18 total)
CLF_METHODS = [
    # Permutation test methods
    "ptest_mc",
    "ptest_rdc",
    # Embedding methods (tree-based)
    "cit",
    "cif",
    "rf",
    "et",
    "xgb",
    "lgbm",
    "cat",
    # R partykit methods
    "r_ctree",
    "r_cforest",
    # Wrapper methods
    "boruta",
    "pi",
    "cpi",
    "shap",
    "rfe",
]


# Regression methods (18 total)
REG_METHODS = [
    # Permutation test methods
    "ptest_pc",
    "ptest_dc",
    "ptest_rdc",
    # Embedding methods (tree-based)
    "cit",
    "cif",
    "rf",
    "et",
    "xgb",
    "lgbm",
    "cat",
    # R partykit methods
    "r_ctree",
    "r_cforest",
    # Wrapper methods
    "boruta",
    "pi",
    "cpi",
    "shap",
    "rfe",
]


# Methods that benefit from multi-threading
THREADED_METHODS = {
    "rf",
    "et",
    "xgb",
    "lgbm",
    "cat",
    "boruta",
    "pi",
    "cpi",
    "shap",
    "rfe",
}


# Embedding methods (have feature_importances_ and can make predictions)
EMBEDDING_METHODS = {"cit", "cif", "rf", "et", "xgb", "lgbm", "cat"}




def get_methods(task: str) -> list[str]:
    """Get method names for a task type.

    Parameters
    ----------
    task : str
        Either "classification" or "regression".

    Returns
    -------
    list[str]
        List of method names.
    """
    return CLF_METHODS if task == "classification" else REG_METHODS


def get_method_info(name: str) -> MethodInfo | None:
    """Get metadata for a method by name.

    Parameters
    ----------
    name : str
        Method name.

    Returns
    -------
    MethodInfo or None
        Method metadata, or None if not found.
    """
    return METHOD_INFO.get(name)


def get_all_method_info(task: str | None = None, category: str | None = None) -> list[MethodInfo]:
    """Get method metadata filtered by task and/or category.

    Parameters
    ----------
    task : str, optional
        Filter by task type: "classification" or "regression".
    category : str, optional
        Filter by category: "filter", "ptest", "embedding", "wrapper".

    Returns
    -------
    list[MethodInfo]
        Filtered method metadata.
    """
    methods = get_methods(task) if task else list(METHOD_INFO.keys())
    result = []
    for name in methods:
        info = METHOD_INFO.get(name)
        if info is None:
            continue
        if category and info.category != category:
            continue
        result.append(info)
    return result


def get_full_method_configs(
    methods: list[str],
    task: str,
) -> list[MethodConfig]:
    """Generate MethodConfig objects from full parameter grids.

    This function uses the full hyperparameter grids from pipeline/config.py
    to generate all valid configurations for each method. Use this for
    comprehensive hyperparameter sweeps.

    Parameters
    ----------
    methods : list[str]
        Base method names.
    task : str
        "classification" or "regression".

    Returns
    -------
    list[MethodConfig]
        All method configurations from full grids.
    """
    from paper.scripts.pipeline.config import get_method_configs

    configs: list[MethodConfig] = []
    for method in methods:
        try:
            method_configs = get_method_configs(method, task)
        except ValueError:
            # Method not in config system, use default params
            configs.append(MethodConfig(name=method))
            continue

        for cfg in method_configs:
            # Extract params (exclude metadata keys)
            params = {k: v for k, v in cfg.items() if k not in {"method", "random_state"}}
            configs.append(
                MethodConfig(
                    name=method,
                    params=tuple(sorted(params.items())),
                )
            )

    return configs


def get_method_config_count(methods: list[str], task: str) -> dict[str, int]:
    """Get the number of configs that would be generated per method.

    Parameters
    ----------
    methods : list[str]
        Base method names.
    task : str
        "classification" or "regression".

    Returns
    -------
    dict[str, int]
        Map of method name to config count.
    """
    from paper.scripts.pipeline.config import get_method_configs

    counts: dict[str, int] = {}
    for method in methods:
        try:
            configs = get_method_configs(method, task)
            counts[method] = len(configs)
        except ValueError:
            # Method not in config system
            counts[method] = 1
    return counts
