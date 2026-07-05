"""Experiment configuration - hyperparameter grids for all methods.

This module defines hyperparameter grids for all feature selection methods
and generates config combinations using itertools.product with conflict filtering.

Usage:
    from paper.benchmark.pipeline.config import get_configs, grid_sizes

    # Get all configs for classification
    configs = get_configs("classification")

    # Print grid sizes
    for method, size in grid_sizes("classification").items():
        print(f"{method}: {size}")
"""

from collections.abc import Callable
from itertools import product
from typing import Any

from paper.benchmark.config.constants import RANDOM_STATE

# Valid threshold_method + max_thresholds combinations
VALID_THRESHOLD_COMBOS: dict[str, list[Any]] = {
    "exact": [None],
    "histogram": [256],
}


def _is_valid_threshold_combo(config: dict[str, Any]) -> bool:
    """Check that threshold_method and max_thresholds are a valid pair."""
    method = config.get("threshold_method")
    if method is None:
        return True
    return config.get("max_thresholds") in VALID_THRESHOLD_COMBOS.get(method, [])


def _generate_filtered_grid(
    param_grid: dict[str, list[Any]],
    method: str,
    filter_fn: Callable[[dict[str, Any]], bool],
) -> list[dict[str, Any]]:
    """Generate combinations from a param grid, excluding invalid ones via filter_fn."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    params: list[dict[str, Any]] = []
    for combo in product(*values):
        config = dict(zip(keys, combo, strict=False))
        if not filter_fn(config):
            continue
        config["method"] = method
        config["random_state"] = RANDOM_STATE
        params.append(config)
    return params


def _generate_simple_grid(
    param_grid: dict[str, list[Any]],
    method: str,
) -> list[dict[str, Any]]:
    """Generate combinations without conflict filtering (for simple methods)."""
    if not param_grid:
        # No parameters - return a single config with just method and random_state
        return [{"method": method, "random_state": RANDOM_STATE}]

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    params: list[dict[str, Any]] = []
    for combo in product(*values):
        config = dict(zip(keys, combo, strict=False))
        config["method"] = method
        config["random_state"] = RANDOM_STATE
        params.append(config)

    return params


# =============================================================================
# CLASSIFICATION PARAMETER GRIDS
# =============================================================================

CLF_PTEST_GRID: dict[str, list[Any]] = {
    "alpha": [0.05],
    "n_resamples": ["auto"],
    "early_stopping": ["adaptive"],
}

CLF_RF_GRID: dict[str, list[Any]] = {
    "max_samples": [None],
    "class_weight": ["balanced"],
    "n_estimators": [100],
    "n_jobs": [-1],
}

CLF_ET_GRID: dict[str, list[Any]] = {
    "class_weight": ["balanced"],
    "n_estimators": [100],
    "n_jobs": [-1],
}

CLF_XGB_GRID: dict[str, list[Any]] = {
    "max_depth": [6],
    "learning_rate": [0.1],
    "importance_type": ["gain", "weight", "cover", "total_gain", "total_cover"],
    "n_estimators": [500],
    "n_jobs": [-1],
}

CLF_LGBM_GRID: dict[str, list[Any]] = {
    "max_depth": [6],
    "learning_rate": [0.1],
    "importance_type": ["split", "gain"],
    "n_estimators": [500],
    "n_jobs": [-1],
}

CLF_CAT_GRID: dict[str, list[Any]] = {
    "depth": [6],
    "learning_rate": [0.1],
    "l2_leaf_reg": [1],
    "n_estimators": [500],
    "allow_writing_files": [False],
}

CLF_BORUTA_GRID: dict[str, list[Any]] = {
    "n_estimators": ["auto"],
    "max_iter": [200],
}

CLF_PI_GRID: dict[str, list[Any]] = {
    "n_repeats": [10],
}

CLF_CPI_GRID: dict[str, list[Any]] = {
    "n_repeats": [10],
}

CLF_RFE_GRID: dict[str, list[Any]] = {}


# =============================================================================
# CIT/CIF PARAMETER GRIDS (shared base + task-specific overrides)
# =============================================================================

_CIT_CIF_BASE: dict[str, list[Any]] = {
    "alpha_selector": [0.05],
    "alpha_splitter": [0.05],
    # Selector
    "n_resamples_selector": ["minimum"],
    "adjust_alpha_selector": [True],
    "early_stopping_selector": ["adaptive"],
    "early_stopping_confidence_selector": [0.95],
    "feature_muting": [True],
    "feature_scanning": [True],
    # Splitter
    "n_resamples_splitter": ["minimum"],
    "adjust_alpha_splitter": [True],
    "early_stopping_splitter": ["adaptive"],
    "early_stopping_confidence_splitter": [0.95],
    "threshold_scanning": [True],
    # Threshold
    "threshold_method": ["histogram"],
    "max_thresholds": [256],
    # Honesty
    "honesty": [False, True],
    "honesty_fraction": [0.5],
}

CLF_CIT_GRID: dict[str, list[Any]] = {
    "selector": ["mc", "rdc"],
    "splitter": ["gini"],
    **_CIT_CIF_BASE,
}
CLF_CIF_GRID: dict[str, list[Any]] = {
    **CLF_CIT_GRID,
    "max_samples": [None],
    "bootstrap": [True],
    "sampling_method": ["stratified"],
    "n_estimators": [100],
    "n_jobs": [-1],
}

REG_CIT_GRID: dict[str, list[Any]] = {
    "selector": ["pc", "rdc"],
    "splitter": ["mse"],
    **_CIT_CIF_BASE,
}
REG_CIF_GRID: dict[str, list[Any]] = {
    **REG_CIT_GRID,
    "max_samples": [None],
    "bootstrap": [True],
    "n_estimators": [100],
    "n_jobs": [-1],
}


# =============================================================================
# REGRESSION PARAMETER GRIDS
# =============================================================================

REG_PTEST_GRID: dict[str, list[Any]] = {
    "alpha": [0.05],
    "n_resamples": ["auto"],
    "early_stopping": ["adaptive"],
}

REG_RF_GRID: dict[str, list[Any]] = {
    "max_samples": [None],
    "n_estimators": [100],
    "n_jobs": [-1],
}

REG_ET_GRID: dict[str, list[Any]] = {
    "n_estimators": [100],
    "n_jobs": [-1],
}

REG_XGB_GRID: dict[str, list[Any]] = {
    "max_depth": [6],
    "learning_rate": [0.1],
    "importance_type": ["gain", "weight", "cover", "total_gain", "total_cover"],
    "n_estimators": [500],
    "n_jobs": [-1],
}

REG_LGBM_GRID: dict[str, list[Any]] = {
    "max_depth": [6],
    "learning_rate": [0.1],
    "importance_type": ["split", "gain"],
    "n_estimators": [500],
    "n_jobs": [-1],
}

REG_CAT_GRID: dict[str, list[Any]] = {
    "depth": [6],
    "learning_rate": [0.1],
    "l2_leaf_reg": [1],
    "n_estimators": [500],
    "allow_writing_files": [False],
}

REG_BORUTA_GRID: dict[str, list[Any]] = {
    "n_estimators": ["auto"],
    "max_iter": [200],
}

REG_PI_GRID: dict[str, list[Any]] = {
    "n_repeats": [10],
}

REG_CPI_GRID: dict[str, list[Any]] = {
    "n_repeats": [10],
}

REG_RFE_GRID: dict[str, list[Any]] = {}


# =============================================================================
# R METHODS PARAMETER GRIDS
# =============================================================================

R_CTREE_GRID: dict[str, list[Any]] = {
    "teststat": ["quadratic"],
    "testtype": ["Bonferroni", "MonteCarlo"],
    "alpha": [0.05],
    "nresample": [9999],
    "minsplit": [20],
    "minbucket": [7],
}

R_CFOREST_GRID: dict[str, list[Any]] = {
    "teststat": ["quadratic"],
    "testtype": ["Bonferroni", "MonteCarlo"],
    "mincriterion": [0.95],
    "nresample": [9999],
    "ntree": [100],
    "mtry": ["sqrt"],
    "replace": [False, True],
    "fraction": [0.632],
    "varimp_conditional": [False],
    "varimp_nperm": [1],
}


# =============================================================================
# CONFIG GENERATORS
# =============================================================================


def clf_ptest_mc() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_PTEST_GRID, "ptest_mc")


def clf_ptest_rdc() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_PTEST_GRID, "ptest_rdc")


def clf_rf() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_RF_GRID, "rf")


def clf_et() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_ET_GRID, "et")


def clf_xgb() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_XGB_GRID, "xgb")


def clf_lgbm() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_LGBM_GRID, "lgbm")


def clf_cat() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_CAT_GRID, "cat")


def clf_cit() -> list[dict[str, Any]]:
    return _generate_filtered_grid(CLF_CIT_GRID, "cit", _is_valid_threshold_combo)


def clf_cif() -> list[dict[str, Any]]:
    return _generate_filtered_grid(CLF_CIF_GRID, "cif", _is_valid_threshold_combo)


def clf_boruta() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_BORUTA_GRID, "boruta")


def clf_pi() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_PI_GRID, "pi")


def clf_cpi() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_CPI_GRID, "cpi")


def clf_rfe() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_RFE_GRID, "rfe")


def clf_r_ctree() -> list[dict[str, Any]]:
    return _generate_simple_grid(R_CTREE_GRID, "r_ctree")


def clf_r_cforest() -> list[dict[str, Any]]:
    return _generate_simple_grid(R_CFOREST_GRID, "r_cforest")


def reg_ptest_pc() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_PTEST_GRID, "ptest_pc")


def reg_ptest_dc() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_PTEST_GRID, "ptest_dc")


def reg_ptest_rdc() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_PTEST_GRID, "ptest_rdc")


def reg_rf() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_RF_GRID, "rf")


def reg_et() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_ET_GRID, "et")


def reg_xgb() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_XGB_GRID, "xgb")


def reg_lgbm() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_LGBM_GRID, "lgbm")


def reg_cat() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_CAT_GRID, "cat")


def reg_cit() -> list[dict[str, Any]]:
    return _generate_filtered_grid(REG_CIT_GRID, "cit", _is_valid_threshold_combo)


def reg_cif() -> list[dict[str, Any]]:
    return _generate_filtered_grid(REG_CIF_GRID, "cif", _is_valid_threshold_combo)


def reg_boruta() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_BORUTA_GRID, "boruta")


def reg_pi() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_PI_GRID, "pi")


def reg_cpi() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_CPI_GRID, "cpi")


def reg_rfe() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_RFE_GRID, "rfe")


def reg_r_ctree() -> list[dict[str, Any]]:
    return _generate_simple_grid(R_CTREE_GRID, "r_ctree")


def reg_r_cforest() -> list[dict[str, Any]]:
    return _generate_simple_grid(R_CFOREST_GRID, "r_cforest")


# =============================================================================
# METHOD REGISTRY
# =============================================================================

CLF_CONFIG_GENERATORS: dict[str, Callable[[], list[dict[str, Any]]]] = {
    # Permutation tests
    "ptest_mc": clf_ptest_mc,
    "ptest_rdc": clf_ptest_rdc,
    # Tree-based
    "rf": clf_rf,
    "et": clf_et,
    "xgb": clf_xgb,
    "lgbm": clf_lgbm,
    "cat": clf_cat,
    "cit": clf_cit,
    "cif": clf_cif,
    # R methods
    "r_ctree": clf_r_ctree,
    "r_cforest": clf_r_cforest,
    # Wrapper
    "boruta": clf_boruta,
    "pi": clf_pi,
    "cpi": clf_cpi,
    "rfe": clf_rfe,
}

REG_CONFIG_GENERATORS: dict[str, Callable[[], list[dict[str, Any]]]] = {
    # Permutation tests
    "ptest_pc": reg_ptest_pc,
    "ptest_dc": reg_ptest_dc,
    "ptest_rdc": reg_ptest_rdc,
    # Tree-based
    "rf": reg_rf,
    "et": reg_et,
    "xgb": reg_xgb,
    "lgbm": reg_lgbm,
    "cat": reg_cat,
    "cit": reg_cit,
    "cif": reg_cif,
    # R methods
    "r_ctree": reg_r_ctree,
    "r_cforest": reg_r_cforest,
    # Wrapper
    "boruta": reg_boruta,
    "pi": reg_pi,
    "cpi": reg_cpi,
    "rfe": reg_rfe,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_method_configs(method: str, task: str) -> list[dict[str, Any]]:
    """Get all configs for a specific method.

    Parameters
    ----------
    method : str
        Method name (e.g., "xgb", "cit").
    task : str
        Either "classification" or "regression".

    Returns
    -------
    list[dict[str, Any]]
        List of configuration dictionaries for the method.
    """
    generators = CLF_CONFIG_GENERATORS if task == "classification" else REG_CONFIG_GENERATORS
    gen = generators.get(method)
    if gen is None:
        raise ValueError(f"Unknown method '{method}' for task type '{task}'")
    return gen()


def get_configs(task: str) -> dict[str, list[dict[str, Any]]]:
    """Get all configs for a task type.

    Parameters
    ----------
    task : str
        Either "classification" or "regression".

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Dictionary mapping method names to lists of configurations.
    """
    generators = CLF_CONFIG_GENERATORS if task == "classification" else REG_CONFIG_GENERATORS
    return {method: gen() for method, gen in generators.items()}


def grid_sizes(task: str) -> dict[str, int]:
    """Return the number of configs per method.

    Parameters
    ----------
    task : str
        Either "classification" or "regression".

    Returns
    -------
    dict[str, int]
        Dictionary mapping method names to config counts.
    """
    configs = get_configs(task)
    return {method: len(cfgs) for method, cfgs in configs.items()}


def total_configs(task: str) -> int:
    """Return total number of configs for a task type.

    Parameters
    ----------
    task : str
        Either "classification" or "regression".

    Returns
    -------
    int
        Total number of configurations.
    """
    return sum(grid_sizes(task).values())


if __name__ == "__main__":
    print("=" * 70)
    print("CLASSIFICATION CONFIG GRID SIZES")
    print("=" * 70)
    clf_sizes = grid_sizes("classification")
    for method, size in sorted(clf_sizes.items(), key=lambda x: -x[1]):
        print(f"  {method:12s}: {size:,}")
    print(f"  {'TOTAL':12s}: {total_configs('classification'):,}")

    print()
    print("=" * 70)
    print("REGRESSION CONFIG GRID SIZES")
    print("=" * 70)
    reg_sizes = grid_sizes("regression")
    for method, size in sorted(reg_sizes.items(), key=lambda x: -x[1]):
        print(f"  {method:12s}: {size:,}")
    print(f"  {'TOTAL':12s}: {total_configs('regression'):,}")
