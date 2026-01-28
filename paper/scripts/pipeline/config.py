"""Experiment configuration - hyperparameter grids for all methods.

This module defines hyperparameter grids for all feature selection methods
and generates config combinations using itertools.product with conflict filtering.

Usage:
    from paper.scripts.pipeline.config import get_configs, grid_sizes

    # Get all configs for classification
    configs = get_configs("classification")

    # Print grid sizes
    for method, size in grid_sizes("classification").items():
        print(f"{method}: {size}")
"""

from __future__ import annotations

from itertools import product
from typing import Any, Callable

from paper.scripts.config.constants import RANDOM_STATE

# =============================================================================
# PARAMETER DEPENDENCY DOCUMENTATION
# =============================================================================
#
# CIT/CIF parameters form a dependency graph. Understanding these dependencies
# is critical for generating valid configs and avoiding redundant experiments.
#
# CONFLICT RULES SUMMARY:
#
# Rule 1: n_resamples_selector = None
#         -> adjust_alpha_selector, early_stopping_selector,
#            early_stopping_confidence_selector, feature_muting have NO effect
#
# Rule 2: early_stopping_selector = None
#         -> early_stopping_confidence_selector, feature_scanning have NO effect
#
# Rule 3: early_stopping_selector = "simple"
#         -> early_stopping_confidence_selector has NO effect
#
# Rule 4: n_resamples_splitter = None
#         -> adjust_alpha_splitter, early_stopping_splitter,
#            early_stopping_confidence_splitter have NO effect
#
# Rule 5: early_stopping_splitter = None
#         -> early_stopping_confidence_splitter, threshold_scanning have NO effect
#
# Rule 6: early_stopping_splitter = "simple"
#         -> early_stopping_confidence_splitter has NO effect
#
# Rule 7: honesty = False
#         -> honesty_fraction has NO effect
#
# Rule 8: threshold_method = "exact"
#         -> max_thresholds has NO effect
#
# Rule 9: threshold_method + max_thresholds must be compatible:
#         - exact: max_thresholds must be None
#         - random: max_thresholds in [0.5, 0.8] (fraction)
#         - percentile: max_thresholds in [10, 50] (num percentiles)
#         - histogram: max_thresholds in [128, 256] (num bins)
#
# =============================================================================


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize config by setting irrelevant params to defaults.

    When a parameter has no effect (due to dependency rules), we set it to
    a canonical default value. This allows detection of functionally
    identical configs that differ only in irrelevant parameters.
    """
    c = config.copy()

    # Rule 1: n_resamples_selector = None
    if c.get("n_resamples_selector") is None:
        c["adjust_alpha_selector"] = True
        c["early_stopping_selector"] = None
        c["early_stopping_confidence_selector"] = 0.95
        c["feature_muting"] = True

    # Rule 2: early_stopping_selector = None
    if c.get("early_stopping_selector") is None:
        c["early_stopping_confidence_selector"] = 0.95
        c["feature_scanning"] = True

    # Rule 3: early_stopping_selector = "simple"
    if c.get("early_stopping_selector") == "simple":
        c["early_stopping_confidence_selector"] = 0.95

    # Rule 4: n_resamples_splitter = None
    if c.get("n_resamples_splitter") is None:
        c["adjust_alpha_splitter"] = True
        c["early_stopping_splitter"] = None
        c["early_stopping_confidence_splitter"] = 0.95

    # Rule 5: early_stopping_splitter = None
    if c.get("early_stopping_splitter") is None:
        c["early_stopping_confidence_splitter"] = 0.95
        c["threshold_scanning"] = True

    # Rule 6: early_stopping_splitter = "simple"
    if c.get("early_stopping_splitter") == "simple":
        c["early_stopping_confidence_splitter"] = 0.95

    # Rule 7: honesty = False
    if not c.get("honesty"):
        c["honesty_fraction"] = 0.5

    # Rule 8: threshold_method = "exact"
    if c.get("threshold_method") == "exact":
        c["max_thresholds"] = None

    return c


def _make_hashable(v: Any) -> Any:
    """Convert value to hashable type (lists become tuples)."""
    if isinstance(v, list):
        return tuple(v)
    return v


def _filter_param_conflicts(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter out functionally identical configs based on dependency rules."""
    seen: set[tuple[tuple[str, Any], ...]] = set()
    unique: list[dict[str, Any]] = []

    for config in configs:
        normalized = _normalize_config(config)
        key = tuple(sorted((k, _make_hashable(v)) for k, v in normalized.items()))
        if key not in seen:
            seen.add(key)
            unique.append(config)

    return unique


# Valid threshold_method + max_thresholds combinations (Rule 9)
VALID_THRESHOLD_COMBOS: dict[str, list[Any]] = {
    "exact": [None],
    "random": [0.8],
    "percentile": [50],
    "histogram": [256],
}


def _is_valid_threshold_combo(threshold_method: str, max_thresholds: Any) -> bool:
    """Check if threshold_method and max_thresholds are compatible."""
    return max_thresholds in VALID_THRESHOLD_COMBOS.get(threshold_method, [])


def _generate_cit_cif_configs(
    selector_options: list[str | list[str]],
    splitter_options: list[str],
    method: str,
    is_forest: bool = False,
    is_classifier: bool = True,
) -> list[dict[str, Any]]:
    """Generate CIT/CIF configs efficiently by only creating valid combinations.

    This avoids generating millions of redundant configs by respecting
    parameter dependencies during generation.
    """
    params: list[dict[str, Any]] = []

    # Selector-side effective combinations (respecting Rules 1-3)
    selector_side_combos: list[dict[str, Any]] = []
    for n_res_sel in ["auto", None]:
        if n_res_sel is None:
            # Rule 1: All dependent params fixed to defaults
            selector_side_combos.append(
                {
                    "n_resamples_selector": None,
                    "adjust_alpha_selector": True,
                    "early_stopping_selector": None,
                    "early_stopping_confidence_selector": 0.95,
                    "feature_muting": True,
                    "feature_scanning": True,
                }
            )
        else:
            for adj_alpha_sel in [True, False]:
                for es_sel in ["adaptive", None]:
                    if es_sel is None:
                        # Rule 2: confidence and scanning fixed
                        selector_side_combos.append(
                            {
                                "n_resamples_selector": n_res_sel,
                                "adjust_alpha_selector": adj_alpha_sel,
                                "early_stopping_selector": None,
                                "early_stopping_confidence_selector": 0.95,
                                "feature_muting": True,
                                "feature_scanning": True,
                            }
                        )
                        selector_side_combos.append(
                            {
                                "n_resamples_selector": n_res_sel,
                                "adjust_alpha_selector": adj_alpha_sel,
                                "early_stopping_selector": None,
                                "early_stopping_confidence_selector": 0.95,
                                "feature_muting": False,
                                "feature_scanning": True,
                            }
                        )
                    else:  # adaptive
                        for conf_sel in [0.95, 0.99]:
                            for feat_mut in [True, False]:
                                for feat_scan in [True, False]:
                                    selector_side_combos.append(
                                        {
                                            "n_resamples_selector": n_res_sel,
                                            "adjust_alpha_selector": adj_alpha_sel,
                                            "early_stopping_selector": "adaptive",
                                            "early_stopping_confidence_selector": conf_sel,
                                            "feature_muting": feat_mut,
                                            "feature_scanning": feat_scan,
                                        }
                                    )

    # Splitter-side effective combinations (respecting Rules 4-6)
    splitter_side_combos: list[dict[str, Any]] = []
    for n_res_spl in ["auto", None]:
        if n_res_spl is None:
            # Rule 4: All dependent params fixed
            splitter_side_combos.append(
                {
                    "n_resamples_splitter": None,
                    "adjust_alpha_splitter": True,
                    "early_stopping_splitter": None,
                    "early_stopping_confidence_splitter": 0.95,
                    "threshold_scanning": True,
                }
            )
        else:
            for adj_alpha_spl in [True, False]:
                for es_spl in ["adaptive", None]:
                    if es_spl is None:
                        # Rule 5: confidence and scanning fixed
                        splitter_side_combos.append(
                            {
                                "n_resamples_splitter": n_res_spl,
                                "adjust_alpha_splitter": adj_alpha_spl,
                                "early_stopping_splitter": None,
                                "early_stopping_confidence_splitter": 0.95,
                                "threshold_scanning": True,
                            }
                        )
                    else:  # adaptive
                        for conf_spl in [0.95, 0.99]:
                            for thresh_scan in [True, False]:
                                splitter_side_combos.append(
                                    {
                                        "n_resamples_splitter": n_res_spl,
                                        "adjust_alpha_splitter": adj_alpha_spl,
                                        "early_stopping_splitter": "adaptive",
                                        "early_stopping_confidence_splitter": conf_spl,
                                        "threshold_scanning": thresh_scan,
                                    }
                                )

    # Threshold combos (Rule 8-9)
    threshold_combos: list[dict[str, Any]] = []
    for thresh_method, max_thresh_opts in VALID_THRESHOLD_COMBOS.items():
        for max_thresh in max_thresh_opts:
            threshold_combos.append(
                {
                    "threshold_method": thresh_method,
                    "max_thresholds": max_thresh,
                }
            )

    # Honesty combos (Rule 7)
    honesty_combos: list[dict[str, Any]] = [
        {"honesty": False, "honesty_fraction": 0.5},
        {"honesty": True, "honesty_fraction": 0.5},
    ]

    # Forest-specific combos
    if is_forest:
        if is_classifier:
            forest_combos = list(
                product(
                    [None, 0.8],  # max_samples
                    ["bayesian", "classic"],  # bootstrap_method
                    [None, "stratified", "undersample", "oversample"],  # sampling_method
                )
            )
        else:
            forest_combos = list(
                product(
                    [None, 0.8],  # max_samples
                    ["bayesian", "classic"],  # bootstrap_method
                )
            )
    else:
        forest_combos = [None]

    # Generate all valid combinations
    for selector in selector_options:
        for splitter in splitter_options:
            for alpha_sel in [0.05, 0.01]:
                for alpha_spl in [0.05, 0.01]:
                    for sel_combo in selector_side_combos:
                        for spl_combo in splitter_side_combos:
                            for thresh_combo in threshold_combos:
                                for hon_combo in honesty_combos:
                                    for forest_combo in forest_combos:
                                        config = {
                                            "selector": selector,
                                            "splitter": splitter,
                                            "alpha_selector": alpha_sel,
                                            "alpha_splitter": alpha_spl,
                                            **sel_combo,
                                            **spl_combo,
                                            **thresh_combo,
                                            **hon_combo,
                                            "method": method,
                                            "random_state": RANDOM_STATE,
                                        }
                                        if is_forest:
                                            if is_classifier:
                                                config["max_samples"] = forest_combo[0]
                                                config["bootstrap_method"] = forest_combo[1]
                                                config["sampling_method"] = forest_combo[2]
                                            else:
                                                config["max_samples"] = forest_combo[0]
                                                config["bootstrap_method"] = forest_combo[1]
                                            config["n_estimators"] = 100
                                            config["n_jobs"] = -1
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

CLF_FILTER_GRID: dict[str, list[Any]] = {}  # No hyperparameters

CLF_PTEST_GRID: dict[str, list[Any]] = {
    "alpha": [0.05, 0.01],
    "n_resamples": ["minimum", "maximum", "auto"],
    "early_stopping": ["adaptive", "simple", None],
}

CLF_RF_GRID: dict[str, list[Any]] = {
    "max_samples": [None, 0.8],
    "class_weight": [None, "balanced"],
    "n_estimators": [100],
    "n_jobs": [-1],
}

CLF_ET_GRID: dict[str, list[Any]] = {
    "class_weight": [None, "balanced"],
    "n_estimators": [100],
    "n_jobs": [-1],
}

CLF_XGB_GRID: dict[str, list[Any]] = {
    "max_depth": [1, 2, 3, 4, 6, 8],
    "learning_rate": [0.001, 0.01, 0.1],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "reg_alpha": [0.001, 0.01, None],
    "reg_lambda": [0.001, 0.01, None],
    "importance_type": ["gain", "weight", "cover", "total_gain", "total_cover"],
    "n_estimators": [100],
    "n_jobs": [-1],
}

CLF_LGBM_GRID: dict[str, list[Any]] = {
    "max_depth": [1, 2, 3, 4, 6, 8],
    "learning_rate": [0.001, 0.01, 0.1],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "reg_alpha": [0.001, 0.01, None],
    "reg_lambda": [0.001, 0.01, None],
    "importance_type": ["split", "gain"],
    "class_weight": [None, "balanced"],
    "n_estimators": [100],
    "n_jobs": [-1],
}

CLF_CAT_GRID: dict[str, list[Any]] = {
    "depth": [1, 2, 3, 4, 6, 8],
    "learning_rate": [0.001, 0.01, 0.1],
    "l2_leaf_reg": [1, 3, 5, 7, 9],
    "colsample_bylevel": [0.8, 0.9, 1.0],
    "auto_class_weights": [None, "Balanced"],
    "n_estimators": [100],
    "allow_writing_files": [False],
}

CLF_BORUTA_GRID: dict[str, list[Any]] = {
    "n_estimators": ["auto", 100, 200],
    "max_iter": [100, 200],
}

CLF_PI_GRID: dict[str, list[Any]] = {
    "n_repeats": [5, 10, 20],
}

CLF_CPI_GRID: dict[str, list[Any]] = {
    "n_repeats": [5, 10, 20],
}

CLF_SHAP_GRID: dict[str, list[Any]] = {
    "max_samples": [100, 500, 1000],
}

CLF_RFE_GRID: dict[str, list[Any]] = {}


# =============================================================================
# REGRESSION PARAMETER GRIDS
# =============================================================================

REG_FILTER_GRID: dict[str, list[Any]] = {}  # No hyperparameters

REG_PTEST_GRID: dict[str, list[Any]] = {
    "alpha": [0.05, 0.01],
    "n_resamples": ["minimum", "maximum", "auto"],
    "early_stopping": ["adaptive", "simple", None],
}

REG_RF_GRID: dict[str, list[Any]] = {
    "max_samples": [None, 0.8],
    "n_estimators": [100],
    "n_jobs": [-1],
}

REG_ET_GRID: dict[str, list[Any]] = {
    "n_estimators": [100],
    "n_jobs": [-1],
}

REG_XGB_GRID: dict[str, list[Any]] = {
    "max_depth": [1, 2, 4, 6, 8],
    "learning_rate": [0.001, 0.01, 0.1],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "reg_alpha": [0.001, 0.01, None],
    "reg_lambda": [0.001, 0.01, None],
    "importance_type": ["gain", "weight", "cover", "total_gain", "total_cover"],
    "n_estimators": [100],
    "n_jobs": [-1],
}

REG_LGBM_GRID: dict[str, list[Any]] = {
    "max_depth": [1, 2, 4, 6, 8],
    "learning_rate": [0.001, 0.01, 0.1],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "reg_alpha": [0.001, 0.01, None],
    "reg_lambda": [0.001, 0.01, None],
    "importance_type": ["split", "gain"],
    "n_estimators": [100],
    "n_jobs": [-1],
}

REG_CAT_GRID: dict[str, list[Any]] = {
    "depth": [1, 2, 4, 6, 8],
    "learning_rate": [0.001, 0.01, 0.1],
    "l2_leaf_reg": [1, 3, 5, 7, 9],
    "colsample_bylevel": [0.8, 0.9, 1.0],
    "n_estimators": [100],
    "allow_writing_files": [False],
}

REG_BORUTA_GRID: dict[str, list[Any]] = {
    "n_estimators": ["auto", 100, 200],
    "max_iter": [100, 200],
}

REG_PI_GRID: dict[str, list[Any]] = {
    "n_repeats": [5, 10, 20],
}

REG_CPI_GRID: dict[str, list[Any]] = {
    "n_repeats": [5, 10, 20],
}

REG_SHAP_GRID: dict[str, list[Any]] = {
    "max_samples": [100, 500, 1000],
}

REG_RFE_GRID: dict[str, list[Any]] = {}


# =============================================================================
# R METHODS PARAMETER GRIDS
# =============================================================================

R_CTREE_GRID: dict[str, list[Any]] = {
    "teststat": ["quadratic", "maximum"],
    "testtype": ["Bonferroni", "MonteCarlo", "Univariate"],
    "alpha": [0.05, 0.01],
    "nresample": [1000, 9999],
    "minsplit": [20, 10],
    "minbucket": [7, 5],
}

R_CFOREST_GRID: dict[str, list[Any]] = {
    "teststat": ["quadratic", "maximum"],
    "testtype": ["Bonferroni", "MonteCarlo", "Univariate"],
    "mincriterion": [0.95, 0.99, 0],
    "nresample": [1000, 9999],
    "ntree": [100],
    "mtry": ["sqrt", "all"],
    "replace": [False, True],
    "fraction": [0.632, 0.8],
    "varimp_conditional": [False, True],
    "varimp_nperm": [1, 5],
}


# =============================================================================
# CONFIG GENERATORS
# =============================================================================


def clf_mc() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_FILTER_GRID, "mc")


def clf_mi() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_FILTER_GRID, "mi")


def clf_rdc() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_FILTER_GRID, "rdc")


def clf_mrmr() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_FILTER_GRID, "mrmr")


def clf_ptest_mc() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_PTEST_GRID, "ptest_mc")


def clf_ptest_mi() -> list[dict[str, Any]]:
    # MI permutation test uses fewer alpha values (computationally expensive)
    grid = {**CLF_PTEST_GRID, "alpha": [0.05]}
    return _generate_simple_grid(grid, "ptest_mi")


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
    return _generate_cit_cif_configs(
        selector_options=["mc", "mi", "rdc"],
        splitter_options=["gini"],
        method="cit",
        is_forest=False,
        is_classifier=True,
    )


def clf_cif() -> list[dict[str, Any]]:
    return _generate_cit_cif_configs(
        selector_options=["mc", "mi", "rdc"],
        splitter_options=["gini"],
        method="cif",
        is_forest=True,
        is_classifier=True,
    )


def clf_boruta() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_BORUTA_GRID, "boruta")


def clf_pi() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_PI_GRID, "pi")


def clf_cpi() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_CPI_GRID, "cpi")


def clf_shap() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_SHAP_GRID, "shap")


def clf_rfe() -> list[dict[str, Any]]:
    return _generate_simple_grid(CLF_RFE_GRID, "rfe")


def clf_r_ctree() -> list[dict[str, Any]]:
    return _generate_simple_grid(R_CTREE_GRID, "r_ctree")


def clf_r_cforest() -> list[dict[str, Any]]:
    return _generate_simple_grid(R_CFOREST_GRID, "r_cforest")


def reg_pc() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_FILTER_GRID, "pc")


def reg_dc() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_FILTER_GRID, "dc")


def reg_rdc() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_FILTER_GRID, "rdc")


def reg_mrmr() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_FILTER_GRID, "mrmr")


def reg_ptest_pc() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_PTEST_GRID, "ptest_pc")


def reg_ptest_dc() -> list[dict[str, Any]]:
    # DC permutation test uses fewer alpha values (computationally expensive)
    grid = {**REG_PTEST_GRID, "alpha": [0.05]}
    return _generate_simple_grid(grid, "ptest_dc")


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
    return _generate_cit_cif_configs(
        selector_options=["pc", "dc", "rdc"],
        splitter_options=["mse"],
        method="cit",
        is_forest=False,
        is_classifier=False,
    )


def reg_cif() -> list[dict[str, Any]]:
    return _generate_cit_cif_configs(
        selector_options=["pc", "dc", "rdc"],
        splitter_options=["mse"],
        method="cif",
        is_forest=True,
        is_classifier=False,
    )


def reg_boruta() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_BORUTA_GRID, "boruta")


def reg_pi() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_PI_GRID, "pi")


def reg_cpi() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_CPI_GRID, "cpi")


def reg_shap() -> list[dict[str, Any]]:
    return _generate_simple_grid(REG_SHAP_GRID, "shap")


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
    # Filter methods
    "mc": clf_mc,
    "mi": clf_mi,
    "rdc": clf_rdc,
    "mrmr": clf_mrmr,
    # Permutation tests
    "ptest_mc": clf_ptest_mc,
    "ptest_mi": clf_ptest_mi,
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
    "shap": clf_shap,
    "rfe": clf_rfe,
}

REG_CONFIG_GENERATORS: dict[str, Callable[[], list[dict[str, Any]]]] = {
    # Filter methods
    "pc": reg_pc,
    "dc": reg_dc,
    "rdc": reg_rdc,
    "mrmr": reg_mrmr,
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
    "shap": reg_shap,
    "rfe": reg_rfe,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_method_configs(method: str, task_type: str) -> list[dict[str, Any]]:
    """Get all configs for a specific method.

    Parameters
    ----------
    method : str
        Method name (e.g., "xgb", "cit").
    task_type : str
        Either "classification" or "regression".

    Returns
    -------
    list[dict[str, Any]]
        List of configuration dictionaries for the method.
    """
    generators = CLF_CONFIG_GENERATORS if task_type == "classification" else REG_CONFIG_GENERATORS
    gen = generators.get(method)
    if gen is None:
        raise ValueError(f"Unknown method '{method}' for task type '{task_type}'")
    return gen()


def get_configs(task_type: str) -> dict[str, list[dict[str, Any]]]:
    """Get all configs for a task type.

    Parameters
    ----------
    task_type : str
        Either "classification" or "regression".

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Dictionary mapping method names to lists of configurations.
    """
    generators = CLF_CONFIG_GENERATORS if task_type == "classification" else REG_CONFIG_GENERATORS
    return {method: gen() for method, gen in generators.items()}


def grid_sizes(task_type: str) -> dict[str, int]:
    """Return the number of configs per method.

    Parameters
    ----------
    task_type : str
        Either "classification" or "regression".

    Returns
    -------
    dict[str, int]
        Dictionary mapping method names to config counts.
    """
    configs = get_configs(task_type)
    return {method: len(cfgs) for method, cfgs in configs.items()}


def total_configs(task_type: str) -> int:
    """Return total number of configs for a task type.

    Parameters
    ----------
    task_type : str
        Either "classification" or "regression".

    Returns
    -------
    int
        Total number of configurations.
    """
    return sum(grid_sizes(task_type).values())


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
