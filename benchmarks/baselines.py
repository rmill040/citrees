from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


CLASSIFIERS: dict[str, type[BaseEstimator]] = {
    "rf": RandomForestClassifier,
    "et": ExtraTreesClassifier,
    "gb": GradientBoostingClassifier,
}

REGRESSORS: dict[str, type[BaseEstimator]] = {
    "rf": RandomForestRegressor,
    "et": ExtraTreesRegressor,
    "gb": GradientBoostingRegressor,
}


def _try_import_xgboost():
    try:
        from xgboost import XGBClassifier, XGBRegressor

        CLASSIFIERS["xgb"] = XGBClassifier
        REGRESSORS["xgb"] = XGBRegressor
    except ImportError:
        pass


def _try_import_lightgbm():
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        CLASSIFIERS["lgbm"] = LGBMClassifier
        REGRESSORS["lgbm"] = LGBMRegressor
    except ImportError:
        pass


def _try_import_catboost():
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor

        CLASSIFIERS["catboost"] = CatBoostClassifier
        REGRESSORS["catboost"] = CatBoostRegressor
    except ImportError:
        pass


_try_import_xgboost()
_try_import_lightgbm()
_try_import_catboost()


DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "rf": {"n_estimators": 100, "n_jobs": -1, "random_state": 42},
    "et": {"n_estimators": 100, "n_jobs": -1, "random_state": 42},
    "gb": {"n_estimators": 100, "random_state": 42},
    "xgb": {"n_estimators": 100, "n_jobs": -1, "random_state": 42, "verbosity": 0},
    "lgbm": {"n_estimators": 100, "n_jobs": -1, "random_state": 42, "verbose": -1},
    "catboost": {"n_estimators": 100, "random_state": 42, "verbose": 0},
}


def get_baseline(
    name: str,
    task: str = "classification",
    **kwargs: Any,
) -> BaseEstimator:
    registry = CLASSIFIERS if task == "classification" else REGRESSORS
    if name not in registry:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(registry.keys())}")

    params = {**DEFAULT_PARAMS.get(name, {}), **kwargs}
    return registry[name](**params)


def list_baselines(task: str = "classification") -> list[str]:
    registry = CLASSIFIERS if task == "classification" else REGRESSORS
    return list(registry.keys())


class BaselineRegistry:
    get = staticmethod(get_baseline)
    list = staticmethod(list_baselines)
