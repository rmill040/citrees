"""Experiment configuration dataclasses.

Provides clean, typed configurations for all experiment methods.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

RANDOM_STATE = 1718
RANDOM_SEEDS = list(range(30))  # 30 seeds (0-29) for robust variance estimation


@dataclass
class BaseConfig:
    """Base configuration with common fields."""

    random_state: int = RANDOM_STATE
    method: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for experiment."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


# =============================================================================
# Conditional Inference Tree/Forest Configs
# =============================================================================


@dataclass
class CITConfig(BaseConfig):
    """Conditional Inference Tree configuration."""

    method: str = "cit"

    # Resampling
    n_resamples_selector: str | int | None = "auto"
    n_resamples_splitter: str | int | None = "auto"

    # Alpha adjustment
    adjust_alpha_selector: bool = True
    adjust_alpha_splitter: bool = True

    # Early stopping
    early_stopping_selector: str | None = "adaptive"
    early_stopping_splitter: str | None = "adaptive"

    # Scanning
    feature_scanning: bool = True
    threshold_scanning: bool = True
    feature_muting: bool = True

    # Thresholds
    threshold_method: Literal["exact", "random", "percentile", "histogram"] = "exact"
    max_thresholds: int | float | None = None

    # Honesty
    honesty: bool = False

    # Other
    verbose: int = 0


@dataclass
class CIFConfig(CITConfig):
    """Conditional Inference Forest configuration."""

    method: str = "cif"

    # Forest-specific
    n_estimators: int = 100
    max_samples: float | None = None
    bootstrap_method: Literal["bayesian", "classic"] = "bayesian"
    sampling_method: Literal["balanced", "stratified"] = "stratified"
    n_jobs: int = 1


# =============================================================================
# Baseline Method Configs
# =============================================================================


@dataclass
class RandomForestConfig(BaseConfig):
    """Random Forest configuration."""

    method: str = "rf"
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | float | None = "sqrt"
    bootstrap: bool = True
    class_weight: str | None = None
    n_jobs: int = 1


@dataclass
class XGBConfig(BaseConfig):
    """XGBoost configuration."""

    method: str = "xgb"
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_alpha: float | None = None
    reg_lambda: float | None = None
    importance_type: str = "gain"
    n_jobs: int = 1


@dataclass
class LightGBMConfig(BaseConfig):
    """LightGBM configuration."""

    method: str = "lightgbm"
    n_estimators: int = 100
    max_depth: int = -1
    learning_rate: float = 0.1
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    importance_type: str = "gain"
    n_jobs: int = 1


# =============================================================================
# Experiment Config
# =============================================================================


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    dataset: str
    method_config: BaseConfig
    cv_folds: int = 5

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for experiment."""
        return {
            "dataset": self.dataset,
            "cv_folds": self.cv_folds,
            **self.method_config.to_dict(),
        }


# =============================================================================
# Config Generators
# =============================================================================


def generate_cit_configs(
    n_resamples_selector: list = ["minimum", "maximum", "auto", None],
    n_resamples_splitter: list = ["minimum", "maximum", "auto", None],
    adjust_alpha_selector: list[bool] = [True, False],
    adjust_alpha_splitter: list[bool] = [True, False],
    feature_scanning: list[bool] = [True, False],
    threshold_scanning: list[bool] = [True, False],
    threshold_method: list[str] = ["exact", "random", "percentile", "histogram"],
    honesty: list[bool] = [True, False],
) -> Iterator[CITConfig]:
    """Generate CIT configurations for hyperparameter search."""
    for nrs in n_resamples_selector:
        for nrsp in n_resamples_splitter:
            for aas in adjust_alpha_selector:
                for aasp in adjust_alpha_splitter:
                    for fs in feature_scanning:
                        for ts in threshold_scanning:
                            for tm in threshold_method:
                                for h in honesty:
                                    # Handle max_thresholds based on threshold_method
                                    if tm == "exact":
                                        max_thresholds_list = [None]
                                    elif tm == "random":
                                        max_thresholds_list = [0.5, 0.8]
                                    elif tm == "percentile":
                                        max_thresholds_list = [10, 50]
                                    else:  # histogram
                                        max_thresholds_list = [64, 128, 256]

                                    for mt in max_thresholds_list:
                                        yield CITConfig(
                                            n_resamples_selector=nrs,
                                            n_resamples_splitter=nrsp,
                                            adjust_alpha_selector=aas,
                                            adjust_alpha_splitter=aasp,
                                            feature_scanning=fs,
                                            threshold_scanning=ts,
                                            threshold_method=tm,
                                            max_thresholds=mt,
                                            honesty=h,
                                        )


def generate_cif_configs(
    n_resamples_selector: list = ["minimum", "maximum", "auto", None],
    n_resamples_splitter: list = ["minimum", "maximum", "auto", None],
    adjust_alpha_selector: list[bool] = [True, False],
    adjust_alpha_splitter: list[bool] = [True, False],
    feature_scanning: list[bool] = [True, False],
    threshold_scanning: list[bool] = [True, False],
    threshold_method: list[str] = ["exact", "random", "percentile", "histogram"],
    max_samples: list = [None, 0.8],
    bootstrap_method: list[str] = ["bayesian", "classic"],
    sampling_method: list[str] = ["balanced", "stratified"],
    honesty: list[bool] = [True, False],
) -> Iterator[CIFConfig]:
    """Generate CIF configurations for hyperparameter search."""
    for nrs in n_resamples_selector:
        for nrsp in n_resamples_splitter:
            for aas in adjust_alpha_selector:
                for aasp in adjust_alpha_splitter:
                    for fs in feature_scanning:
                        for ts in threshold_scanning:
                            for ms in max_samples:
                                for bm in bootstrap_method:
                                    for sm in sampling_method:
                                        for tm in threshold_method:
                                            for h in honesty:
                                                # Handle max_thresholds
                                                if tm == "exact":
                                                    max_thresholds_list = [None]
                                                elif tm == "random":
                                                    max_thresholds_list = [0.5, 0.8]
                                                elif tm == "percentile":
                                                    max_thresholds_list = [10, 50]
                                                else:
                                                    max_thresholds_list = [64, 128, 256]

                                                for mt in max_thresholds_list:
                                                    yield CIFConfig(
                                                        n_resamples_selector=nrs,
                                                        n_resamples_splitter=nrsp,
                                                        adjust_alpha_selector=aas,
                                                        adjust_alpha_splitter=aasp,
                                                        feature_scanning=fs,
                                                        threshold_scanning=ts,
                                                        threshold_method=tm,
                                                        max_thresholds=mt,
                                                        max_samples=ms,
                                                        bootstrap_method=bm,
                                                        sampling_method=sm,
                                                        honesty=h,
                                                    )


def generate_xgb_configs(
    max_depth: list[int] = [1, 2, 3, 4, 6, 8],
    learning_rate: list[float] = [0.001, 0.01, 0.1],
    subsample: list[float] = [0.8, 0.9, 1.0],
    colsample_bytree: list[float] = [0.8, 0.9, 1.0],
    reg_alpha: list = [0.001, 0.01, None],
    reg_lambda: list = [0.001, 0.01, None],
    importance_type: list[str] = ["gain", "weight", "cover", "total_gain", "total_cover"],
) -> Iterator[XGBConfig]:
    """Generate XGBoost configurations for hyperparameter search."""
    for md in max_depth:
        for lr in learning_rate:
            for ss in subsample:
                for cb in colsample_bytree:
                    for ra in reg_alpha:
                        for rl in reg_lambda:
                            for it in importance_type:
                                yield XGBConfig(
                                    max_depth=md,
                                    learning_rate=lr,
                                    subsample=ss,
                                    colsample_bytree=cb,
                                    reg_alpha=ra,
                                    reg_lambda=rl,
                                    importance_type=it,
                                )
