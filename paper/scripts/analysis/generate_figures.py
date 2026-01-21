"""Generate figures for citrees paper.

Creates visualizations comparing feature selection behavior across methods:
- Bar plots showing feature selection counts (informative vs noise)
- Critical difference diagrams for method rankings
- Box plots for downstream performance

Usage:
    uv sync --group paper
    UV_CACHE_DIR=$PWD/.uv-cache uv run python paper/scripts/analysis/generate_figures.py --profile paper

Profiles:
  - quick: fast sanity run (small n, few repeats)
  - paper: larger synthetic datasets for publication figures
  - huge: very large runs (slow; use with care)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

_paper_dir = Path(__file__).resolve().parents[2]
# Ensure Matplotlib/fontconfig caches are writable in sandboxed environments.
_cache_root = Path(tempfile.gettempdir()) / "citrees-paper-cache"
_mpl_dir = _cache_root / "mplconfig"
_xdg_dir = _cache_root / "xdg-cache"
_mpl_dir.mkdir(parents=True, exist_ok=True)
_xdg_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_friedman1, make_regression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = _paper_dir / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
CACHE_DIR = RESULTS_DIR / "cache"


@dataclass(frozen=True)
class FigureRunConfig:
    profile: str
    seed: int
    n_repeats: int
    n_jobs: int
    n_samples_main: int
    n_estimators_forest: int
    train_fraction: float
    timing_repeats: int
    timing_n_samples: int
    timing_n_features: int
    highdim_repeats: int
    highdim_n: int
    sample_size_grid: tuple[int, ...]


PROFILES: dict[str, FigureRunConfig] = {
    "quick": FigureRunConfig(
        profile="quick",
        seed=1718,
        n_repeats=5,
        n_jobs=1,
        n_samples_main=500,
        n_estimators_forest=50,
        train_fraction=0.8,
        timing_repeats=1,
        timing_n_samples=500,
        timing_n_features=100,
        highdim_repeats=3,
        highdim_n=200,
        sample_size_grid=(100, 200, 500, 1000),
    ),
    "paper": FigureRunConfig(
        profile="paper",
        seed=1718,
        n_repeats=10,
        n_jobs=-1,
        n_samples_main=2000,
        n_estimators_forest=100,
        train_fraction=0.8,
        timing_repeats=3,
        timing_n_samples=2000,
        timing_n_features=200,
        highdim_repeats=5,
        highdim_n=500,
        sample_size_grid=(200, 500, 1000, 2000, 5000),
    ),
    "huge": FigureRunConfig(
        profile="huge",
        seed=1718,
        n_repeats=20,
        n_jobs=-1,
        n_samples_main=5000,
        n_estimators_forest=200,
        train_fraction=0.8,
        timing_repeats=5,
        timing_n_samples=5000,
        timing_n_features=300,
        highdim_repeats=10,
        highdim_n=1000,
        sample_size_grid=(200, 500, 1000, 2000, 5000, 10000),
    ),
}

FIGURE_IDS: tuple[str, ...] = (
    "feature_selection",
    "timing",
    "correlated",
    "complexity",
    "highdim",
    "signal",
    "redundant",
    "multiclass",
    "imbalanced",
    "sample_size",
    "regression",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES.keys()), default="paper")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-repeats", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=None, help="Base n for main synthetic figures.")
    parser.add_argument("--n-estimators", type=int, default=None, help="n_estimators for forests (RF/ET/CIF).")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel jobs for sklearn and CIForest.")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        choices=list(FIGURE_IDS),
        help=(
            "Run only a subset of figures. Options: feature_selection, timing, correlated, complexity, highdim, "
            "signal, redundant, multiclass, imbalanced, sample_size, regression"
        ),
    )
    return parser.parse_args()


def _resolve_config(args: argparse.Namespace) -> FigureRunConfig:
    cfg = PROFILES[str(args.profile)]
    cfg = replace(cfg, profile=str(args.profile))
    if args.seed is not None:
        cfg = replace(cfg, seed=int(args.seed))
    if args.n_repeats is not None:
        cfg = replace(cfg, n_repeats=int(args.n_repeats))
    if args.n_samples is not None:
        cfg = replace(cfg, n_samples_main=int(args.n_samples))
    if args.n_estimators is not None:
        cfg = replace(cfg, n_estimators_forest=int(args.n_estimators))
    if args.n_jobs is not None:
        cfg = replace(cfg, n_jobs=int(args.n_jobs))

    # Some sandboxed environments disallow `os.sysconf(...)`, which joblib/loky uses to validate
    # system semaphore limits before spawning processes. Fall back to single-threaded execution
    # rather than crashing, while still allowing `n_jobs != 1` on normal systems.
    if cfg.n_jobs != 1:
        try:
            os.sysconf("SC_SEM_NSEMS_MAX")
        except PermissionError:
            print(
                "Parallel backends unavailable in this environment; falling back to n_jobs=1.",
                file=sys.stderr,
                flush=True,
            )
            cfg = replace(cfg, n_jobs=1)

    return cfg


def count_splits_sklearn(estimator) -> Counter:
    """Count feature splits in sklearn tree/forest."""
    counts: Counter = Counter()
    if hasattr(estimator, "estimators_"):
        for tree in estimator.estimators_:
            features = tree.tree_.feature
            for f in features:
                if f >= 0:
                    counts[int(f)] += 1
    else:
        features = estimator.tree_.feature
        for f in features:
            if f >= 0:
                counts[int(f)] += 1
    return counts


def count_splits_citree(tree_dict: dict, counts: Counter | None = None) -> Counter:
    """Count feature splits in citrees tree structure."""
    if counts is None:
        counts = Counter()
    if "value" not in tree_dict:
        counts[int(tree_dict["feature"])] += 1
        count_splits_citree(tree_dict["left_child"], counts)
        count_splits_citree(tree_dict["right_child"], counts)
    return counts


def run_feature_selection_experiment(task: str, cfg: FigureRunConfig) -> pd.DataFrame:
    """Run feature selection experiment across methods.

    Parameters
    ----------
    task : str
        Either 'classification' or 'regression'.

    Returns
    -------
    pd.DataFrame
        Results with columns: method, feature, count, repeat, is_informative
    """
    results = []
    n_informative = 5
    n_features = 20

    for repeat in range(cfg.n_repeats):
        print(f"  Repeat {repeat + 1}/{cfg.n_repeats}")

        # Generate data with explicit informative features
        # shuffle=False keeps informative features at indices 0 to n_informative-1
        if task == "classification":
            X, y = make_classification(
                n_samples=cfg.n_samples_main,
                n_features=n_features,
                n_informative=n_informative,
                n_redundant=0,
                n_clusters_per_class=1,
                flip_y=0.05,
                shuffle=False,
                random_state=cfg.seed + repeat,
            )
        else:
            X, y = make_regression(
                n_samples=cfg.n_samples_main,
                n_features=n_features,
                n_informative=n_informative,
                noise=10.0,
                shuffle=False,
                random_state=cfg.seed + repeat,
            )

        if task == "classification":
            models = {
                "cit": ConditionalInferenceTreeClassifier(random_state=cfg.seed + repeat, verbose=0),
                "cif": ConditionalInferenceForestClassifier(
                    n_estimators=cfg.n_estimators_forest,
                    random_state=cfg.seed + repeat,
                    n_jobs=cfg.n_jobs,
                    verbose=0,
                ),
                "dt": DecisionTreeClassifier(random_state=cfg.seed + repeat),
                "rf": RandomForestClassifier(
                    n_estimators=cfg.n_estimators_forest, random_state=cfg.seed + repeat, n_jobs=cfg.n_jobs
                ),
                "et": ExtraTreesClassifier(
                    n_estimators=cfg.n_estimators_forest, random_state=cfg.seed + repeat, n_jobs=cfg.n_jobs
                ),
            }
        else:
            models = {
                "cit": ConditionalInferenceTreeRegressor(random_state=cfg.seed + repeat, verbose=0),
                "cif": ConditionalInferenceForestRegressor(
                    n_estimators=cfg.n_estimators_forest,
                    random_state=cfg.seed + repeat,
                    n_jobs=cfg.n_jobs,
                    verbose=0,
                ),
                "dt": DecisionTreeRegressor(random_state=cfg.seed + repeat),
                "rf": RandomForestRegressor(
                    n_estimators=cfg.n_estimators_forest, random_state=cfg.seed + repeat, n_jobs=cfg.n_jobs
                ),
                "et": ExtraTreesRegressor(
                    n_estimators=cfg.n_estimators_forest, random_state=cfg.seed + repeat, n_jobs=cfg.n_jobs
                ),
            }

        for name, model in models.items():
            model.fit(X, y)

            # Count splits
            if name in ["cit", "cif"]:
                if name == "cif":
                    counts: Counter = Counter()
                    for tree in model.estimators_:
                        count_splits_citree(tree.tree_, counts)
                else:
                    counts = count_splits_citree(model.tree_)
            else:
                counts = count_splits_sklearn(model)

            # Record results
            for feature in range(n_features):
                results.append(
                    {
                        "method": name,
                        "feature": feature,
                        "count": counts.get(feature, 0),
                        "repeat": repeat,
                        "is_informative": feature < n_informative,
                    }
                )

    return pd.DataFrame(results)


def plot_feature_selection_bars(df: pd.DataFrame, task: str, output_path: Path) -> None:
    """Create bar plot showing feature selection counts by method."""
    methods = ["cit", "cif", "dt", "rf", "et"]
    method_labels = {
        "cit": "CITree",
        "cif": "CIForest",
        "dt": "Decision Tree",
        "rf": "Random Forest",
        "et": "Extra Trees",
    }

    fig, axes = plt.subplots(1, len(methods), figsize=(15, 4), sharey=True)

    for ax, method in zip(axes, methods):
        method_df = df[df["method"] == method]

        # Aggregate across repeats
        agg = (
            method_df.groupby("feature")
            .agg({"count": ["mean", "std"], "is_informative": "first"})
            .reset_index()
        )
        agg.columns = ["feature", "mean", "std", "is_informative"]

        # Colors: blue for informative, gray for noise
        colors = ["#2ecc71" if inf else "#bdc3c7" for inf in agg["is_informative"]]

        ax.bar(
            agg["feature"],
            agg["mean"],
            yerr=agg["std"],
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            capsize=2,
        )
        ax.set_xlabel("Feature ID")
        ax.set_title(method_labels[method])
        ax.set_xticks([0, 4, 9, 14, 19])

        # Add vertical line separating informative from noise
        ax.axvline(x=4.5, color="red", linestyle="--", alpha=0.7, linewidth=1)

    axes[0].set_ylabel("Split Count")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", edgecolor="black", label="Informative"),
        Patch(facecolor="#bdc3c7", edgecolor="black", label="Noise"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.99, 0.95))

    plt.suptitle(f"Feature Selection Counts ({task.title()})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_informative_ratio(df: pd.DataFrame, output_path: Path) -> None:
    """Create bar plot showing % of splits on informative features."""
    methods = ["cit", "cif", "dt", "rf", "et"]
    method_labels = ["CITree", "CIForest", "Decision Tree", "Random Forest", "Extra Trees"]

    ratios = []
    for method in methods:
        method_df = df[df["method"] == method]
        informative = method_df[method_df["is_informative"]]["count"].sum()
        total = method_df["count"].sum()
        ratios.append(100 * informative / total if total > 0 else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#27ae60" if r > 50 else "#e74c3c" for r in ratios]
    bars = ax.bar(method_labels, ratios, color=colors, edgecolor="black", linewidth=1)

    ax.axhline(y=50, color="black", linestyle="--", alpha=0.5, label="Random (50%)")
    ax.axhline(y=25, color="gray", linestyle=":", alpha=0.5, label="Expected if 5/20 features")

    ax.set_ylabel("% Splits on Informative Features")
    ax.set_ylim(0, 100)
    ax.legend()

    # Add value labels on bars
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{ratio:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.title("Feature Selection Bias: Informative vs Noise Features")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics and significance tests."""
    methods = df["method"].unique()
    stats_data = []

    for method in methods:
        method_df = df[df["method"] == method]

        # Per-repeat informative ratio
        repeat_ratios = []
        for repeat in method_df["repeat"].unique():
            repeat_df = method_df[method_df["repeat"] == repeat]
            informative = repeat_df[repeat_df["is_informative"]]["count"].sum()
            total = repeat_df["count"].sum()
            if total > 0:
                repeat_ratios.append(100 * informative / total)

        stats_data.append(
            {
                "method": method,
                "informative_ratio_mean": np.mean(repeat_ratios),
                "informative_ratio_std": np.std(repeat_ratios),
                "total_splits_mean": method_df.groupby("repeat")["count"].sum().mean(),
            }
        )

    return pd.DataFrame(stats_data)


def generate_latex_table(stats_df: pd.DataFrame, output_path: Path) -> None:
    """Generate LaTeX table for paper."""
    method_labels = {
        "cit": "CITree",
        "cif": "CIForest",
        "dt": "Decision Tree",
        "rf": "Random Forest",
        "et": "Extra Trees",
    }

    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(
        r"\caption{Feature selection bias across tree-based methods. Higher informative ratio indicates better selection of relevant features.}"
    )
    latex.append(r"\label{tab:feature_selection_bias}")
    latex.append(r"\begin{tabular}{lcc}")
    latex.append(r"\toprule")
    latex.append(r"Method & Informative Ratio (\%) & Total Splits \\")
    latex.append(r"\midrule")

    for _, row in stats_df.sort_values("informative_ratio_mean", ascending=False).iterrows():
        method = method_labels.get(row["method"], row["method"])
        ratio = f"{row['informative_ratio_mean']:.1f} $\\pm$ {row['informative_ratio_std']:.1f}"
        splits = f"{row['total_splits_mean']:.0f}"
        latex.append(f"{method} & {ratio} & {splits} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(latex))
    print(f"Saved: {output_path}")


def run_timing_experiment(cfg: FigureRunConfig) -> pd.DataFrame:
    """Run timing experiments varying key hyperparameters."""
    import time

    results = []
    n_samples = cfg.timing_n_samples
    n_features = cfg.timing_n_features

    # Generate data once
    X, y = make_friedman1(
        n_samples=n_samples, n_features=n_features, noise=1.0, random_state=cfg.seed
    )
    y_binary = (y > np.median(y)).astype(int)

    # Key hyperparameters to vary
    configs = [
        # Baseline configurations
        {"name": "default", "params": {}},
        # Alpha variations
        {"name": "alpha=0.01", "params": {"alpha_selector": 0.01, "alpha_splitter": 0.01}},
        {"name": "alpha=0.10", "params": {"alpha_selector": 0.10, "alpha_splitter": 0.10}},
        {"name": "alpha=0.20", "params": {"alpha_selector": 0.20, "alpha_splitter": 0.20}},
        # Early stopping
        {
            "name": "no_early_stop",
            "params": {"early_stopping_selector": None, "early_stopping_splitter": None},
        },
        # Resamples
        {
            "name": "resamples=100",
            "params": {"n_resamples_selector": 100, "n_resamples_splitter": 100},
        },
        {
            "name": "resamples=1000",
            "params": {"n_resamples_selector": 1000, "n_resamples_splitter": 1000},
        },
        {
            "name": "resamples=auto",
            "params": {"n_resamples_selector": "auto", "n_resamples_splitter": "auto"},
        },
        # Threshold method
        {"name": "histogram", "params": {"threshold_method": "histogram", "max_thresholds": 128}},
        # Feature muting
        {"name": "no_muting", "params": {"feature_muting": False}},
        # Scanning
        {"name": "no_scanning", "params": {"feature_scanning": False, "threshold_scanning": False}},
        # Combined fast
        {
            "name": "fast",
            "params": {
                "early_stopping_selector": "adaptive",
                "early_stopping_splitter": "adaptive",
                "threshold_method": "histogram",
                "max_thresholds": 128,
                "n_resamples_selector": "auto",
                "n_resamples_splitter": "auto",
            },
        },
    ]

    for repeat in range(cfg.timing_repeats):
        print(f"  Timing repeat {repeat + 1}/{cfg.timing_repeats}")
        for config in configs:
            params = {"random_state": cfg.seed + repeat, "verbose": 0, **config["params"]}
            clf = ConditionalInferenceTreeClassifier(**params)

            tic = time.time()
            clf.fit(X, y_binary)
            toc = time.time()

            results.append(
                {
                    "config": config["name"],
                    "repeat": repeat,
                    "time": toc - tic,
                    "depth": clf.depth_,
                }
            )

    return pd.DataFrame(results)


def plot_timing_bars(df: pd.DataFrame, output_path: Path) -> None:
    """Create bar plot showing timing for different configurations."""
    # Aggregate
    agg = (
        df.groupby("config")
        .agg(
            {
                "time": ["mean", "std"],
            }
        )
        .reset_index()
    )
    agg.columns = ["config", "mean", "std"]
    agg = agg.sort_values("mean")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by category
    def get_color(name):
        if "alpha" in name:
            return "#3498db"
        elif "resample" in name:
            return "#e74c3c"
        elif "early" in name or "fast" in name:
            return "#27ae60"
        elif "histogram" in name or "muting" in name or "scanning" in name:
            return "#9b59b6"
        return "#95a5a6"

    colors = [get_color(name) for name in agg["config"]]

    bars = ax.barh(
        agg["config"],
        agg["mean"],
        xerr=agg["std"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=3,
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Configuration")
    ax.set_title("CITree Training Time by Hyperparameter Configuration")

    # Add value labels
    for bar, mean in zip(bars, agg["mean"]):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.2f}s",
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_timing_speedup(df: pd.DataFrame, output_path: Path) -> None:
    """Create bar plot showing speedup relative to slowest config."""
    agg = df.groupby("config")["time"].mean().reset_index()
    baseline = agg[agg["config"] == "no_early_stop"]["time"].values[0]
    agg["speedup"] = baseline / agg["time"]
    agg = agg.sort_values("speedup", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#27ae60" if s > 1.5 else "#f39c12" if s > 1 else "#e74c3c" for s in agg["speedup"]]
    bars = ax.barh(agg["config"], agg["speedup"], color=colors, edgecolor="black", linewidth=0.5)

    ax.axvline(x=1.0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Speedup (relative to no_early_stop)")
    ax.set_ylabel("Configuration")
    ax.set_title("CITree Training Speedup by Configuration")

    for bar, speedup in zip(bars, agg["speedup"]):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{speedup:.1f}x",
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def run_correlated_features_experiment(cfg: FigureRunConfig) -> pd.DataFrame:
    """Test feature selection with correlated features.

    Creates dataset where:
    - Features 0-4: informative
    - Features 5-9: highly correlated with features 0-4 (duplicates with noise)
    - Features 10-19: pure noise

    RF/ET tend to split importance between correlated features.
    CITree should still prefer the original informative features.
    """
    results = []
    n_samples = cfg.n_samples_main
    n_informative = 5

    for repeat in range(cfg.n_repeats):
        print(f"  Correlated repeat {repeat + 1}/{cfg.n_repeats}")
        rng = np.random.RandomState(cfg.seed + repeat)

        # Generate base classification data with explicit informative features
        X_base, y = make_classification(
            n_samples=n_samples,
            n_features=n_informative,
            n_informative=n_informative,
            n_redundant=0,
            n_clusters_per_class=1,
            shuffle=False,
            random_state=cfg.seed + repeat,
        )

        # Create correlated features (copies with small noise)
        X_correlated = X_base + rng.normal(0, 0.1, X_base.shape)

        # Create pure noise features
        X_noise = rng.normal(0, 1, (n_samples, 10))

        # Combine: [informative, correlated, noise]
        X = np.hstack([X_base, X_correlated, X_noise])

        models = {
            "cit": ConditionalInferenceTreeClassifier(random_state=cfg.seed + repeat, verbose=0),
            "cif": ConditionalInferenceForestClassifier(
                n_estimators=cfg.n_estimators_forest,
                random_state=cfg.seed + repeat,
                n_jobs=cfg.n_jobs,
                verbose=0,
            ),
            "rf": RandomForestClassifier(
                n_estimators=cfg.n_estimators_forest, random_state=cfg.seed + repeat, n_jobs=cfg.n_jobs
            ),
            "et": ExtraTreesClassifier(
                n_estimators=cfg.n_estimators_forest, random_state=cfg.seed + repeat, n_jobs=cfg.n_jobs
            ),
        }

        for name, model in models.items():
            model.fit(X, y)  # y is already binary from make_classification

            if name in ["cit", "cif"]:
                if name == "cif":
                    counts: Counter = Counter()
                    for tree in model.estimators_:
                        count_splits_citree(tree.tree_, counts)
                else:
                    counts = count_splits_citree(model.tree_)
            else:
                counts = count_splits_sklearn(model)

            # Categorize features
            informative_count = sum(counts.get(i, 0) for i in range(5))
            correlated_count = sum(counts.get(i, 0) for i in range(5, 10))
            noise_count = sum(counts.get(i, 0) for i in range(10, 20))
            total = informative_count + correlated_count + noise_count

            results.append(
                {
                    "method": name,
                    "repeat": repeat,
                    "informative": informative_count,
                    "correlated": correlated_count,
                    "noise": noise_count,
                    "total": total,
                    "informative_pct": 100 * informative_count / max(total, 1),
                    "correlated_pct": 100 * correlated_count / max(total, 1),
                    "noise_pct": 100 * noise_count / max(total, 1),
                }
            )

    return pd.DataFrame(results)


def plot_correlated_features(df: pd.DataFrame, output_path: Path) -> None:
    """Create stacked bar plot showing feature type selection."""
    methods = ["cit", "cif", "rf", "et"]
    method_labels = ["CITree", "CIForest", "Random Forest", "Extra Trees"]

    # Aggregate
    agg = (
        df.groupby("method")
        .agg(
            {
                "informative_pct": "mean",
                "correlated_pct": "mean",
                "noise_pct": "mean",
            }
        )
        .reindex(methods)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    width = 0.6

    # Stacked bars
    bars1 = ax.bar(x, agg["informative_pct"], width, label="Informative (0-4)", color="#27ae60")
    bars2 = ax.bar(
        x,
        agg["correlated_pct"],
        width,
        bottom=agg["informative_pct"],
        label="Correlated copies (5-9)",
        color="#f39c12",
    )
    bars3 = ax.bar(
        x,
        agg["noise_pct"],
        width,
        bottom=agg["informative_pct"] + agg["correlated_pct"],
        label="Pure noise (10-19)",
        color="#e74c3c",
    )

    ax.set_ylabel("% of Splits")
    ax.set_xlabel("Method")
    ax.set_title(
        "Feature Selection with Correlated Features\n(Features 5-9 are noisy copies of 0-4)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)

    # Add percentage labels
    for i, method in enumerate(methods):
        row = agg.loc[method]
        ax.text(
            i,
            row["informative_pct"] / 2,
            f"{row['informative_pct']:.0f}%",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def run_complexity_vs_accuracy_experiment(cfg: FigureRunConfig) -> pd.DataFrame:
    """Compare tree complexity (depth, n_splits) vs accuracy."""
    results = []

    for repeat in range(cfg.n_repeats):
        print(f"  Complexity repeat {repeat + 1}/{cfg.n_repeats}")

        # Use make_classification with explicit informative features
        X, y = make_classification(
            n_samples=cfg.n_samples_main,
            n_features=20,
            n_informative=5,
            n_redundant=0,
            n_clusters_per_class=1,
            flip_y=0.05,
            shuffle=False,
            random_state=cfg.seed + repeat,
        )

        # Train/test split
        n_train = int(cfg.train_fraction * X.shape[0])
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        models = {
            "cit": ConditionalInferenceTreeClassifier(random_state=cfg.seed + repeat, verbose=0),
            "dt": DecisionTreeClassifier(random_state=cfg.seed + repeat),
            "rf": RandomForestClassifier(
                n_estimators=cfg.n_estimators_forest, random_state=cfg.seed + repeat, n_jobs=cfg.n_jobs
            ),
            "et": ExtraTreesClassifier(
                n_estimators=cfg.n_estimators_forest, random_state=cfg.seed + repeat, n_jobs=cfg.n_jobs
            ),
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            accuracy = np.mean(model.predict(X_test) == y_test)

            # Get complexity metrics
            if name == "cit":
                counts = count_splits_citree(model.tree_)
                n_splits = int(sum(counts.values()))
                depth = model.depth_
            elif name == "dt":
                n_splits = model.tree_.node_count
                depth = model.tree_.max_depth
            else:  # forests
                n_splits = sum(t.tree_.node_count for t in model.estimators_)
                depth = np.mean([t.tree_.max_depth for t in model.estimators_])

            results.append(
                {
                    "method": name,
                    "repeat": repeat,
                    "accuracy": accuracy,
                    "n_splits": n_splits,
                    "depth": depth,
                }
            )

    return pd.DataFrame(results)


def plot_complexity_vs_accuracy(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter plot of complexity vs accuracy."""
    method_labels = {
        "cit": "CITree",
        "dt": "Decision Tree",
        "rf": "Random Forest",
        "et": "Extra Trees",
    }
    colors = {"cit": "#27ae60", "dt": "#3498db", "rf": "#e74c3c", "et": "#9b59b6"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: n_splits vs accuracy
    ax = axes[0]
    for method in ["cit", "dt"]:  # Only single trees for fair comparison
        method_df = df[df["method"] == method]
        ax.scatter(
            method_df["n_splits"],
            method_df["accuracy"],
            label=method_labels[method],
            color=colors[method],
            s=80,
            alpha=0.7,
        )
    ax.set_xlabel("Number of Splits")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Tree Complexity vs Accuracy (Single Trees)")
    ax.legend()

    # Plot 2: Aggregate comparison
    ax = axes[1]
    agg = (
        df.groupby("method")
        .agg(
            {
                "accuracy": ["mean", "std"],
                "n_splits": ["mean", "std"],
            }
        )
        .reset_index()
    )
    agg.columns = ["method", "acc_mean", "acc_std", "splits_mean", "splits_std"]

    for _, row in agg.iterrows():
        method = row["method"]
        ax.errorbar(
            row["splits_mean"],
            row["acc_mean"],
            xerr=row["splits_std"],
            yerr=row["acc_std"],
            fmt="o",
            markersize=12,
            label=method_labels[method],
            color=colors[method],
            capsize=5,
        )

    ax.set_xlabel("Mean Number of Splits")
    ax.set_ylabel("Mean Test Accuracy")
    ax.set_title("Complexity vs Accuracy Trade-off")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def run_signal_strength_experiment(cfg: FigureRunConfig) -> pd.DataFrame:
    """Test performance with varying signal strength (class_sep).

    Uses make_classification with different class_sep values. Higher class_sep = more separable
    classes = easier problem.
    """
    results = []

    # Vary class separation (signal strength)
    class_seps = [0.2, 0.5, 1.0, 2.0, 4.0]

    for class_sep in class_seps:
        print(f"  Signal strength: class_sep={class_sep}")

        for repeat in range(cfg.n_repeats):
            X, y = make_classification(
                n_samples=cfg.n_samples_main,
                n_features=20,
                n_informative=5,
                n_redundant=0,
                n_clusters_per_class=1,
                class_sep=class_sep,
                flip_y=0.01,
                shuffle=False,
                random_state=cfg.seed + repeat,
            )

            # Train/test split
            n_train = int(cfg.train_fraction * X.shape[0])
            X_train, X_test = X[:n_train], X[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]

            models = {
                "cit": ConditionalInferenceTreeClassifier(random_state=cfg.seed + repeat, verbose=0),
                "cif": ConditionalInferenceForestClassifier(
                    n_estimators=cfg.n_estimators_forest,
                    random_state=cfg.seed + repeat,
                    n_jobs=cfg.n_jobs,
                    verbose=0,
                ),
                "dt": DecisionTreeClassifier(random_state=cfg.seed + repeat),
                "rf": RandomForestClassifier(
                    n_estimators=cfg.n_estimators_forest, random_state=cfg.seed + repeat, n_jobs=cfg.n_jobs
                ),
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                accuracy = np.mean(model.predict(X_test) == y_test)

                # Count informative feature selection
                if name in ["cit"]:
                    counts = count_splits_citree(model.tree_)
                elif name == "cif":
                    counts = Counter()
                    for tree in model.estimators_:
                        count_splits_citree(tree.tree_, counts)
                else:
                    counts = count_splits_sklearn(model)

                # Informative feature ratio
                informative = sum(counts.get(i, 0) for i in range(5))
                total = sum(counts.values())
                informative_ratio = informative / total if total > 0 else 0

                results.append(
                    {
                        "method": name,
                        "class_sep": class_sep,
                        "repeat": repeat,
                        "accuracy": accuracy,
                        "informative_ratio": informative_ratio,
                    }
                )

    return pd.DataFrame(results)


def plot_signal_strength(df: pd.DataFrame, output_path: Path) -> None:
    """Plot performance vs signal strength."""
    method_labels = {
        "cit": "CITree",
        "cif": "CIForest",
        "dt": "Decision Tree",
        "rf": "Random Forest",
    }
    colors = {"cit": "#27ae60", "cif": "#2ecc71", "dt": "#3498db", "rf": "#e74c3c"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Accuracy vs class_sep
    ax = axes[0]
    for method in ["cit", "cif", "dt", "rf"]:
        method_df = df[df["method"] == method]
        agg = method_df.groupby("class_sep").agg({"accuracy": ["mean", "std"]}).reset_index()
        agg.columns = ["class_sep", "mean", "std"]
        ax.errorbar(
            agg["class_sep"],
            agg["mean"],
            yerr=agg["std"],
            label=method_labels[method],
            color=colors[method],
            marker="o",
            markersize=8,
            capsize=5,
            linewidth=2,
        )
    ax.set_xlabel("Class Separation (Signal Strength)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy vs Signal Strength")
    ax.legend()

    # Plot 2: Informative ratio vs class_sep
    ax = axes[1]
    for method in ["cit", "cif", "dt", "rf"]:
        method_df = df[df["method"] == method]
        agg = (
            method_df.groupby("class_sep").agg({"informative_ratio": ["mean", "std"]}).reset_index()
        )
        agg.columns = ["class_sep", "mean", "std"]
        ax.errorbar(
            agg["class_sep"],
            agg["mean"],
            yerr=agg["std"],
            label=method_labels[method],
            color=colors[method],
            marker="o",
            markersize=8,
            capsize=5,
            linewidth=2,
        )
    ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="Random (5/20)")
    ax.set_xlabel("Class Separation (Signal Strength)")
    ax.set_ylabel("Informative Feature Ratio")
    ax.set_title("Feature Selection Quality vs Signal Strength")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def run_redundant_features_experiment(cfg: FigureRunConfig) -> pd.DataFrame:
    """Test feature selection with sklearn's redundant feature generation.

    n_redundant creates linear combinations of informative features. This is different from our
    manual correlated features experiment.
    """
    results = []

    # Vary number of redundant features
    n_redundant_values = [0, 2, 5, 10]

    for n_redundant in n_redundant_values:
        print(f"  Redundant features: n_redundant={n_redundant}")
        n_features = 5 + n_redundant + 10  # informative + redundant + noise

        for repeat in range(cfg.n_repeats):
            X, y = make_classification(
                n_samples=cfg.n_samples_main,
                n_features=n_features,
                n_informative=5,
                n_redundant=n_redundant,
                n_repeated=0,
                n_clusters_per_class=1,
                flip_y=0.05,
                shuffle=False,
                random_state=cfg.seed + repeat,
            )

            # Feature layout: [informative (0-4), redundant (5-5+n_redundant), noise (rest)]
            n_informative = 5

            models = {
                "cit": ConditionalInferenceTreeClassifier(random_state=cfg.seed + repeat, verbose=0),
                "cif": ConditionalInferenceForestClassifier(
                    n_estimators=cfg.n_estimators_forest,
                    random_state=cfg.seed + repeat,
                    n_jobs=cfg.n_jobs,
                    verbose=0,
                ),
                "rf": RandomForestClassifier(
                    n_estimators=cfg.n_estimators_forest, random_state=cfg.seed + repeat, n_jobs=cfg.n_jobs
                ),
            }

            for name, model in models.items():
                model.fit(X, y)

                if name in ["cit"]:
                    counts = count_splits_citree(model.tree_)
                elif name == "cif":
                    counts = Counter()
                    for tree in model.estimators_:
                        count_splits_citree(tree.tree_, counts)
                else:
                    counts = count_splits_sklearn(model)

                # Categorize features
                informative = sum(counts.get(i, 0) for i in range(n_informative))
                redundant = sum(
                    counts.get(i, 0) for i in range(n_informative, n_informative + n_redundant)
                )
                noise = sum(
                    counts.get(i, 0) for i in range(n_informative + n_redundant, n_features)
                )
                total = informative + redundant + noise

                results.append(
                    {
                        "method": name,
                        "n_redundant": n_redundant,
                        "repeat": repeat,
                        "informative_pct": 100 * informative / max(total, 1),
                        "redundant_pct": 100 * redundant / max(total, 1),
                        "noise_pct": 100 * noise / max(total, 1),
                    }
                )

    return pd.DataFrame(results)


def plot_redundant_features(df: pd.DataFrame, output_path: Path) -> None:
    """Plot feature selection with varying redundant features."""
    methods = ["cit", "cif", "rf"]
    method_labels = {"cit": "CITree", "cif": "CIForest", "rf": "Random Forest"}

    fig, axes = plt.subplots(1, len(methods), figsize=(14, 5), sharey=True)

    for ax, method in zip(axes, methods):
        method_df = df[df["method"] == method]
        agg = (
            method_df.groupby("n_redundant")
            .agg(
                {
                    "informative_pct": "mean",
                    "redundant_pct": "mean",
                    "noise_pct": "mean",
                }
            )
            .reset_index()
        )

        x = np.arange(len(agg))
        width = 0.6

        bars1 = ax.bar(x, agg["informative_pct"], width, label="Informative", color="#27ae60")
        bars2 = ax.bar(
            x,
            agg["redundant_pct"],
            width,
            bottom=agg["informative_pct"],
            label="Redundant",
            color="#f39c12",
        )
        bars3 = ax.bar(
            x,
            agg["noise_pct"],
            width,
            bottom=agg["informative_pct"] + agg["redundant_pct"],
            label="Noise",
            color="#e74c3c",
        )

        ax.set_xlabel("Number of Redundant Features")
        ax.set_title(method_labels[method])
        ax.set_xticks(x)
        ax.set_xticklabels(agg["n_redundant"].astype(int))
        ax.set_ylim(0, 100)

    axes[0].set_ylabel("% of Splits")
    axes[0].legend(loc="upper right")

    plt.suptitle(
        "Feature Selection with Redundant Features\n(Redundant = linear combinations of informative)",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def run_multiclass_experiment(cfg: FigureRunConfig) -> pd.DataFrame:
    """Test performance on multi-class problems."""
    results = []

    # Vary number of classes
    n_classes_values = [2, 3, 5, 10]

    for n_classes in n_classes_values:
        print(f"  Multi-class: n_classes={n_classes}")

        for repeat in range(cfg.n_repeats):
            X, y = make_classification(
                n_samples=cfg.n_samples_main,
                n_features=20,
                n_informative=10,
                n_redundant=0,
                n_classes=n_classes,
                n_clusters_per_class=1,
                flip_y=0.05,
                shuffle=False,
                random_state=cfg.seed + repeat,
            )

            # Train/test split
            n_train = int(cfg.train_fraction * X.shape[0])
            X_train, X_test = X[:n_train], X[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]

            models = {
                "cit": ConditionalInferenceTreeClassifier(random_state=cfg.seed + repeat, verbose=0),
                "cif": ConditionalInferenceForestClassifier(
                    n_estimators=cfg.n_estimators_forest,
                    random_state=cfg.seed + repeat,
                    n_jobs=cfg.n_jobs,
                    verbose=0,
                ),
                "dt": DecisionTreeClassifier(random_state=cfg.seed + repeat),
                "rf": RandomForestClassifier(
                    n_estimators=cfg.n_estimators_forest, random_state=cfg.seed + repeat, n_jobs=cfg.n_jobs
                ),
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                accuracy = np.mean(model.predict(X_test) == y_test)

                # Count informative feature selection
                if name in ["cit"]:
                    counts = count_splits_citree(model.tree_)
                elif name == "cif":
                    counts = Counter()
                    for tree in model.estimators_:
                        count_splits_citree(tree.tree_, counts)
                else:
                    counts = count_splits_sklearn(model)

                informative = sum(counts.get(i, 0) for i in range(10))
                total = sum(counts.values())
                informative_ratio = informative / total if total > 0 else 0

                results.append(
                    {
                        "method": name,
                        "n_classes": n_classes,
                        "repeat": repeat,
                        "accuracy": accuracy,
                        "informative_ratio": informative_ratio,
                    }
                )

    return pd.DataFrame(results)


def plot_multiclass(df: pd.DataFrame, output_path: Path) -> None:
    """Plot performance on multi-class problems."""
    method_labels = {
        "cit": "CITree",
        "cif": "CIForest",
        "dt": "Decision Tree",
        "rf": "Random Forest",
    }
    colors = {"cit": "#27ae60", "cif": "#2ecc71", "dt": "#3498db", "rf": "#e74c3c"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Accuracy vs n_classes
    ax = axes[0]
    for method in ["cit", "cif", "dt", "rf"]:
        method_df = df[df["method"] == method]
        agg = method_df.groupby("n_classes").agg({"accuracy": ["mean", "std"]}).reset_index()
        agg.columns = ["n_classes", "mean", "std"]
        ax.errorbar(
            agg["n_classes"],
            agg["mean"],
            yerr=agg["std"],
            label=method_labels[method],
            color=colors[method],
            marker="o",
            markersize=8,
            capsize=5,
            linewidth=2,
        )
    ax.set_xlabel("Number of Classes")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy vs Number of Classes")
    ax.legend()

    # Plot 2: Informative ratio vs n_classes
    ax = axes[1]
    for method in ["cit", "cif", "dt", "rf"]:
        method_df = df[df["method"] == method]
        agg = (
            method_df.groupby("n_classes").agg({"informative_ratio": ["mean", "std"]}).reset_index()
        )
        agg.columns = ["n_classes", "mean", "std"]
        ax.errorbar(
            agg["n_classes"],
            agg["mean"],
            yerr=agg["std"],
            label=method_labels[method],
            color=colors[method],
            marker="o",
            markersize=8,
            capsize=5,
            linewidth=2,
        )
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random (10/20)")
    ax.set_xlabel("Number of Classes")
    ax.set_ylabel("Informative Feature Ratio")
    ax.set_title("Feature Selection Quality vs Number of Classes")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def run_imbalanced_experiment(cfg: FigureRunConfig) -> pd.DataFrame:
    """Test performance with class imbalance."""
    results = []

    # Vary imbalance ratio (minority class weight)
    weights_values = [
        [0.5, 0.5],  # Balanced
        [0.7, 0.3],  # Moderate imbalance
        [0.9, 0.1],  # Heavy imbalance
        [0.95, 0.05],  # Extreme imbalance
    ]

    for weights in weights_values:
        imbalance_ratio = max(weights) / min(weights)
        print(f"  Imbalance: {weights[0]:.2f}/{weights[1]:.2f} (ratio={imbalance_ratio:.1f})")

        for repeat in range(cfg.n_repeats):
            X, y = make_classification(
                n_samples=cfg.n_samples_main,
                n_features=20,
                n_informative=5,
                n_redundant=0,
                n_clusters_per_class=1,
                weights=weights,
                flip_y=0.01,
                shuffle=False,
                random_state=cfg.seed + repeat,
            )

            # Train/test split
            n_train = int(cfg.train_fraction * X.shape[0])
            X_train, X_test = X[:n_train], X[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]

            models = {
                "cit": ConditionalInferenceTreeClassifier(random_state=cfg.seed + repeat, verbose=0),
                "cif": ConditionalInferenceForestClassifier(
                    n_estimators=cfg.n_estimators_forest,
                    random_state=cfg.seed + repeat,
                    n_jobs=cfg.n_jobs,
                    verbose=0,
                ),
                "dt": DecisionTreeClassifier(random_state=cfg.seed + repeat),
                "rf": RandomForestClassifier(
                    n_estimators=cfg.n_estimators_forest, random_state=cfg.seed + repeat, n_jobs=cfg.n_jobs
                ),
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = np.mean(y_pred == y_test)

                # Balanced accuracy (average of recalls)
                from sklearn.metrics import balanced_accuracy_score

                balanced_acc = balanced_accuracy_score(y_test, y_pred)

                results.append(
                    {
                        "method": name,
                        "imbalance_ratio": imbalance_ratio,
                        "repeat": repeat,
                        "accuracy": accuracy,
                        "balanced_accuracy": balanced_acc,
                    }
                )

    return pd.DataFrame(results)


def plot_imbalanced(df: pd.DataFrame, output_path: Path) -> None:
    """Plot performance with class imbalance."""
    method_labels = {
        "cit": "CITree",
        "cif": "CIForest",
        "dt": "Decision Tree",
        "rf": "Random Forest",
    }
    colors = {"cit": "#27ae60", "cif": "#2ecc71", "dt": "#3498db", "rf": "#e74c3c"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Accuracy vs imbalance
    ax = axes[0]
    for method in ["cit", "cif", "dt", "rf"]:
        method_df = df[df["method"] == method]
        agg = method_df.groupby("imbalance_ratio").agg({"accuracy": ["mean", "std"]}).reset_index()
        agg.columns = ["imbalance_ratio", "mean", "std"]
        ax.errorbar(
            agg["imbalance_ratio"],
            agg["mean"],
            yerr=agg["std"],
            label=method_labels[method],
            color=colors[method],
            marker="o",
            markersize=8,
            capsize=5,
            linewidth=2,
        )
    ax.set_xlabel("Class Imbalance Ratio")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy vs Class Imbalance")
    ax.legend()
    ax.set_xscale("log")

    # Plot 2: Balanced accuracy vs imbalance
    ax = axes[1]
    for method in ["cit", "cif", "dt", "rf"]:
        method_df = df[df["method"] == method]
        agg = (
            method_df.groupby("imbalance_ratio")
            .agg({"balanced_accuracy": ["mean", "std"]})
            .reset_index()
        )
        agg.columns = ["imbalance_ratio", "mean", "std"]
        ax.errorbar(
            agg["imbalance_ratio"],
            agg["mean"],
            yerr=agg["std"],
            label=method_labels[method],
            color=colors[method],
            marker="o",
            markersize=8,
            capsize=5,
            linewidth=2,
        )
    ax.set_xlabel("Class Imbalance Ratio")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Balanced Accuracy vs Class Imbalance")
    ax.legend()
    ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def run_sample_size_experiment(cfg: FigureRunConfig) -> pd.DataFrame:
    """Test performance with varying sample sizes."""
    results = []

    # Vary sample size
    n_samples_values = list(cfg.sample_size_grid)

    for n_samples in n_samples_values:
        print(f"  Sample size: n={n_samples}")

        for repeat in range(cfg.n_repeats):
            X, y = make_classification(
                n_samples=n_samples,
                n_features=20,
                n_informative=5,
                n_redundant=0,
                n_clusters_per_class=1,
                flip_y=0.05,
                shuffle=False,
                random_state=cfg.seed + repeat,
            )

            # Train/test split (80/20)
            n_train = int(cfg.train_fraction * n_samples)
            X_train, X_test = X[:n_train], X[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]

            models = {
                "cit": ConditionalInferenceTreeClassifier(random_state=cfg.seed + repeat, verbose=0),
                "cif": ConditionalInferenceForestClassifier(
                    n_estimators=cfg.n_estimators_forest,
                    random_state=cfg.seed + repeat,
                    n_jobs=cfg.n_jobs,
                    verbose=0,
                ),
                "dt": DecisionTreeClassifier(random_state=cfg.seed + repeat),
                "rf": RandomForestClassifier(
                    n_estimators=cfg.n_estimators_forest, random_state=cfg.seed + repeat, n_jobs=cfg.n_jobs
                ),
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                accuracy = np.mean(model.predict(X_test) == y_test)

                # Count informative feature selection
                if name in ["cit"]:
                    counts = count_splits_citree(model.tree_)
                elif name == "cif":
                    counts = Counter()
                    for tree in model.estimators_:
                        count_splits_citree(tree.tree_, counts)
                else:
                    counts = count_splits_sklearn(model)

                informative = sum(counts.get(i, 0) for i in range(5))
                total = sum(counts.values())
                informative_ratio = informative / total if total > 0 else 0

                results.append(
                    {
                        "method": name,
                        "n_samples": n_samples,
                        "repeat": repeat,
                        "accuracy": accuracy,
                        "informative_ratio": informative_ratio,
                    }
                )

    return pd.DataFrame(results)


def plot_sample_size(df: pd.DataFrame, output_path: Path) -> None:
    """Plot performance vs sample size."""
    method_labels = {
        "cit": "CITree",
        "cif": "CIForest",
        "dt": "Decision Tree",
        "rf": "Random Forest",
    }
    colors = {"cit": "#27ae60", "cif": "#2ecc71", "dt": "#3498db", "rf": "#e74c3c"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Accuracy vs n_samples
    ax = axes[0]
    for method in ["cit", "cif", "dt", "rf"]:
        method_df = df[df["method"] == method]
        agg = method_df.groupby("n_samples").agg({"accuracy": ["mean", "std"]}).reset_index()
        agg.columns = ["n_samples", "mean", "std"]
        ax.errorbar(
            agg["n_samples"],
            agg["mean"],
            yerr=agg["std"],
            label=method_labels[method],
            color=colors[method],
            marker="o",
            markersize=8,
            capsize=5,
            linewidth=2,
        )
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy vs Sample Size")
    ax.legend()

    # Plot 2: Informative ratio vs n_samples
    ax = axes[1]
    for method in ["cit", "cif", "dt", "rf"]:
        method_df = df[df["method"] == method]
        agg = (
            method_df.groupby("n_samples").agg({"informative_ratio": ["mean", "std"]}).reset_index()
        )
        agg.columns = ["n_samples", "mean", "std"]
        ax.errorbar(
            agg["n_samples"],
            agg["mean"],
            yerr=agg["std"],
            label=method_labels[method],
            color=colors[method],
            marker="o",
            markersize=8,
            capsize=5,
            linewidth=2,
        )
    ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="Random (5/20)")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Informative Feature Ratio")
    ax.set_title("Feature Selection Quality vs Sample Size")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def run_regression_experiment(cfg: FigureRunConfig) -> pd.DataFrame:
    """Run feature selection experiment for regression.

    Uses make_regression with shuffle=False to have known informative features.
    """
    results = []
    n_informative = 5
    n_features = 20

    for repeat in range(cfg.n_repeats):
        print(f"  Regression repeat {repeat + 1}/{cfg.n_repeats}")

        X, y = make_regression(
            n_samples=cfg.n_samples_main,
            n_features=n_features,
            n_informative=n_informative,
            noise=10.0,
            shuffle=False,
            random_state=cfg.seed + repeat,
        )

        # Train/test split
        n_train = int(cfg.train_fraction * X.shape[0])
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        models = {
            "cit": ConditionalInferenceTreeRegressor(random_state=cfg.seed + repeat, verbose=0),
            "cif": ConditionalInferenceForestRegressor(
                n_estimators=cfg.n_estimators_forest,
                random_state=cfg.seed + repeat,
                n_jobs=cfg.n_jobs,
                verbose=0,
            ),
            "dt": DecisionTreeRegressor(random_state=cfg.seed + repeat),
            "rf": RandomForestRegressor(
                n_estimators=cfg.n_estimators_forest, random_state=cfg.seed + repeat, n_jobs=cfg.n_jobs
            ),
            "et": ExtraTreesRegressor(
                n_estimators=cfg.n_estimators_forest, random_state=cfg.seed + repeat, n_jobs=cfg.n_jobs
            ),
        }

        for name, model in models.items():
            model.fit(X_train, y_train)

            # R2 score
            from sklearn.metrics import r2_score

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            # Count splits
            if name in ["cit", "cif"]:
                if name == "cif":
                    counts: Counter = Counter()
                    for tree in model.estimators_:
                        count_splits_citree(tree.tree_, counts)
                else:
                    counts = count_splits_citree(model.tree_)
            else:
                counts = count_splits_sklearn(model)

            # Informative feature ratio
            informative = sum(counts.get(i, 0) for i in range(n_informative))
            total = sum(counts.values())

            results.append(
                {
                    "method": name,
                    "repeat": repeat,
                    "r2": r2,
                    "informative_count": informative,
                    "total_splits": total,
                    "informative_ratio": informative / total if total > 0 else 0,
                }
            )

    return pd.DataFrame(results)


def plot_regression_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Plot regression feature selection comparison."""
    methods = ["cit", "cif", "dt", "rf", "et"]
    method_labels = ["CITree", "CIForest", "Decision Tree", "Random Forest", "Extra Trees"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Informative ratio
    ax = axes[0]
    agg = df.groupby("method").agg({"informative_ratio": ["mean", "std"]}).reset_index()
    agg.columns = ["method", "mean", "std"]
    agg = agg.set_index("method").reindex(methods).reset_index()

    colors = ["#27ae60" if r > 0.5 else "#e74c3c" for r in agg["mean"]]
    bars = ax.bar(
        method_labels,
        agg["mean"] * 100,
        yerr=agg["std"] * 100,
        color=colors,
        edgecolor="black",
        linewidth=1,
        capsize=5,
    )
    ax.axhline(y=25, color="gray", linestyle="--", alpha=0.5, label="Random (5/20)")
    ax.set_ylabel("% Splits on Informative Features")
    ax.set_title("Regression: Feature Selection Quality")
    ax.set_ylim(0, 100)

    for bar, mean in zip(bars, agg["mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{mean * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 2: R2 score
    ax = axes[1]
    agg = df.groupby("method").agg({"r2": ["mean", "std"]}).reset_index()
    agg.columns = ["method", "mean", "std"]
    agg = agg.set_index("method").reindex(methods).reset_index()

    bars = ax.bar(
        method_labels,
        agg["mean"],
        yerr=agg["std"],
        color="#3498db",
        edgecolor="black",
        linewidth=1,
        capsize=5,
    )
    ax.set_ylabel("R² Score")
    ax.set_title("Regression: Prediction Quality")
    ax.set_ylim(0, 1)

    for bar, mean in zip(bars, agg["mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def run_high_dimensional_experiment(cfg: FigureRunConfig) -> pd.DataFrame:
    """Test performance on high-dimensional data (p >> n).

    Uses make_classification with shuffle=False so first k features are guaranteed to be informative
    (ground truth known).
    """
    results = []

    # Vary p/n ratio (p >> n regimes)
    n = cfg.highdim_n
    k = 5
    p_over_n_grid = [0.25, 1.0, 2.5, 5.0]
    p_grid = [max(10, int(round(r * n))) for r in p_over_n_grid]

    for p in p_grid:
        print(f"  High-dim: n={n}, p={p}")

        for repeat in range(cfg.highdim_repeats):
            # Use make_classification with explicit informative features
            # shuffle=False ensures features 0 to k-1 are informative
            X, y = make_classification(
                n_samples=n,
                n_features=p,
                n_informative=k,
                n_redundant=0,
                n_clusters_per_class=1,
                flip_y=0.05,
                shuffle=False,
                random_state=cfg.seed + repeat,
            )

            # Train/test split
            n_train = int(cfg.train_fraction * n)
            X_train, X_test = X[:n_train], X[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]

            models = {
                "cit": ConditionalInferenceTreeClassifier(random_state=cfg.seed + repeat, verbose=0),
                "cif": ConditionalInferenceForestClassifier(
                    n_estimators=cfg.n_estimators_forest,
                    random_state=cfg.seed + repeat,
                    n_jobs=cfg.n_jobs,
                    verbose=0,
                ),
                "rf": RandomForestClassifier(
                    n_estimators=cfg.n_estimators_forest, random_state=cfg.seed + repeat, n_jobs=cfg.n_jobs
                ),
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                accuracy = np.mean(model.predict(X_test) == y_test)

                # Count how many of top-k selected features are truly informative
                if name in ["cit"]:
                    counts = count_splits_citree(model.tree_)
                elif name == "cif":
                    counts = Counter()
                    for tree in model.estimators_:
                        count_splits_citree(tree.tree_, counts)
                else:
                    counts = count_splits_sklearn(model)

                # Precision: of selected features, how many are informative?
                top_features = [f for f, _ in counts.most_common(k)]
                true_positives = sum(1 for f in top_features if f < k)
                precision_at_k = true_positives / k if top_features else 0

                results.append(
                    {
                        "method": name,
                        "n": n,
                        "p": p,
                        "p_over_n": p / n,
                        "repeat": repeat,
                        "accuracy": accuracy,
                        "precision_at_k": precision_at_k,
                    }
                )

    return pd.DataFrame(results)


def plot_high_dimensional(df: pd.DataFrame, output_path: Path) -> None:
    """Plot performance vs dimensionality."""
    method_labels = {"cit": "CITree", "cif": "CIForest", "rf": "Random Forest"}
    colors = {"cit": "#27ae60", "cif": "#2ecc71", "rf": "#e74c3c"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Accuracy vs p/n ratio
    ax = axes[0]
    for method in ["cit", "cif", "rf"]:
        method_df = df[df["method"] == method]
        agg = method_df.groupby("p_over_n").agg({"accuracy": ["mean", "std"]}).reset_index()
        agg.columns = ["p_over_n", "mean", "std"]
        ax.errorbar(
            agg["p_over_n"],
            agg["mean"],
            yerr=agg["std"],
            label=method_labels[method],
            color=colors[method],
            marker="o",
            markersize=8,
            capsize=5,
            linewidth=2,
        )
    ax.set_xlabel("p/n Ratio (Features / Samples)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy in High Dimensions")
    ax.legend()
    ax.set_xscale("log")

    # Plot 2: Feature selection precision vs p/n ratio
    ax = axes[1]
    for method in ["cit", "cif", "rf"]:
        method_df = df[df["method"] == method]
        agg = method_df.groupby("p_over_n").agg({"precision_at_k": ["mean", "std"]}).reset_index()
        agg.columns = ["p_over_n", "mean", "std"]
        ax.errorbar(
            agg["p_over_n"],
            agg["mean"],
            yerr=agg["std"],
            label=method_labels[method],
            color=colors[method],
            marker="o",
            markersize=8,
            capsize=5,
            linewidth=2,
        )
    ax.set_xlabel("p/n Ratio (Features / Samples)")
    ax.set_ylabel("Precision@k (True Informative in Top-k)")
    ax.set_title("Feature Selection Quality in High Dimensions")
    ax.legend()
    ax.set_xscale("log")
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    """Generate figures (optionally a subset)."""
    args = _parse_args()
    cfg = _resolve_config(args)

    selected = set(args.only) if args.only else set(FIGURE_IDS)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_figures": sorted(selected),
        **asdict(cfg),
    }
    (CACHE_DIR / "generate_figures_run.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True))

    print("=" * 60)
    print(f"Profile: {cfg.profile}")
    print(
        f"n_samples_main={cfg.n_samples_main} | n_repeats={cfg.n_repeats} | "
        f"n_estimators_forest={cfg.n_estimators_forest} | n_jobs={cfg.n_jobs}"
    )
    print(f"Selected figures: {', '.join(sorted(selected))}")

    # 1. Feature selection bias figures (classification)
    if "feature_selection" in selected:
        print("=" * 60)
        print("1. Running feature selection bias experiment...")
        clf_df = run_feature_selection_experiment("classification", cfg)

        print("\nGenerating classification figures...")
        plot_feature_selection_bars(clf_df, "classification", FIGURES_DIR / "feature_selection_clf.png")
        plot_informative_ratio(clf_df, FIGURES_DIR / "informative_ratio.png")

        print("\nComputing statistics...")
        stats_df = compute_statistics(clf_df)
        print(stats_df.to_string(index=False))

        print("\nGenerating LaTeX table...")
        generate_latex_table(stats_df, TABLES_DIR / "feature_selection_table.tex")
        clf_df.to_parquet(CACHE_DIR / "feature_selection_data.parquet")

    # 2. Timing figures
    if "timing" in selected:
        print("\n" + "=" * 60)
        print("2. Running timing experiment...")
        timing_df = run_timing_experiment(cfg)

        print("\nGenerating timing figures...")
        plot_timing_bars(timing_df, FIGURES_DIR / "timing_bars.png")
        plot_timing_speedup(timing_df, FIGURES_DIR / "timing_speedup.png")
        timing_df.to_parquet(CACHE_DIR / "timing_data.parquet")

    # 3. Correlated features experiment
    if "correlated" in selected:
        print("\n" + "=" * 60)
        print("3. Running correlated features experiment...")
        corr_df = run_correlated_features_experiment(cfg)

        print("\nGenerating correlated features figure...")
        plot_correlated_features(corr_df, FIGURES_DIR / "correlated_features.png")
        corr_df.to_parquet(CACHE_DIR / "correlated_features_data.parquet")

        print("\nCorrelated features summary:")
        print(
            corr_df.groupby("method")[["informative_pct", "correlated_pct", "noise_pct"]]
            .mean()
            .round(1)
        )

    # 4. Complexity vs accuracy
    if "complexity" in selected:
        print("\n" + "=" * 60)
        print("4. Running complexity vs accuracy experiment...")
        complexity_df = run_complexity_vs_accuracy_experiment(cfg)

        print("\nGenerating complexity figure...")
        plot_complexity_vs_accuracy(complexity_df, FIGURES_DIR / "complexity_vs_accuracy.png")
        complexity_df.to_parquet(CACHE_DIR / "complexity_data.parquet")

    # 5. High-dimensional experiment
    if "highdim" in selected:
        print("\n" + "=" * 60)
        print("5. Running high-dimensional experiment...")
        highdim_df = run_high_dimensional_experiment(cfg)

        print("\nGenerating high-dimensional figure...")
        plot_high_dimensional(highdim_df, FIGURES_DIR / "high_dimensional.png")
        highdim_df.to_parquet(CACHE_DIR / "high_dimensional_data.parquet")

    # 6. Signal strength experiment
    if "signal" in selected:
        print("\n" + "=" * 60)
        print("6. Running signal strength experiment...")
        signal_df = run_signal_strength_experiment(cfg)

        print("\nGenerating signal strength figure...")
        plot_signal_strength(signal_df, FIGURES_DIR / "signal_strength.png")
        signal_df.to_parquet(CACHE_DIR / "signal_strength_data.parquet")

    # 7. Redundant features experiment
    if "redundant" in selected:
        print("\n" + "=" * 60)
        print("7. Running redundant features experiment...")
        redundant_df = run_redundant_features_experiment(cfg)

        print("\nGenerating redundant features figure...")
        plot_redundant_features(redundant_df, FIGURES_DIR / "redundant_features.png")
        redundant_df.to_parquet(CACHE_DIR / "redundant_features_data.parquet")

    # 8. Multi-class experiment
    if "multiclass" in selected:
        print("\n" + "=" * 60)
        print("8. Running multi-class experiment...")
        multiclass_df = run_multiclass_experiment(cfg)

        print("\nGenerating multi-class figure...")
        plot_multiclass(multiclass_df, FIGURES_DIR / "multiclass.png")
        multiclass_df.to_parquet(CACHE_DIR / "multiclass_data.parquet")

    # 9. Class imbalance experiment
    if "imbalanced" in selected:
        print("\n" + "=" * 60)
        print("9. Running class imbalance experiment...")
        imbalance_df = run_imbalanced_experiment(cfg)

        print("\nGenerating imbalance figure...")
        plot_imbalanced(imbalance_df, FIGURES_DIR / "imbalanced.png")
        imbalance_df.to_parquet(CACHE_DIR / "imbalanced_data.parquet")

    # 10. Sample size experiment
    if "sample_size" in selected:
        print("\n" + "=" * 60)
        print("10. Running sample size experiment...")
        sample_df = run_sample_size_experiment(cfg)

        print("\nGenerating sample size figure...")
        plot_sample_size(sample_df, FIGURES_DIR / "sample_size.png")
        sample_df.to_parquet(CACHE_DIR / "sample_size_data.parquet")

    # 11. Regression experiment
    if "regression" in selected:
        print("\n" + "=" * 60)
        print("11. Running regression experiment...")
        reg_df = run_regression_experiment(cfg)

        print("\nGenerating regression figure...")
        plot_regression_comparison(reg_df, FIGURES_DIR / "regression_comparison.png")
        reg_df.to_parquet(CACHE_DIR / "regression_data.parquet")

        print("\nRegression summary:")
        print(reg_df.groupby("method")[["informative_ratio", "r2"]].mean().round(3))

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Figures directory: {FIGURES_DIR}")
    print(f"Tables directory: {TABLES_DIR}")
    print(f"Cache directory: {CACHE_DIR}")
    print("\nFigures created:")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
