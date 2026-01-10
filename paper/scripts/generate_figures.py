"""Generate figures for citrees paper.

Creates visualizations comparing feature selection behavior across methods:
- Bar plots showing feature selection counts (informative vs noise)
- Critical difference diagrams for method rankings
- Box plots for downstream performance

Usage:
    uv run python scripts/generate_figures.py
"""
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import make_classification, make_friedman1
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
)

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "figures"
RANDOM_STATE = 1718
N_REPEATS = 10


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


def run_feature_selection_experiment(task: str = "classification") -> pd.DataFrame:
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

    for repeat in range(N_REPEATS):
        print(f"  Repeat {repeat + 1}/{N_REPEATS}")

        # Generate data - Friedman1 has 5 informative features
        X, y = make_friedman1(
            n_samples=500,
            n_features=n_features,
            noise=1.0,
            random_state=RANDOM_STATE + repeat,
        )

        if task == "classification":
            y = (y > np.median(y)).astype(int)
            models = {
                "cit": ConditionalInferenceTreeClassifier(random_state=RANDOM_STATE + repeat),
                "cif": ConditionalInferenceForestClassifier(
                    n_estimators=100, random_state=RANDOM_STATE + repeat, n_jobs=-1, verbose=0
                ),
                "dt": DecisionTreeClassifier(random_state=RANDOM_STATE + repeat),
                "rf": RandomForestClassifier(
                    n_estimators=100, random_state=RANDOM_STATE + repeat, n_jobs=-1
                ),
                "et": ExtraTreesClassifier(
                    n_estimators=100, random_state=RANDOM_STATE + repeat, n_jobs=-1
                ),
            }
        else:
            models = {
                "cit": ConditionalInferenceTreeRegressor(random_state=RANDOM_STATE + repeat),
                "cif": ConditionalInferenceForestRegressor(
                    n_estimators=100, random_state=RANDOM_STATE + repeat, n_jobs=-1, verbose=0
                ),
                "dt": DecisionTreeRegressor(random_state=RANDOM_STATE + repeat),
                "rf": RandomForestRegressor(
                    n_estimators=100, random_state=RANDOM_STATE + repeat, n_jobs=-1
                ),
                "et": ExtraTreesRegressor(
                    n_estimators=100, random_state=RANDOM_STATE + repeat, n_jobs=-1
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
                results.append({
                    "method": name,
                    "feature": feature,
                    "count": counts.get(feature, 0),
                    "repeat": repeat,
                    "is_informative": feature < n_informative,
                })

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
        agg = method_df.groupby("feature").agg({
            "count": ["mean", "std"],
            "is_informative": "first"
        }).reset_index()
        agg.columns = ["feature", "mean", "std", "is_informative"]

        # Colors: blue for informative, gray for noise
        colors = ["#2ecc71" if inf else "#bdc3c7" for inf in agg["is_informative"]]

        ax.bar(agg["feature"], agg["mean"], yerr=agg["std"], color=colors,
               edgecolor="black", linewidth=0.5, capsize=2)
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
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{ratio:.1f}%", ha="center", va="bottom", fontsize=10)

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

        stats_data.append({
            "method": method,
            "informative_ratio_mean": np.mean(repeat_ratios),
            "informative_ratio_std": np.std(repeat_ratios),
            "total_splits_mean": method_df.groupby("repeat")["count"].sum().mean(),
        })

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
    latex.append(r"\caption{Feature selection bias across tree-based methods. Higher informative ratio indicates better selection of relevant features.}")
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


def run_timing_experiment() -> pd.DataFrame:
    """Run timing experiments varying key hyperparameters."""
    import time

    results = []
    n_samples = 500
    n_features = 100

    # Generate data once
    X, y = make_friedman1(n_samples=n_samples, n_features=n_features, noise=1.0, random_state=RANDOM_STATE)
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
        {"name": "no_early_stop", "params": {"early_stopping_selector": False, "early_stopping_splitter": False}},
        # Resamples
        {"name": "resamples=100", "params": {"n_resamples_selector": 100, "n_resamples_splitter": 100}},
        {"name": "resamples=1000", "params": {"n_resamples_selector": 1000, "n_resamples_splitter": 1000}},
        {"name": "resamples=auto", "params": {"n_resamples_selector": "auto", "n_resamples_splitter": "auto"}},
        # Threshold method
        {"name": "histogram", "params": {"threshold_method": "histogram", "max_thresholds": 128}},
        # Feature muting
        {"name": "no_muting", "params": {"feature_muting": False}},
        # Scanning
        {"name": "no_scanning", "params": {"feature_scanning": False, "threshold_scanning": False}},
        # Combined fast
        {"name": "fast", "params": {
            "early_stopping_selector": True, "early_stopping_splitter": True,
            "threshold_method": "histogram", "max_thresholds": 128,
            "n_resamples_selector": "auto", "n_resamples_splitter": "auto",
        }},
    ]

    for repeat in range(5):
        print(f"  Timing repeat {repeat + 1}/5")
        for config in configs:
            params = {"random_state": RANDOM_STATE + repeat, **config["params"]}
            clf = ConditionalInferenceTreeClassifier(**params)

            tic = time.time()
            clf.fit(X, y_binary)
            toc = time.time()

            results.append({
                "config": config["name"],
                "repeat": repeat,
                "time": toc - tic,
                "depth": clf.depth_,
            })

    return pd.DataFrame(results)


def plot_timing_bars(df: pd.DataFrame, output_path: Path) -> None:
    """Create bar plot showing timing for different configurations."""
    # Aggregate
    agg = df.groupby("config").agg({
        "time": ["mean", "std"],
    }).reset_index()
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

    bars = ax.barh(agg["config"], agg["mean"], xerr=agg["std"], color=colors,
                   edgecolor="black", linewidth=0.5, capsize=3)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Configuration")
    ax.set_title("CITree Training Time by Hyperparameter Configuration")

    # Add value labels
    for bar, mean in zip(bars, agg["mean"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"{mean:.2f}s", ha="left", va="center", fontsize=9)

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
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"{speedup:.1f}x", ha="left", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    """Generate all figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Feature selection bias figures
    print("Running classification experiment...")
    clf_df = run_feature_selection_experiment("classification")

    print("\nGenerating classification figures...")
    plot_feature_selection_bars(clf_df, "classification", OUTPUT_DIR / "feature_selection_clf.png")
    plot_informative_ratio(clf_df, OUTPUT_DIR / "informative_ratio.png")

    print("\nComputing statistics...")
    stats_df = compute_statistics(clf_df)
    print(stats_df.to_string(index=False))

    print("\nGenerating LaTeX table...")
    generate_latex_table(stats_df, OUTPUT_DIR / "feature_selection_table.tex")

    clf_df.to_parquet(OUTPUT_DIR / "feature_selection_data.parquet")

    # Timing figures
    print("\n" + "=" * 60)
    print("Running timing experiment...")
    timing_df = run_timing_experiment()

    print("\nGenerating timing figures...")
    plot_timing_bars(timing_df, OUTPUT_DIR / "timing_bars.png")
    plot_timing_speedup(timing_df, OUTPUT_DIR / "timing_speedup.png")

    timing_df.to_parquet(OUTPUT_DIR / "timing_data.parquet")

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
