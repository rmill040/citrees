"""Run the executable JSS estimator tutorial on Breast Cancer Wisconsin data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceTreeClassifier,
)

Profile = Literal["smoke", "quick", "full"]

BASE_SEED = 1718
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "tutorial"


@dataclass(frozen=True)
class TutorialSettings:
    """Workload settings for one tutorial replication profile."""

    forest_estimators: int
    search_folds: int
    search_alphas: tuple[float, ...]
    search_depths: tuple[int, ...]


@dataclass(frozen=True)
class TutorialRun:
    """Deterministic tutorial outputs before serialization."""

    input_sha256: str
    n_samples: int
    n_features: int
    target_names: tuple[str, ...]
    transcript: str
    inspection: pd.DataFrame
    feature_importances: pd.DataFrame
    holdout_metrics: pd.DataFrame
    search_results: pd.DataFrame


class TreeParameters(TypedDict):
    """Typed keyword arguments shared by the tutorial estimators."""

    selector: str
    n_resamples_selector: str
    n_resamples_splitter: str
    threshold_method: str
    max_thresholds: int
    max_depth: int
    random_state: int
    verbose: int


def _settings(profile: Profile) -> TutorialSettings:
    if profile == "smoke":
        return TutorialSettings(
            forest_estimators=3,
            search_folds=2,
            search_alphas=(0.05,),
            search_depths=(2, 3),
        )
    if profile == "quick":
        return TutorialSettings(
            forest_estimators=25,
            search_folds=3,
            search_alphas=(0.01, 0.05),
            search_depths=(3, 5),
        )
    if profile == "full":
        return TutorialSettings(
            forest_estimators=100,
            search_folds=5,
            search_alphas=(0.01, 0.05),
            search_depths=(3, 5),
        )
    raise ValueError(f"unknown tutorial profile: {profile}")


def _input_sha256(X: pd.DataFrame, y: pd.Series) -> str:
    """Hash the exact feature matrix, target, and column identities."""
    descriptor = {
        "columns": [str(column) for column in X.columns],
        "feature_shape": list(X.shape),
        "target_name": str(y.name),
        "target_shape": list(y.shape),
    }
    digest = hashlib.sha256(
        json.dumps(descriptor, separators=(",", ":"), sort_keys=True).encode("ascii")
    )
    digest.update(np.ascontiguousarray(X.to_numpy(), dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(y.to_numpy(), dtype="<i8").tobytes())
    return digest.hexdigest()


def _tree_parameters(*, max_depth: int, random_state: int) -> TreeParameters:
    """Return the explicit tutorial tree configuration."""
    return {
        "selector": "mc",
        "n_resamples_selector": "minimum",
        "n_resamples_splitter": "minimum",
        "threshold_method": "histogram",
        "max_thresholds": 32,
        "max_depth": max_depth,
        "random_state": random_state,
        "verbose": 0,
    }


def _rank_importances(model: str, names: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    """Return all feature importances in stable descending order."""
    order = np.argsort(-values, kind="stable")
    return pd.DataFrame(
        {
            "model": model,
            "rank": np.arange(1, len(order) + 1, dtype=np.int64),
            "feature": names[order],
            "importance": values[order],
        }
    )


def _format_top_features(frame: pd.DataFrame, n: int = 5) -> str:
    """Format the leading named importances for the transcript."""
    return ", ".join(
        f"{row.feature}={row.importance:.4f}" for row in frame.head(n).itertuples(index=False)
    )


def _search_summary(search: GridSearchCV) -> pd.DataFrame:
    """Return the parameter-search results in rank order."""
    results = search.cv_results_
    frame = pd.DataFrame(
        {
            "alpha_selector": [
                float(params["tree__alpha_selector"]) for params in results["params"]
            ],
            "max_depth": [int(params["tree__max_depth"]) for params in results["params"]],
            "mean_balanced_accuracy": results["mean_test_score"].astype(float),
            "standard_deviation": results["std_test_score"].astype(float),
            "rank": results["rank_test_score"].astype(np.int64),
        }
    )
    return frame.sort_values(
        ["rank", "alpha_selector", "max_depth"],
        kind="stable",
        ignore_index=True,
    )


def run_tutorial(profile: Profile, *, base_seed: int = BASE_SEED) -> TutorialRun:
    """Run the complete estimator, inspection, pipeline, and search workflow."""
    settings = _settings(profile)
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=base_seed,
    )

    tree = ConditionalInferenceTreeClassifier(
        **_tree_parameters(max_depth=3, random_state=base_seed)
    )
    tree.fit(X_train, y_train)
    tree_predictions = tree.predict(X_test)
    tree_probabilities = tree.predict_proba(X_test)[:, 1]
    tree_path = tree.decision_path(X_test)
    tree_leaves = tree.apply(X_test)
    tree_counts = tree.realized_permutation_counts_

    root = tree.tree_
    if "value" in root:
        root_feature = "terminal"
        root_threshold = float("nan")
        root_feature_p_value = float("nan")
        root_threshold_p_value = float("nan")
    else:
        root_feature = str(tree.feature_names_in_[int(root["feature"])])
        root_threshold = float(root["threshold"])
        root_feature_p_value = float(root["pval_feature"])
        root_threshold_p_value = float(root["pval_threshold"])

    forest = ConditionalInferenceForestClassifier(
        n_estimators=settings.forest_estimators,
        bootstrap=True,
        oob_score=True,
        n_jobs=1,
        **_tree_parameters(max_depth=5, random_state=base_seed),
    )
    forest.fit(X_train, y_train)
    forest_predictions = forest.predict(X_test)
    forest_probabilities = forest.predict_proba(X_test)[:, 1]
    forest_leaves = forest.apply(X_test)
    forest_path, forest_node_offsets = forest.decision_path(X_test)
    forest_counts = {
        stage: sum(tree_.realized_permutation_counts_[stage] for tree_ in forest.estimators_)
        for stage in ("selector", "splitter")
    }

    scaler = StandardScaler().set_output(transform="pandas")
    pipeline = Pipeline(
        [
            ("scale", scaler),
            (
                "tree",
                ConditionalInferenceTreeClassifier(
                    **_tree_parameters(max_depth=3, random_state=base_seed)
                ),
            ),
        ]
    )
    folds = StratifiedKFold(
        n_splits=settings.search_folds,
        shuffle=True,
        random_state=base_seed,
    )
    search = GridSearchCV(
        pipeline,
        {
            "tree__alpha_selector": list(settings.search_alphas),
            "tree__max_depth": list(settings.search_depths),
        },
        cv=folds,
        scoring="balanced_accuracy",
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    tuned_predictions = search.predict(X_test)
    tuned_probabilities = search.predict_proba(X_test)[:, 1]
    fitted_pipeline_tree = search.best_estimator_.named_steps["tree"]

    holdout_metrics = pd.DataFrame(
        [
            {
                "model": "tree",
                "balanced_accuracy": balanced_accuracy_score(y_test, tree_predictions),
                "roc_auc": roc_auc_score(y_test, tree_probabilities),
            },
            {
                "model": "forest",
                "balanced_accuracy": balanced_accuracy_score(y_test, forest_predictions),
                "roc_auc": roc_auc_score(y_test, forest_probabilities),
            },
            {
                "model": "tuned_pipeline",
                "balanced_accuracy": balanced_accuracy_score(y_test, tuned_predictions),
                "roc_auc": roc_auc_score(y_test, tuned_probabilities),
            },
        ]
    )

    feature_importances = pd.concat(
        [
            _rank_importances("tree", tree.feature_names_in_, tree.feature_importances_),
            _rank_importances("forest", forest.feature_names_in_, forest.feature_importances_),
        ],
        ignore_index=True,
    )
    tree_importances = feature_importances[feature_importances["model"] == "tree"]
    forest_importances = feature_importances[feature_importances["model"] == "forest"]

    inspection = pd.DataFrame(
        [
            {"object": "tree", "measure": "depth", "value": int(tree.depth_)},
            {"object": "tree", "measure": "nodes", "value": int(tree_path.shape[1])},
            {
                "object": "tree",
                "measure": "unique_test_leaves",
                "value": int(np.unique(tree_leaves).size),
            },
            {
                "object": "tree",
                "measure": "selector_permutations",
                "value": int(tree_counts["selector"]),
            },
            {
                "object": "tree",
                "measure": "splitter_permutations",
                "value": int(tree_counts["splitter"]),
            },
            {
                "object": "forest",
                "measure": "estimators",
                "value": len(forest.estimators_),
            },
            {
                "object": "forest",
                "measure": "path_columns",
                "value": int(forest_path.shape[1]),
            },
            {
                "object": "forest",
                "measure": "node_offset_entries",
                "value": int(len(forest_node_offsets)),
            },
            {
                "object": "forest",
                "measure": "leaf_matrix_columns",
                "value": int(forest_leaves.shape[1]),
            },
            {
                "object": "forest",
                "measure": "selector_permutations",
                "value": int(forest_counts["selector"]),
            },
            {
                "object": "forest",
                "measure": "splitter_permutations",
                "value": int(forest_counts["splitter"]),
            },
        ]
    )

    best_params = {key.removeprefix("tree__"): value for key, value in search.best_params_.items()}
    class_counts = y.value_counts().sort_index()
    metrics = holdout_metrics.set_index("model")
    lines = [
        f"data: {len(X)} samples, {X.shape[1]} features",
        (
            f"split: {len(X_train)} train, {len(X_test)} test; "
            f"classes {int(class_counts.iloc[0])}/{int(class_counts.iloc[1])}"
        ),
        f"tree: depth={tree.depth_}, nodes={tree_path.shape[1]}, root={root_feature}",
        (
            f"root: threshold={root_threshold:.4f}, feature p={root_feature_p_value:.4f}, "
            f"threshold p={root_threshold_p_value:.4f}"
        ),
        (
            "tree holdout: "
            f"balanced accuracy={metrics.loc['tree', 'balanced_accuracy']:.4f}, "
            f"ROC AUC={metrics.loc['tree', 'roc_auc']:.4f}"
        ),
        (
            "tree permutations: "
            f"selector={tree_counts['selector']}, splitter={tree_counts['splitter']}"
        ),
        f"tree leading importances: {_format_top_features(tree_importances)}",
        (
            f"forest: estimators={len(forest.estimators_)}, "
            f"leaf matrix={forest_leaves.shape}, OOB score={forest.oob_score_:.4f}"
        ),
        (
            "forest holdout: "
            f"balanced accuracy={metrics.loc['forest', 'balanced_accuracy']:.4f}, "
            f"ROC AUC={metrics.loc['forest', 'roc_auc']:.4f}"
        ),
        (
            "forest permutations: "
            f"selector={forest_counts['selector']}, splitter={forest_counts['splitter']}"
        ),
        f"forest leading importances: {_format_top_features(forest_importances)}",
        (
            "pipeline feature names: "
            + ", ".join(str(name) for name in fitted_pipeline_tree.feature_names_in_[:3])
        ),
        f"search best parameters: {json.dumps(best_params, sort_keys=True)}",
        f"search cross-validated balanced accuracy={search.best_score_:.4f}",
        (
            "tuned pipeline holdout: "
            f"balanced accuracy={metrics.loc['tuned_pipeline', 'balanced_accuracy']:.4f}, "
            f"ROC AUC={metrics.loc['tuned_pipeline', 'roc_auc']:.4f}"
        ),
    ]

    return TutorialRun(
        input_sha256=_input_sha256(X, y),
        n_samples=len(X),
        n_features=X.shape[1],
        target_names=tuple(str(name) for name in dataset.target_names),
        transcript="\n".join(lines) + "\n",
        inspection=inspection,
        feature_importances=feature_importances,
        holdout_metrics=holdout_metrics,
        search_results=_search_summary(search),
    )


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_results(
    run: TutorialRun,
    output_dir: Path,
    *,
    profile: Profile,
    base_seed: int,
    elapsed_seconds: float,
) -> None:
    """Write tutorial tables, transcript, and a machine-readable receipt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: list[Path] = []
    for name, frame in (
        ("inspection", run.inspection),
        ("feature_importances", run.feature_importances),
        ("holdout_metrics", run.holdout_metrics),
        ("search_results", run.search_results),
    ):
        parquet_path = output_dir / f"{name}.parquet"
        csv_path = output_dir / f"{name}.csv"
        frame.to_parquet(parquet_path, index=False)
        frame.to_csv(csv_path, index=False)
        artifact_paths.extend((parquet_path, csv_path))

    transcript_path = output_dir / "transcript.txt"
    transcript_path.write_text(run.transcript, encoding="ascii")
    artifact_paths.append(transcript_path)

    repo_root = Path(__file__).resolve().parents[3]
    source_files = [
        Path(__file__).resolve(),
        repo_root / "citrees" / "_forest.py",
        repo_root / "citrees" / "_permutation.py",
        repo_root / "citrees" / "_selector.py",
        repo_root / "citrees" / "_sequential.py",
        repo_root / "citrees" / "_splitter.py",
        repo_root / "citrees" / "_threshold_method.py",
        repo_root / "citrees" / "_tree.py",
        repo_root / "citrees" / "_types.py",
        repo_root / "pyproject.toml",
        repo_root / "uv.lock",
    ]
    versions = {
        package: importlib.metadata.version(package)
        for package in ("citrees", "numpy", "pandas", "scikit-learn", "scipy")
    }
    receipt = {
        "analysis": "tutorial",
        "schema_version": 1,
        "profile": profile,
        "base_seed": base_seed,
        "settings": asdict(_settings(profile)),
        "inputs": {
            "sklearn_breast_cancer": {
                "sha256": run.input_sha256,
                "n_samples": run.n_samples,
                "n_features": run.n_features,
                "target_names": list(run.target_names),
            }
        },
        "created_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_sha256": {str(path.relative_to(repo_root)): _sha256(path) for path in source_files},
        "versions": versions,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(artifact_paths)
        },
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("smoke", "quick", "full"),
        default="quick",
        help="Tutorial workload.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for tables, transcript, and receipt.",
    )
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Base random seed.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.profile == "full" and _git_dirty():
        raise RuntimeError("The full tutorial profile requires a clean source tree")
    started = time.perf_counter()
    run = run_tutorial(args.profile, base_seed=args.seed)
    elapsed = time.perf_counter() - started
    write_results(
        run,
        args.output_dir,
        profile=args.profile,
        base_seed=args.seed,
        elapsed_seconds=elapsed,
    )
    print(f"Wrote tutorial artifacts to {args.output_dir} in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
