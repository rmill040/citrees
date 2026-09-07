"""TCGA-BRCA receptor-status application for the citrees JSS article.

The design is fixed in ``tcga_brca-specification.json`` and is read at run time
rather than restated here, so the code cannot drift from the preregistration.

Predictors are protein-coding gene expression; outcomes are receptor statuses
determined by immunohistochemistry and FISH. The outcome assays are independent
of the RNA-seq that supplies the predictors, which is why PAM50 subtype is not
used: it is called from the same expression data and would be circular.

Every global filter applied when the matrix was derived is unsupervised and
never reads an outcome, so it sits outside the cross-validation loop. The
candidate screen and the standardization are fit on training folds only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from citrees import ConditionalInferenceForestClassifier, ConditionalInferenceTreeClassifier

MODULE_PATH: Final = "paper/jss/replication/tcga_brca.py"
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
SPECIFICATION: Final = Path(__file__).with_name("tcga_brca-specification.json")
DEFAULT_DATA_DIR: Final = REPO_ROOT / "paper" / "jss" / "data" / "tcga_brca"
DEFAULT_OUTPUT_DIR: Final = REPO_ROOT / "paper" / "jss" / "results" / "tcga-brca"
METHODS: Final = ("marginal", "l2_logistic", "cit", "cif")
ANALYSIS: Final = "tcga_brca"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


@dataclass(frozen=True)
class Cohort:
    """One outcome: its design matrix, labels, and gene identifiers."""

    outcome: str
    X: np.ndarray
    y: np.ndarray
    genes: tuple[str, ...]
    symbols: tuple[str, ...]


def _load(data_dir: Path, outcomes: tuple[str, ...]) -> tuple[dict[str, Cohort], dict[str, str]]:
    expression = pd.read_parquet(data_dir / "brca_expression.parquet").set_index("sample")
    labels = pd.read_parquet(data_dir / "brca_labels.parquet").set_index("sample")
    symbols = pd.read_parquet(data_dir / "brca_gene_symbols.parquet")
    lookup = dict(zip(symbols["ensembl_id"], symbols["symbol"], strict=True))
    genes = tuple(str(column) for column in expression.columns)

    cohorts: dict[str, Cohort] = {}
    for outcome in outcomes:
        keep = labels[outcome].notna()
        index = labels.index[keep]
        cohorts[outcome] = Cohort(
            outcome=outcome,
            X=expression.loc[index].to_numpy(dtype=np.float64),
            y=labels.loc[index, outcome].to_numpy(dtype=np.int64),
            genes=genes,
            symbols=tuple(lookup.get(gene, gene) for gene in genes),
        )
    data_hashes = {
        f"paper/jss/data/tcga_brca/{name}": _sha256(data_dir / name)
        for name in ("brca_expression.parquet", "brca_labels.parquet", "brca_gene_symbols.parquet")
    }
    return cohorts, data_hashes


def _screen(X: np.ndarray, y: np.ndarray, count: int) -> np.ndarray:
    """Rank features by absolute point-biserial correlation, ties by gene index."""
    centered = X - X.mean(axis=0)
    scale = np.sqrt((centered**2).sum(axis=0))
    scale[scale == 0.0] = 1.0
    target = y - y.mean()
    denominator = np.sqrt((target**2).sum())
    correlation = (centered * target[:, None]).sum(axis=0) / (scale * max(denominator, 1e-12))
    # lexsort is stable and ascending, so negate the magnitude to rank descending
    # and break ties on ascending gene index exactly as the specification requires.
    order = np.lexsort((np.arange(X.shape[1]), -np.abs(correlation)))
    return order[:count]


def _rank_within_candidates(
    X: np.ndarray, y: np.ndarray, trees: int, projections: int, seed: int
) -> dict[str, np.ndarray]:
    """Order the candidate columns by each method's importance, best first."""
    rankings: dict[str, np.ndarray] = {}
    # `marginal` keeps the screen order, which already encodes the tie-breaker.
    rankings["marginal"] = np.arange(X.shape[1])

    logistic = LogisticRegression(C=1.0, max_iter=5000, random_state=seed)
    logistic.fit(X, y)
    coefficients = np.abs(logistic.coef_).max(axis=0)
    rankings["l2_logistic"] = np.lexsort((np.arange(X.shape[1]), -coefficients))

    tree = ConditionalInferenceTreeClassifier(
        selector="rdc", rdc_n_projections=projections, random_state=seed
    )
    tree.fit(X, y)
    rankings["cit"] = np.lexsort((np.arange(X.shape[1]), -tree.feature_importances_))

    forest = ConditionalInferenceForestClassifier(
        selector="rdc",
        rdc_n_projections=projections,
        n_estimators=trees,
        n_jobs=1,
        random_state=seed,
    )
    forest.fit(X, y)
    rankings["cif"] = np.lexsort((np.arange(X.shape[1]), -forest.feature_importances_))
    return rankings


def _score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    model = LogisticRegression(C=1.0, max_iter=5000, random_state=seed)
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    scores = {
        "log_loss": float(log_loss(y_test, probability, labels=(0, 1))),
        "brier_score": float(brier_score_loss(y_test, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, probability >= 0.5)),
    }
    # A fold can be single-class only in degenerate profiles; AUC is undefined there.
    scores["roc_auc"] = (
        float(roc_auc_score(y_test, probability)) if len(np.unique(y_test)) == 2 else float("nan")
    )
    return scores


def _run_outcome(
    cohort: Cohort, spec: dict[str, Any], profile: dict[str, Any], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts = [k for k in spec["prediction"]["feature_counts"] if k <= profile["candidate_count"]]
    splitter = RepeatedStratifiedKFold(
        n_splits=profile["folds"], n_repeats=profile["repeats"], random_state=seed
    )
    metrics: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []

    for fold, (train, test) in enumerate(splitter.split(cohort.X, cohort.y)):
        X_train_raw, X_test_raw = cohort.X[train], cohort.X[test]
        y_train, y_test = cohort.y[train], cohort.y[test]

        # Standardization and the candidate screen are fit on the training fold only.
        scaler = StandardScaler().fit(X_train_raw)
        X_train_all = scaler.transform(X_train_raw)
        X_test_all = scaler.transform(X_test_raw)
        candidates = _screen(X_train_all, y_train, profile["candidate_count"])
        X_train, X_test = X_train_all[:, candidates], X_test_all[:, candidates]

        rankings = _rank_within_candidates(
            X_train,
            y_train,
            profile["trees"],
            spec["conditional_methods"]["rdc_n_projections"],
            seed + fold,
        )
        for method, order in rankings.items():
            for rank, position in enumerate(order[: max(counts)]):
                selections.append(
                    {
                        "outcome": cohort.outcome,
                        "fold": fold,
                        "method": method,
                        "rank": rank,
                        "ensembl_id": cohort.genes[candidates[position]],
                        "symbol": cohort.symbols[candidates[position]],
                    }
                )
            for k in counts:
                chosen = order[:k]
                row = {
                    "outcome": cohort.outcome,
                    "fold": fold,
                    "repeat": fold // profile["folds"],
                    "method": method,
                    "n_features": k,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                }
                row.update(
                    _score(X_train[:, chosen], y_train, X_test[:, chosen], y_test, seed + fold)
                )
                metrics.append(row)
    return metrics, selections


def _corrected_t(
    differences: np.ndarray, n_train: np.ndarray, n_test: np.ndarray, folds: int, repeats: int
) -> dict[str, Any]:
    """Bouckaert and Frank (2004) corrected repeated k-fold t test, one sided."""
    mean = float(differences.mean())
    variance = float(differences.var(ddof=1))
    inflation = 1.0 / (folds * repeats) + float(np.mean(n_test / n_train))
    standard_error = float(np.sqrt(inflation * variance))
    degrees = folds * repeats - 1
    statistic = mean / standard_error if standard_error > 0.0 else 0.0
    return {
        "mean_difference": mean,
        "standard_error": standard_error,
        "statistic": statistic,
        "degrees_of_freedom": degrees,
        # alternative is `cif_lower`: evidence is a negative difference
        "p_value": float(stats.t.cdf(statistic, degrees)),
    }


def _inference(
    metrics: pd.DataFrame, spec: dict[str, Any], profile: dict[str, Any]
) -> pd.DataFrame:
    hypothesis = spec["primary_test"]
    rows: list[dict[str, Any]] = []
    for outcome in profile["outcomes"]:
        subset = metrics[
            (metrics["outcome"] == outcome) & (metrics["n_features"] == hypothesis["feature_count"])
        ]
        if subset.empty:
            continue
        pivot = subset.pivot_table(
            index="fold", columns="method", values="log_loss", aggfunc="mean"
        )
        sizes = subset.groupby("fold")[["n_train", "n_test"]].first().loc[pivot.index]
        result = _corrected_t(
            (pivot["cif"] - pivot["marginal"]).to_numpy(),
            sizes["n_train"].to_numpy(dtype=float),
            sizes["n_test"].to_numpy(dtype=float),
            profile["folds"],
            profile["repeats"],
        )
        result.update(
            {
                "outcome": outcome,
                "role": "primary" if outcome == spec["outcomes"]["primary"] else "secondary",
                "contrast": hypothesis["contrast"],
                "alternative": hypothesis["alternative"],
                "metric": hypothesis["metric"],
                "feature_count": hypothesis["feature_count"],
                "inference_status": profile["inference_status"],
            }
        )
        rows.append(result)

    frame = pd.DataFrame(rows)
    # Holm adjustment covers the secondary family only; the primary stands alone.
    secondary = frame["role"] == "secondary"
    frame["p_value_adjusted"] = frame["p_value"]
    if secondary.any():
        order = frame.loc[secondary, "p_value"].sort_values()
        size = len(order)
        running = 0.0
        for position, (index, value) in enumerate(order.items()):
            running = max(running, min(1.0, (size - position) * value))
            frame.loc[index, "p_value_adjusted"] = running
    return frame


def _stability(selections: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for count in spec["stability"]["feature_counts"]:
        for (outcome, method), group in selections.groupby(["outcome", "method"]):
            sets = [
                frozenset(fold_group.loc[fold_group["rank"] < count, "ensembl_id"])
                for _, fold_group in group.groupby("fold")
            ]
            pairs = [len(a & b) / count for i, a in enumerate(sets) for b in sets[i + 1 :]]
            rows.append(
                {
                    "outcome": outcome,
                    "method": method,
                    "n_features": count,
                    "n_folds": len(sets),
                    "mean_pairwise_overlap": float(np.mean(pairs)) if pairs else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _positive_control(selections: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    expected = spec["positive_control"]["expected"]
    rows: list[dict[str, Any]] = []
    for (outcome, method), group in selections.groupby(["outcome", "method"]):
        marker = expected[outcome]
        folds = group["fold"].nunique()
        hits = sum(
            marker in set(fold_group.loc[fold_group["rank"] < 10, "symbol"])
            for _, fold_group in group.groupby("fold")
        )
        rows.append(
            {
                "outcome": outcome,
                "method": method,
                "marker": marker,
                "folds": folds,
                "folds_with_marker_in_top10": hits,
                "fraction": hits / folds if folds else float("nan"),
                "passes_majority_rule": bool(folds and hits / folds > 0.5),
            }
        )
    return pd.DataFrame(rows)


def _write_receipt(
    output_dir: Path, profile_name: str, artifacts: dict[str, Path], data_hashes: dict[str, str]
) -> None:
    versions = {
        package: metadata.version(package)
        for package in ("citrees", "numpy", "pandas", "scikit-learn", "scipy", "numba")
    }
    sources = {
        MODULE_PATH: _sha256(REPO_ROOT / MODULE_PATH),
        "pyproject.toml": _sha256(REPO_ROOT / "pyproject.toml"),
        "uv.lock": _sha256(REPO_ROOT / "uv.lock"),
        f"paper/jss/replication/{SPECIFICATION.name}": _sha256(SPECIFICATION),
        **data_hashes,
    }
    receipt = {
        "analysis": ANALYSIS,
        "profile": profile_name,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
        "source_sha256": sources,
        "artifacts": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in artifacts.values()
        },
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="ascii"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "quick", "smoke"), default="full")
    parser.add_argument("--seed", type=int, default=1718)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()

    spec = json.loads(SPECIFICATION.read_text(encoding="utf-8"))
    profile = spec["profiles"][arguments.profile]
    outcomes = tuple(profile["outcomes"])
    print(f"profile={arguments.profile} outcomes={outcomes} seed={arguments.seed}", flush=True)

    cohorts, data_hashes = _load(arguments.data_dir, outcomes)
    metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        cohort = cohorts[outcome]
        print(f"  {outcome}: n={cohort.X.shape[0]} p={cohort.X.shape[1]}", flush=True)
        metrics, selections = _run_outcome(cohort, spec, profile, arguments.seed)
        metric_rows.extend(metrics)
        selection_rows.extend(selections)

    fold_metrics = pd.DataFrame(metric_rows)
    selected = pd.DataFrame(selection_rows)
    inference = _inference(fold_metrics, spec, profile)
    stability = _stability(selected, spec)
    control = _positive_control(selected, spec)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in arguments.output_dir.iterdir():
        if stale.is_file():
            stale.unlink()
    artifacts: dict[str, Path] = {}
    for name, frame in (
        ("fold_metrics", fold_metrics),
        ("selected_genes", selected),
        ("primary_inference", inference),
        ("stability", stability),
        ("positive_control", control),
    ):
        for suffix in ("parquet", "csv"):
            path = arguments.output_dir / f"{name}.{suffix}"
            frame.to_parquet(path, index=False) if suffix == "parquet" else frame.to_csv(
                path, index=False
            )
            artifacts[path.name] = path

    _write_receipt(arguments.output_dir, arguments.profile, artifacts, data_hashes)
    print("\n== inference ==", flush=True)
    print(
        inference[
            ["outcome", "role", "mean_difference", "statistic", "p_value", "p_value_adjusted"]
        ].to_string(index=False),
        flush=True,
    )
    print("\n== positive control ==", flush=True)
    print(control.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
