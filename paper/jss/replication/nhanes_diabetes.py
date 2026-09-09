"""NHANES diabetes application for the citrees JSS article.

The design is fixed in ``nhanes_diabetes-specification.json`` and read at run
time, so the code cannot drift from the preregistration.

The question is about variable ranking, not prediction: on predictors whose
number of distinct values runs from 2 to roughly n, impurity-based random-forest
importance is known to favor the high-cardinality columns (Strobl et al. 2007).
Three shuffled copies of real columns, at binary, medium, and high cardinality,
are noise by construction and make the effect measurable. citrees and the
reference partykit::cforest implementation are compared with scikit-learn's
RandomForestClassifier on identical numeric input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from citrees import ConditionalInferenceForestClassifier

MODULE_PATH: Final = "paper/jss/replication/nhanes_diabetes.py"
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
SPECIFICATION: Final = Path(__file__).with_name("nhanes_diabetes-specification.json")
DEFAULT_DATA_DIR: Final = REPO_ROOT / "paper" / "jss" / "data" / "nhanes_2021_2023"
DEFAULT_OUTPUT_DIR: Final = REPO_ROOT / "paper" / "jss" / "results" / "nhanes-diabetes"
METHODS: Final = ("rf", "cif", "cforest")
ANALYSIS: Final = "nhanes_diabetes"


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
    """Design matrix with the shuffled controls appended, labels, and column names."""

    X: np.ndarray
    y: np.ndarray
    columns: tuple[str, ...]


def acquire(data_dir: Path, spec: dict[str, Any]) -> dict[str, str]:
    """Download any missing source file and verify every pinned digest."""
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, expected in spec["dataset"]["source_sha256"].items():
        path = raw / name
        if not path.exists():
            url = spec["dataset"]["url_template"].format(file=name.removesuffix(".xpt"))
            print(f"  downloading {name}", flush=True)
            urllib.request.urlretrieve(url, path)  # noqa: S310 - pinned https URL, digest verified below
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"{name}: sha256 {actual} does not match the pinned {expected}")
        hashes[f"paper/jss/data/nhanes_2021_2023/raw/{name}"] = actual
    return hashes


def build_cohort(data_dir: Path, spec: dict[str, Any], seed: int) -> Cohort:
    raw = data_dir / "raw"
    tables: dict[str, pd.DataFrame] = {}

    def table(name: str) -> pd.DataFrame:
        if name not in tables:
            tables[name] = pd.read_sas(raw / f"{name}.xpt", format="xport").set_index("SEQN")
        return tables[name]

    demo = table("DEMO_L")
    adults = demo.index[demo["RIDAGEYR"] >= 20]
    ghb = table("GHB_L")["LBXGH"].reindex(adults).dropna()
    index = ghb.index
    diagnosed = table("DIQ_L")["DIQ010"].reindex(index) == 1
    y = ((ghb >= 6.5) | diagnosed).astype(np.int64)

    X = pd.DataFrame(
        {
            name: table(file)[column].reindex(index)
            for name, (file, column) in spec["predictors"].items()
        }
    )
    for column, codes in spec["missing_codes"].items():
        X.loc[X[column].isin(codes), column] = np.nan
    # SAS transport files store zero as a denormal (about 5e-79); make it exact.
    X = X.mask(X.abs() < 1e-30, 0.0)
    complete = X.notna().all(axis=1)
    X, y = X[complete], y[complete]

    rng = np.random.default_rng(seed)
    for name, control in spec["shuffled_controls"].items():
        if name == "construction":
            continue
        X[name] = rng.permutation(X[control["source"]].to_numpy())
    return Cohort(X=X.to_numpy(dtype=np.float64), y=y.to_numpy(), columns=tuple(X.columns))


def _rank(importance: np.ndarray) -> np.ndarray:
    """Rank 1 = largest importance; tied importances receive their midrank."""
    return np.asarray(stats.rankdata(-importance, method="average"), dtype=np.float64)


def _importances(
    X: np.ndarray, y: np.ndarray, trees: int, seed: int
) -> dict[str, tuple[np.ndarray, float]]:
    from paper.benchmark.pipeline.r_methods import r_cforest_importance

    out: dict[str, tuple[np.ndarray, float]] = {}
    start = time.perf_counter()
    # Every forest is fitted with all available cores so the timings are comparable.
    rf = RandomForestClassifier(n_estimators=trees, n_jobs=-1, random_state=seed).fit(X, y)
    out["rf"] = (rf.feature_importances_, time.perf_counter() - start)

    start = time.perf_counter()
    cif = ConditionalInferenceForestClassifier(
        n_estimators=trees,
        selector="mc",
        threshold_method="histogram",
        max_thresholds=32,
        n_jobs=-1,
        random_state=seed,
    ).fit(X, y)
    out["cif"] = (cif.feature_importances_, time.perf_counter() - start)

    start = time.perf_counter()
    # mtry="sqrt" is partykit's own default, ceiling(sqrt(p)); the wrapper maps an
    # omitted mtry to all predictors. Ten importance permutations match Sections
    # 5.2 and 5.3 of the article.
    importance = r_cforest_importance(
        X, y, ntree=trees, mtry="sqrt", varimp_nperm=10, cores=-1, random_state=seed
    )
    out["cforest"] = (np.asarray(importance, dtype=np.float64), time.perf_counter() - start)
    return out


def _predict(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, seed: int
) -> dict[str, float]:
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(C=1.0, max_iter=5000, random_state=seed).fit(
        scaler.transform(X_train), y_train
    )
    p = model.predict_proba(scaler.transform(X_test))[:, 1]
    return {
        "log_loss": float(log_loss(y_test, p, labels=(0, 1))),
        "roc_auc": float(roc_auc_score(y_test, p)) if len(np.unique(y_test)) == 2 else float("nan"),
        "brier_score": float(brier_score_loss(y_test, p)),
    }


def run_folds(
    cohort: Cohort, spec: dict[str, Any], profile: dict[str, Any], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = RepeatedStratifiedKFold(
        n_splits=profile["folds"], n_repeats=profile["repeats"], random_state=seed
    )
    rank_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(splitter.split(cohort.X, cohort.y)):
        X_train, y_train, X_test, y_test = (
            cohort.X[train],
            cohort.y[train],
            cohort.X[test],
            cohort.y[test],
        )
        n_unique = np.array([len(np.unique(X_train[:, j])) for j in range(X_train.shape[1])])
        results = _importances(X_train, y_train, profile["trees"], seed + fold)
        for method, (importance, seconds) in results.items():
            ranks = _rank(importance)
            for j, name in enumerate(cohort.columns):
                rank_rows.append(
                    {
                        "fold": fold,
                        "repeat": fold // profile["folds"],
                        "method": method,
                        "feature": name,
                        "rank": float(ranks[j]),
                        "importance": float(importance[j]),
                        "n_unique_train": int(n_unique[j]),
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                        "fit_seconds": seconds,
                    }
                )
            for k in spec["descriptive"]["prediction"]["feature_counts"]:
                top = np.argsort(ranks)[:k]
                row: dict[str, Any] = {
                    "fold": fold,
                    "repeat": fold // profile["folds"],
                    "method": method,
                    "n_features": k,
                }
                row.update(_predict(X_train[:, top], y_train, X_test[:, top], y_test, seed + fold))
                metric_rows.append(row)
        print(f"  fold {fold + 1}/{profile['folds'] * profile['repeats']}", flush=True)
    return pd.DataFrame(rank_rows), pd.DataFrame(metric_rows)


def corrected_t(
    differences: np.ndarray,
    n_train: np.ndarray,
    n_test: np.ndarray,
    folds: int,
    repeats: int,
    alternative: str,
) -> dict[str, Any]:
    """Bouckaert and Frank (2004) corrected repeated k-fold t test, one sided."""
    mean = float(differences.mean())
    variance = float(differences.var(ddof=1)) if differences.size > 1 else 0.0
    inflation = 1.0 / (folds * repeats) + float(np.mean(n_test / n_train))
    standard_error = float(np.sqrt(inflation * variance))
    degrees = folds * repeats - 1
    statistic = mean / standard_error if standard_error > 0.0 else 0.0
    p_value = (
        float(stats.t.cdf(statistic, degrees))
        if alternative == "less"
        else float(stats.t.sf(statistic, degrees))
    )
    return {
        "mean_difference": mean,
        "standard_error": standard_error,
        "statistic": statistic,
        "degrees_of_freedom": degrees,
        "p_value": p_value,
    }


def _fold_sizes(ranks: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sizes = ranks.groupby("fold")[["n_train", "n_test"]].first().sort_index()
    return sizes["n_train"].to_numpy(dtype=float), sizes["n_test"].to_numpy(dtype=float)


def _per_fold_rank(ranks: pd.DataFrame, method: str, feature: str) -> pd.Series:
    subset = ranks[(ranks["method"] == method) & (ranks["feature"] == feature)]
    return subset.set_index("fold")["rank"].sort_index().astype(float)


def _per_fold_cardinality_spearman(ranks: pd.DataFrame, method: str) -> pd.Series:
    subset = ranks[ranks["method"] == method]
    return (
        subset.groupby("fold")
        .apply(
            lambda g: float(stats.spearmanr(g["importance"], g["n_unique_train"])[0]),
            include_groups=False,
        )
        .sort_index()
    )


def _per_fold_block_mean_rank(ranks: pd.DataFrame, method: str, block: list[str]) -> pd.Series:
    subset = ranks[(ranks["method"] == method) & (ranks["feature"].isin(block))]
    return subset.groupby("fold")["rank"].mean().sort_index()


def inference(ranks: pd.DataFrame, spec: dict[str, Any], profile: dict[str, Any]) -> pd.DataFrame:
    n_train, n_test = _fold_sizes(ranks)
    folds, repeats = profile["folds"], profile["repeats"]
    rows: list[dict[str, Any]] = []

    primary = _per_fold_rank(ranks, "rf", "shuffled_mec_weight") - _per_fold_rank(
        ranks, "cif", "shuffled_mec_weight"
    )
    row = corrected_t(primary.to_numpy(), n_train, n_test, folds, repeats, "less")
    row.update(
        {
            "hypothesis": "shuffled_mec_weight_rank",
            "role": "primary",
            "contrast": "rf_minus_cif",
            "alternative": spec["primary_test"]["alternative"],
        }
    )
    rows.append(row)

    secondary = spec["secondary_inference"]["hypotheses"]
    card = _per_fold_cardinality_spearman(ranks, "rf") - _per_fold_cardinality_spearman(
        ranks, "cif"
    )
    row = corrected_t(card.to_numpy(), n_train, n_test, folds, repeats, "greater")
    row.update(
        {
            "hypothesis": "cardinality_correlation",
            "role": "secondary",
            "contrast": "rf_minus_cif",
            "alternative": secondary["cardinality_correlation"]["alternative"],
        }
    )
    rows.append(row)

    block = secondary["low_cardinality_clinical_block"]["block"]
    blk = _per_fold_block_mean_rank(ranks, "rf", block) - _per_fold_block_mean_rank(
        ranks, "cif", block
    )
    row = corrected_t(blk.to_numpy(), n_train, n_test, folds, repeats, "greater")
    row.update(
        {
            "hypothesis": "low_cardinality_clinical_block",
            "role": "secondary",
            "contrast": "rf_minus_cif",
            "alternative": secondary["low_cardinality_clinical_block"]["alternative"],
        }
    )
    rows.append(row)

    frame = pd.DataFrame(rows)
    frame["inference_status"] = profile["inference_status"]
    frame["p_value_adjusted"] = frame["p_value"]
    is_secondary = frame["role"] == "secondary"
    order = frame.loc[is_secondary, "p_value"].sort_values()
    running = 0.0
    for position, (index, value) in enumerate(order.items()):
        running = max(running, min(1.0, (len(order) - position) * value))
        frame.loc[index, "p_value_adjusted"] = running
    return frame


def rank_summary(ranks: pd.DataFrame) -> pd.DataFrame:
    summary = ranks.groupby(["feature", "method"]).agg(
        mean_rank=("rank", "mean"), sd_rank=("rank", "std"), n_unique=("n_unique_train", "median")
    )
    wide = summary["mean_rank"].unstack("method")
    wide.columns = [f"mean_rank_{m}" for m in wide.columns]
    wide["n_unique"] = summary["n_unique"].groupby("feature").first()
    return wide.reset_index().sort_values("mean_rank_rf")


def reference_agreement(ranks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, group in ranks.groupby("fold"):
        wide = group.pivot(index="feature", columns="method", values="rank")
        rows.append(
            {
                "fold": fold,
                "repeat": int(group["repeat"].iloc[0]),
                "spearman_cif_cforest": float(stats.spearmanr(wide["cif"], wide["cforest"])[0]),
                "spearman_rf_cforest": float(stats.spearmanr(wide["rf"], wide["cforest"])[0]),
                "spearman_rf_cif": float(stats.spearmanr(wide["rf"], wide["cif"])[0]),
            }
        )
    return pd.DataFrame(rows)


def positive_control(ranks: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    metabolic = {"waist", "total_chol", "triglycerides", "hdl"}
    rows = []
    for method, group in ranks.groupby("method"):
        folds = group["fold"].nunique()
        top5 = sum(
            bool(metabolic & set(g.loc[g["rank"] <= 5, "feature"]))
            for _, g in group.groupby("fold")
        )
        sex_out = sum(
            bool((g.loc[g["feature"] == "shuffled_sex", "rank"] > 20).all())
            for _, g in group.groupby("fold")
        )
        rows.append(
            {
                "method": method,
                "folds": folds,
                "folds_metabolic_in_top5": top5,
                "folds_shuffled_sex_outside_top20": sex_out,
                "passes": bool(top5 > folds / 2 and sex_out > folds / 2),
            }
        )
    return pd.DataFrame(rows)


def write_receipt(
    output_dir: Path, profile_name: str, artifacts: dict[str, Path], data_hashes: dict[str, str]
) -> None:
    versions = {
        p: metadata.version(p)
        for p in ("citrees", "numpy", "pandas", "scikit-learn", "scipy", "numba", "rpy2")
    }
    try:
        versions["partykit"] = subprocess.run(
            ("Rscript", "-e", "cat(as.character(packageVersion('partykit')))"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        versions["partykit"] = "unavailable"
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
            p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in artifacts.values()
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
    args = parser.parse_args()

    spec = json.loads(SPECIFICATION.read_text(encoding="utf-8"))
    profile = spec["profiles"][args.profile]
    print(f"profile={args.profile} seed={args.seed}", flush=True)
    data_hashes = acquire(args.data_dir, spec)
    cohort = build_cohort(args.data_dir, spec, args.seed)
    n, p = cohort.X.shape
    print(
        f"  cohort n={n} p={p} (27 predictors + 3 shuffled controls) prevalence={cohort.y.mean():.3f}",
        flush=True,
    )
    if n != spec["dataset"]["expected_n"] or p != spec["dataset"]["expected_p"] + 3:
        raise RuntimeError(
            f"cohort {n}x{p} does not match the preregistered {spec['dataset']['expected_n']}x{spec['dataset']['expected_p'] + 3}"
        )

    ranks, metrics = run_folds(cohort, spec, profile, args.seed)
    frames = {
        "fold_ranks": ranks,
        "fold_prediction_metrics": metrics,
        "primary_inference": inference(ranks, spec, profile),
        "rank_summary": rank_summary(ranks),
        "reference_agreement": reference_agreement(ranks),
        "positive_control": positive_control(ranks, spec),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.iterdir():
        if stale.is_file():
            stale.unlink()
    artifacts: dict[str, Path] = {}
    for name, frame in frames.items():
        for suffix in ("parquet", "csv"):
            path = args.output_dir / f"{name}.{suffix}"
            frame.to_parquet(path, index=False) if suffix == "parquet" else frame.to_csv(
                path, index=False
            )
            artifacts[path.name] = path
    write_receipt(args.output_dir, args.profile, artifacts, data_hashes)

    print("\n== inference ==")
    print(
        frames["primary_inference"][
            ["hypothesis", "role", "mean_difference", "statistic", "p_value", "p_value_adjusted"]
        ].to_string(index=False)
    )
    print("\n== mean ranks (controls) ==")
    print(
        frames["rank_summary"][
            frames["rank_summary"]["feature"].str.startswith("shuffled")
        ].to_string(index=False)
    )
    print("\n== positive control ==")
    print(frames["positive_control"].to_string(index=False))


if __name__ == "__main__":
    main()
