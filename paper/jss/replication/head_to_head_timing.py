"""Head-to-head forest timing protocol behind the JSS head-to-head timing tables.

Times 100-tree classification forests, fit-only, in a fresh child process per
fit: the benchmark ``citrees`` configuration (``citrees``: minimum budgets,
256-bin histogram, per-threshold Bonferroni test), the same without the
Bonferroni adjustment over candidate thresholds (``citrees_splitgate_nobonf``),
the same with the max-type threshold test (``citrees_maxt``),
``partykit::cforest`` with the Monte Carlo Bonferroni test (B = 999) on 1 and 32
cores (``partykit_sqrt_1``, ``partykit_sqrt_32``), and a scikit-learn random
forest (``sklearn_rf``). Trees have depth at most 3, minimum split 20, minimum
leaf 7, alpha 0.05; the forests run with ``n_jobs=-1``. Each citrees fit is timed
as the second of two identical fits, so compilation is excluded.

The ``full`` profile is the manuscript protocol: synthetic Gaussian predictors
with a weak or strong planted signal at n in {500, 2000, 8000, 32000} x p in
{20, 100}, plus p = 500 at n in {500, 2000}, and the ten real benchmark datasets
with at least 500 observations; two repeats per cell, one on letter, isolet, and
gisette; a fit is censored after ``--fit-timeout`` seconds. It requires a host
with at least ``REFERENCE_LOGICAL_CPUS`` logical CPUs, R, and ``partykit``.
``quick`` and ``smoke`` reduce the cells to validate the pipeline and make no
timing claim.

Usage: ``python -m paper.jss.replication.head_to_head_timing --profile full
--output-dir <new directory>``. ``--shard-index`` and ``--num-shards`` split the
fits round-robin across hosts; the replication suite runs one shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pyarrow.parquet as pq

from paper.benchmark.utils import get_hardware_metadata

ANALYSIS: Final = "head_to_head_timing"
MODULE_PATH: Final = "paper/jss/replication/head_to_head_timing.py"
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR: Final = REPO_ROOT / "paper" / "data" / "classification" / "real"
REFERENCE_LOGICAL_CPUS: Final = 32
DEFAULT_FIT_TIMEOUT: Final = 1200
MAX_DEPTH, MIN_SPLIT, MIN_LEAF, ALPHA = 3, 20, 7, 0.05
CONFIGS: Final = (
    "citrees",
    "citrees_splitgate_nobonf",
    "citrees_maxt",
    "partykit_sqrt_1",
    "partykit_sqrt_32",
    "sklearn_rf",
)
SINGLE_REPEAT_DATASETS: Final = frozenset({"gisette", "isolet", "letter"})
REAL_DATASETS: Final = (
    "letter",
    "gamma",
    "pendigits",
    "isolet",
    "musk",
    "gisette",
    "page-blocks",
    "spam",
    "madelon",
    "vowel-context",
)
SIGNALS: Final = ("weak", "strong")

PROFILES: Final[dict[str, dict[str, Any]]] = {
    "full": {
        "synthetic": tuple((n, p) for n in (500, 2000, 8000, 32000) for p in (20, 100))
        + ((500, 500), (2000, 500)),
        "signals": SIGNALS,
        "real": REAL_DATASETS,
        "repeats": 2,
        "trees": 100,
    },
    "quick": {
        "synthetic": ((500, 20), (500, 100), (2000, 20), (2000, 100)),
        "signals": SIGNALS,
        "real": ("vowel-context", "page-blocks", "spam"),
        "repeats": 1,
        "trees": 100,
    },
    "smoke": {
        "synthetic": ((500, 20),),
        "signals": ("weak",),
        "real": ("vowel-context",),
        "repeats": 1,
        "trees": 4,
    },
}

# One fit per process. argv: kind name n p signal config trees seed data_dir.
CHILD: Final = r"""
import sys, time, json, numpy as np, pandas as pd
kind, name, n, p, signal, config = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
TREES, seed, data_dir = int(sys.argv[7]), int(sys.argv[8]), sys.argv[9]
MAX_DEPTH, MIN_SPLIT, MIN_LEAF, ALPHA = 3, 20, 7, 0.05
if kind == "synthetic":
    rng = np.random.default_rng(seed + n + p)
    X = rng.standard_normal((n, p)); active = min(5, p)
    coef = np.linspace(1.0, 0.25, active) if signal == "weak" else np.ones(active)
    noise = 1.0 if signal == "weak" else 0.5
    latent = X[:, :active] @ coef + noise * rng.standard_normal(n)
    y = (latent >= np.median(latent)).astype(np.int64)
else:
    d = pd.read_parquet(f"{data_dir}/clf_{name}.parquet")
    y = pd.factorize(d["y"])[0].astype(np.int64); X = d.drop(columns=["y"]).to_numpy(dtype=np.float64)
    n, p = X.shape
wi = np.random.default_rng(0).permutation(n)[:150]  # warm-up rows: random, so both classes are present
if config.startswith("citrees"):
    from citrees import ConditionalInferenceForestClassifier as F
    rec = dict(selector="mc", alpha_selector=ALPHA, adjust_alpha_selector=True, n_resamples_selector="minimum",
               early_stopping_selector="adaptive", early_stopping_confidence_selector=0.95, n_resamples_splitter="minimum",
               adjust_alpha_splitter=True, early_stopping_splitter="adaptive", early_stopping_confidence_splitter=0.95,
               feature_muting=True, feature_scanning=True, threshold_scanning=True, threshold_method="histogram",
               max_thresholds=256, max_depth=MAX_DEPTH, min_samples_split=MIN_SPLIT, min_samples_leaf=MIN_LEAF, verbose=0)
    if config == "citrees_splitgate_nobonf": rec.update(adjust_alpha_splitter=False)
    if config == "citrees_maxt": rec.update(threshold_test="maxt")
    F(**rec, n_estimators=3, n_jobs=-1, random_state=1).fit(X[wi][:, :min(p, 8)], y[wi])  # warm-up: JIT + worker pool
    # one identical full fit first: the small warm-up does not reach kernels selected by size
    F(**rec, n_estimators=TREES, n_jobs=-1, random_state=0).fit(X, y)
    t = time.time(); F(**rec, n_estimators=TREES, n_jobs=-1, random_state=0).fit(X, y); el = time.time() - t
elif config.startswith("partykit"):
    cores = int(config.rsplit("_", 1)[1])
    from paper.benchmark.pipeline.r_methods import r_cforest_fit
    kw = dict(task="classification", teststat="quadratic", testtype=("Bonferroni", "MonteCarlo"), mincriterion=1 - ALPHA,
              nresample=999, mtry="sqrt", maxdepth=MAX_DEPTH, minsplit=MIN_SPLIT, minbucket=MIN_LEAF, replace=True, fraction=1.0, cores=cores)
    r_cforest_fit(X[wi][:, :min(p, 8)], y[wi], ntree=3, random_state=1, **kw)  # warm-up: R session
    t = time.time(); r_cforest_fit(X, y, ntree=TREES, random_state=0, **kw); el = time.time() - t
else:
    from sklearn.ensemble import RandomForestClassifier as RF
    t = time.time(); RF(n_estimators=TREES, max_depth=MAX_DEPTH, min_samples_split=MIN_SPLIT, min_samples_leaf=MIN_LEAF, n_jobs=-1, random_state=0).fit(X, y); el = time.time() - t
print(json.dumps({"fit_s": round(el, 2), "n": int(n), "p": int(p), "classes": int(len(np.unique(y)))}))
"""

Cell = tuple[str, str, int, int, str]


def build_cells(profile: dict[str, Any]) -> list[Cell]:
    """The ordered (kind, name, n, p, signal) cells of one profile."""
    cells: list[Cell] = [
        ("synthetic", "syn", n, p, signal)
        for n, p in profile["synthetic"]
        for signal in profile["signals"]
    ]
    cells.extend(("real", name, 0, 0, "real") for name in profile["real"])
    return cells


def shard_jobs(
    profile: dict[str, Any], shard_index: int, num_shards: int
) -> list[tuple[Cell, str]]:
    """The (cell, configuration) fits assigned round-robin to one shard."""
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard index {shard_index} is outside 0..{num_shards - 1}")
    jobs = [(cell, config) for cell in build_cells(profile) for config in CONFIGS]
    return jobs[shard_index::num_shards]


def _real_shape(path: Path) -> tuple[int, int]:
    """Rows and predictor columns of one real dataset (the response is column ``y``)."""
    parquet = pq.ParquetFile(path)
    return parquet.metadata.num_rows, len(parquet.schema_arrow.names) - 1


def fit_once(
    cell: Cell, config: str, *, trees: int, seed: int, data_dir: Path, fit_timeout: int
) -> dict[str, Any]:
    """Time one fit in a fresh process; a fit past ``fit_timeout`` is censored."""
    kind, name, n, p, signal = cell
    if kind == "real":
        n, p = _real_shape(data_dir / f"clf_{name}.parquet")
    record: dict[str, Any] = {"kind": kind, "dataset": name, "n": n, "p": p, "signal": signal}
    record["config"] = config
    try:
        completed = subprocess.run(
            [sys.executable, "-c", CHILD, kind, name, str(n), str(p), signal, config]
            + [str(trees), str(seed), str(data_dir)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=fit_timeout,
        )
    except subprocess.TimeoutExpired:
        return {**record, "fit_s": None, "classes": None, "timeout": True}
    if completed.returncode != 0:
        raise RuntimeError(
            f"{config} fit on {name} (n={n}, p={p}, {signal}) failed with exit code "
            f"{completed.returncode}:\n{completed.stderr[-2000:]}"
        )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    if (result["n"], result["p"]) != (n, p):
        raise RuntimeError(
            f"{name} shape differs: expected {(n, p)}, fit {(result['n'], result['p'])}"
        )
    return {**record, "fit_s": result["fit_s"], "classes": result["classes"], "timeout": False}


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Median fit seconds per cell and configuration; censored fits are excluded."""
    return (
        raw.pivot_table(
            index=["kind", "dataset", "n", "p", "signal"],
            columns="config",
            values="fit_s",
            aggfunc="median",
        )
        .reindex(columns=[c for c in CONFIGS if c in set(raw["config"])])
        .reset_index()
        .rename_axis(columns=None)
    )


def run(
    profile: dict[str, Any],
    *,
    seed: int,
    data_dir: Path,
    fit_timeout: int,
    shard_index: int,
    num_shards: int,
) -> pd.DataFrame:
    """Time every fit of one shard and return one row per fit."""
    jobs = shard_jobs(profile, shard_index, num_shards)
    print(f"shard {shard_index}/{num_shards}: {len(jobs)} fits", flush=True)
    rows = []
    for cell, config in jobs:
        repeats = 1 if cell[1] in SINGLE_REPEAT_DATASETS else profile["repeats"]
        for rep in range(repeats):
            row = fit_once(
                cell,
                config,
                trees=profile["trees"],
                seed=seed,
                data_dir=data_dir,
                fit_timeout=fit_timeout,
            )
            row["rep"] = rep
            rows.append(row)
            print(
                f"{row['kind']:9s} {row['dataset']:13s} n={row['n']:>6} p={row['p']:>5} "
                f"{row['signal']:6s} {config:24s} rep{rep}: "
                f"{'censored' if row['timeout'] else row['fit_s']}",
                flush=True,
            )
    return pd.DataFrame(rows)


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


def _r_versions() -> dict[str, str]:
    from paper.benchmark.pipeline.r_methods import get_r_runtime_versions

    return get_r_runtime_versions()


def write_receipt(
    output_dir: Path,
    args: argparse.Namespace,
    hardware: dict[str, Any],
    elapsed_seconds: float,
    artifacts: list[Path],
) -> None:
    """Write the suite receipt: revision, environment, inputs, sources, and artifact hashes."""
    profile = PROFILES[args.profile]
    versions = {
        p: metadata.version(p)
        for p in ("citrees", "numpy", "pandas", "scikit-learn", "scipy", "numba", "rpy2")
    }
    versions.update(_r_versions())
    receipt = {
        "analysis": ANALYSIS,
        "profile": args.profile,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "fit_timeout_seconds": args.fit_timeout,
        "design": {**profile, "configs": CONFIGS},
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
        "hardware": hardware,
        "elapsed_seconds": elapsed_seconds,
        "versions": versions,
        "input_sha256": {
            f"clf_{name}.parquet": _sha256(args.data_dir / f"clf_{name}.parquet")
            for name in profile["real"]
        },
        "source_sha256": {
            MODULE_PATH: _sha256(REPO_ROOT / MODULE_PATH),
            "paper/benchmark/pipeline/r_methods.py": _sha256(
                REPO_ROOT / "paper/benchmark/pipeline/r_methods.py"
            ),
            "pyproject.toml": _sha256(REPO_ROOT / "pyproject.toml"),
            "uv.lock": _sha256(REPO_ROOT / "uv.lock"),
        },
        "artifacts": {p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in artifacts},
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="ascii"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="full")
    parser.add_argument("--seed", type=int, default=1718, help="Synthetic-data seed base.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fit-timeout", type=int, default=DEFAULT_FIT_TIMEOUT)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    hardware = get_hardware_metadata()
    if args.profile == "full" and int(hardware["logical_cpus"]) < REFERENCE_LOGICAL_CPUS:
        raise RuntimeError(
            f"the full profile requires {REFERENCE_LOGICAL_CPUS} logical CPUs; "
            f"this host has {hardware['logical_cpus']}"
        )
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    started = time.perf_counter()
    raw = run(
        PROFILES[args.profile],
        seed=args.seed,
        data_dir=args.data_dir,
        fit_timeout=args.fit_timeout,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    args.output_dir.mkdir(parents=True)
    raw_path = args.output_dir / "head_to_head_raw.csv"
    summary_path = args.output_dir / "head_to_head_summary.csv"
    raw.to_csv(raw_path, index=False)
    summary = summarize(raw)
    summary.to_csv(summary_path, index=False)
    write_receipt(
        args.output_dir, args, hardware, time.perf_counter() - started, [raw_path, summary_path]
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
