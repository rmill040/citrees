"""Run and merge deterministic shards for the large JSS experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd

from paper.benchmark.pipeline.r_methods import get_r_runtime_versions
from paper.jss.replication import behavior, calibration, performance

Profile = Literal["smoke", "quick", "full"]
TargetAnalysis = Literal["calibration", "behavior", "performance"]
CalibrationComponent = Literal[
    "selector",
    "root",
    "cardinality_tree",
    "cardinality_forest",
]
RUNTIME_PACKAGES = (
    "citrees",
    "dcor",
    "numba",
    "numpy",
    "pandas",
    "pyarrow",
    "rpy2",
    "scikit-learn",
    "scipy",
)
R_ENVIRONMENT_KEYS = frozenset({"r", "partykit", "libcoin", "mvtnorm"})
THREAD_ENVIRONMENT_KEYS = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_DISABLE_JIT",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
    "R_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_FULL_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

BASE_SEED = 1718
REPO_ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_COMPONENTS: tuple[CalibrationComponent, ...] = (
    "selector",
    "root",
    "cardinality_tree",
    "cardinality_forest",
)
CALIBRATION_TABLES: dict[CalibrationComponent, str] = {
    "selector": "selector_null_raw",
    "root": "root_null_raw",
    "cardinality_tree": "cardinality_tree_raw",
    "cardinality_forest": "cardinality_forest_raw",
}


@dataclass(frozen=True)
class ShardSpec:
    """Complete identity of one deterministic experiment shard."""

    target_analysis: TargetAnalysis
    component: str
    profile: Profile
    shard_index: int
    num_shards: int
    base_seed: int


@dataclass(frozen=True)
class VerifiedShard:
    """One verified receipt and its local artifact directory."""

    receipt_path: Path
    output_dir: Path
    spec: ShardSpec
    receipt: dict[str, object]
    receipt_sha256: str


@dataclass(frozen=True)
class ExecutionContext:
    """Source, runtime, and input identity captured before shard execution."""

    git_sha: str
    git_dirty: bool
    python: str
    platform: str
    hardware: dict[str, object]
    blas_configuration: dict[str, object]
    thread_environment: dict[str, str]
    r_environment: dict[str, str]
    source_sha256: dict[str, str]
    input_sha256: dict[str, str]
    versions: dict[str, str]


_SHARD_RECEIPT_FIELDS = frozenset(
    {
        "analysis",
        "spec",
        "spec_sha256",
        "assignments",
        "assignment_count",
        "assignments_sha256",
        "created_utc",
        "elapsed_seconds",
        "execution_context_sha256",
        "scientific_context_sha256",
        *ExecutionContext.__dataclass_fields__,
        "tables",
        "artifacts",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _scientific_context_payload(
    context: Mapping[str, object],
    target_analysis: TargetAnalysis,
) -> dict[str, object]:
    hardware = context["hardware"]
    if not isinstance(hardware, Mapping):
        raise TypeError("execution-context hardware must be a mapping")
    payload = {
        "git_sha": context["git_sha"],
        "git_dirty": context["git_dirty"],
        "python": context["python"],
        "machine": hardware["machine"],
        "logical_cpus": hardware["logical_cpus"],
        "blas_configuration": context["blas_configuration"],
        "thread_environment": context["thread_environment"],
        "r_environment": context["r_environment"],
        "source_sha256": context["source_sha256"],
        "input_sha256": context["input_sha256"],
        "versions": context["versions"],
    }
    if target_analysis == "performance":
        payload["platform"] = context["platform"]
        payload["processor"] = hardware["processor"]
    return payload


def _git_sha() -> str:
    expected = os.environ.get("GIT_SHA")
    if expected is not None and not _FULL_GIT_SHA_PATTERN.fullmatch(expected):
        raise RuntimeError("GIT_SHA must be a lowercase 40-character Git revision")
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        observed = completed.stdout.strip()
        if not _FULL_GIT_SHA_PATTERN.fullmatch(observed):
            raise RuntimeError(f"Git returned an invalid source revision: {observed!r}")
        if expected is not None and expected != observed:
            raise RuntimeError(
                f"GIT_SHA {expected!r} differs from the checked-out revision {observed!r}"
            )
        return observed
    if expected is None:
        raise RuntimeError(
            "Source revision is unavailable: run inside a Git worktree or set GIT_SHA"
        )
    return expected


def _git_dirty() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return bool(completed.stdout.strip())
    if os.environ.get("GIT_SHA") is not None and os.environ.get("CITREES_SOURCE_CLEAN") == "1":
        return False
    raise RuntimeError(
        "Source cleanliness is unavailable outside a Git worktree; "
        "set GIT_SHA and CITREES_SOURCE_CLEAN=1 for a verified image"
    )


def _source_files(target_analysis: TargetAnalysis) -> tuple[Path, ...]:
    if target_analysis == "performance":
        return tuple(
            dict.fromkeys(
                (
                    Path(__file__).resolve(),
                    REPO_ROOT / "paper" / "jss" / "replication" / "cloud.py",
                    *performance.performance_source_files(),
                )
            )
        )
    analysis_file = Path(
        calibration.__file__ if target_analysis == "calibration" else behavior.__file__
    ).resolve()
    package_files = tuple(sorted((REPO_ROOT / "citrees").glob("*.py")))
    return (
        Path(__file__).resolve(),
        REPO_ROOT / "paper" / "jss" / "replication" / "cloud.py",
        analysis_file,
        REPO_ROOT / "paper" / "benchmark" / "pipeline" / "r_methods.py",
        *package_files,
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "uv.lock",
    )


def _versions() -> dict[str, str]:
    return {package: importlib.metadata.version(package) for package in RUNTIME_PACKAGES}


def _processor_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="ascii").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor()


def _hardware() -> dict[str, object]:
    logical_cpus = os.cpu_count()
    if logical_cpus is None or logical_cpus < 1:
        raise RuntimeError("logical CPU count is unavailable")
    return {
        "machine": platform.machine(),
        "processor": _processor_name(),
        "logical_cpus": logical_cpus,
    }


def _blas_configuration() -> dict[str, object]:
    return cast(
        dict[str, object],
        json.loads(json.dumps(np.__config__.CONFIG, sort_keys=True)),
    )


def _thread_environment() -> dict[str, str]:
    return {name: os.environ.get(name, "") for name in THREAD_ENVIRONMENT_KEYS}


def capture_execution_context(
    target_analysis: TargetAnalysis,
) -> ExecutionContext:
    """Capture the complete context that must remain fixed for one shard."""
    source_files = _source_files(target_analysis)
    input_sha256 = (
        {dataset.name: dataset.sha256 for dataset in behavior.load_behavior_datasets()}
        if target_analysis == "behavior"
        else {}
    )
    return ExecutionContext(
        git_sha=_git_sha(),
        git_dirty=_git_dirty(),
        python=platform.python_version(),
        platform=platform.platform(),
        hardware=_hardware(),
        blas_configuration=_blas_configuration(),
        thread_environment=_thread_environment(),
        r_environment=get_r_runtime_versions(),
        source_sha256={str(path.relative_to(REPO_ROOT)): _sha256(path) for path in source_files},
        input_sha256=input_sha256,
        versions=_versions(),
    )


def _calibration_replicate_count(
    profile: Profile,
    component: CalibrationComponent,
) -> int:
    settings = calibration._settings(profile)
    return {
        "selector": settings.selector_replicates,
        "root": settings.root_replicates,
        "cardinality_tree": settings.cardinality_replicates,
        "cardinality_forest": settings.cardinality_forest_replicates,
    }[component]


def _assignment_count(spec: ShardSpec) -> int:
    if spec.target_analysis == "calibration":
        component = cast(CalibrationComponent, spec.component)
        return _calibration_replicate_count(spec.profile, component)
    if spec.target_analysis == "behavior":
        return len(behavior.behavior_cell_inventory(spec.profile))
    return len(performance.build_performance_grid(spec.profile, base_seed=spec.base_seed))


def validate_shard_spec(spec: ShardSpec) -> None:
    """Require one nonempty shard within the target's exact inventory."""
    if spec.profile not in ("smoke", "quick", "full"):
        raise ValueError(f"unknown shard profile: {spec.profile!r}")
    if spec.target_analysis == "calibration":
        if spec.component not in CALIBRATION_COMPONENTS:
            raise ValueError(f"unknown calibration shard component: {spec.component!r}")
    elif spec.target_analysis == "behavior":
        if spec.component != "behavior":
            raise ValueError("behavior shards must use component='behavior'")
    elif spec.target_analysis == "performance":
        if spec.component != "performance":
            raise ValueError("performance shards must use component='performance'")
    else:
        raise ValueError(f"unknown shard target: {spec.target_analysis!r}")
    if isinstance(spec.base_seed, bool) or not isinstance(spec.base_seed, int):
        raise TypeError("base_seed must be an integer")
    if spec.base_seed < 0:
        raise ValueError("base_seed must be nonnegative")
    if isinstance(spec.num_shards, bool) or not isinstance(spec.num_shards, int):
        raise TypeError("num_shards must be an integer")
    if isinstance(spec.shard_index, bool) or not isinstance(spec.shard_index, int):
        raise TypeError("shard_index must be an integer")
    if spec.num_shards < 1:
        raise ValueError("num_shards must be at least one")
    if spec.shard_index < 0 or spec.shard_index >= spec.num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    total = _assignment_count(spec)
    if spec.num_shards > total:
        raise ValueError(f"num_shards must not exceed the {total} available assignments")


def calibration_shard_replicates(spec: ShardSpec) -> tuple[int, ...]:
    """Return this calibration shard's complete replicate assignment."""
    validate_shard_spec(spec)
    if spec.target_analysis != "calibration":
        raise ValueError("calibration replicate assignment requires a calibration shard")
    total = _assignment_count(spec)
    replicates = tuple(range(spec.shard_index, total, spec.num_shards))
    if not replicates:
        raise RuntimeError("calibration shard assignment is empty")
    return replicates


def behavior_shard_cells(spec: ShardSpec) -> tuple[behavior.BehaviorCell, ...]:
    """Return this behavior shard's complete paired-cell assignment."""
    validate_shard_spec(spec)
    if spec.target_analysis != "behavior":
        raise ValueError("behavior cell assignment requires a behavior shard")
    inventory = behavior.behavior_cell_inventory(spec.profile)
    start = len(inventory) * spec.shard_index // spec.num_shards
    stop = len(inventory) * (spec.shard_index + 1) // spec.num_shards
    cells = inventory[start:stop]
    if not cells:
        raise RuntimeError("behavior shard assignment is empty")
    return cells


def performance_shard_cells(spec: ShardSpec) -> tuple[performance.PerformanceCell, ...]:
    """Return this performance shard's complete isolated-cell assignment."""
    validate_shard_spec(spec)
    if spec.target_analysis != "performance":
        raise ValueError("performance cell assignment requires a performance shard")
    inventory = performance.build_performance_grid(spec.profile, base_seed=spec.base_seed)
    start = len(inventory) * spec.shard_index // spec.num_shards
    stop = len(inventory) * (spec.shard_index + 1) // spec.num_shards
    cells = inventory[start:stop]
    if not cells:
        raise RuntimeError("performance shard assignment is empty")
    return cells


def _frame_keys(frame: pd.DataFrame, columns: Sequence[str]) -> set[tuple[object, ...]]:
    return set(frame.loc[:, columns].itertuples(index=False, name=None))


def _validate_exact_rows(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    expected: set[tuple[object, ...]],
    label: str,
) -> None:
    if frame.duplicated(list(key_columns)).any():
        raise ValueError(f"{label} contains duplicate keys")
    observed = _frame_keys(frame, key_columns)
    if observed != expected or len(frame) != len(expected):
        missing = sorted(expected.difference(observed), key=repr)
        extra = sorted(observed.difference(expected), key=repr)
        raise ValueError(f"{label} inventory differs: missing={missing}, extra={extra}")


def _validate_calibration_component(
    component: CalibrationComponent,
    frame: pd.DataFrame,
    *,
    profile: Profile,
    base_seed: int,
    replicates: Sequence[int],
) -> None:
    table_name = CALIBRATION_TABLES[component]
    expected_schema = calibration.CALIBRATION_RESULT_SCHEMAS[table_name]
    if tuple(frame.columns) != expected_schema:
        raise ValueError(f"{table_name} differs from its required schema")
    if component == "selector":
        calibration.validate_selector_null_raw(
            frame,
            profile,
            base_seed=base_seed,
            replicate_indices=replicates,
        )
        return
    if component == "root":
        calibration.validate_root_null_raw(
            frame,
            profile,
            base_seed=base_seed,
            replicate_indices=replicates,
        )
        return
    if component == "cardinality_tree":
        calibration.validate_cardinality_tree_experiment(
            frame,
            profile,
            base_seed=base_seed,
            replicate_indices=replicates,
        )
        return
    calibration.validate_cardinality_forest_experiment(
        frame,
        profile,
        base_seed=base_seed,
        replicate_indices=replicates,
    )


def run_calibration_shard(spec: ShardSpec) -> dict[str, pd.DataFrame]:
    """Run and validate one calibration component shard."""
    replicates = calibration_shard_replicates(spec)
    component = cast(CalibrationComponent, spec.component)
    if component == "selector":
        frame = calibration.run_selector_null(
            spec.profile,
            base_seed=spec.base_seed,
            replicate_indices=replicates,
        )
    elif component == "root":
        frame = calibration.run_root_null(
            spec.profile,
            base_seed=spec.base_seed,
            replicate_indices=replicates,
        )
    elif component == "cardinality_tree":
        frame = calibration.run_cardinality_trees(
            spec.profile,
            base_seed=spec.base_seed,
            replicate_indices=replicates,
        )
    else:
        frame = calibration.run_cardinality_forests(
            spec.profile,
            base_seed=spec.base_seed,
            replicate_indices=replicates,
        )
    _validate_calibration_component(
        component,
        frame,
        profile=spec.profile,
        base_seed=spec.base_seed,
        replicates=replicates,
    )
    return {CALIBRATION_TABLES[component]: frame}


def _behavior_cell_keys(
    cells: Sequence[behavior.BehaviorCell],
) -> set[tuple[object, ...]]:
    return {
        (
            cell.task,
            cell.dataset,
            cell.model_family,
            cell.repeat,
            cell.fold,
        )
        for cell in cells
    }


def _validate_behavior_raw(
    tables: Mapping[str, pd.DataFrame],
    cells: Sequence[behavior.BehaviorCell],
) -> None:
    if set(tables) != set(behavior.BEHAVIOR_RAW_TABLES):
        raise ValueError("behavior shard tables differ from the required inventory")
    for name in behavior.BEHAVIOR_RAW_TABLES:
        if tuple(tables[name].columns) != behavior.BEHAVIOR_RESULT_SCHEMAS[name]:
            raise ValueError(f"{name} differs from its required schema")
    cell_keys = _behavior_cell_keys(cells)
    fits = tables["behavior_fit_raw"]
    expected_fits = {(*cell_key, method) for cell_key in cell_keys for method in behavior.METHODS}
    fit_key = ("task", "dataset", "model_family", "repeat", "fold", "method")
    _validate_exact_rows(
        fits,
        key_columns=fit_key,
        expected=expected_fits,
        label="behavior_fit_raw",
    )
    folds = tables["behavior_fold_summary"]
    _validate_exact_rows(
        folds,
        key_columns=("task", "dataset", "model_family", "repeat", "fold"),
        expected=cell_keys,
        label="behavior_fold_summary",
    )
    features = tables["behavior_feature_raw"]
    predictions = tables["behavior_prediction_raw"]
    probabilities = tables["behavior_probability_raw"]
    if _frame_keys(features, fit_key) != expected_fits:
        raise ValueError("behavior feature groups differ from fitted cells")
    if _frame_keys(predictions, fit_key) != expected_fits:
        raise ValueError("behavior prediction groups differ from fitted cells")
    classification_fits = {key for key in expected_fits if key[0] == "classification"}
    if _frame_keys(probabilities, fit_key) != classification_fits:
        raise ValueError("behavior probability groups differ from classification fits")
    if features.duplicated([*fit_key, "feature_id"]).any():
        raise ValueError("behavior feature rows contain duplicate keys")
    if predictions.duplicated([*fit_key, "sample_id"]).any():
        raise ValueError("behavior prediction rows contain duplicate keys")
    if probabilities.duplicated([*fit_key, "sample_id", "class_label"]).any():
        raise ValueError("behavior probability rows contain duplicate keys")


def run_behavior_shard(spec: ShardSpec) -> dict[str, pd.DataFrame]:
    """Run and validate one behavior shard."""
    cells = behavior_shard_cells(spec)
    tables = behavior.run_behavior_raw(
        spec.profile,
        base_seed=spec.base_seed,
        selected_cells=cells,
    )
    _validate_behavior_raw(tables, cells)
    return tables


def run_performance_shard(spec: ShardSpec) -> dict[str, pd.DataFrame]:
    """Run and validate one performance shard."""
    cells = performance_shard_cells(spec)
    rows = [performance._run_cell_subprocess(cell) for cell in cells]
    raw = pd.DataFrame(rows).loc[:, performance.PERFORMANCE_RAW_SCHEMA]
    performance.validate_performance_raw(
        raw,
        cells,
        profile=spec.profile,
        require_unique_worker_pids=True,
    )
    return {"performance_raw": raw}


def _assignment_payload(spec: ShardSpec) -> list[object]:
    if spec.target_analysis == "calibration":
        return list(calibration_shard_replicates(spec))
    if spec.target_analysis == "behavior":
        return [asdict(cell) for cell in behavior_shard_cells(spec)]
    return [asdict(cell) for cell in performance_shard_cells(spec)]


def _validate_written_shard(
    spec: ShardSpec,
    tables: Mapping[str, pd.DataFrame],
) -> None:
    if spec.target_analysis == "calibration":
        component = cast(CalibrationComponent, spec.component)
        _validate_calibration_component(
            component,
            tables[CALIBRATION_TABLES[component]],
            profile=spec.profile,
            base_seed=spec.base_seed,
            replicates=calibration_shard_replicates(spec),
        )
    elif spec.target_analysis == "behavior":
        _validate_behavior_raw(tables, behavior_shard_cells(spec))
    else:
        if set(tables) != {"performance_raw"}:
            raise ValueError("performance shard tables differ from the required inventory")
        performance.validate_performance_raw(
            tables["performance_raw"],
            performance_shard_cells(spec),
            profile=spec.profile,
            require_unique_worker_pids=True,
        )


def _normalize_shard_tables(
    spec: ShardSpec,
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    if spec.target_analysis == "calibration":
        return {
            name: calibration.normalize_calibration_table(name, frame)
            for name, frame in tables.items()
        }
    if spec.target_analysis == "behavior":
        return {
            name: behavior.normalize_behavior_table(name, frame) for name, frame in tables.items()
        }
    return {"performance_raw": tables["performance_raw"].loc[:, performance.PERFORMANCE_RAW_SCHEMA]}


def write_shard(
    spec: ShardSpec,
    tables: Mapping[str, pd.DataFrame],
    output_dir: Path,
    *,
    elapsed_seconds: float,
    execution_context: ExecutionContext,
) -> Path:
    """Atomically write one shard and its complete provenance receipt."""
    validate_shard_spec(spec)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"shard output already exists: {output_dir}")
    if spec.profile == "full" and execution_context.git_dirty:
        raise RuntimeError("full-profile shards require a clean source tree")
    if capture_execution_context(spec.target_analysis) != execution_context:
        raise RuntimeError("shard execution context changed during computation")
    tables = _normalize_shard_tables(spec, tables)
    _validate_written_shard(spec, tables)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-",
        dir=output_dir.parent,
    ) as temporary:
        staging = Path(temporary) / output_dir.name
        staging.mkdir()
        artifacts: dict[str, dict[str, object]] = {}
        written: dict[str, pd.DataFrame] = {}
        for name, frame in tables.items():
            artifact_path = staging / f"{name}.parquet"
            frame.to_parquet(artifact_path, index=False)
            written[name] = pd.read_parquet(artifact_path)
            artifacts[artifact_path.name] = {
                "bytes": artifact_path.stat().st_size,
                "sha256": _sha256(artifact_path),
            }
        _validate_written_shard(spec, written)
        assignments = _assignment_payload(spec)
        context_payload = asdict(execution_context)
        receipt = {
            "analysis": "jss_shard",
            "spec": asdict(spec),
            "spec_sha256": _json_sha256(asdict(spec)),
            "assignments": assignments,
            "assignment_count": len(assignments),
            "assignments_sha256": _json_sha256(assignments),
            "created_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "execution_context_sha256": _json_sha256(context_payload),
            "scientific_context_sha256": _json_sha256(
                _scientific_context_payload(context_payload, spec.target_analysis)
            ),
            **context_payload,
            "tables": {
                name: {"rows": len(frame), "columns": list(frame.columns)}
                for name, frame in written.items()
            },
            "artifacts": artifacts,
        }
        (staging / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        staging.rename(output_dir)
    return output_dir


def run_shard_to_directory(spec: ShardSpec, output_dir: Path) -> Path:
    """Execute and write one requested shard."""
    validate_shard_spec(spec)
    execution_context = capture_execution_context(spec.target_analysis)
    if spec.profile == "full" and execution_context.git_dirty:
        raise RuntimeError("full-profile shards require a clean source tree")
    started = time.perf_counter()
    if spec.target_analysis == "calibration":
        tables = run_calibration_shard(spec)
    elif spec.target_analysis == "behavior":
        tables = run_behavior_shard(spec)
    else:
        tables = run_performance_shard(spec)
    return write_shard(
        spec,
        tables,
        output_dir,
        elapsed_seconds=time.perf_counter() - started,
        execution_context=execution_context,
    )


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _require_json_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _require_canonical_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} is invalid") from error
    if parsed.tzinfo is None or parsed.astimezone(UTC).isoformat() != value:
        raise ValueError(f"{label} must be canonical UTC")
    return parsed


def _expected_artifact_names(spec: ShardSpec) -> set[str]:
    if spec.target_analysis == "calibration":
        component = cast(CalibrationComponent, spec.component)
        return {f"{CALIBRATION_TABLES[component]}.parquet"}
    if spec.target_analysis == "behavior":
        return {f"{name}.parquet" for name in behavior.BEHAVIOR_RAW_TABLES}
    return {"performance_raw.parquet"}


def _verify_receipt_artifacts(
    receipt: Mapping[str, object],
    output_dir: Path,
    spec: ShardSpec,
) -> None:
    artifacts = _require_mapping(receipt.get("artifacts"), "shard artifacts")
    if set(artifacts) != _expected_artifact_names(spec):
        raise ValueError("shard artifact inventory differs from its specification")
    expected_files = {"receipt.json", *artifacts}
    observed_files = {
        str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()
    }
    if observed_files != expected_files:
        raise ValueError("shard output inventory differs from its receipt")
    for name, raw_metadata in artifacts.items():
        metadata = _require_mapping(raw_metadata, f"artifact metadata for {name}")
        if set(metadata) != {"bytes", "sha256"}:
            raise ValueError(f"shard artifact metadata fields differ for {name}")
        artifact_path = output_dir / name
        if artifact_path.parent != output_dir or not artifact_path.is_file():
            raise ValueError(f"invalid shard artifact path: {name!r}")
        byte_count = metadata.get("bytes")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, Integral)
            or int(byte_count) != artifact_path.stat().st_size
        ):
            raise ValueError(f"shard artifact byte count differs for {name}")
        if metadata.get("sha256") != _sha256(artifact_path):
            raise ValueError(f"shard artifact hash differs for {name}")


def _verify_receipt_tables(
    receipt: Mapping[str, object],
    output_dir: Path,
    spec: ShardSpec,
) -> None:
    tables = _require_mapping(receipt.get("tables"), "shard tables")
    expected_tables = {name.removesuffix(".parquet") for name in _expected_artifact_names(spec)}
    if set(tables) != expected_tables:
        raise ValueError("shard table inventory differs from its specification")
    for name, raw_metadata in tables.items():
        metadata = _require_mapping(raw_metadata, f"table metadata for {name}")
        if set(metadata) != {"rows", "columns"}:
            raise ValueError(f"shard table metadata fields differ for {name}")
        rows = metadata["rows"]
        columns = metadata["columns"]
        if isinstance(rows, bool) or not isinstance(rows, Integral) or int(rows) < 0:
            raise ValueError(f"shard table row count is invalid for {name}")
        if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
            raise TypeError(f"shard table columns must be a JSON string array for {name}")
        frame = pd.read_parquet(output_dir / f"{name}.parquet")
        if int(rows) != len(frame) or columns != list(frame.columns):
            raise ValueError(f"shard table metadata differs from the artifact for {name}")


def _verify_receipt_sources(
    receipt: Mapping[str, object],
    spec: ShardSpec,
) -> None:
    source_hashes = _require_mapping(receipt.get("source_sha256"), "shard source hashes")
    expected_sources = {
        str(path.relative_to(REPO_ROOT)) for path in _source_files(spec.target_analysis)
    }
    if set(source_hashes) != expected_sources:
        raise ValueError("shard source hash inventory differs from its specification")
    for name, expected_sha256 in source_hashes.items():
        if not isinstance(name, str) or not isinstance(expected_sha256, str):
            raise TypeError("shard source hashes must map strings to strings")
        source_path = (REPO_ROOT / name).resolve()
        if not source_path.is_relative_to(REPO_ROOT) or not source_path.is_file():
            raise ValueError(f"invalid shard source path: {name!r}")
        if _sha256(source_path) != expected_sha256:
            raise ValueError(f"shard source hash differs for {name}")


def _verify_execution_context(
    receipt: Mapping[str, object],
    spec: ShardSpec,
) -> None:
    python_version = receipt.get("python")
    platform_name = receipt.get("platform")
    if not isinstance(python_version, str) or not python_version:
        raise TypeError("shard Python version must be a nonempty string")
    if not isinstance(platform_name, str) or not platform_name:
        raise TypeError("shard platform must be a nonempty string")

    hardware = _require_mapping(receipt.get("hardware"), "shard hardware")
    if set(hardware) != {"machine", "processor", "logical_cpus"}:
        raise ValueError("shard hardware fields differ from the required inventory")
    if not isinstance(hardware["machine"], str) or not hardware["machine"]:
        raise ValueError("shard machine identity must be a nonempty string")
    if not isinstance(hardware["processor"], str):
        raise TypeError("shard processor identity must be a string")
    logical_cpus = hardware["logical_cpus"]
    if (
        isinstance(logical_cpus, bool)
        or not isinstance(logical_cpus, Integral)
        or int(logical_cpus) < 1
    ):
        raise ValueError("shard logical CPU count must be a positive integer")
    blas_configuration = _require_mapping(
        receipt.get("blas_configuration"),
        "shard BLAS configuration",
    )
    if not blas_configuration:
        raise ValueError("shard BLAS configuration must not be empty")
    thread_environment = _require_mapping(
        receipt.get("thread_environment"),
        "shard thread environment",
    )
    if set(thread_environment) != set(THREAD_ENVIRONMENT_KEYS) or not all(
        isinstance(value, str) for value in thread_environment.values()
    ):
        raise ValueError("shard thread environment differs from the required inventory")

    r_environment = _require_mapping(
        receipt.get("r_environment"),
        "shard R environment",
    )
    if set(r_environment) != R_ENVIRONMENT_KEYS or not all(
        isinstance(value, str) and value for value in r_environment.values()
    ):
        raise ValueError("shard R environment differs from the required inventory")
    versions = _require_mapping(receipt.get("versions"), "shard package versions")
    if set(versions) != set(RUNTIME_PACKAGES) or not all(
        isinstance(value, str) and value for value in versions.values()
    ):
        raise ValueError("shard package versions differ from the required inventory")

    source_sha256 = _require_mapping(
        receipt.get("source_sha256"),
        "shard source hashes",
    )
    input_sha256 = _require_mapping(
        receipt.get("input_sha256"),
        "shard input hashes",
    )
    expected_inputs = (
        {dataset.name: dataset.sha256 for dataset in behavior.load_behavior_datasets()}
        if spec.target_analysis == "behavior"
        else {}
    )
    if dict(input_sha256) != expected_inputs:
        raise ValueError("shard input hashes differ from the frozen datasets")

    context_payload = {
        "git_sha": receipt.get("git_sha"),
        "git_dirty": receipt.get("git_dirty"),
        "python": python_version,
        "platform": platform_name,
        "hardware": dict(hardware),
        "blas_configuration": dict(blas_configuration),
        "thread_environment": dict(thread_environment),
        "r_environment": dict(r_environment),
        "source_sha256": dict(source_sha256),
        "input_sha256": dict(input_sha256),
        "versions": dict(versions),
    }
    if receipt.get("execution_context_sha256") != _json_sha256(context_payload):
        raise ValueError("shard execution context hash differs")
    if receipt.get("scientific_context_sha256") != _json_sha256(
        _scientific_context_payload(context_payload, spec.target_analysis)
    ):
        raise ValueError("shard scientific context hash differs")


def _parse_shard_spec(raw_spec: object) -> ShardSpec:
    spec_values = _require_mapping(raw_spec, "shard spec")
    expected = set(ShardSpec.__dataclass_fields__)
    if set(spec_values) != expected:
        raise ValueError("shard spec fields differ from the required schema")
    target_analysis = spec_values["target_analysis"]
    component = spec_values["component"]
    profile = spec_values["profile"]
    if not isinstance(target_analysis, str):
        raise TypeError("target_analysis must be a string")
    if not isinstance(component, str):
        raise TypeError("component must be a string")
    if not isinstance(profile, str):
        raise TypeError("profile must be a string")
    spec = ShardSpec(
        target_analysis=cast(TargetAnalysis, target_analysis),
        component=component,
        profile=cast(Profile, profile),
        shard_index=_require_json_integer(spec_values["shard_index"], "shard_index"),
        num_shards=_require_json_integer(spec_values["num_shards"], "num_shards"),
        base_seed=_require_json_integer(spec_values["base_seed"], "base_seed"),
    )
    validate_shard_spec(spec)
    return spec


def _verify_shard_receipt(
    receipt_path: Path,
    *,
    expected_spec: ShardSpec | None,
    expected_git_sha: str,
    allow_cloud_execution: bool,
    require_cloud_execution: bool,
) -> VerifiedShard:
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    if not isinstance(receipt, dict) or receipt.get("analysis") != "jss_shard":
        raise ValueError("shard receipt has an invalid analysis identity")
    has_cloud_execution = "cloud_execution" in receipt
    if has_cloud_execution and not allow_cloud_execution:
        raise ValueError("base shard receipt contains cloud execution metadata")
    if require_cloud_execution and not has_cloud_execution:
        raise ValueError("cloud shard receipt omits cloud execution metadata")
    expected_fields = set(_SHARD_RECEIPT_FIELDS)
    if has_cloud_execution:
        expected_fields.add("cloud_execution")
    if set(receipt) != expected_fields:
        raise ValueError("shard receipt fields differ from the required schema")

    spec = _parse_shard_spec(receipt.get("spec"))
    if expected_spec is not None and spec != expected_spec:
        raise ValueError("shard receipt specification differs from the requested shard")
    if receipt.get("spec_sha256") != _json_sha256(asdict(spec)):
        raise ValueError("shard spec hash differs")
    assignments = receipt.get("assignments")
    if not isinstance(assignments, list):
        raise TypeError("shard assignments must be a JSON array")
    if assignments != _assignment_payload(spec):
        raise ValueError("shard assignments differ from the deterministic inventory")
    if _require_json_integer(receipt.get("assignment_count"), "assignment_count") != len(
        assignments
    ):
        raise ValueError("shard assignment count differs")
    if receipt.get("assignments_sha256") != _json_sha256(assignments):
        raise ValueError("shard assignment hash differs")
    if receipt.get("git_sha") != expected_git_sha:
        raise ValueError("shard source revision differs from the expected revision")
    if not isinstance(receipt.get("git_dirty"), bool):
        raise TypeError("shard git_dirty must be boolean")
    if spec.profile == "full" and receipt["git_dirty"]:
        raise ValueError("full-profile shard records a dirty source tree")
    _require_canonical_utc(receipt.get("created_utc"), "created_utc")
    elapsed_seconds = receipt.get("elapsed_seconds")
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, Real)
        or not 0.0 <= float(elapsed_seconds) < float("inf")
    ):
        raise ValueError("shard elapsed_seconds must be finite and nonnegative")
    _verify_receipt_sources(receipt, spec)
    _verify_execution_context(receipt, spec)
    output_dir = receipt_path.parent
    _verify_receipt_artifacts(receipt, output_dir, spec)
    _verify_receipt_tables(receipt, output_dir, spec)
    return VerifiedShard(
        receipt_path=receipt_path,
        output_dir=output_dir,
        spec=spec,
        receipt=receipt,
        receipt_sha256=_sha256(receipt_path),
    )


def verify_shard_directory(
    output_dir: Path,
    *,
    expected_spec: ShardSpec,
    expected_git_sha: str | None = None,
    allow_cloud_execution: bool = False,
    require_cloud_execution: bool = False,
) -> VerifiedShard:
    """Verify one exact shard directory against its requested specification."""
    output_dir = output_dir.resolve()
    if not output_dir.is_dir():
        raise NotADirectoryError(f"shard output is not a directory: {output_dir}")
    validate_shard_spec(expected_spec)
    return _verify_shard_receipt(
        output_dir / "receipt.json",
        expected_spec=expected_spec,
        expected_git_sha=expected_git_sha or _git_sha(),
        allow_cloud_execution=allow_cloud_execution,
        require_cloud_execution=require_cloud_execution,
    )


def discover_verified_shards(
    shard_root: Path,
    *,
    target_analysis: TargetAnalysis,
    profile: Profile,
    base_seed: int,
) -> tuple[VerifiedShard, ...]:
    """Discover and verify every matching shard receipt below one root."""
    shard_root = shard_root.resolve()
    if not shard_root.is_dir():
        raise NotADirectoryError(f"shard root is not a directory: {shard_root}")
    current_git_sha = _git_sha()
    verified: list[VerifiedShard] = []
    for receipt_path in sorted(shard_root.rglob("receipt.json")):
        if any(part.startswith(".") for part in receipt_path.relative_to(shard_root).parts):
            continue
        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
        if not isinstance(receipt, dict) or receipt.get("analysis") != "jss_shard":
            continue
        spec = _parse_shard_spec(receipt.get("spec"))
        if spec.target_analysis != target_analysis:
            continue
        if spec.profile != profile or spec.base_seed != base_seed:
            raise ValueError("shard profile or seed differs from the requested merge")
        verified.append(
            _verify_shard_receipt(
                receipt_path,
                expected_spec=spec,
                expected_git_sha=current_git_sha,
                allow_cloud_execution=True,
                require_cloud_execution=False,
            )
        )
    if not verified:
        raise ValueError(f"no {target_analysis} shard receipts found below {shard_root}")
    context_hashes = {shard.receipt.get("scientific_context_sha256") for shard in verified}
    if len(context_hashes) != 1:
        raise ValueError("shards were produced by different scientific contexts")
    return tuple(verified)


def _validate_complete_shard_group(
    shards: Sequence[VerifiedShard],
    *,
    component: str,
) -> None:
    if not shards:
        raise ValueError(f"missing shard group for {component}")
    counts = {shard.spec.num_shards for shard in shards}
    if len(counts) != 1:
        raise ValueError(f"{component} shards disagree on num_shards")
    num_shards = counts.pop()
    indices = [shard.spec.shard_index for shard in shards]
    if len(set(indices)) != len(indices):
        raise ValueError(f"{component} contains duplicate shard indices")
    if set(indices) != set(range(num_shards)):
        raise ValueError(f"{component} shard index inventory is incomplete")


def _read_shard_table(shard: VerifiedShard, table_name: str) -> pd.DataFrame:
    artifact_name = f"{table_name}.parquet"
    artifacts = _require_mapping(shard.receipt.get("artifacts"), "shard artifacts")
    if set(artifacts) != _expected_artifact_names(shard.spec):
        raise ValueError(f"{shard.spec.component} shard contains unexpected artifacts")
    artifact_path = shard.output_dir / artifact_name
    if not artifact_path.is_file():
        raise ValueError(f"shard omits {artifact_name}")
    return pd.read_parquet(artifact_path)


def _order_calibration_component(
    frame: pd.DataFrame,
    component: CalibrationComponent,
    profile: Profile,
) -> pd.DataFrame:
    """Reconstruct the exact row order emitted by the one-shot runner."""
    if component == "selector":
        selector_scenarios = calibration._selector_scenarios(
            profile,
            calibration._settings(profile),
        )
        scenario_order = {
            scenario.scenario: index for index, scenario in enumerate(selector_scenarios)
        }
        ordered = frame.assign(
            _scenario_order=frame["scenario"].map(scenario_order),
        ).sort_values(
            ["_scenario_order", "replicate"],
            kind="stable",
        )
        return ordered.drop(columns="_scenario_order").reset_index(drop=True)
    if component == "root":
        root_scenarios = calibration._root_scenarios(
            profile,
            calibration._settings(profile),
        )
        scenario_order = {scenario.scenario: index for index, scenario in enumerate(root_scenarios)}
        ordered = frame.assign(
            _scenario_order=frame["scenario"].map(scenario_order),
        ).sort_values(
            ["_scenario_order", "replicate"],
            kind="stable",
        )
        return ordered.drop(columns="_scenario_order").reset_index(drop=True)

    task_order = {"classification": 0, "regression": 1}
    method_order = (
        {"citrees": 0, "partykit": 1, "cart": 2}
        if component == "cardinality_tree"
        else {
            "citrees_cif": 0,
            "partykit_cforest": 1,
            "sklearn_rf": 2,
        }
    )
    ordered = frame.assign(
        _task_order=frame["task"].map(task_order),
        _method_order=frame["method"].map(method_order),
    ).sort_values(
        ["_task_order", "replicate", "_method_order", "feature_id"],
        kind="stable",
    )
    return ordered.drop(columns=["_task_order", "_method_order"]).reset_index(drop=True)


def _order_behavior_table(
    frame: pd.DataFrame,
    table_name: str,
    profile: Profile,
) -> pd.DataFrame:
    """Reconstruct exact one-shot cell and method order for one raw table."""
    cell_order: dict[tuple[str, str, str, int, int], int] = {
        (
            cell.task,
            cell.dataset,
            cell.model_family,
            cell.repeat,
            cell.fold,
        ): index
        for index, cell in enumerate(behavior.behavior_cell_inventory(profile))
    }
    observed_cell_order = [
        cell_order[
            (
                str(row.task),
                str(row.dataset),
                str(row.model_family),
                int(row.repeat),
                int(row.fold),
            )
        ]
        for row in frame.itertuples(index=False)
    ]
    ordered = frame.assign(_cell_order=observed_cell_order)
    sort_columns = ["_cell_order"]
    if table_name != "behavior_fold_summary":
        method_order = {method: index for index, method in enumerate(behavior.METHODS)}
        ordered = ordered.assign(_method_order=ordered["method"].map(method_order))
        sort_columns.append("_method_order")
    ordered = ordered.sort_values(sort_columns, kind="stable")
    return ordered.drop(columns=sort_columns).reset_index(drop=True)


def merge_calibration_shards(
    shard_root: Path,
    *,
    profile: Profile,
    base_seed: int = BASE_SEED,
) -> tuple[dict[str, pd.DataFrame], tuple[VerifiedShard, ...]]:
    """Merge a complete calibration shard inventory into final tables."""
    shards = discover_verified_shards(
        shard_root,
        target_analysis="calibration",
        profile=profile,
        base_seed=base_seed,
    )
    by_component: dict[str, list[VerifiedShard]] = defaultdict(list)
    for shard in shards:
        by_component[shard.spec.component].append(shard)
    if set(by_component) != set(CALIBRATION_COMPONENTS):
        raise ValueError("calibration shard components are incomplete")
    merged: dict[str, pd.DataFrame] = {}
    for component in CALIBRATION_COMPONENTS:
        component_shards = by_component[component]
        _validate_complete_shard_group(component_shards, component=component)
        table_name = CALIBRATION_TABLES[component]
        frame = pd.concat(
            [
                _read_shard_table(shard, table_name)
                for shard in sorted(component_shards, key=lambda item: item.spec.shard_index)
            ],
            ignore_index=True,
        )
        frame = _order_calibration_component(frame, component, profile)
        _validate_calibration_component(
            component,
            frame,
            profile=profile,
            base_seed=base_seed,
            replicates=range(_calibration_replicate_count(profile, component)),
        )
        merged[table_name] = frame
    results = calibration.assemble_calibration_results(
        merged["selector_null_raw"],
        merged["root_null_raw"],
        merged["cardinality_tree_raw"],
        merged["cardinality_forest_raw"],
        profile=profile,
        base_seed=base_seed,
    )
    return results, shards


def merge_behavior_shards(
    shard_root: Path,
    *,
    profile: Profile,
    base_seed: int = BASE_SEED,
) -> tuple[dict[str, pd.DataFrame], tuple[VerifiedShard, ...]]:
    """Merge a complete behavior shard inventory into final tables."""
    shards = discover_verified_shards(
        shard_root,
        target_analysis="behavior",
        profile=profile,
        base_seed=base_seed,
    )
    if {shard.spec.component for shard in shards} != {"behavior"}:
        raise ValueError("behavior shard components are invalid")
    _validate_complete_shard_group(shards, component="behavior")
    raw: dict[str, pd.DataFrame] = {}
    for table_name in behavior.BEHAVIOR_RAW_TABLES:
        frame = pd.concat(
            [
                pd.read_parquet(shard.output_dir / f"{table_name}.parquet")
                for shard in sorted(shards, key=lambda item: item.spec.shard_index)
            ],
            ignore_index=True,
        )
        raw[table_name] = _order_behavior_table(frame, table_name, profile)
    cells = behavior.behavior_cell_inventory(profile)
    _validate_behavior_raw(raw, cells)
    results = behavior.assemble_behavior_results(
        raw,
        profile,
        base_seed=base_seed,
    )
    return results, shards


def merge_performance_shards(
    shard_root: Path,
    *,
    profile: Profile,
    base_seed: int = BASE_SEED,
) -> tuple[dict[str, pd.DataFrame], tuple[VerifiedShard, ...]]:
    """Merge a complete performance shard inventory into final tables."""
    verified = discover_verified_shards(
        shard_root,
        target_analysis="performance",
        profile=profile,
        base_seed=base_seed,
    )
    if {shard.spec.component for shard in verified} != {"performance"}:
        raise ValueError("performance shard components are invalid")
    _validate_complete_shard_group(verified, component="performance")
    raw = pd.concat(
        [
            _read_shard_table(shard, "performance_raw")
            for shard in sorted(verified, key=lambda item: item.spec.shard_index)
        ],
        ignore_index=True,
    )
    cell_order = {
        cell.cell_id: index
        for index, cell in enumerate(
            performance.build_performance_grid(profile, base_seed=base_seed)
        )
    }
    raw = (
        raw.assign(_cell_order=raw["cell_id"].map(cell_order))
        .sort_values("_cell_order", kind="stable")
        .drop(columns="_cell_order")
        .reset_index(drop=True)
    )
    expected_cells = performance.build_performance_grid(profile, base_seed=base_seed)
    performance.validate_performance_raw(
        raw,
        expected_cells,
        profile=profile,
        require_unique_worker_pids=False,
    )
    summary = performance.summarize_performance(raw)
    performance.validate_performance_results(
        raw,
        summary,
        profile=profile,
        base_seed=base_seed,
        require_unique_worker_pids=False,
    )
    return {"performance_raw": raw, "performance_summary": summary}, verified


def _augment_final_receipt(
    output_dir: Path,
    shards: Sequence[VerifiedShard],
    *,
    shard_root: Path,
    merge_elapsed_seconds: float,
) -> None:
    receipt_path = output_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    if not isinstance(receipt, dict):
        raise TypeError("final analysis receipt must be a JSON object")
    source_hashes = receipt.setdefault("source_sha256", {})
    if not isinstance(source_hashes, dict):
        raise TypeError("final analysis source hashes must be a JSON object")
    source_hashes[str(Path(__file__).resolve().relative_to(REPO_ROOT))] = _sha256(
        Path(__file__).resolve()
    )
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("final analysis artifacts must be a JSON object")
    shard_elapsed_seconds = [
        float(cast(Real, shard.receipt["elapsed_seconds"])) for shard in shards
    ]
    shard_receipts: list[dict[str, object]] = []
    for shard in sorted(
        shards,
        key=lambda item: (
            item.spec.component,
            item.spec.shard_index,
        ),
    ):
        receipt_artifact = (
            f"distributed_shard_receipt__{shard.spec.component}__{shard.spec.shard_index:05d}.json"
        )
        receipt_artifact_path = output_dir / receipt_artifact
        shutil.copyfile(shard.receipt_path, receipt_artifact_path)
        artifacts[receipt_artifact] = {
            "bytes": receipt_artifact_path.stat().st_size,
            "sha256": _sha256(receipt_artifact_path),
        }
        shard_receipts.append(
            {
                "source_receipt": str(shard.receipt_path.relative_to(shard_root.resolve())),
                "receipt_artifact": receipt_artifact,
                "receipt_sha256": shard.receipt_sha256,
                "spec": asdict(shard.spec),
            }
        )
    receipt["distributed_execution"] = {
        "n_shards": len(shards),
        "sum_shard_elapsed_seconds": sum(shard_elapsed_seconds),
        "maximum_shard_elapsed_seconds": max(shard_elapsed_seconds),
        "merge_elapsed_seconds": merge_elapsed_seconds,
        "shards": shard_receipts,
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def write_merged_results(
    target_analysis: TargetAnalysis,
    results: dict[str, pd.DataFrame],
    shards: Sequence[VerifiedShard],
    output_dir: Path,
    *,
    shard_root: Path,
    profile: Profile,
    base_seed: int,
    merge_elapsed_seconds: float,
) -> Path:
    """Atomically write one merged analysis and its shard provenance."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"merged output already exists: {output_dir}")
    if profile == "full" and _git_dirty():
        raise RuntimeError("full-profile shard merges require a clean source tree")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-",
        dir=output_dir.parent,
    ) as temporary:
        staging = Path(temporary) / output_dir.name
        elapsed = merge_elapsed_seconds
        if target_analysis == "calibration":
            calibration.write_results(
                results,
                staging,
                profile=profile,
                base_seed=base_seed,
                elapsed_seconds=elapsed,
            )
        elif target_analysis == "behavior":
            behavior.write_results(
                results,
                staging,
                profile=profile,
                base_seed=base_seed,
                elapsed_seconds=elapsed,
            )
        else:
            performance.write_results(
                results["performance_raw"],
                results["performance_summary"],
                staging,
                profile=profile,
                base_seed=base_seed,
                elapsed_seconds=elapsed,
                require_unique_worker_pids=False,
            )
        _augment_final_receipt(
            staging,
            shards,
            shard_root=shard_root,
            merge_elapsed_seconds=merge_elapsed_seconds,
        )
        staging.rename(output_dir)
    return output_dir


def merge_shards_to_directory(
    target_analysis: TargetAnalysis,
    shard_root: Path,
    output_dir: Path,
    *,
    profile: Profile,
    base_seed: int = BASE_SEED,
) -> Path:
    """Merge, validate, and write one complete distributed analysis."""
    started = time.perf_counter()
    if target_analysis == "calibration":
        results, shards = merge_calibration_shards(
            shard_root,
            profile=profile,
            base_seed=base_seed,
        )
    elif target_analysis == "behavior":
        results, shards = merge_behavior_shards(
            shard_root,
            profile=profile,
            base_seed=base_seed,
        )
    else:
        results, shards = merge_performance_shards(
            shard_root,
            profile=profile,
            base_seed=base_seed,
        )
    return write_merged_results(
        target_analysis,
        results,
        shards,
        output_dir,
        shard_root=shard_root,
        profile=profile,
        base_seed=base_seed,
        merge_elapsed_seconds=time.perf_counter() - started,
    )


def _profile_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=("smoke", "quick", "full"),
        default="full",
        help="Replication workload.",
    )
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Base random seed.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibration_shard = subparsers.add_parser(
        "calibration-shard",
        help="Run one calibration component shard.",
    )
    _profile_argument(calibration_shard)
    calibration_shard.add_argument("--component", choices=CALIBRATION_COMPONENTS, required=True)
    calibration_shard.add_argument("--shard-index", type=int, required=True)
    calibration_shard.add_argument("--num-shards", type=int, required=True)
    calibration_shard.add_argument("--output-dir", type=Path, required=True)

    behavior_shard = subparsers.add_parser(
        "behavior-shard",
        help="Run one matched-behavior shard.",
    )
    _profile_argument(behavior_shard)
    behavior_shard.add_argument("--shard-index", type=int, required=True)
    behavior_shard.add_argument("--num-shards", type=int, required=True)
    behavior_shard.add_argument("--output-dir", type=Path, required=True)

    performance_shard = subparsers.add_parser(
        "performance-shard",
        help="Run one isolated performance-cell shard.",
    )
    _profile_argument(performance_shard)
    performance_shard.add_argument("--shard-index", type=int, required=True)
    performance_shard.add_argument("--num-shards", type=int, required=True)
    performance_shard.add_argument("--output-dir", type=Path, required=True)

    for name in ("calibration-merge", "behavior-merge", "performance-merge"):
        merge = subparsers.add_parser(name, help=f"Merge complete {name[:-6]} shards.")
        _profile_argument(merge)
        merge.add_argument("--shard-root", type=Path, required=True)
        merge.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profile = cast(Profile, args.profile)
    if args.command == "calibration-shard":
        spec = ShardSpec(
            target_analysis="calibration",
            component=args.component,
            profile=profile,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            base_seed=args.seed,
        )
        output = run_shard_to_directory(spec, args.output_dir)
    elif args.command == "behavior-shard":
        spec = ShardSpec(
            target_analysis="behavior",
            component="behavior",
            profile=profile,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            base_seed=args.seed,
        )
        output = run_shard_to_directory(spec, args.output_dir)
    elif args.command == "performance-shard":
        spec = ShardSpec(
            target_analysis="performance",
            component="performance",
            profile=profile,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            base_seed=args.seed,
        )
        output = run_shard_to_directory(spec, args.output_dir)
    else:
        target = args.command.removesuffix("-merge")
        output = merge_shards_to_directory(
            cast(TargetAnalysis, target),
            args.shard_root,
            args.output_dir,
            profile=profile,
            base_seed=args.seed,
        )
    print(f"Wrote verified {args.command} output to {output}.")


if __name__ == "__main__":
    main()
