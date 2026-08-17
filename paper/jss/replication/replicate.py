"""Run the complete Journal of Statistical Software replication suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from paper.jss.replication import cloud

Profile = Literal["smoke", "quick", "full"]

BASE_SEED = 1718
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"
DEFAULT_DGRP_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "dgrp"
DISTRIBUTED_ANALYSES = frozenset({"calibration", "behavior", "performance"})


@dataclass(frozen=True)
class AnalysisSpec:
    """One independently validated analysis dispatched by the suite."""

    name: str
    module: str
    expected_analysis: str
    profiled: bool


@dataclass(frozen=True)
class AnalysisExecution:
    """Captured process result for one analysis command."""

    command: tuple[str, ...]
    returncode: int
    elapsed_seconds: float
    stdout: str
    stderr: str


ANALYSES = (
    AnalysisSpec(
        name="calibration",
        module="paper.jss.replication.calibration",
        expected_analysis="calibration",
        profiled=True,
    ),
    AnalysisSpec(
        name="behavior",
        module="paper.jss.replication.behavior",
        expected_analysis="behavior",
        profiled=True,
    ),
    AnalysisSpec(
        name="performance",
        module="paper.jss.replication.performance",
        expected_analysis="performance",
        profiled=True,
    ),
    AnalysisSpec(
        name="tutorial",
        module="paper.jss.replication.tutorial",
        expected_analysis="tutorial",
        profiled=True,
    ),
    AnalysisSpec(
        name="dgrp",
        module="paper.jss.replication.dgrp",
        expected_analysis="dgrp",
        profiled=True,
    ),
    AnalysisSpec(
        name="rdc_sensitivity",
        module="paper.jss.replication.rdc_sensitivity",
        expected_analysis="rdc_sensitivity",
        profiled=True,
    ),
)

AnalysisRunner = Callable[
    [AnalysisSpec, Profile, Path, int, Path, Path | None],
    AnalysisExecution,
]


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _analysis_command(
    spec: AnalysisSpec,
    profile: Profile,
    output_dir: Path,
    base_seed: int,
    dgrp_data_dir: Path,
    shard_root: Path | None,
) -> tuple[str, ...]:
    if shard_root is not None and spec.name in DISTRIBUTED_ANALYSES:
        return (
            sys.executable,
            "-m",
            "paper.jss.replication.shards",
            f"{spec.name}-merge",
            "--profile",
            profile,
            "--seed",
            str(base_seed),
            "--shard-root",
            str(shard_root / spec.name),
            "--output-dir",
            str(output_dir),
        )
    command = [
        sys.executable,
        "-m",
        spec.module,
        "--output-dir",
        str(output_dir),
    ]
    if spec.profiled:
        command.extend(("--profile", profile, "--seed", str(base_seed)))
    if spec.name == "dgrp":
        command.extend(("--data-dir", str(dgrp_data_dir)))
    return tuple(command)


def _run_analysis(
    spec: AnalysisSpec,
    profile: Profile,
    output_dir: Path,
    base_seed: int,
    dgrp_data_dir: Path,
    shard_root: Path | None,
) -> AnalysisExecution:
    command = _analysis_command(
        spec,
        profile,
        output_dir,
        base_seed,
        dgrp_data_dir,
        shard_root,
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return AnalysisExecution(
        command=command,
        returncode=completed.returncode,
        elapsed_seconds=time.perf_counter() - started,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _require_artifact_metadata(
    artifacts: object,
) -> Mapping[str, Mapping[str, object]]:
    if not isinstance(artifacts, dict):
        raise TypeError("child receipt artifacts must be a JSON object")
    for name, metadata in artifacts.items():
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise TypeError("child receipt artifact entries must be JSON objects")
    return artifacts


def _require_source_hashes(
    spec: AnalysisSpec,
    receipt: Mapping[str, object],
) -> Mapping[str, str]:
    source_hashes = receipt.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise TypeError(f"{spec.name} receipt source_sha256 must be a nonempty JSON object")
    required_sources = {
        f"{spec.module.replace('.', '/')}.py",
        "pyproject.toml",
        "uv.lock",
    }
    missing = sorted(required_sources.difference(source_hashes))
    if missing:
        raise ValueError(f"{spec.name} receipt omits required sources: {missing}")
    for name, expected_sha256 in source_hashes.items():
        if not isinstance(name, str) or not isinstance(expected_sha256, str):
            raise TypeError(f"{spec.name} receipt source hashes must map strings to strings")
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError(f"{spec.name} receipt has a malformed source hash for {name}")
        path = (REPO_ROOT / name).resolve()
        if not path.is_relative_to(REPO_ROOT) or not path.is_file():
            raise ValueError(f"{spec.name} receipt source path is invalid: {name!r}")
        if _sha256(path) != expected_sha256:
            raise ValueError(f"{spec.name} receipt source hash differs for {name}")
    return source_hashes


def _verify_child_output(
    spec: AnalysisSpec,
    output_dir: Path,
    *,
    profile: Profile,
    expected_git_sha: str,
    require_clean: bool,
) -> dict[str, object]:
    receipt_path = output_dir / "receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"{spec.name} wrote no receipt")
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    if not isinstance(receipt, dict):
        raise TypeError(f"{spec.name} receipt must be a JSON object")
    if receipt.get("analysis") != spec.expected_analysis:
        raise ValueError(
            f"{spec.name} receipt analysis differs: "
            f"expected={spec.expected_analysis!r}, observed={receipt.get('analysis')!r}"
        )
    if spec.profiled and receipt.get("profile") != profile:
        raise ValueError(
            f"{spec.name} receipt profile differs: "
            f"expected={profile!r}, observed={receipt.get('profile')!r}"
        )
    if receipt.get("git_sha") != expected_git_sha:
        raise ValueError(
            f"{spec.name} receipt git SHA differs: "
            f"expected={expected_git_sha!r}, observed={receipt.get('git_sha')!r}"
        )
    git_dirty = receipt.get("git_dirty")
    if not isinstance(git_dirty, bool):
        raise TypeError(f"{spec.name} receipt git_dirty must be boolean")
    if require_clean and git_dirty:
        raise ValueError(f"{spec.name} receipt records a dirty full-profile source")
    for field in ("python", "platform"):
        if not isinstance(receipt.get(field), str) or not receipt[field]:
            raise TypeError(f"{spec.name} receipt {field} must be a nonempty string")
    versions = receipt.get("versions")
    if not isinstance(versions, dict) or not versions:
        raise TypeError(f"{spec.name} receipt versions must be a nonempty JSON object")
    if not all(
        isinstance(package, str) and package and isinstance(version, str) and version
        for package, version in versions.items()
    ):
        raise TypeError(f"{spec.name} receipt versions must map nonempty strings")
    _require_source_hashes(spec, receipt)

    artifacts = _require_artifact_metadata(receipt.get("artifacts"))
    expected_files = {"receipt.json", *artifacts}
    observed_files = {
        str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()
    }
    if observed_files != expected_files:
        missing = sorted(expected_files.difference(observed_files))
        extra = sorted(observed_files.difference(expected_files))
        raise ValueError(f"{spec.name} output inventory differs: missing={missing}, extra={extra}")
    for name, metadata in artifacts.items():
        path = output_dir / name
        if path.parent != output_dir:
            raise ValueError(f"{spec.name} artifact path must be flat: {name!r}")
        observed_bytes = path.stat().st_size
        observed_sha256 = _sha256(path)
        if metadata.get("bytes") != observed_bytes:
            raise ValueError(f"{spec.name} artifact byte count differs for {name}")
        if metadata.get("sha256") != observed_sha256:
            raise ValueError(f"{spec.name} artifact hash differs for {name}")
    return receipt


def _transcript_section(
    index: int,
    total: int,
    spec: AnalysisSpec,
    execution: AnalysisExecution,
) -> str:
    command = shlex.join(execution.command)
    stdout = execution.stdout.rstrip()
    stderr = execution.stderr.rstrip()
    return "\n".join(
        (
            f"[{index}/{total}] {spec.name}",
            f"command: {command}",
            f"returncode: {execution.returncode}",
            f"elapsed_seconds: {execution.elapsed_seconds:.6f}",
            "stdout:",
            stdout if stdout else "(empty)",
            "stderr:",
            stderr if stderr else "(empty)",
            "",
        )
    )


def _output_hashes(root: Path) -> dict[str, dict[str, object]]:
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "receipt.json"
    }


def run_replication(
    profile: Profile,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    base_seed: int = BASE_SEED,
    dgrp_data_dir: Path = DEFAULT_DGRP_DATA_DIR,
    shard_root: Path | None = None,
    runner: AnalysisRunner = _run_analysis,
) -> Path:
    """Run all analyses and atomically publish one verified result tree."""
    if profile not in ("smoke", "quick", "full"):
        raise ValueError(f"unknown replication profile: {profile}")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"replication output already exists: {output_dir}")
    if shard_root is not None:
        shard_root = shard_root.resolve()
        if not shard_root.is_dir():
            raise NotADirectoryError(f"distributed shard root is not a directory: {shard_root}")
        missing_shard_roots = sorted(
            analysis for analysis in DISTRIBUTED_ANALYSES if not (shard_root / analysis).is_dir()
        )
        if missing_shard_roots:
            raise ValueError(f"distributed shard root omits analyses: {missing_shard_roots}")
    cloud_accounting: dict[str, object] | None = None
    if shard_root is not None:
        accounting_path = shard_root / cloud.COMPUTE_ACCOUNTING_FILENAME
        provenance_path = shard_root / cloud.CLOUD_PROVENANCE_DIRECTORY
        if accounting_path.exists() != provenance_path.exists():
            raise ValueError("distributed shard root has incomplete cloud provenance")
        if accounting_path.exists():
            cloud_accounting = cloud.validate_compute_accounting(shard_root)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    scratch_dir = REPO_ROOT / "scratch"
    scratch_dir.mkdir(exist_ok=True)
    if scratch_dir.stat().st_dev != output_dir.parent.stat().st_dev:
        raise RuntimeError("replication output must share a filesystem with the repository")

    starting_git_sha = _git_sha()
    if cloud_accounting is not None:
        if cloud_accounting.get("profile") != profile:
            raise ValueError("cloud accounting profile differs from replication")
        if cloud_accounting.get("base_seed") != base_seed:
            raise ValueError("cloud accounting seed differs from replication")
        if cloud_accounting.get("git_sha") != starting_git_sha:
            raise ValueError("cloud accounting source revision differs from replication")
    if profile == "full" and _git_dirty():
        raise RuntimeError("The full replication profile requires a clean source tree")

    suite_started = time.perf_counter()
    transcript_sections = [
        "citrees Journal of Statistical Software replication",
        f"profile: {profile}",
        f"base_seed: {base_seed}",
        f"git_sha: {starting_git_sha}",
        f"platform: {platform.platform()}",
        f"execution_mode: {'distributed' if shard_root is not None else 'serial'}",
        "",
    ]
    child_receipts: dict[str, dict[str, object]] = {}
    child_executions: dict[str, dict[str, object]] = {}

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-",
        dir=scratch_dir,
    ) as temporary:
        staging = Path(temporary) / output_dir.name
        staging.mkdir()
        for index, spec in enumerate(ANALYSES, start=1):
            child_dir = staging / spec.name
            execution = runner(
                spec,
                profile,
                child_dir,
                base_seed,
                dgrp_data_dir,
                shard_root,
            )
            transcript_sections.append(_transcript_section(index, len(ANALYSES), spec, execution))
            if execution.returncode != 0:
                raise RuntimeError(
                    f"{spec.name} failed with exit code {execution.returncode}\n"
                    f"stdout:\n{execution.stdout}\nstderr:\n{execution.stderr}"
                )
            child_receipts[spec.name] = _verify_child_output(
                spec,
                child_dir,
                profile=profile,
                expected_git_sha=starting_git_sha,
                require_clean=profile == "full",
            )
            child_executions[spec.name] = {
                "command": list(execution.command),
                "elapsed_seconds": execution.elapsed_seconds,
            }
            if profile == "full" and (_git_sha() != starting_git_sha or _git_dirty()):
                raise RuntimeError("Source tree changed during the full replication run")

        transcript_path = staging / "transcript.txt"
        transcript_path.write_text(
            "\n".join(transcript_sections).rstrip() + "\n",
            encoding="utf-8",
        )
        source_files = [
            Path(__file__).resolve(),
            Path(__file__).resolve().with_name("__main__.py"),
            REPO_ROOT / "pyproject.toml",
            REPO_ROOT / "uv.lock",
        ]
        if shard_root is not None:
            source_files.append(Path(__file__).resolve().with_name("shards.py"))
        cloud_accounting_metadata: dict[str, object] | None = None
        if cloud_accounting is not None and shard_root is not None:
            if cloud.validate_compute_accounting(shard_root) != cloud_accounting:
                raise RuntimeError("cloud accounting changed during distributed replication")
            accounting_source = shard_root / cloud.COMPUTE_ACCOUNTING_FILENAME
            accounting_artifact = staging / cloud.COMPUTE_ACCOUNTING_FILENAME
            shutil.copyfile(accounting_source, accounting_artifact)
            shutil.copytree(
                shard_root / cloud.CLOUD_PROVENANCE_DIRECTORY,
                staging / cloud.CLOUD_PROVENANCE_DIRECTORY,
            )
            cloud_accounting_metadata = {
                "analysis": cloud_accounting["analysis"],
                "campaign_sha256": cloud_accounting["campaign_sha256"],
                "artifact": cloud.COMPUTE_ACCOUNTING_FILENAME,
                "bytes": accounting_artifact.stat().st_size,
                "sha256": _sha256(accounting_artifact),
            }
            source_files.append(Path(cloud.__file__).resolve())
        receipt = {
            "analysis": "jss_replication",
            "profile": profile,
            "base_seed": base_seed,
            "execution_mode": "distributed" if shard_root is not None else "serial",
            "distributed_analyses": (
                sorted(DISTRIBUTED_ANALYSES) if shard_root is not None else []
            ),
            "analyses": [asdict(spec) for spec in ANALYSES],
            "created_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - suite_started,
            "git_sha": starting_git_sha,
            "git_dirty": _git_dirty(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "source_sha256": {
                str(path.relative_to(REPO_ROOT)): _sha256(path) for path in source_files
            },
            "executions": child_executions,
            "child_receipts": {
                name: {
                    "analysis": child_receipt["analysis"],
                    "sha256": _sha256(staging / name / "receipt.json"),
                }
                for name, child_receipt in child_receipts.items()
            },
            "cloud_compute_accounting": cloud_accounting_metadata,
            "artifacts": _output_hashes(staging),
        }
        (staging / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        staging.rename(output_dir)
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("smoke", "quick", "full"),
        default="quick",
        help="Replication workload.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="New directory that will receive the complete result tree.",
    )
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Base random seed.")
    parser.add_argument(
        "--dgrp-data-dir",
        type=Path,
        default=DEFAULT_DGRP_DATA_DIR,
        help="Directory containing or receiving the pinned DGRP sources.",
    )
    parser.add_argument(
        "--shard-root",
        type=Path,
        help=(
            "Root containing complete calibration/, behavior/, and performance/ shard trees. "
            "When supplied, those analyses are merged through the canonical suite."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = run_replication(
        args.profile,
        args.output_dir,
        base_seed=args.seed,
        dgrp_data_dir=args.dgrp_data_dir,
        shard_root=args.shard_root,
    )
    print(f"Wrote verified JSS replication outputs to {output_dir}.")


if __name__ == "__main__":
    main()
