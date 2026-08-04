"""Tests for the package-level JSS replication entry point."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from paper.jss.replication import replicate
from paper.jss.replication.replicate import (
    ANALYSES,
    AnalysisExecution,
    AnalysisSpec,
    _analysis_command,
    _verify_child_output,
    run_replication,
)

pytestmark = pytest.mark.paper


def _fake_runner(
    spec: AnalysisSpec,
    profile: replicate.Profile,
    output_dir: Path,
    base_seed: int,
    dgrp_data_dir: Path,
) -> AnalysisExecution:
    assert base_seed == 7
    assert dgrp_data_dir.name == "dgrp-data"
    output_dir.mkdir(parents=True)
    artifact = output_dir / f"{spec.name}.txt"
    artifact.write_text(f"{spec.name}\n", encoding="ascii")
    source_paths = (
        replicate.REPO_ROOT / f"{spec.module.replace('.', '/')}.py",
        replicate.REPO_ROOT / "pyproject.toml",
        replicate.REPO_ROOT / "uv.lock",
    )
    receipt: dict[str, object] = {
        "analysis": spec.expected_analysis,
        "schema_version": 1,
        "git_sha": replicate._git_sha(),
        "git_dirty": replicate._git_dirty(),
        "python": "3.12.0",
        "platform": "test-platform",
        "source_sha256": {
            str(path.relative_to(replicate.REPO_ROOT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in source_paths
        },
        "versions": {"fixture": "1.0"},
        "artifacts": {
            artifact.name: {
                "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        },
    }
    if spec.profiled:
        receipt["profile"] = profile
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return AnalysisExecution(
        command=("fake-analysis", spec.name, profile),
        returncode=0,
        elapsed_seconds=1.0,
        stdout=f"completed {spec.name}\n",
        stderr="",
    )


def test_analysis_inventory_covers_every_implemented_replication_module() -> None:
    assert [spec.name for spec in ANALYSES] == [
        "calibration",
        "behavior",
        "performance",
        "tutorial",
        "dgrp",
    ]
    assert [spec.expected_analysis for spec in ANALYSES] == [
        "calibration",
        "behavior",
        "performance",
        "tutorial",
        "dgrp_phenotypes",
    ]
    assert all(spec.profiled for spec in ANALYSES[:-1])
    assert not ANALYSES[-1].profiled


def test_child_commands_forward_profile_seed_output_and_dgrp_data(tmp_path: Path) -> None:
    profiled = _analysis_command(
        ANALYSES[0],
        "quick",
        tmp_path / "calibration",
        7,
        tmp_path / "dgrp-data",
    )
    assert profiled[:3] == (
        replicate.sys.executable,
        "-m",
        "paper.jss.replication.calibration",
    )
    assert profiled[-4:] == ("--profile", "quick", "--seed", "7")

    dgrp = _analysis_command(
        ANALYSES[-1],
        "quick",
        tmp_path / "dgrp",
        7,
        tmp_path / "dgrp-data",
    )
    assert "--profile" not in dgrp
    assert "--seed" not in dgrp
    assert dgrp[-2:] == ("--data-dir", str(tmp_path / "dgrp-data"))


def test_replication_atomically_publishes_receipts_hashes_and_transcript(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "replication-output"
    dgrp_data_dir = tmp_path / "dgrp-data"
    dgrp_data_dir.mkdir()
    observed = run_replication(
        "smoke",
        output_dir,
        base_seed=7,
        dgrp_data_dir=dgrp_data_dir,
        runner=_fake_runner,
    )

    assert observed == output_dir.resolve()
    assert output_dir.is_dir()
    receipt = json.loads((output_dir / "receipt.json").read_text(encoding="ascii"))
    assert receipt["analysis"] == "jss_replication"
    assert receipt["schema_version"] == 1
    assert receipt["profile"] == "smoke"
    assert receipt["base_seed"] == 7
    assert set(receipt["child_receipts"]) == {spec.name for spec in ANALYSES}
    assert set(receipt["executions"]) == {spec.name for spec in ANALYSES}
    assert "paper/jss/replication/replicate.py" in receipt["source_sha256"]
    assert "paper/jss/replication/__main__.py" in receipt["source_sha256"]

    transcript = (output_dir / "transcript.txt").read_text(encoding="utf-8")
    assert "profile: smoke" in transcript
    for index, spec in enumerate(ANALYSES, start=1):
        assert f"[{index}/{len(ANALYSES)}] {spec.name}" in transcript
        child_receipt = output_dir / spec.name / "receipt.json"
        assert (
            receipt["child_receipts"][spec.name]["sha256"]
            == hashlib.sha256(child_receipt.read_bytes()).hexdigest()
        )
    for path, metadata in receipt["artifacts"].items():
        artifact = output_dir / path
        assert artifact.is_file()
        assert metadata["bytes"] == artifact.stat().st_size
        assert metadata["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()


def test_replication_rejects_existing_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_replication(
            "smoke",
            output_dir,
            base_seed=7,
            dgrp_data_dir=tmp_path / "dgrp-data",
            runner=_fake_runner,
        )


def test_failed_child_is_not_published(tmp_path: Path) -> None:
    output_dir = tmp_path / "failed-output"
    dgrp_data_dir = tmp_path / "dgrp-data"
    dgrp_data_dir.mkdir()

    def fail_runner(
        spec: AnalysisSpec,
        profile: replicate.Profile,
        child_dir: Path,
        base_seed: int,
        observed_data_dir: Path,
    ) -> AnalysisExecution:
        if spec.name == "behavior":
            return AnalysisExecution(
                command=("fake-analysis", spec.name),
                returncode=3,
                elapsed_seconds=0.1,
                stdout="",
                stderr="intentional failure",
            )
        return _fake_runner(
            spec,
            profile,
            child_dir,
            base_seed,
            observed_data_dir,
        )

    with pytest.raises(RuntimeError, match="behavior failed"):
        run_replication(
            "smoke",
            output_dir,
            base_seed=7,
            dgrp_data_dir=dgrp_data_dir,
            runner=fail_runner,
        )
    assert not output_dir.exists()


def test_child_receipt_validation_rejects_artifact_corruption(tmp_path: Path) -> None:
    spec = ANALYSES[0]
    output_dir = tmp_path / spec.name
    data_dir = tmp_path / "dgrp-data"
    data_dir.mkdir()
    _fake_runner(spec, "smoke", output_dir, 7, data_dir)
    artifact = output_dir / f"{spec.name}.txt"
    artifact.write_text("corrupted\n", encoding="ascii")

    with pytest.raises(ValueError, match="byte count differs"):
        _verify_child_output(
            spec,
            output_dir,
            profile="smoke",
            expected_git_sha=replicate._git_sha(),
            require_clean=False,
        )


def test_child_receipt_validation_rejects_source_or_revision_mismatch(tmp_path: Path) -> None:
    spec = ANALYSES[0]
    data_dir = tmp_path / "dgrp-data"
    data_dir.mkdir()

    wrong_source_dir = tmp_path / "wrong-source"
    _fake_runner(spec, "smoke", wrong_source_dir, 7, data_dir)
    wrong_source_receipt = json.loads(
        (wrong_source_dir / "receipt.json").read_text(encoding="ascii")
    )
    module_path = f"{spec.module.replace('.', '/')}.py"
    wrong_source_receipt["source_sha256"][module_path] = "0" * 64
    (wrong_source_dir / "receipt.json").write_text(
        json.dumps(wrong_source_receipt, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="source hash differs"):
        _verify_child_output(
            spec,
            wrong_source_dir,
            profile="smoke",
            expected_git_sha=replicate._git_sha(),
            require_clean=False,
        )

    wrong_revision_dir = tmp_path / "wrong-revision"
    _fake_runner(spec, "smoke", wrong_revision_dir, 7, data_dir)
    wrong_revision_receipt = json.loads(
        (wrong_revision_dir / "receipt.json").read_text(encoding="ascii")
    )
    wrong_revision_receipt["git_sha"] = "0" * 40
    (wrong_revision_dir / "receipt.json").write_text(
        json.dumps(wrong_revision_receipt, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="git SHA differs"):
        _verify_child_output(
            spec,
            wrong_revision_dir,
            profile="smoke",
            expected_git_sha=replicate._git_sha(),
            require_clean=False,
        )


def test_unknown_profile_is_rejected_before_execution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown replication profile"):
        run_replication(
            "invalid",  # type: ignore[arg-type]
            tmp_path / "output",
            runner=_fake_runner,
        )
