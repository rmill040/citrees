"""Tests for benchmark artifact provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from paper.benchmark.pipeline.types import DatasetIdentity, ExperimentConfig, MethodConfig
from paper.benchmark.utils.env import (
    get_benchmark_scope,
    get_container_image,
    get_hardware_metadata,
    get_library_versions,
)

pytestmark = pytest.mark.paper

OPENSSL_PACKAGE_VERSION = "3.0.13-0ubuntu3.12"


class _CaptureStore:
    def __init__(self) -> None:
        self.saved: pd.DataFrame | None = None

    def exists(self, stage: str, config: ExperimentConfig) -> bool:
        return False

    def save(self, stage: str, config: ExperimentConfig, frame: pd.DataFrame) -> str:
        self.saved = frame.copy()
        return "s3://bucket/isolated/rankings/result.parquet"


def test_library_versions_cover_the_python_benchmark_runtime() -> None:
    versions = get_library_versions()

    assert {
        "python",
        "citrees",
        "dcor",
        "numba",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "sklearn",
    }.issubset(versions)
    assert all(value for value in versions.values())


def test_docker_runtime_pins_and_verifies_the_openssl_cli_package() -> None:
    dockerfile = Path("paper/benchmark/infra/docker/Dockerfile").read_text(encoding="ascii")

    assert f"    openssl={OPENSSL_PACKAGE_VERSION} \\" in dockerfile
    assert f"    libssl-dev={OPENSSL_PACKAGE_VERSION} \\" in dockerfile
    assert (
        "test \"$(dpkg-query --show --showformat='${Version}' openssl)\" "
        f'= "{OPENSSL_PACKAGE_VERSION}"'
    ) in dockerfile


def test_hardware_and_image_metadata_use_explicit_worker_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EC2_INSTANCE_TYPE", "c6a.8xlarge")
    monkeypatch.setenv(
        "CITREES_IMAGE_URI",
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/citrees@sha256:" + "a" * 64,
    )
    monkeypatch.setattr("paper.benchmark.utils.env.os.cpu_count", lambda: 32)

    hardware = get_hardware_metadata()

    assert hardware["logical_cpus"] == 32
    assert hardware["ec2_instance_type"] == "c6a.8xlarge"
    assert hardware["platform"]
    assert hardware["machine"]
    assert get_container_image().endswith("sha256:" + "a" * 64)


def test_benchmark_scope_requires_complete_distributed_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CITREES_ARTIFACT_PREFIX", "repairs/run-001")
    monkeypatch.setenv("CITREES_CAMPAIGN_SHA256", "e" * 64)
    monkeypatch.setenv("CITREES_CANONICAL_MANIFEST_SHA256", "c" * 64)
    monkeypatch.setenv("CITREES_GATE_RECEIPT_SHA256", "d" * 64)
    monkeypatch.setenv("CITREES_MANIFEST_SHA256", "b" * 64)
    monkeypatch.setenv("CITREES_RUNTIME_CONTRACT_SHA256", "f" * 64)
    monkeypatch.setenv("AWS_ACCOUNT_ID", "999999999999")
    monkeypatch.setattr(
        "paper.benchmark.infra.aws.get_aws_account_id",
        lambda: "123456789012",
    )

    assert get_benchmark_scope() == {
        "artifact_prefix": "repairs/run-001",
        "campaign_sha256": "e" * 64,
        "canonical_manifest_sha256": "c" * 64,
        "gate_receipt_sha256": "d" * 64,
        "manifest_sha256": "b" * 64,
        "runtime_contract_sha256": "f" * 64,
        "aws_account_id": "123456789012",
    }


@pytest.mark.parametrize("value", [None, "", "   "])
def test_unknown_container_identity_is_explicit(
    value: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if value is None:
        monkeypatch.delenv("CITREES_IMAGE_URI", raising=False)
    else:
        monkeypatch.setenv("CITREES_IMAGE_URI", value)

    assert get_container_image() == "unknown"


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_ranking_artifact_records_complete_r_execution_provenance(
    task: Literal["classification", "regression"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from paper.benchmark.pipeline import r_methods, stage1

    rng = np.random.default_rng(1718)
    X = rng.standard_normal((50, 4))
    y = np.array([0, 1] * 25) if task == "classification" else rng.standard_normal(50)
    params = (("alpha", 0.05), ("testtype", "Bonferroni"))
    config = ExperimentConfig(
        method=MethodConfig("r_ctree", params=params),
        dataset="fixture",
        seed=7,
        task=task,
        dataset_identity=DatasetIdentity("d" * 64, n_samples=50, n_features=4),
    )
    store = _CaptureStore()

    def ranking(X_train: np.ndarray, y_train: np.ndarray, **kwargs: object) -> np.ndarray:
        return np.arange(X_train.shape[1])

    monkeypatch.setattr(r_methods, "r_ctree_ranking", ranking)
    monkeypatch.setattr(
        stage1,
        "load_dataset",
        lambda dataset, requested_task, *, identity: (X, y),
    )
    monkeypatch.setattr(
        stage1,
        "get_dataset_metadata",
        lambda dataset, requested_task, *, identity: {
            "dataset_source": "fixture",
            "dataset_type": "real",
            "dataset_family": "test",
            "n_informative": 1,
        },
    )
    monkeypatch.setattr(stage1, "get_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(
        stage1,
        "get_library_versions",
        lambda: {"python": "3.12.7", "rpy2": "3.6.7"},
    )
    monkeypatch.setattr(
        r_methods,
        "get_r_runtime_versions",
        lambda: {"r": "R version 4.5.2", "partykit": "1.2.24"},
    )
    monkeypatch.setattr(
        stage1,
        "get_hardware_metadata",
        lambda: {"logical_cpus": 32, "ec2_instance_type": "c6a.8xlarge"},
    )
    monkeypatch.setattr(
        stage1,
        "get_container_image",
        lambda: "repository@sha256:" + "a" * 64,
    )
    monkeypatch.setattr(
        stage1,
        "get_benchmark_scope",
        lambda: {
            "artifact_prefix": "repairs/run-001",
            "campaign_sha256": "e" * 64,
            "canonical_manifest_sha256": "c" * 64,
            "gate_receipt_sha256": "d" * 64,
            "manifest_sha256": "b" * 64,
            "runtime_contract_sha256": "f" * 64,
            "aws_account_id": "123456789012",
        },
    )

    result = stage1._run_selection(config, store)  # type: ignore[arg-type]

    assert result.is_success
    assert store.saved is not None
    saved = store.saved
    assert saved["fold_random_state"].tolist() == [7000, 7001, 7002, 7003, 7004]
    assert saved["git_sha"].unique().tolist() == ["a" * 40]
    assert saved["container_image"].unique().tolist() == ["repository@sha256:" + "a" * 64]
    assert saved["dataset_sha256"].unique().tolist() == ["d" * 64]
    assert saved["method_params_json"].unique().tolist() == [
        '{"alpha":0.05,"testtype":"Bonferroni"}'
    ]
    assert saved["artifact_prefix"].unique().tolist() == ["repairs/run-001"]
    assert saved["campaign_sha256"].unique().tolist() == ["e" * 64]
    assert saved["canonical_manifest_sha256"].unique().tolist() == ["c" * 64]
    assert saved["gate_receipt_sha256"].unique().tolist() == ["d" * 64]
    assert saved["manifest_sha256"].unique().tolist() == ["b" * 64]
    assert saved["runtime_contract_sha256"].unique().tolist() == ["f" * 64]
    assert saved["aws_account_id"].unique().tolist() == ["123456789012"]
    assert all(
        value
        == {
            "python": "3.12.7",
            "rpy2": "3.6.7",
            "r": "R version 4.5.2",
            "partykit": "1.2.24",
        }
        for value in saved["library_versions"]
    )
    assert all(
        value == {"logical_cpus": 32, "ec2_instance_type": "c6a.8xlarge"}
        for value in saved["hardware"]
    )

    class ExistingStore:
        def exists(self, stage: str, existing_config: ExperimentConfig) -> bool:
            return True

        def load(
            self,
            stage: str,
            existing_config: ExperimentConfig,
        ) -> pd.DataFrame:
            return saved.assign(manifest_sha256="c" * 64)

    rejected = stage1._run_selection(config, ExistingStore())  # type: ignore[arg-type]

    assert rejected.is_failure
    assert rejected.error_type == "ArtifactValidationError"
    assert "manifest_sha256" in str(rejected.error)
