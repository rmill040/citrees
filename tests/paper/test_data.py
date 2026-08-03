"""Tests for immutable benchmark dataset identity and storage."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from botocore.exceptions import ClientError

from paper.benchmark.adapters import data
from paper.benchmark.infra import aws as aws_infra
from paper.benchmark.pipeline.types import DatasetIdentity

pytestmark = pytest.mark.paper


def _parquet_payload(*, offset: float = 0.0) -> bytes:
    frame = pd.DataFrame(
        {
            "x0": np.arange(12, dtype=float) + offset,
            "x1": np.arange(12, dtype=float) * 2.0,
            "y": np.array([0, 1] * 6),
        }
    )
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _missing_object() -> ClientError:
    return ClientError(
        {
            "Error": {"Code": "NoSuchKey", "Message": "missing"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        },
        "GetObject",
    )


def test_dataset_identity_binds_bytes_and_dimensions() -> None:
    payload = _parquet_payload()
    changed_payload = _parquet_payload(offset=1.0)
    identity = data.get_dataset_payload_identity(payload)

    assert identity.n_samples == 12
    assert identity.n_features == 2
    assert identity.sha256 != data.get_dataset_payload_identity(changed_payload).sha256
    data.validate_dataset_payload(payload, identity, location="memory")
    with pytest.raises(ValueError, match="identity mismatch"):
        data.validate_dataset_payload(changed_payload, identity, location="memory")


@pytest.mark.parametrize(
    ("sha256", "n_samples", "n_features"),
    [
        ("not-a-sha", 12, 2),
        ("d" * 64, True, 2),
        ("d" * 64, 12.0, 2),
        ("d" * 64, 12, False),
        ("d" * 64, 12, 2.0),
        ("d" * 64, 0, 2),
        ("d" * 64, 12, -1),
    ],
)
def test_dataset_identity_rejects_noncanonical_values(
    sha256: str,
    n_samples: object,
    n_features: object,
) -> None:
    with pytest.raises(ValueError):
        DatasetIdentity(sha256, n_samples, n_features)  # type: ignore[arg-type]


def test_local_dataset_is_rehashed_on_every_load(tmp_path: Path) -> None:
    path = tmp_path / "classification" / "real" / "clf_fixture.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(_parquet_payload())
    identity = data.get_dataset_identity(
        "fixture",
        "classification",
        source="real",
        data_dir=tmp_path,
    )

    X, y = data.load_dataset(
        "fixture",
        "classification",
        identity=identity,
        source="real",
        data_dir=tmp_path,
    )
    assert X.shape == (12, 2)
    assert y.shape == (12,)

    path.write_bytes(_parquet_payload(offset=1.0))
    with pytest.raises(ValueError, match="identity mismatch"):
        data.load_dataset(
            "fixture",
            "classification",
            identity=identity,
            source="real",
            data_dir=tmp_path,
        )


def test_local_load_validates_and_parses_the_same_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _parquet_payload()
    changed_payload = _parquet_payload(offset=100.0)
    path = tmp_path / "classification" / "real" / "clf_fixture.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    identity = data.get_dataset_payload_identity(payload)
    original_read_bytes = Path.read_bytes
    reads = 0

    def changing_read_bytes(observed_path: Path) -> bytes:
        nonlocal reads
        if observed_path != path:
            return original_read_bytes(observed_path)
        reads += 1
        return payload if reads == 1 else changed_payload

    monkeypatch.setattr(Path, "read_bytes", changing_read_bytes)

    X, _ = data.load_dataset(
        "fixture",
        "classification",
        identity=identity,
        source="real",
        data_dir=tmp_path,
    )

    assert reads == 1
    assert X[0, 0] == 0.0


def test_content_addressed_cache_validates_download_and_every_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _parquet_payload()
    identity = data.get_dataset_payload_identity(payload)
    client = type(
        "Client",
        (),
        {"get_object": lambda self, **kwargs: {"Body": io.BytesIO(payload)}},
    )()
    monkeypatch.setattr(data, "get_data_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(data, "_get_s3_bucket", lambda: "citrees-test")
    monkeypatch.setattr(data, "_get_s3_client", lambda **kwargs: client)

    cached = data.ensure_dataset_cached(
        "fixture",
        "classification",
        "real",
        identity,
    )

    assert cached == tmp_path / "sha256" / f"{identity.sha256}.parquet"
    assert cached.read_bytes() == payload
    cached.write_bytes(_parquet_payload(offset=1.0))
    with pytest.raises(ValueError, match="identity mismatch"):
        data.ensure_dataset_cached(
            "fixture",
            "classification",
            "real",
            identity,
        )


def test_invalid_download_is_rejected_before_cache_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = data.get_dataset_payload_identity(_parquet_payload())
    changed_payload = _parquet_payload(offset=1.0)
    client = type(
        "Client",
        (),
        {"get_object": lambda self, **kwargs: {"Body": io.BytesIO(changed_payload)}},
    )()
    monkeypatch.setattr(data, "get_data_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(data, "_get_s3_bucket", lambda: "citrees-test")
    monkeypatch.setattr(data, "_get_s3_client", lambda **kwargs: client)

    with pytest.raises(ValueError, match="identity mismatch"):
        data.ensure_dataset_cached(
            "fixture",
            "classification",
            "real",
            identity,
        )

    assert not (tmp_path / "sha256" / f"{identity.sha256}.parquet").exists()


class _DatasetS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.put_calls: list[dict[str, Any]] = []

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        del Bucket
        if Key not in self.objects:
            raise _missing_object()
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, **kwargs: Any) -> None:
        self.put_calls.append(kwargs)
        key = kwargs["Key"]
        if key in self.objects:
            raise ClientError(
                {
                    "Error": {"Code": "PreconditionFailed", "Message": "exists"},
                    "ResponseMetadata": {"HTTPStatusCode": 412},
                },
                "PutObject",
            )
        self.objects[key] = kwargs["Body"]


def test_dataset_upload_is_content_addressed_immutable_and_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_dir = tmp_path / "classification" / "real"
    real_dir.mkdir(parents=True)
    parquet = real_dir / "clf_fixture.parquet"
    parquet.write_bytes(_parquet_payload())
    identity = data.get_dataset_file_identity(parquet)
    client = _DatasetS3()

    monkeypatch.setattr(aws_infra, "ensure_s3_bucket", lambda: "citrees-test")
    monkeypatch.setattr(aws_infra.boto3, "client", lambda *args, **kwargs: client)
    monkeypatch.setattr(
        data,
        "get_data_dir",
        lambda task, source: tmp_path / task / source,
    )

    first = aws_infra.upload_datasets(task="classification")
    second = aws_infra.upload_datasets(task="classification")

    key = data.get_dataset_s3_key(
        "fixture",
        "classification",
        "real",
        identity,
    )
    assert first == {"uploaded": 1, "skipped": 0}
    assert second == {"uploaded": 0, "skipped": 1}
    assert set(client.objects) == {key}
    assert client.put_calls[0]["IfNoneMatch"] == "*"
    assert client.put_calls[0]["Metadata"]["sha256"] == identity.sha256

    client.objects[key] = _parquet_payload(offset=1.0)
    with pytest.raises(ValueError, match="identity mismatch"):
        aws_infra.upload_datasets(task="classification")


def test_dataset_upload_hashes_the_exact_published_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_payload = _parquet_payload()
    second_payload = _parquet_payload(offset=1.0)
    real_dir = tmp_path / "classification" / "real"
    real_dir.mkdir(parents=True)
    parquet = real_dir / "clf_fixture.parquet"
    parquet.write_bytes(first_payload)
    first_identity = data.get_dataset_payload_identity(first_payload)
    client = _DatasetS3()
    original_read_bytes = Path.read_bytes
    reads = 0

    def changing_read_bytes(observed_path: Path) -> bytes:
        nonlocal reads
        if observed_path != parquet:
            return original_read_bytes(observed_path)
        reads += 1
        return first_payload if reads == 1 else second_payload

    monkeypatch.setattr(Path, "read_bytes", changing_read_bytes)
    monkeypatch.setattr(aws_infra, "ensure_s3_bucket", lambda: "citrees-test")
    monkeypatch.setattr(aws_infra.boto3, "client", lambda *args, **kwargs: client)
    monkeypatch.setattr(
        data,
        "get_data_dir",
        lambda task, source: tmp_path / task / source,
    )

    result = aws_infra.upload_datasets(task="classification")

    expected_key = data.get_dataset_s3_key(
        "fixture",
        "classification",
        "real",
        first_identity,
    )
    assert result == {"uploaded": 1, "skipped": 0}
    assert reads == 1
    assert client.objects == {expected_key: first_payload}
