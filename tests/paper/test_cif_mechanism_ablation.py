"""Focused tests for immutable CIF mechanism-study artifacts."""

from __future__ import annotations

import io
import re
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from paper.benchmark.experiments import cif_mechanism_ablation as mechanism

pytestmark = pytest.mark.paper

IMAGE_URI = (
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/citrees"
    "@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)


def _specification_sha256(*, source: str = "real") -> str:
    return mechanism.mechanism_specification_sha256(
        tasks=("classification", "regression"),
        source=source,
        datasets=(),
        seeds=(0, 1, 2, 3, 4),
        folds=(0, 1, 2, 3, 4),
        model_variants=("cif_default",),
        ranking_variants=("split_importance", "split_count"),
        n_jobs=-1,
        downstream_n_jobs=1,
    )


def test_distributed_prefix_binds_exact_image_and_canonical_specification() -> None:
    specification_sha256 = _specification_sha256()

    assert re.fullmatch(r"[0-9a-f]{64}", specification_sha256)
    assert mechanism.distributed_output_uri(
        bucket="citrees-123456789012",
        image_uri=IMAGE_URI,
        specification_sha256=specification_sha256,
    ) == (
        "s3://citrees-123456789012/experiments/cif_mechanism_ablation"
        f"/image-sha256/{'a' * 64}/spec-sha256/{specification_sha256}"
    )
    assert _specification_sha256(source="synthetic") != specification_sha256


def test_runner_has_no_mutable_output_or_overwrite_options() -> None:
    args = mechanism.parse_args([])

    assert args.distributed is False
    assert not hasattr(args, "output_uri")
    assert not hasattr(args, "force")

    with pytest.raises(SystemExit):
        mechanism.parse_args(["--output-uri", "s3://other/prefix"])
    with pytest.raises(SystemExit):
        mechanism.parse_args(["--force"])


def test_s3_write_is_conditional() -> None:
    client = MagicMock()
    frame = pd.DataFrame({"value": [1]})
    uri = "s3://citrees-123456789012/experiments/mechanism/rankings.parquet"

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(mechanism, "_s3_client", lambda: client)
        mechanism.save_frame(frame, uri)

    call = client.put_object.call_args
    assert call.kwargs["IfNoneMatch"] == "*"
    assert call.kwargs["Bucket"] == "citrees-123456789012"
    assert call.kwargs["Key"] == "experiments/mechanism/rankings.parquet"


def test_ambiguous_s3_response_accepts_only_exact_existing_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    stored: dict[str, bytes] = {}

    def response_lost(**kwargs: object) -> None:
        stored["payload"] = kwargs["Body"]  # type: ignore[assignment]
        raise TimeoutError("response lost")

    client.put_object.side_effect = response_lost
    client.get_object.side_effect = lambda **kwargs: {"Body": io.BytesIO(stored["payload"])}
    monkeypatch.setattr(mechanism, "_s3_client", lambda: client)

    mechanism.save_frame(
        pd.DataFrame({"value": [1]}),
        "s3://citrees-123456789012/experiments/mechanism/rankings.parquet",
    )


def test_s3_write_rejects_existing_different_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.put_object.side_effect = RuntimeError("precondition failed")
    client.get_object.return_value = {"Body": io.BytesIO(b"different")}
    monkeypatch.setattr(mechanism, "_s3_client", lambda: client)

    with pytest.raises(RuntimeError, match="Immutable mechanism artifact collision"):
        mechanism.save_frame(
            pd.DataFrame({"value": [1]}),
            "s3://citrees-123456789012/experiments/mechanism/rankings.parquet",
        )

    client.put_object.assert_called_once()


def test_local_write_behavior_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "rankings.parquet"
    frame = pd.DataFrame({"value": [1]})

    mechanism.save_frame(frame, str(path))
    pd.testing.assert_frame_equal(pd.read_parquet(path), frame)

    replacement = pd.DataFrame({"value": [2]})
    mechanism.save_frame(replacement, str(path))
    pd.testing.assert_frame_equal(pd.read_parquet(path), replacement)
