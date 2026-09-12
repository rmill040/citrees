"""Tests for per-dataset checkpoints in the ablation experiments."""

from __future__ import annotations

from pathlib import Path

import pytest

from paper.benchmark.experiments import experiment_common as ec

pytestmark = pytest.mark.paper


@pytest.fixture
def data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(ec, "DATA_DIR", tmp_path)
    return tmp_path


def test_checkpoint_round_trip_and_safe_names(data_dir: Path) -> None:
    rows = [
        {"experiment": "x", "task": "clf", "dataset_type": "real/a b", "variant": "v", "p10": 0.5}
    ]
    path = ec.save_checkpoint("exp", "clf", "real/a b", rows)
    assert path.parent == data_dir / "exp.partials" and path.name == "clf__real_a_b.csv"
    assert ec.load_checkpoint("exp", "clf", "real/a b") == rows
    assert ec.load_checkpoint("exp", "clf", "other") is None


def test_run_dataset_checkpointed_runs_once_then_resumes(
    data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[int] = []

    def work(local: list[dict]) -> None:
        calls.append(1)
        local.extend([{"variant": "a", "value": 1.0}, {"variant": "b", "value": 2.0}])

    rows: list[dict] = []
    assert ec.run_dataset_checkpointed("exp", "reg", "ds", rows, work) is False
    assert len(rows) == 2 and calls == [1]
    resumed: list[dict] = [{"variant": "earlier", "value": 0.0}]
    assert ec.run_dataset_checkpointed("exp", "reg", "ds", resumed, work) is True
    assert calls == [1]
    assert [r["variant"] for r in resumed] == ["earlier", "a", "b"]
    assert "resumed 2 rows" in capsys.readouterr().out
