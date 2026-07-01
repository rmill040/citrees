"""Tests for API queue completion logic."""

from __future__ import annotations

import pandas as pd
import pytest

from paper.scripts.api.server import _needs_metrics_work
from paper.scripts.pipeline.types import ExperimentConfig, MethodConfig

pytestmark = pytest.mark.paper


class _FakeStore:
    def __init__(self, rankings_df: pd.DataFrame, metrics_df: pd.DataFrame):
        self._rankings_df = rankings_df
        self._metrics_df = metrics_df

    def load(self, stage: str, config: ExperimentConfig) -> pd.DataFrame:
        del config
        if stage == "rankings":
            return self._rankings_df
        if stage == "metrics":
            return self._metrics_df
        raise AssertionError(f"unexpected stage: {stage}")


def test_needs_metrics_work_when_no_metrics_artifact_exists():
    cfg = ExperimentConfig(
        method=MethodConfig("cif"),
        dataset="arcene",
        seed=0,
        task="classification",
    )
    store = _FakeStore(
        rankings_df=pd.DataFrame({"feature_ranking": [list(range(1000))]}),
        metrics_df=pd.DataFrame({"k": []}),
    )
    assert _needs_metrics_work(store, cfg, completed_metrics=set())


def test_needs_metrics_work_when_existing_metrics_miss_high_p_bridge():
    cfg = ExperimentConfig(
        method=MethodConfig("cif"),
        dataset="arcene",
        seed=0,
        task="classification",
    )
    store = _FakeStore(
        rankings_df=pd.DataFrame({"feature_ranking": [list(range(1200))]}),
        metrics_df=pd.DataFrame({"k": [5, 10, 25, 50, 100, 1200]}),
    )
    assert _needs_metrics_work(store, cfg, completed_metrics={cfg.key})


def test_existing_metrics_covering_high_p_bridge_are_complete():
    cfg = ExperimentConfig(
        method=MethodConfig("cif"),
        dataset="arcene",
        seed=0,
        task="classification",
    )
    store = _FakeStore(
        rankings_df=pd.DataFrame({"feature_ranking": [list(range(1200))]}),
        metrics_df=pd.DataFrame(
            {"k": [5, 10, 25, 50, 100, 150, 200, 300, 500, 600, 750, 900, 1000, 1200]}
        ),
    )
    assert not _needs_metrics_work(store, cfg, completed_metrics={cfg.key})
