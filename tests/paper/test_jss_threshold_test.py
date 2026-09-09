"""Tests for the Bonferroni-versus-max-type threshold test analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from paper.jss.replication import threshold_test as tt

pytestmark = pytest.mark.paper


class TestDesign:
    def test_module_path_points_at_this_module(self) -> None:
        assert (tt.REPO_ROOT / tt.MODULE_PATH).samefile(tt.__file__)

    def test_only_linear_selectors_are_used(self) -> None:
        clf = tt._estimator("classification", "maxt", None, 16, 0)
        reg = tt._estimator("regression", "bonferroni", "adaptive", 16, 0)
        assert clf.selector == "mc" and reg.selector == "pc"

    def test_profiles_share_one_schema(self) -> None:
        keys = {frozenset(p) for p in tt.PROFILES.values()}
        assert len(keys) == 1

    def test_real_datasets_exist_and_are_mid_size(self) -> None:
        for task, names in tt.REAL_DATASETS.items():
            for name in names:
                X, y = tt._load_real(task, name)
                assert 200 <= X.shape[0] <= 6000 and X.shape[1] <= 110, (name, X.shape)
                assert len(y) == X.shape[0]


class TestHelpers:
    def test_depth_counts_edges(self) -> None:
        leaf: dict = {"value": 1.0}
        assert tt._depth(leaf) == 0
        assert (
            tt._depth(
                {"left_child": leaf, "right_child": {"left_child": leaf, "right_child": leaf}}
            )
            == 2
        )

    def test_split_made_requires_children(self) -> None:
        assert not tt._split_made({})
        assert tt._split_made({"left_child": {"x": 1}})

    def test_wilson_interval_is_clipped(self) -> None:
        low, high = tt._wilson(0.0, 100)
        assert low == 0.0 and high == pytest.approx(0.0)


class TestSmokeRun:
    @pytest.fixture(scope="class")
    def frames(self) -> dict[str, pd.DataFrame]:
        profile = {
            "null_reps": 4,
            "null_n": (60,),
            "null_k": (8,),
            "scale_n": (120,),
            "scale_k": (8,),
            "scale_reps": 1,
            "real_folds": 2,
            "real_k": 8,
        }
        real, agreement = tt.real_subset({**profile, "real_folds": 2}, 1)
        return {
            "null": tt.null_calibration(profile, 1),
            "scale": tt.synthetic_scaling(profile, 1),
            "real": real,
            "agreement": agreement,
        }

    def test_null_grid_covers_every_cell(self, frames) -> None:
        f = frames["null"]
        assert len(f) == 2 * 1 * 1 * 2 * 2
        assert set(f["threshold_test"]) == set(tt.TESTS)
        assert ((f["false_split_rate"] >= 0) & (f["false_split_rate"] <= 1)).all()

    def test_scaling_reports_positive_times(self, frames) -> None:
        assert (frames["scale"]["seconds_per_fit"] > 0).all()

    def test_real_subset_has_both_tests_per_dataset(self, frames) -> None:
        counts = frames["real"].groupby(["dataset", "stopping"])["threshold_test"].nunique()
        assert (counts == 2).all()

    def test_agreement_is_a_correlation(self, frames) -> None:
        a = frames["agreement"]["importance_spearman_mean"]
        assert ((a >= -1) & (a <= 1)).all()


class TestReceipt:
    def test_receipt_satisfies_the_driver_contract(self, tmp_path: Path) -> None:
        artifact = tmp_path / "null_calibration.csv"
        artifact.write_text("a\n1\n", encoding="utf-8")
        tt.write_receipt(tmp_path, "smoke", {"null_calibration.csv": artifact})
        receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="ascii"))
        assert receipt["analysis"] == "threshold_test" and receipt["profile"] == "smoke"
        assert {tt.MODULE_PATH, "pyproject.toml", "uv.lock"} <= set(receipt["source_sha256"])
        assert receipt["artifacts"]["null_calibration.csv"]["bytes"] == artifact.stat().st_size
        assert isinstance(receipt["git_dirty"], bool) and len(receipt["git_sha"]) == 40
