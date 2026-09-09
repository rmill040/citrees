"""Tests for the NHANES diabetes application analysis.

The design lives in ``nhanes_diabetes-specification.json``; these tests check
that the code honors it: the ranking uses midranks for ties, the corrected t
test matches Bouckaert and Frank (2004) by hand, Holm covers the secondary
family only, the shuffled controls are permutations of their sources, and the
receipt satisfies the contract the replication driver enforces.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper.jss.replication import nhanes_diabetes as nhanes

pytestmark = pytest.mark.paper


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(nhanes.SPECIFICATION.read_text(encoding="utf-8"))


class TestSpecification:
    def test_module_path_constant_points_at_this_module(self) -> None:
        assert (nhanes.REPO_ROOT / nhanes.MODULE_PATH).samefile(nhanes.__file__)

    def test_outcome_defining_files_are_not_predictors(self, spec: dict) -> None:
        files = {file for file, _ in spec["predictors"].values()}
        assert not files & {"GHB_L", "GLU_L", "DIQ_L"}

    def test_every_pinned_source_is_a_sha256(self, spec: dict) -> None:
        for name, digest in spec["dataset"]["source_sha256"].items():
            assert name.endswith(".xpt") and len(digest) == 64 and digest == digest.lower()

    def test_every_predictor_file_is_pinned(self, spec: dict) -> None:
        pinned = {n.removesuffix(".xpt") for n in spec["dataset"]["source_sha256"]}
        assert {file for file, _ in spec["predictors"].values()} <= pinned
        assert {"GHB_L", "DIQ_L"} <= pinned

    def test_missing_codes_only_name_predictors(self, spec: dict) -> None:
        assert set(spec["missing_codes"]) <= set(spec["predictors"])

    def test_shuffled_controls_span_three_cardinality_classes(self, spec: dict) -> None:
        controls = {k: v for k, v in spec["shuffled_controls"].items() if k != "construction"}
        assert set(controls) == {"shuffled_mec_weight", "shuffled_age", "shuffled_sex"}
        assert {v["source"] for v in controls.values()} <= set(spec["predictors"])

    def test_clinical_block_members_are_predictors(self, spec: dict) -> None:
        block = spec["secondary_inference"]["hypotheses"]["low_cardinality_clinical_block"]["block"]
        assert set(block) <= set(spec["predictors"])
        assert not any(b.startswith("shuffled") for b in block)

    def test_only_the_full_profile_claims_manuscript_inference(self, spec: dict) -> None:
        status = {k: v["inference_status"] for k, v in spec["profiles"].items()}
        assert status["full"] == "manuscript_inference"
        assert "manuscript_inference" not in (status["quick"], status["smoke"])


class TestRank:
    def test_largest_importance_is_rank_one(self) -> None:
        assert list(nhanes._rank(np.array([0.1, 0.5, 0.3]))) == [3.0, 1.0, 2.0]

    def test_ties_receive_their_midrank(self) -> None:
        # Four zero-importance features share ranks 2..5, midrank 3.5.
        ranks = nhanes._rank(np.array([0.0, 0.9, 0.0, 0.0, 0.0]))
        assert ranks[1] == 1.0
        assert all(r == 3.5 for i, r in enumerate(ranks) if i != 1)

    def test_ranks_are_a_valid_ranking(self) -> None:
        importance = np.random.default_rng(0).random(30)
        assert sorted(nhanes._rank(importance)) == list(map(float, range(1, 31)))


class TestCorrectedT:
    def test_matches_hand_computation(self) -> None:
        d = np.array([-3.0, -5.0, -2.0, -4.0, -6.0])
        out = nhanes.corrected_t(d, np.full(5, 80.0), np.full(5, 20.0), 5, 1, "less")
        expected = d.mean() / np.sqrt((1 / 5 + 0.25) * d.var(ddof=1))
        assert out["statistic"] == pytest.approx(expected)
        assert out["degrees_of_freedom"] == 4

    def test_alternatives_are_one_sided_and_complementary(self) -> None:
        d = np.array([-3.0, -5.0, -2.0, -4.0, -6.0])
        args = (np.full(5, 80.0), np.full(5, 20.0), 5, 1)
        less = nhanes.corrected_t(d, *args, "less")["p_value"]
        greater = nhanes.corrected_t(d, *args, "greater")["p_value"]
        assert less < 0.05 < greater
        assert less + greater == pytest.approx(1.0)

    def test_zero_variance_is_safe(self) -> None:
        out = nhanes.corrected_t(np.zeros(4), np.full(4, 80.0), np.full(4, 20.0), 4, 1, "less")
        assert out["statistic"] == 0.0 and out["p_value"] == pytest.approx(0.5)


def _ranks_frame(
    spec: dict, rf_noise_rank: float, cif_noise_rank: float, folds: int = 6
) -> pd.DataFrame:
    features = list(spec["predictors"]) + ["shuffled_mec_weight", "shuffled_age", "shuffled_sex"]
    rows = []
    rng = np.random.default_rng(1)
    for fold in range(folds):
        for method in nhanes.METHODS:
            imp = rng.random(len(features))
            ranks = nhanes._rank(imp)
            for j, f in enumerate(features):
                r = ranks[j]
                if f == "shuffled_mec_weight":
                    # small fold jitter so the paired differences have variance
                    r = rf_noise_rank + 0.5 * (fold % 3) if method == "rf" else cif_noise_rank
                rows.append(
                    {
                        "fold": fold,
                        "repeat": fold // 3,
                        "method": method,
                        "feature": f,
                        "rank": r,
                        "importance": imp[j],
                        "n_unique_train": 2 if f.endswith("sex") else 50 + j,
                        "n_train": 80,
                        "n_test": 20,
                        "fit_seconds": 0.1,
                    }
                )
    return pd.DataFrame(rows)


class TestInference:
    profile = {"folds": 3, "repeats": 2, "inference_status": "exploratory"}

    def test_primary_detects_rf_ranking_noise_higher(self, spec: dict) -> None:
        frame = nhanes.inference(
            _ranks_frame(spec, rf_noise_rank=5, cif_noise_rank=25), spec, self.profile
        )
        primary = frame[frame["role"] == "primary"].iloc[0]
        assert primary["mean_difference"] == pytest.approx(-19.5)
        assert primary["p_value"] < 0.05

    def test_primary_is_null_when_methods_agree(self, spec: dict) -> None:
        frame = nhanes.inference(
            _ranks_frame(spec, rf_noise_rank=20, cif_noise_rank=20.5), spec, self.profile
        )
        primary = frame[frame["role"] == "primary"].iloc[0]
        assert abs(primary["mean_difference"]) < 1.0 and primary["p_value"] > 0.2

    def test_holm_covers_the_secondary_family_only(self, spec: dict) -> None:
        frame = nhanes.inference(_ranks_frame(spec, 5, 25), spec, self.profile).set_index(
            "hypothesis"
        )
        assert (
            frame.loc["shuffled_mec_weight_rank", "p_value_adjusted"]
            == frame.loc["shuffled_mec_weight_rank", "p_value"]
        )
        sec = frame[frame["role"] == "secondary"]
        assert len(sec) == 2 and (sec["p_value_adjusted"] >= sec["p_value"]).all()

    def test_records_inference_status(self, spec: dict) -> None:
        frame = nhanes.inference(_ranks_frame(spec, 5, 25), spec, self.profile)
        assert set(frame["inference_status"]) == {"exploratory"}


class TestDescriptives:
    def test_reference_agreement_is_one_for_identical_rankings(self, spec: dict) -> None:
        frame = _ranks_frame(spec, 10, 10, folds=2)
        base = frame[frame["method"] == "rf"].copy()
        same = pd.concat([base.assign(method=m) for m in nhanes.METHODS])
        out = nhanes.reference_agreement(same)
        assert np.allclose(out["spearman_cif_cforest"], 1.0)

    def test_positive_control_requires_both_rules(self, spec: dict) -> None:
        frame = _ranks_frame(spec, 10, 10, folds=2)
        frame.loc[frame["feature"] == "waist", "rank"] = 1.0
        frame.loc[frame["feature"] == "shuffled_sex", "rank"] = 30.0
        assert nhanes.positive_control(frame, spec)["passes"].all()
        frame.loc[frame["feature"] == "shuffled_sex", "rank"] = 3.0
        assert not nhanes.positive_control(frame, spec)["passes"].any()

    def test_rank_summary_has_one_row_per_feature(self, spec: dict) -> None:
        out = nhanes.rank_summary(_ranks_frame(spec, 10, 10, folds=2))
        assert len(out) == len(spec["predictors"]) + 3
        assert {"mean_rank_rf", "mean_rank_cif", "mean_rank_cforest", "n_unique"} <= set(
            out.columns
        )


class TestReceipt:
    @pytest.fixture
    def receipt(self, tmp_path: Path) -> dict:
        artifact = tmp_path / "fold_ranks.csv"
        artifact.write_text("fold,rank\n0,1\n", encoding="utf-8")
        nhanes.write_receipt(tmp_path, "smoke", {"fold_ranks.csv": artifact}, {})
        return json.loads((tmp_path / "receipt.json").read_text(encoding="ascii"))

    def test_declares_analysis_and_profile(self, receipt: dict) -> None:
        assert (
            receipt["analysis"] == nhanes.ANALYSIS == "nhanes_diabetes"
            and receipt["profile"] == "smoke"
        )

    def test_carries_driver_verified_fields(self, receipt: dict) -> None:
        assert isinstance(receipt["git_dirty"], bool) and len(receipt["git_sha"]) == 40
        assert receipt["python"] and receipt["platform"]
        assert receipt["versions"] and all(receipt["versions"].values())

    def test_source_hashes_cover_module_lock_and_specification(self, receipt: dict) -> None:
        required = {
            nhanes.MODULE_PATH,
            "pyproject.toml",
            "uv.lock",
            f"paper/jss/replication/{nhanes.SPECIFICATION.name}",
        }
        assert required <= set(receipt["source_sha256"])
        for path, digest in receipt["source_sha256"].items():
            assert len(digest) == 64 and (nhanes.REPO_ROOT / path).is_file(), path

    def test_artifact_entries_match_disk(self, receipt: dict, tmp_path: Path) -> None:
        path = tmp_path / "fold_ranks.csv"
        entry = receipt["artifacts"]["fold_ranks.csv"]
        assert entry["bytes"] == path.stat().st_size and entry["sha256"] == nhanes._sha256(path)
