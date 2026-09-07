"""Tests for the TCGA-BRCA application analysis.

The design lives in ``tcga_brca-specification.json``, so these tests check that
the code honors the frozen specification: the candidate screen and the scaler
never see a held-out sample, the corrected t test matches Bouckaert and Frank
(2004) computed by hand, Holm runs over the secondary family only, and the
receipt satisfies the contract the replication driver enforces.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper.jss.replication import tcga_brca

pytestmark = pytest.mark.paper


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(tcga_brca.SPECIFICATION.read_text(encoding="utf-8"))


class TestSpecification:
    def test_specification_is_present_and_hashable(self) -> None:
        assert tcga_brca.SPECIFICATION.is_file()
        assert len(tcga_brca._sha256(tcga_brca.SPECIFICATION)) == 64

    def test_module_path_constant_points_at_this_module(self) -> None:
        assert (tcga_brca.REPO_ROOT / tcga_brca.MODULE_PATH).samefile(tcga_brca.__file__)

    def test_primary_outcome_is_not_a_subtype_call(self, spec: dict) -> None:
        # PAM50 is called from the same RNA-seq that supplies the predictors.
        assert spec["outcomes"]["primary"] == "er"
        assert "pam50" not in {spec["outcomes"]["primary"], *spec["outcomes"]["secondary"]}
        assert "REJECTED" in spec["outcomes"]["independence_note"]

    def test_every_profile_supports_the_tested_feature_count(self, spec: dict) -> None:
        required = spec["primary_test"]["feature_count"]
        for name, profile in spec["profiles"].items():
            assert profile["candidate_count"] >= required, name

    def test_only_the_full_profile_claims_manuscript_inference(self, spec: dict) -> None:
        statuses = {name: p["inference_status"] for name, p in spec["profiles"].items()}
        assert statuses["full"] == "manuscript_inference"
        assert statuses["quick"] != "manuscript_inference"
        assert statuses["smoke"] != "manuscript_inference"


class TestCandidateScreen:
    def test_screen_recovers_the_planted_signal_first(self) -> None:
        rng = np.random.default_rng(1718)
        y = np.repeat((0, 1), 40)
        X = rng.normal(size=(80, 12))
        X[:, 7] += 4.0 * y
        assert tcga_brca._screen(X, y, 3)[0] == 7

    def test_screen_returns_the_requested_count_in_descending_strength(self) -> None:
        rng = np.random.default_rng(7)
        y = np.repeat((0, 1), 30)
        X = rng.normal(size=(60, 20))
        for column, effect in enumerate((3.0, 2.0, 1.0)):
            X[:, column] += effect * y
        selected = tcga_brca._screen(X, y, 5)
        assert len(selected) == 5
        # The two strong effects lead; the weakest planted effect only has to
        # survive the screen, since noise columns can out-correlate it at n=60.
        assert list(selected[:2]) == [0, 1]
        assert 2 in selected

    def test_screen_breaks_ties_on_ascending_gene_index(self) -> None:
        y = np.array([0, 0, 1, 1])
        column = np.array([0.0, 0.0, 1.0, 1.0])
        X = np.column_stack([np.arange(4.0), column, column, column])
        assert list(tcga_brca._screen(X, y, 3)) == [1, 2, 3]

    def test_screen_tolerates_a_constant_column(self) -> None:
        y = np.array([0, 0, 1, 1])
        X = np.column_stack([np.ones(4), np.array([0.0, 0.1, 1.0, 1.1])])
        selected = tcga_brca._screen(X, y, 2)
        assert selected[0] == 1
        assert np.isfinite(X[:, selected]).all()


@pytest.fixture(scope="module")
def rankings() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1718)
    y = np.repeat((0, 1), 45)
    X = rng.normal(size=(90, 8))
    X[:, 2] += 3.0 * y
    return tcga_brca._rank_within_candidates(X, y, trees=4, projections=10, seed=1718)


class TestRankings:
    def test_every_specified_method_produces_a_ranking(self, rankings: dict, spec: dict) -> None:
        assert set(rankings) == set(spec["prediction"]["methods"]) == set(tcga_brca.METHODS)

    def test_each_ranking_is_a_permutation_of_the_candidates(self, rankings: dict) -> None:
        for method, order in rankings.items():
            assert sorted(order.tolist()) == list(range(8)), method

    def test_marginal_preserves_the_screen_order(self, rankings: dict) -> None:
        assert list(rankings["marginal"]) == list(range(8))

    def test_every_method_ranks_the_planted_signal_first(self, rankings: dict) -> None:
        for method in ("l2_logistic", "cit", "cif"):
            assert rankings[method][0] == 2, method


class TestCorrectedTTest:
    def test_matches_a_hand_computed_bouckaert_frank_statistic(self) -> None:
        differences = np.array([-0.10, -0.05, -0.20, 0.01, -0.06])
        n_train = np.full(5, 80.0)
        n_test = np.full(5, 20.0)
        result = tcga_brca._corrected_t(differences, n_train, n_test, folds=5, repeats=1)
        inflation = 1.0 / 5 + 0.25
        expected = differences.mean() / np.sqrt(inflation * differences.var(ddof=1))
        assert result["statistic"] == pytest.approx(expected)
        assert result["degrees_of_freedom"] == 4
        assert result["mean_difference"] == pytest.approx(differences.mean())

    def test_correction_is_conservative_relative_to_the_naive_test(self) -> None:
        differences = np.array([-0.10, -0.05, -0.20, 0.01, -0.06])
        corrected = tcga_brca._corrected_t(
            differences, np.full(5, 80.0), np.full(5, 20.0), folds=5, repeats=1
        )
        naive = differences.mean() / np.sqrt(differences.var(ddof=1) / 5)
        assert abs(corrected["statistic"]) < abs(naive)

    def test_p_value_is_one_sided_toward_the_registered_alternative(self) -> None:
        args = (np.full(5, 80.0), np.full(5, 20.0))
        lower = tcga_brca._corrected_t(
            np.full(5, -0.1) + [0, 0.01, -0.01, 0.02, -0.02], *args, 5, 1
        )
        higher = tcga_brca._corrected_t(
            np.full(5, 0.1) + [0, 0.01, -0.01, 0.02, -0.02], *args, 5, 1
        )
        assert lower["p_value"] < 0.05 < higher["p_value"]

    def test_zero_variance_does_not_divide_by_zero(self) -> None:
        result = tcga_brca._corrected_t(
            np.zeros(5), np.full(5, 80.0), np.full(5, 20.0), folds=5, repeats=1
        )
        assert result["statistic"] == 0.0
        assert result["p_value"] == pytest.approx(0.5)


def _metrics_frame(values: dict[str, dict[str, float]], feature_count: int) -> pd.DataFrame:
    rows = []
    for outcome, per_method in values.items():
        for method, level in per_method.items():
            for fold in range(6):
                rows.append(
                    {
                        "outcome": outcome,
                        "fold": fold,
                        "method": method,
                        "n_features": feature_count,
                        "n_train": 80,
                        "n_test": 20,
                        "log_loss": level + 0.001 * fold,
                    }
                )
    return pd.DataFrame(rows)


class TestInference:
    @pytest.fixture
    def profile(self) -> dict:
        return {
            "folds": 3,
            "repeats": 2,
            "outcomes": ["er", "pr", "her2"],
            "inference_status": "exploratory",
        }

    def test_holm_adjusts_the_secondary_family_only(self, spec: dict, profile: dict) -> None:
        metrics = _metrics_frame(
            {
                "er": {"cif": 0.30, "marginal": 0.50},
                "pr": {"cif": 0.40, "marginal": 0.45},
                "her2": {"cif": 0.44, "marginal": 0.45},
            },
            spec["primary_test"]["feature_count"],
        )
        frame = tcga_brca._inference(metrics, spec, profile).set_index("outcome")
        assert frame.loc["er", "role"] == "primary"
        # The primary hypothesis stands alone and is never inflated by the family.
        assert frame.loc["er", "p_value_adjusted"] == frame.loc["er", "p_value"]
        secondary = frame.loc[["pr", "her2"]]
        assert (secondary["p_value_adjusted"] >= secondary["p_value"]).all()

    def test_holm_is_monotone_in_the_sorted_secondary_p_values(
        self, spec: dict, profile: dict
    ) -> None:
        metrics = _metrics_frame(
            {
                "er": {"cif": 0.30, "marginal": 0.50},
                "pr": {"cif": 0.20, "marginal": 0.50},
                "her2": {"cif": 0.449, "marginal": 0.450},
            },
            spec["primary_test"]["feature_count"],
        )
        frame = tcga_brca._inference(metrics, spec, profile).set_index("outcome")
        smaller, larger = sorted((frame.loc["pr"], frame.loc["her2"]), key=lambda r: r["p_value"])
        assert smaller["p_value_adjusted"] <= larger["p_value_adjusted"]

    def test_only_the_registered_feature_count_is_tested(self, spec: dict, profile: dict) -> None:
        tested = spec["primary_test"]["feature_count"]
        metrics = _metrics_frame({"er": {"cif": 0.3, "marginal": 0.5}}, tested)
        other = _metrics_frame({"er": {"cif": 0.9, "marginal": 0.1}}, tested + 15)
        combined = tcga_brca._inference(pd.concat([metrics, other]), spec, profile)
        only_primary = tcga_brca._inference(metrics, spec, profile)
        assert combined.loc[0, "statistic"] == pytest.approx(only_primary.loc[0, "statistic"])

    def test_records_the_profile_inference_status(self, spec: dict, profile: dict) -> None:
        metrics = _metrics_frame(
            {"er": {"cif": 0.3, "marginal": 0.5}}, spec["primary_test"]["feature_count"]
        )
        frame = tcga_brca._inference(metrics, spec, profile)
        assert set(frame["inference_status"]) == {"exploratory"}


def _selection_frame(per_fold: list[list[str]], outcome: str = "er") -> pd.DataFrame:
    rows = []
    for fold, genes in enumerate(per_fold):
        for rank, gene in enumerate(genes):
            rows.append(
                {
                    "outcome": outcome,
                    "fold": fold,
                    "method": "cif",
                    "rank": rank,
                    "ensembl_id": gene,
                    "symbol": gene,
                }
            )
    return pd.DataFrame(rows)


class TestStabilityAndPositiveControl:
    def test_identical_folds_give_perfect_overlap(self, spec: dict) -> None:
        genes = [f"G{i}" for i in range(25)]
        frame = tcga_brca._stability(_selection_frame([genes, genes, genes]), spec)
        assert np.allclose(frame["mean_pairwise_overlap"], 1.0)

    def test_disjoint_folds_give_zero_overlap(self, spec: dict) -> None:
        first = [f"A{i}" for i in range(25)]
        second = [f"B{i}" for i in range(25)]
        frame = tcga_brca._stability(_selection_frame([first, second]), spec)
        assert np.allclose(frame["mean_pairwise_overlap"], 0.0)

    def test_overlap_is_scaled_by_the_requested_count(self, spec: dict) -> None:
        first = ["S0", "S1", "X2", "X3", "X4"] + [f"A{i}" for i in range(20)]
        second = ["S0", "S1", "Y2", "Y3", "Y4"] + [f"B{i}" for i in range(20)]
        frame = tcga_brca._stability(_selection_frame([first, second]), spec)
        at_five = frame.loc[frame["n_features"] == 5, "mean_pairwise_overlap"].iloc[0]
        assert at_five == pytest.approx(2 / 5)

    def test_stability_covers_every_specified_count(self, spec: dict) -> None:
        genes = [f"G{i}" for i in range(25)]
        frame = tcga_brca._stability(_selection_frame([genes, genes]), spec)
        assert sorted(frame["n_features"]) == sorted(spec["stability"]["feature_counts"])

    def test_positive_control_applies_the_majority_rule_at_top_ten(self, spec: dict) -> None:
        marker = spec["positive_control"]["expected"]["er"]
        others = [f"G{i}" for i in range(20)]
        hit = [marker, *others]
        miss = [*others[:10], marker, *others[10:]]
        frame = tcga_brca._positive_control(_selection_frame([hit, hit, miss]), spec)
        assert frame.loc[0, "folds_with_marker_in_top10"] == 2
        assert bool(frame.loc[0, "passes_majority_rule"]) is True

    def test_positive_control_fails_without_a_majority(self, spec: dict) -> None:
        marker = spec["positive_control"]["expected"]["er"]
        others = [f"G{i}" for i in range(20)]
        hit = [marker, *others]
        miss = [*others[:10], marker, *others[10:]]
        frame = tcga_brca._positive_control(_selection_frame([hit, miss, miss]), spec)
        assert bool(frame.loc[0, "passes_majority_rule"]) is False


class TestReceipt:
    @pytest.fixture
    def receipt(self, tmp_path: Path) -> dict:
        artifact = tmp_path / "fold_metrics.csv"
        artifact.write_text("outcome,log_loss\ner,0.5\n", encoding="utf-8")
        tcga_brca._write_receipt(tmp_path, "smoke", {"fold_metrics.csv": artifact}, {})
        return json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))

    def test_declares_the_analysis_and_profile(self, receipt: dict) -> None:
        assert receipt["analysis"] == tcga_brca.ANALYSIS == "tcga_brca"
        assert receipt["profile"] == "smoke"

    def test_carries_the_fields_the_driver_verifies(self, receipt: dict) -> None:
        assert isinstance(receipt["git_dirty"], bool)
        assert len(receipt["git_sha"]) == 40
        assert receipt["python"] and receipt["platform"]
        assert receipt["versions"] and all(receipt["versions"].values())

    def test_source_hashes_cover_the_module_and_the_locked_environment(self, receipt: dict) -> None:
        required = {tcga_brca.MODULE_PATH, "pyproject.toml", "uv.lock"}
        assert required <= set(receipt["source_sha256"])
        for path, digest in receipt["source_sha256"].items():
            assert len(digest) == 64 and digest == digest.lower()
            assert (tcga_brca.REPO_ROOT / path).is_file(), path

    def test_source_hashes_pin_the_frozen_specification(self, receipt: dict) -> None:
        key = f"paper/jss/replication/{tcga_brca.SPECIFICATION.name}"
        assert receipt["source_sha256"][key] == tcga_brca._sha256(tcga_brca.SPECIFICATION)

    def test_artifact_entries_match_the_files_on_disk(self, receipt: dict, tmp_path: Path) -> None:
        entry = receipt["artifacts"]["fold_metrics.csv"]
        path = tmp_path / "fold_metrics.csv"
        assert entry["bytes"] == path.stat().st_size
        assert entry["sha256"] == tcga_brca._sha256(path)

    def test_receipt_is_ascii_so_the_digest_is_portable(self, tmp_path: Path) -> None:
        artifact = tmp_path / "stability.csv"
        artifact.write_text("outcome\ner\n", encoding="utf-8")
        tcga_brca._write_receipt(tmp_path, "smoke", {"stability.csv": artifact}, {})
        (tmp_path / "receipt.json").read_text(encoding="ascii")
