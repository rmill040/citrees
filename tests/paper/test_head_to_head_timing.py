"""The head-to-head timing and performance decomposition protocols are fixed."""

from __future__ import annotations

import pandas as pd
import pytest

from paper.jss.replication import head_to_head_timing as h2h
from paper.jss.replication import performance_decomposition as decomp

pytestmark = pytest.mark.paper


def test_full_protocol_has_thirty_cells_and_six_configurations() -> None:
    profile = h2h.PROFILES["full"]
    cells = h2h.build_cells(profile)
    assert len(cells) == 30
    assert sum(1 for c in cells if c[0] == "synthetic") == 20
    assert tuple(c[1] for c in cells if c[0] == "real") == h2h.REAL_DATASETS
    assert h2h.CONFIGS == (
        "citrees",
        "citrees_splitgate_nobonf",
        "citrees_maxt",
        "partykit_sqrt_1",
        "partykit_sqrt_32",
        "sklearn_rf",
    )
    assert (h2h.MAX_DEPTH, h2h.MIN_SPLIT, h2h.MIN_LEAF, h2h.ALPHA) == (3, 20, 7, 0.05)
    assert (profile["trees"], profile["repeats"]) == (100, 2)
    assert "citrees_maxt" in h2h.CHILD and "threshold_test" in h2h.CHILD


def test_shards_partition_the_full_protocol() -> None:
    profile = h2h.PROFILES["full"]
    everything = h2h.shard_jobs(profile, 0, 1)
    assert len(everything) == 30 * len(h2h.CONFIGS)
    shards = [h2h.shard_jobs(profile, i, 4) for i in range(4)]
    assert sorted(job for shard in shards for job in shard) == sorted(everything)
    with pytest.raises(ValueError, match="outside"):
        h2h.shard_jobs(profile, 4, 4)


def test_head_to_head_summary_takes_median_per_cell_and_configuration() -> None:
    raw = pd.DataFrame(
        {
            "kind": ["synthetic"] * 4,
            "dataset": ["syn"] * 4,
            "n": [500] * 4,
            "p": [20] * 4,
            "signal": ["weak"] * 4,
            "config": ["sklearn_rf", "sklearn_rf", "citrees", "citrees"],
            "fit_s": [1.0, 3.0, 4.0, None],
        }
    )
    summary = h2h.summarize(raw)
    assert list(summary.columns) == ["kind", "dataset", "n", "p", "signal", "citrees", "sklearn_rf"]
    assert summary.loc[0, "sklearn_rf"] == 2.0 and summary.loc[0, "citrees"] == 4.0


def test_decomposition_profiles_and_summary_schema() -> None:
    assert decomp.PROFILES["full"] == {
        "n_samples": 1000,
        "n_features": 50,
        "n_trees": 100,
        "repeats": 3,
    }
    assert len(decomp.VARIANTS) == 9
    raw = pd.DataFrame(
        {
            "task": ["classification"] * 3,
            "variant": ["recommended forest, serial"] * 3,
            "kind": ["forest"] * 3,
            "n_jobs": [1] * 3,
            "seconds": [2.0, 1.0, 3.0],
            "depth_mean": [3.0, 3.0, 3.0],
            "internal_nodes_mean": [5.0, 6.0, 7.0],
        }
    )
    summary = decomp.summarize(raw)
    assert summary.loc[0, ["seconds_median", "seconds_min", "seconds_max", "repeats"]].tolist() == [
        2.0,
        1.0,
        3.0,
        3,
    ]
    assert summary.loc[0, "internal_nodes_mean"] == 6.0
