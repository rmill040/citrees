"""The head-to-head timing protocol's cell list and configurations are fixed."""

from __future__ import annotations

import pytest

from paper.jss.replication import head_to_head_timing as h2h

pytestmark = pytest.mark.paper


def test_protocol_has_thirty_cells_and_six_configurations():
    cells = h2h.build_cells(real=h2h.DEFAULT_REAL, ns=h2h.DEFAULT_NS)
    assert len(cells) == 30
    assert sum(1 for c in cells if c[0] == "synthetic") == 20
    assert [c[1] for c in cells if c[0] == "real"] == h2h.DEFAULT_REAL.split(",")
    assert h2h.DEFAULT_CONFIGS.split(",") == [
        "citrees",
        "citrees_splitgate_nobonf",
        "citrees_maxt",
        "partykit_sqrt_1",
        "partykit_sqrt_32",
        "sklearn_rf",
    ]
    assert (h2h.MAX_DEPTH, h2h.MIN_SPLIT, h2h.MIN_LEAF, h2h.TREES, h2h.ALPHA) == (
        3,
        20,
        7,
        100,
        0.05,
    )
    assert "citrees_maxt" in h2h.CHILD and "threshold_test" in h2h.CHILD
