import numpy as np
import pandas as pd
import pytest

from paper.scripts.analysis.stats import friedman_test, pairwise_wilcoxon_holm


@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide")
def test_friedman_uses_complete_case_rows():
    data = pd.DataFrame(
        {
            "a_m": [1.0, 2.0, np.nan, 4.0],
            "b_m": [1.0, np.nan, 3.0, 4.0],
            "c_m": [1.0, 2.0, 3.0, 4.0],
        }
    )
    chi2, p, n_datasets, k_methods = friedman_test(data, ["a", "b", "c"], "m")
    assert k_methods == 3
    assert n_datasets == 2
    assert np.isfinite(chi2) or np.isnan(chi2)


@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide")
def test_pairwise_wilcoxon_uses_aligned_pairs():
    n = 12
    a = np.arange(n, dtype=float)
    b = np.arange(n, dtype=float)
    c = np.arange(n, dtype=float)
    a[1] = np.nan
    b[2] = np.nan
    data = pd.DataFrame({"a_m": a, "b_m": b, "c_m": c})

    results = pairwise_wilcoxon_holm(data, ["a", "b", "c"], "m")
    assert not results.empty

    ab = results[(results["method1"] == "a") & (results["method2"] == "b")].iloc[0]
    assert ab["n_pairs"] == n - 2
