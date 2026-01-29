from citrees import ConditionalInferenceTreeClassifier


def test_bonferroni_correction_resets_when_single_test() -> None:
    """Regression test: per-node Bonferroni must not leak across nodes.

    In particular, calling _bonferroni_correction with n_tests=1 should reset the
    private per-node attributes to their unadjusted values.
    """
    clf = ConditionalInferenceTreeClassifier(
        alpha_selector=0.05,
        alpha_splitter=0.05,
        adjust_alpha_selector=True,
        adjust_alpha_splitter=True,
        n_resamples_selector=100,
        n_resamples_splitter=100,
        random_state=0,
    )

    clf._bonferroni_correction(adjust="selector", n_tests=10)
    assert clf._alpha_selector == 0.05 / 10
    assert clf._n_resamples_selector == 100 * 10

    clf._bonferroni_correction(adjust="selector", n_tests=1)
    assert clf._alpha_selector == 0.05
    assert clf._n_resamples_selector == 100

    clf._bonferroni_correction(adjust="splitter", n_tests=5)
    assert clf._alpha_splitter == 0.05 / 5
    assert clf._n_resamples_splitter == 100 * 5

    clf._bonferroni_correction(adjust="splitter", n_tests=1)
    assert clf._alpha_splitter == 0.05
    assert clf._n_resamples_splitter == 100
