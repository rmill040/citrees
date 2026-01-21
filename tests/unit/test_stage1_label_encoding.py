import numpy as np

from paper.scripts.experiments.ray_feature_selection import filter_selector


def test_filter_selector_label_invariance():
    n_per = 5
    x0 = np.vstack(
        [
            np.full((n_per, 1), 1.0),
            np.full((n_per, 1), 0.0),
            np.full((n_per, 1), 0.0),
        ]
    )
    x1 = np.vstack(
        [
            np.full((n_per, 1), 0.0),
            np.full((n_per, 1), 0.0),
            np.full((n_per, 1), 100.0),
        ]
    )
    X = np.hstack([x0, x1])
    y = np.array([0] * n_per + [1] * n_per + [2] * n_per)

    rank_base = filter_selector(X, y, method="mc", task_type="classification", random_state=0)
    rank_shift = filter_selector(X, y + 5, method="mc", task_type="classification", random_state=0)

    assert np.array_equal(rank_base, rank_shift)
