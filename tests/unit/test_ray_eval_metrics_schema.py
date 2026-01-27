import numpy as np

from paper.scripts.pipeline.stage2 import evaluate_fold


def test_evaluate_fold_classification_metrics_schema():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y = (X[:, 0] + rng.normal(scale=0.1, size=40) > 0).astype(int)

    X_train, X_test = X[:30], X[30:]
    y_train, y_test = y[:30], y[30:]
    ranking = np.arange(X.shape[1])

    results = evaluate_fold(
        X_train,
        y_train,
        X_test,
        y_test,
        ranking,
        task_type="classification",
        random_state=0,
        n_jobs=1,
    )

    row = results[0]
    required = {
        "accuracy",
        "f1",
        "f1_macro",
        "balanced_accuracy",
        "roc_auc",
        "auc",
    }
    missing = required - set(row.keys())
    assert not missing


def test_evaluate_fold_regression_metrics_schema():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y = X[:, 0] * 0.5 + rng.normal(scale=0.1, size=40)

    X_train, X_test = X[:30], X[30:]
    y_train, y_test = y[:30], y[30:]
    ranking = np.arange(X.shape[1])

    results = evaluate_fold(
        X_train,
        y_train,
        X_test,
        y_test,
        ranking,
        task_type="regression",
        random_state=0,
        n_jobs=1,
    )

    row = results[0]
    required = {"r2", "mse", "rmse", "mae"}
    missing = required - set(row.keys())
    assert not missing
