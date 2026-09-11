"""Tests for the max-type Stage B grid aliases (cit_maxt, cif_maxt)."""

from __future__ import annotations

import numpy as np
import pytest

from citrees import ConditionalInferenceForestClassifier, ConditionalInferenceTreeRegressor
from paper.analysis.config_resolution import resolve_method_config
from paper.benchmark.pipeline.config import get_method_configs
from paper.benchmark.pipeline.methods import (
    BASE_METHOD_ALIASES,
    EMBEDDING_METHODS,
    base_method,
    get_full_method_configs,
    get_methods,
)
from paper.benchmark.pipeline.selectors import get_embedding_model

pytestmark = pytest.mark.paper

TASKS = ("classification", "regression")


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("alias", ["cit_maxt", "cif_maxt"])
def test_maxt_grid_mirrors_base_grid_with_explicit_maxt_axis(task: str, alias: str) -> None:
    base = base_method(alias)
    alias_configs = get_method_configs(alias, task)
    base_configs = get_method_configs(base, task)

    assert len(alias_configs) == len(base_configs) == 4
    for cfg in alias_configs:
        assert cfg["method"] == alias
        assert cfg["threshold_test"] == "maxt"
    stripped = [
        {k: v for k, v in cfg.items() if k not in {"method", "threshold_test"}}
        for cfg in alias_configs
    ]
    assert stripped == [{k: v for k, v in cfg.items() if k != "method"} for cfg in base_configs]


@pytest.mark.parametrize("task", TASKS)
def test_maxt_identities_are_disjoint_from_bonferroni_identities(task: str) -> None:
    maxt = {c.label for c in get_full_method_configs(["cit_maxt", "cif_maxt"], task)}
    bonferroni = {c.label for c in get_full_method_configs(["cit", "cif"], task)}
    assert len(maxt) == 8
    assert maxt.isdisjoint(bonferroni)
    assert all(label.split("__")[0] in {"cit_maxt", "cif_maxt"} for label in maxt)


def test_aliases_are_registered_everywhere_dispatch_needs_them() -> None:
    assert BASE_METHOD_ALIASES == {"cit_maxt": "cit", "cif_maxt": "cif"}
    assert base_method("rf") == "rf"
    for task in TASKS:
        assert {"cit_maxt", "cif_maxt"} <= set(get_methods(task))
    assert {"cit_maxt", "cif_maxt"} <= EMBEDDING_METHODS


def test_alias_models_carry_the_maxt_threshold_test_and_fit() -> None:
    rng = np.random.default_rng(1718)
    X = rng.normal(size=(60, 5))
    y_clf = (X[:, 0] > 0).astype(int)
    y_reg = X[:, 1] + 0.1 * rng.normal(size=60)

    clf_params = dict(get_full_method_configs(["cif_maxt"], "classification")[0].params_dict)
    clf_params.update(n_estimators=3, n_jobs=1)
    clf = get_embedding_model("cif_maxt", "classification", random_state=1718, params=clf_params)
    assert isinstance(clf, ConditionalInferenceForestClassifier)
    assert clf.get_params()["threshold_test"] == "maxt"
    clf.fit(X, y_clf)
    assert clf.feature_importances_.shape == (5,)

    reg_params = dict(get_full_method_configs(["cit_maxt"], "regression")[0].params_dict)
    reg = get_embedding_model("cit_maxt", "regression", random_state=1718, params=reg_params)
    assert isinstance(reg, ConditionalInferenceTreeRegressor)
    assert reg.get_params()["threshold_test"] == "maxt"
    reg.fit(X, y_reg)
    assert reg.feature_importances_.shape == (5,)


def test_analysis_resolves_alias_identities_back_to_parameters() -> None:
    config = get_full_method_configs(["cif_maxt"], "regression")[3]
    resolved = resolve_method_config("regression", "cif_maxt", config.label)
    assert resolved == config
    assert resolved.params_dict["threshold_test"] == "maxt"
