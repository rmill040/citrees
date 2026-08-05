"""Tests for the complete benchmark experiment grid."""

import pytest

from paper.benchmark.pipeline.grid import ExperimentGrid
from paper.benchmark.pipeline.methods import get_full_method_configs
from paper.benchmark.pipeline.types import DatasetIdentity
from paper.maintenance import audit_hash_alias_manifest


def test_grid_includes_every_high_dimensional_r_and_cit_cell() -> None:
    r_configs = get_full_method_configs(["r_ctree", "r_cforest"], "classification")
    ctree_monte_carlo = next(
        config
        for config in r_configs
        if config.name == "r_ctree" and config.params_dict["testtype"] == "MonteCarlo"
    )
    cit_rdc = next(
        config
        for config in get_full_method_configs(["cit"], "classification")
        if config.params_dict["selector"] == "rdc" and not config.params_dict["honesty"]
    )
    grid = ExperimentGrid(
        task="classification",
        methods=[*r_configs, cit_rdc],
        datasets=["dexter", "gisette", "isolet", "orlraws10P"],
        seeds=list(range(5)),
        dataset_identities={
            dataset: DatasetIdentity("d" * 64, n_samples=10, n_features=4)
            for dataset in ["dexter", "gisette", "isolet", "orlraws10P"]
        },
    )

    keys = {config.key for config in grid}
    required_cells = {
        ("classification", "dexter", config.label, seed)
        for config in r_configs
        for seed in range(5)
    }
    required_cells.update(
        {
            ("classification", "gisette", ctree_monte_carlo.label, 3),
            ("classification", "isolet", ctree_monte_carlo.label, 2),
            ("classification", "isolet", ctree_monte_carlo.label, 3),
            ("classification", "gisette", cit_rdc.label, 0),
            ("classification", "gisette", cit_rdc.label, 1),
            ("classification", "gisette", cit_rdc.label, 3),
            ("classification", "orlraws10P", cit_rdc.label, 1),
        }
    )

    assert len(required_cells) == 37
    assert required_cells <= keys
    assert len(grid) == len(grid.as_list()) == len(grid.methods) * 4 * 5


def test_historical_grid_loader_uses_constant_from_same_revision(monkeypatch) -> None:
    sources = {
        "abc123:paper/scripts/pipeline/config.py": """
from paper.scripts.config.constants import RANDOM_STATE

def get_configs(task):
    return {"rf": [{"method": "rf", "random_state": RANDOM_STATE, "task": task}]}
""",
        "abc123:paper/scripts/config/constants.py": "RANDOM_STATE = 31415\n",
    }

    def read_git_object(command: list[str], *, text: bool) -> str:
        assert command[:2] == ["git", "show"]
        assert text is True
        return sources[command[2]]

    monkeypatch.setattr(
        audit_hash_alias_manifest.subprocess,
        "check_output",
        read_git_object,
    )

    grids = audit_hash_alias_manifest._load_old_grid("abc123:paper/scripts/pipeline/config.py")

    assert grids["classification"]["rf"][0]["random_state"] == 31415
    assert grids["regression"]["rf"][0]["random_state"] == 31415


def test_grid_copies_and_freezes_its_identity_inputs() -> None:
    method = get_full_method_configs(["rf"], "classification")[0]
    methods = [method]
    datasets = ["glass"]
    seeds = [0]
    original_identity = DatasetIdentity("d" * 64, n_samples=214, n_features=9)
    identities = {"glass": original_identity}
    grid = ExperimentGrid(
        task="classification",
        methods=methods,
        datasets=datasets,
        seeds=seeds,
        dataset_identities=identities,
    )

    methods.clear()
    datasets.append("wine")
    seeds.append(1)
    identities["glass"] = DatasetIdentity("e" * 64, n_samples=1, n_features=1)

    assert len(grid) == 1
    assert next(iter(grid)).dataset_identity == original_identity
    with pytest.raises(TypeError):
        grid.dataset_identities["glass"] = identities["glass"]  # type: ignore[index]


@pytest.mark.parametrize(
    ("methods", "datasets", "seeds", "match"),
    [
        ([get_full_method_configs(["rf"], "classification")[0]] * 2, ["glass"], [0], "methods"),
        (
            [get_full_method_configs(["rf"], "classification")[0]],
            ["glass", "glass"],
            [0],
            "datasets",
        ),
        ([get_full_method_configs(["rf"], "classification")[0]], ["glass"], [0, 0], "seeds"),
    ],
)
def test_grid_rejects_duplicate_axes(methods, datasets, seeds, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ExperimentGrid(
            task="classification",
            methods=methods,
            datasets=datasets,
            seeds=seeds,
            dataset_identities={"glass": DatasetIdentity("d" * 64, n_samples=214, n_features=9)},
        )
