"""Tests for extension campaign manifests built on a frozen inventory."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from paper.benchmark.cli.manifest import app as manifest_app
from paper.benchmark.experiments import r_cforest_reproducibility as gate
from paper.benchmark.pipeline.extension_manifest import (
    ExtensionExclusion,
    build_extension_cells,
    counterpart_label,
    load_extension_exclusions,
    write_extension_campaign,
)
from paper.benchmark.pipeline.manifest import (
    ManifestCell,
    RerunManifest,
    compute_campaign_sha256,
    parse_rerun_manifest,
    serialize_rerun_manifest,
)
from paper.benchmark.pipeline.methods import get_full_method_configs
from paper.benchmark.pipeline.runtime_contract import serialize_runtime_contract
from paper.benchmark.pipeline.types import ExperimentConfig
from tests.paper.test_manifest import _runtime_contract
from tests.paper.test_r_cforest_reproducibility import ACCOUNT_A, _manifest

pytestmark = pytest.mark.paper

RUNTIME_CONTRACT_SHA256 = "f" * 64
REASON = "max_type_stage_b_extension"


def _source_with_bonferroni_cells() -> RerunManifest:
    """Gate-complete fixture inventory plus completed cit/cif cells on two datasets per task."""
    base = _manifest(RUNTIME_CONTRACT_SHA256)
    cells: list[ManifestCell] = list(base.cells)
    for task in ("classification", "regression"):
        datasets = sorted(
            {
                (cell.config.dataset, cell.config.dataset_identity, cell.dataset_source)
                for cell in base.cells
                if cell.config.task == task
            },
            key=lambda item: item[0],
        )[:2]
        for config in get_full_method_configs(["cit", "cif"], task):
            for dataset, identity, dataset_source in datasets:
                for seed in gate.EXPECTED_SEEDS:
                    cells.append(
                        ManifestCell(
                            config=ExperimentConfig(
                                method=config,
                                dataset=dataset,
                                seed=seed,
                                task=task,
                                dataset_identity=identity,
                            ),
                            target_aws_account_id=ACCOUNT_A,
                            dataset_source=dataset_source,
                            rerun_reason="completed_benchmark",
                            historically_omitted=False,
                            stage1_required=False,
                            stage2_required=False,
                        )
                    )
    frozen = tuple(cells)
    payload = serialize_rerun_manifest(
        frozen,
        campaign_sha256=compute_campaign_sha256(
            frozen, runtime_contract_sha256=RUNTIME_CONTRACT_SHA256
        ),
        runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
    )
    return parse_rerun_manifest(payload)


def test_counterpart_label_drops_only_the_alias_axis() -> None:
    alias = get_full_method_configs(["cif_maxt"], "classification")[0]
    base = next(
        c
        for c in get_full_method_configs(["cif"], "classification")
        if c.params_dict == {k: v for k, v in alias.params_dict.items() if k != "threshold_test"}
    )
    assert counterpart_label(alias, "cif") == base.label


def test_extension_carries_inventory_and_adds_paired_cells() -> None:
    source = _source_with_bonferroni_cells()
    build = build_extension_cells(
        source,
        methods=["cit_maxt", "cif_maxt"],
        rerun_reason=REASON,
        target_aws_account_id=ACCOUNT_A,
    )

    # Every source cell is carried with execution switched off.
    carried = {cell.identity: cell for cell in build.cells}
    for cell in source.cells:
        kept = carried[cell.identity]
        assert not kept.stage1_required and not kept.stage2_required
        assert kept.rerun_reason == cell.rerun_reason
    # 8 alias configurations x 2 datasets x 5 seeds per task.
    assert len(build.added) == 2 * 8 * 2 * 5
    assert build.added_counts() == {
        "classification": {"cif_maxt": 40, "cit_maxt": 40},
        "regression": {"cif_maxt": 40, "cit_maxt": 40},
    }
    assert all(cell.stage1_required and cell.stage2_required for cell in build.added)
    assert all(cell.rerun_reason == REASON for cell in build.added)
    assert all(cell.config.method.params_dict["threshold_test"] == "maxt" for cell in build.added)
    assert len({cell.identity for cell in build.cells}) == len(build.cells)
    assert build.excluded == ()


def test_extension_manifest_still_satisfies_the_reproducibility_gate(tmp_path: Path) -> None:
    source = _source_with_bonferroni_cells()
    build = build_extension_cells(
        source, methods=["cif_maxt"], rerun_reason=REASON, target_aws_account_id=ACCOUNT_A
    )
    receipt = write_extension_campaign(
        build, runtime_contract_sha256=RUNTIME_CONTRACT_SHA256, output_dir=tmp_path
    )
    manifest = parse_rerun_manifest((tmp_path / "manifest.csv").read_bytes())

    inventory = gate._replacement_inventory(manifest)
    assert sum(len(cells) for datasets in inventory.values() for cells in datasets.values()) == (
        gate.EXPECTED_REPLACEMENT_CELLS
    )
    assert manifest.campaign_sha256 == receipt["campaign_sha256"]
    assert receipt["stage1_required"] == receipt["stage2_required"] == len(build.added)
    assert (tmp_path / f"account-{ACCOUNT_A}.csv").exists()
    assert json.loads((tmp_path / "receipt.json").read_text())["added"] == len(build.added)


def test_exclusions_skip_counterparts_and_must_all_apply() -> None:
    source = _source_with_bonferroni_cells()
    target = next(cell for cell in source.cells if cell.config.method.name == "cit")
    exclusion = ExtensionExclusion(
        task=target.config.task,
        dataset=target.config.dataset,
        method_id=target.config.method.label,
        seed=target.config.seed,
        reason="never_completed",
    )
    build = build_extension_cells(
        source,
        methods=["cit_maxt"],
        rerun_reason=REASON,
        target_aws_account_id=ACCOUNT_A,
        exclusions=[exclusion],
    )
    assert len(build.excluded) == 1
    (key, reason) = build.excluded[0]
    assert reason == "never_completed"
    assert (
        key[0] == target.config.task
        and key[1] == target.config.dataset
        and key[3] == target.config.seed
    )
    assert key[2].startswith("cit_maxt__")
    assert all(cell.identity != key for cell in build.added)

    with pytest.raises(ValueError, match="did not match"):
        build_extension_cells(
            source,
            methods=["cit_maxt"],
            rerun_reason=REASON,
            target_aws_account_id=ACCOUNT_A,
            exclusions=[replace(exclusion, dataset="no_such_dataset")],
        )


def test_extension_rejects_non_alias_methods_and_grid_drift() -> None:
    source = _source_with_bonferroni_cells()
    with pytest.raises(ValueError, match="not a grid alias"):
        build_extension_cells(
            source, methods=["cif"], rerun_reason=REASON, target_aws_account_id=ACCOUNT_A
        )
    drifted_cells = tuple(cell for cell in source.cells if cell.config.method.name != "cif")
    drifted = replace(
        source,
        cells=drifted_cells,
        campaign_sha256=compute_campaign_sha256(
            drifted_cells, runtime_contract_sha256=RUNTIME_CONTRACT_SHA256
        ),
    )
    with pytest.raises(ValueError, match="grids have drifted"):
        build_extension_cells(
            drifted, methods=["cif_maxt"], rerun_reason=REASON, target_aws_account_id=ACCOUNT_A
        )


def test_exclusion_table_is_strict() -> None:
    good = b"task,dataset,method_id,seed,reason\nclassification,glass,cit__abc,0,never_completed\n"
    rows = load_extension_exclusions(good)
    assert rows[0].key == ("classification", "glass", "cit__abc", 0)
    with pytest.raises(ValueError, match="columns"):
        load_extension_exclusions(b"task,dataset\nclassification,glass\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_extension_exclusions(good + b"classification,glass,cit__abc,0,other\n")
    with pytest.raises(ValueError, match="seed"):
        load_extension_exclusions(
            b"task,dataset,method_id,seed,reason\nclassification,glass,x,a,r\n"
        )


def test_tracked_exclusion_table_parses_and_names_only_bonferroni_cells() -> None:
    root = Path(__file__).resolve().parents[2]
    rows = load_extension_exclusions(
        (root / "paper/benchmark/config/maxt_extension_exclusions.csv").read_bytes()
    )
    assert rows
    labels = {
        task: {c.label for c in get_full_method_configs(["cit", "cif"], task)}
        for task in ("classification", "regression")
    }
    assert all(row.method_id in labels[row.task] for row in rows)
    assert {row.reason for row in rows} <= {
        "rdc_host_hard_lock",
        "bonferroni_counterpart_never_completed",
    }
    assert all(
        row.dataset in {"isolet", "gisette"} for row in rows if row.reason == "rdc_host_hard_lock"
    )


def test_cli_extend_writes_campaign_files(tmp_path: Path) -> None:
    source = _source_with_bonferroni_cells()
    contract = _runtime_contract()
    from paper.benchmark.pipeline.runtime_contract import runtime_contract_sha256

    source_payload = serialize_rerun_manifest(
        source.cells,
        campaign_sha256=compute_campaign_sha256(
            source.cells, runtime_contract_sha256=runtime_contract_sha256(contract)
        ),
        runtime_contract_sha256=runtime_contract_sha256(contract),
    )
    (tmp_path / "source.csv").write_bytes(source_payload)
    (tmp_path / "runtime.json").write_bytes(serialize_runtime_contract(contract))
    (tmp_path / "exclusions.csv").write_bytes(b"task,dataset,method_id,seed,reason\n")

    result = CliRunner().invoke(
        manifest_app,
        [
            "extend",
            "--source",
            str(tmp_path / "source.csv"),
            "--runtime-contract",
            str(tmp_path / "runtime.json"),
            "--methods",
            "cit_maxt,cif_maxt",
            "--rerun-reason",
            REASON,
            "--exclusions",
            str(tmp_path / "exclusions.csv"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    receipt = json.loads((tmp_path / "out" / "receipt.json").read_text())
    assert receipt["added"] == 160
    assert receipt["runtime_contract_sha256"] == runtime_contract_sha256(contract)
    manifest = parse_rerun_manifest((tmp_path / "out" / "manifest.csv").read_bytes())
    assert manifest.method_counts("rankings") == {"cif_maxt": 80, "cit_maxt": 80}


def test_stage2_masks_censored_cells_and_keeps_the_inventory(tmp_path: Path) -> None:
    from paper.benchmark.pipeline.extension_manifest import stage2_cells, write_stage2_campaign

    source = _source_with_bonferroni_cells()
    build = build_extension_cells(
        source, methods=["cit_maxt"], rerun_reason=REASON, target_aws_account_id=ACCOUNT_A
    )
    write_extension_campaign(
        build, runtime_contract_sha256=RUNTIME_CONTRACT_SHA256, output_dir=tmp_path
    )
    stage1 = parse_rerun_manifest((tmp_path / "manifest.csv").read_bytes())
    added = [c.identity for c in stage1.cells if c.stage1_required]
    completed = set(added[:-2])  # two cells censored
    cells = stage2_cells(stage1, completed)
    assert len(cells) == len(stage1.cells)
    assert not any(c.stage1_required for c in cells)
    assert sum(c.stage2_required for c in cells) == len(completed)
    assert all(c.identity in completed for c in cells if c.stage2_required)
    receipt = write_stage2_campaign(
        cells,
        source_manifest_sha256=stage1.sha256,
        runtime_contract_sha256=RUNTIME_CONTRACT_SHA256,
        output_dir=tmp_path / "s2",
    )
    stage2 = parse_rerun_manifest((tmp_path / "s2" / "manifest.csv").read_bytes())
    assert receipt["stage2_required"] == len(completed)
    assert stage2.campaign_sha256 != stage1.campaign_sha256
    assert gate._replacement_inventory(stage2)  # gate scope intact
    with pytest.raises(ValueError, match="not Stage 1 cells"):
        stage2_cells(
            stage1,
            {stage1.cells[0].identity}
            if not stage1.cells[0].stage1_required
            else set(added[:1]) | {next(c.identity for c in stage1.cells if not c.stage1_required)},
        )
