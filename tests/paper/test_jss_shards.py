"""Tests for deterministic distributed JSS replication shards."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from paper.jss.replication import behavior, calibration, replicate, shards

pytestmark = pytest.mark.paper


def _patch_runtime_receipts(monkeypatch: pytest.MonkeyPatch) -> None:
    r_environment = {
        "r": "fixture",
        "partykit": "fixture",
        "libcoin": "fixture",
        "mvtnorm": "fixture",
    }
    monkeypatch.setattr(shards, "get_r_runtime_versions", lambda: r_environment)
    monkeypatch.setattr(
        shards,
        "_versions",
        lambda: {package: "fixture" for package in shards.RUNTIME_PACKAGES},
    )
    monkeypatch.setattr(behavior, "get_r_runtime_versions", lambda: r_environment)
    monkeypatch.setattr(calibration, "_r_environment", lambda: r_environment)


def _assert_result_sets_equal(
    expected: dict[str, pd.DataFrame],
    observed: dict[str, pd.DataFrame],
) -> None:
    assert set(observed) == set(expected)
    for name in expected:
        assert tuple(observed[name].columns) == tuple(expected[name].columns)
        pd.testing.assert_frame_equal(
            observed[name].convert_dtypes(),
            expected[name].convert_dtypes(),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )


def _read_parquet_results(
    output_dir: Path,
    table_names: set[str],
) -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(output_dir / f"{name}.parquet") for name in table_names}


def _assert_persisted_results_equal(
    expected: dict[str, pd.DataFrame],
    observed: dict[str, pd.DataFrame],
) -> None:
    assert set(observed) == set(expected)
    for name in expected:
        pd.testing.assert_frame_equal(observed[name], expected[name])


def _assert_table_artifacts_identical(
    expected_dir: Path,
    observed_dir: Path,
) -> None:
    expected_names = {
        path.name for pattern in ("*.parquet", "*.csv") for path in expected_dir.glob(pattern)
    }
    observed_names = {
        path.name for pattern in ("*.parquet", "*.csv") for path in observed_dir.glob(pattern)
    }
    assert observed_names == expected_names
    for name in expected_names:
        assert (observed_dir / name).read_bytes() == (expected_dir / name).read_bytes()


def _mock_behavior_datasets() -> tuple[behavior.DatasetSpec, behavior.DatasetSpec]:
    sample_ids = np.arange(8, dtype=np.float64)
    classification_X = np.column_stack([sample_ids, sample_ids / 8.0, np.sin(sample_ids)])
    classification_y = (sample_ids.astype(np.int64) % 2).astype(np.int64)
    regression_X = np.column_stack([sample_ids, sample_ids**2, np.cos(sample_ids)])
    regression_y = sample_ids * 1.5
    feature_names = ("sample", "trend", "wave")
    return (
        behavior.DatasetSpec(
            task="classification",
            name="mock_classification",
            X=classification_X,
            y=classification_y,
            feature_names=feature_names,
            sha256=behavior._array_sha256(
                classification_X,
                classification_y,
                feature_names,
            ),
        ),
        behavior.DatasetSpec(
            task="regression",
            name="mock_regression",
            X=regression_X,
            y=regression_y,
            feature_names=feature_names,
            sha256=behavior._array_sha256(
                regression_X,
                regression_y,
                feature_names,
            ),
        ),
    )


def _mock_behavior_fit(
    method: behavior.Method,
    task: behavior.Task,
    family: behavior.ModelFamily,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    settings: behavior.BehaviorSettings,
    seed: int,
) -> behavior.ModelBehavior:
    del X_train, y_train, settings, seed
    if family == "tree":
        root_feature = 0 if method == "citrees" else 1
        feature_values = (
            np.array([2.0, 1.0, 0.0]) if method == "citrees" else np.array([1.0, 2.0, 0.0])
        )
    else:
        root_feature = None
        feature_values = (
            np.array([0.6, 0.3, 0.1]) if method == "citrees" else np.array([0.5, np.nan, 0.2])
        )

    sample_ids = X_test[:, 0].astype(np.int64)
    if task == "classification":
        predictions = sample_ids % 2
        probabilities = np.full((len(sample_ids), 2), 0.2, dtype=np.float64)
        probabilities[np.arange(len(sample_ids)), predictions] = 0.8
        if method == "partykit":
            probabilities = probabilities * 0.875 + 0.0625
        return behavior.ModelBehavior(
            root_feature=root_feature,
            feature_values=feature_values,
            predictions=predictions,
            probabilities=probabilities,
            classes=np.array([0, 1]),
        )

    predictions = sample_ids * 1.5 + (0.0 if method == "citrees" else 0.25)
    return behavior.ModelBehavior(
        root_feature=root_feature,
        feature_values=feature_values,
        predictions=predictions,
        probabilities=None,
        classes=None,
    )


def _plus_one_values(indices: np.ndarray, n_resamples: int) -> np.ndarray:
    return (indices + 1) / (n_resamples + 1)


def _synthetic_cardinality_tree_raw(base_seed: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for task_index, task in enumerate(("classification", "regression")):
        _, features = calibration._cardinality_matrix(
            np.random.default_rng(100 + task_index),
            calibration.CARDINALITY_N_SAMPLES,
        )
        data_seed = calibration._stream_seed(
            base_seed,
            f"cardinality_bias__{task}__n{calibration.CARDINALITY_N_SAMPLES}",
            0,
            "data",
        )
        citrees_indices = np.asarray([100, 200, 300, 400, 500], dtype=np.int64)
        citrees_budget = calibration.CARDINALITY_B * len(calibration.CARDINALITY_SUPPORTS)
        citrees_p_values = _plus_one_values(citrees_indices, citrees_budget)
        frames.append(
            calibration.build_cardinality_tree_raw(
                features,
                calibration.CardinalityCandidateDiagnostics(
                    statistics=np.linspace(
                        0.1,
                        0.5,
                        len(calibration.CARDINALITY_SUPPORTS),
                    ),
                    p_values=citrees_p_values,
                    native_p_values=citrees_p_values.copy(),
                    realized_permutations=np.full(
                        len(calibration.CARDINALITY_SUPPORTS),
                        citrees_budget,
                        dtype=np.int64,
                    ),
                    native_selected_position=None,
                    native_p_value_rule="plus_one",
                ),
                task=cast(calibration.Task, task),
                method="citrees",
                selector="mc" if task == "classification" else "pc",
                replicate=0,
                data_seed=data_seed,
                model_seed=calibration._stream_seed(
                    base_seed,
                    f"{task}__citrees__n{calibration.CARDINALITY_N_SAMPLES}",
                    0,
                    "model",
                ),
            )
        )

        partykit_counts = np.asarray([100, 200, 300, 400, 500], dtype=np.int64)
        frames.append(
            calibration.build_cardinality_tree_raw(
                features,
                calibration.CardinalityCandidateDiagnostics(
                    statistics=np.linspace(
                        1.0,
                        5.0,
                        len(calibration.CARDINALITY_SUPPORTS),
                    ),
                    p_values=_plus_one_values(
                        partykit_counts,
                        calibration.CARDINALITY_B,
                    ),
                    native_p_values=partykit_counts / calibration.CARDINALITY_B,
                    realized_permutations=np.full(
                        len(calibration.CARDINALITY_SUPPORTS),
                        calibration.CARDINALITY_B,
                        dtype=np.int64,
                    ),
                    native_selected_position=None,
                    native_p_value_rule="r_monte_carlo",
                ),
                task=cast(calibration.Task, task),
                method="partykit",
                selector="quadratic",
                replicate=0,
                data_seed=data_seed,
                model_seed=calibration._stream_seed(
                    base_seed,
                    f"{task}__partykit__n{calibration.CARDINALITY_N_SAMPLES}",
                    0,
                    "model",
                ),
            )
        )

        scores = np.linspace(0.1, 0.5, len(calibration.CARDINALITY_SUPPORTS))
        frames.append(
            calibration.build_cardinality_cart_raw(
                features,
                scores,
                task=cast(calibration.Task, task),
                replicate=0,
                data_seed=data_seed,
                model_seed=calibration._stream_seed(
                    base_seed,
                    f"{task}__cart__n{calibration.CARDINALITY_N_SAMPLES}",
                    0,
                    "model",
                ),
                native_selected_position=int(np.argmax(scores)),
            )
        )
    raw = pd.concat(frames, ignore_index=True)
    calibration.validate_cardinality_tree_raw(raw)
    return raw


def _position_aligned_values(
    features: tuple[calibration.CardinalityFeatureDesign, ...],
    feature_values: list[float],
) -> np.ndarray:
    values = np.empty(len(features), dtype=np.float64)
    for feature in features:
        values[feature.position] = feature_values[feature.feature_id]
    return values


def _synthetic_cardinality_forest_raw(base_seed: int) -> pd.DataFrame:
    patterns = {
        "citrees_cif": [1.0, 2.0, 3.0, 4.0, 5.0],
        "partykit_cforest": [3.0, 2.0, 5.0, 1.0, 4.0],
        "sklearn_rf": [5.0, 4.0, 3.0, 2.0, 1.0],
    }
    frames: list[pd.DataFrame] = []
    for task_index, task in enumerate(("classification", "regression")):
        _, features = calibration._cardinality_matrix(
            np.random.default_rng(200 + task_index),
            calibration.CARDINALITY_N_SAMPLES,
        )
        data_seed = calibration._stream_seed(
            base_seed,
            f"cardinality_bias__{task}__n{calibration.CARDINALITY_N_SAMPLES}",
            0,
            "data",
        )
        for method, pattern in patterns.items():
            frames.append(
                calibration.build_cardinality_forest_raw(
                    features,
                    _position_aligned_values(features, pattern),
                    task=cast(calibration.Task, task),
                    method=cast(calibration.ForestMethod, method),
                    replicate=0,
                    data_seed=data_seed,
                    model_seed=calibration._stream_seed(
                        base_seed,
                        f"{task}__{method}__n{calibration.CARDINALITY_N_SAMPLES}",
                        0,
                        "model",
                    ),
                )
            )
    raw = pd.concat(frames, ignore_index=True)
    calibration.validate_cardinality_forest_raw(raw)
    return raw


@pytest.fixture(scope="module")
def calibration_smoke_raw() -> dict[str, pd.DataFrame]:
    return {
        "selector_null_raw": calibration.run_selector_null("smoke", base_seed=7),
        "root_null_raw": calibration.run_root_null("smoke", base_seed=7),
        "cardinality_tree_raw": _synthetic_cardinality_tree_raw(7),
        "cardinality_forest_raw": _synthetic_cardinality_forest_raw(7),
    }


def _verified_shard(index: int, num_shards: int) -> shards.VerifiedShard:
    spec = shards.ShardSpec(
        target_analysis="behavior",
        component="behavior",
        profile="smoke",
        shard_index=index,
        num_shards=num_shards,
        base_seed=7,
    )
    return shards.VerifiedShard(
        receipt_path=Path(f"/receipt-{num_shards}-{index}.json"),
        output_dir=Path(f"/shard-{num_shards}-{index}"),
        spec=spec,
        receipt={},
        receipt_sha256=str(index),
    )


def test_assignments_are_complete_disjoint_and_deterministic() -> None:
    for component in shards.CALIBRATION_COMPONENTS:
        probe = shards.ShardSpec(
            target_analysis="calibration",
            component=component,
            profile="smoke",
            shard_index=0,
            num_shards=1,
            base_seed=7,
        )
        total = shards._assignment_count(probe)
        num_shards = min(3, total)
        assignments = [
            shards.calibration_shard_replicates(
                shards.ShardSpec(
                    target_analysis="calibration",
                    component=component,
                    profile="smoke",
                    shard_index=index,
                    num_shards=num_shards,
                    base_seed=7,
                )
            )
            for index in range(num_shards)
        ]
        flattened = [replicate for assignment in assignments for replicate in assignment]
        assert sorted(flattened) == list(range(total))
        assert len(flattened) == len(set(flattened))

    inventory = behavior.behavior_cell_inventory("smoke")
    behavior_assignments = [
        shards.behavior_shard_cells(
            shards.ShardSpec(
                target_analysis="behavior",
                component="behavior",
                profile="smoke",
                shard_index=index,
                num_shards=3,
                base_seed=7,
            )
        )
        for index in range(3)
    ]
    flattened_cells = [cell for assignment in behavior_assignments for cell in assignment]
    assert set(flattened_cells) == set(inventory)
    assert len(flattened_cells) == len(set(flattened_cells)) == len(inventory)
    assert behavior_assignments == [
        shards.behavior_shard_cells(
            shards.ShardSpec(
                target_analysis="behavior",
                component="behavior",
                profile="smoke",
                shard_index=index,
                num_shards=3,
                base_seed=7,
            )
        )
        for index in range(3)
    ]


@pytest.mark.parametrize(
    ("spec", "exception", "match"),
    [
        (
            shards.ShardSpec(
                target_analysis=cast(shards.TargetAnalysis, "unknown"),
                component="selector",
                profile="smoke",
                shard_index=0,
                num_shards=1,
                base_seed=7,
            ),
            ValueError,
            "unknown shard target",
        ),
        (
            shards.ShardSpec(
                target_analysis="calibration",
                component="unknown",
                profile="smoke",
                shard_index=0,
                num_shards=1,
                base_seed=7,
            ),
            ValueError,
            "unknown calibration",
        ),
        (
            shards.ShardSpec(
                target_analysis="behavior",
                component="selector",
                profile="smoke",
                shard_index=0,
                num_shards=1,
                base_seed=7,
            ),
            ValueError,
            "component='behavior'",
        ),
        (
            shards.ShardSpec(
                target_analysis="calibration",
                component="selector",
                profile=cast(shards.Profile, "unknown"),
                shard_index=0,
                num_shards=1,
                base_seed=7,
            ),
            ValueError,
            "unknown shard profile",
        ),
        (
            shards.ShardSpec(
                target_analysis="calibration",
                component="selector",
                profile="smoke",
                shard_index=0,
                num_shards=0,
                base_seed=7,
            ),
            ValueError,
            "at least one",
        ),
        (
            shards.ShardSpec(
                target_analysis="calibration",
                component="selector",
                profile="smoke",
                shard_index=-1,
                num_shards=1,
                base_seed=7,
            ),
            ValueError,
            "0 <= shard_index",
        ),
        (
            shards.ShardSpec(
                target_analysis="calibration",
                component="selector",
                profile="smoke",
                shard_index=0,
                num_shards=5,
                base_seed=7,
            ),
            ValueError,
            "available assignments",
        ),
        (
            shards.ShardSpec(
                target_analysis="calibration",
                component="selector",
                profile="smoke",
                shard_index=0,
                num_shards=cast(int, True),
                base_seed=7,
            ),
            TypeError,
            "num_shards must be an integer",
        ),
        (
            shards.ShardSpec(
                target_analysis="calibration",
                component="selector",
                profile="smoke",
                shard_index=cast(int, True),
                num_shards=1,
                base_seed=7,
            ),
            TypeError,
            "shard_index must be an integer",
        ),
        (
            shards.ShardSpec(
                target_analysis="calibration",
                component="selector",
                profile="smoke",
                shard_index=0,
                num_shards=1,
                base_seed=cast(int, True),
            ),
            TypeError,
            "base_seed must be an integer",
        ),
    ],
)
def test_invalid_shard_specs_are_rejected(
    spec: shards.ShardSpec,
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        shards.validate_shard_spec(spec)


def test_missing_duplicate_and_mixed_shard_groups_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing shard group"):
        shards._validate_complete_shard_group([], component="behavior")
    with pytest.raises(ValueError, match="inventory is incomplete"):
        shards._validate_complete_shard_group(
            [_verified_shard(0, 3), _verified_shard(2, 3)],
            component="behavior",
        )
    with pytest.raises(ValueError, match="duplicate shard indices"):
        shards._validate_complete_shard_group(
            [_verified_shard(0, 2), _verified_shard(0, 2)],
            component="behavior",
        )
    with pytest.raises(ValueError, match="disagree on num_shards"):
        shards._validate_complete_shard_group(
            [_verified_shard(0, 2), _verified_shard(1, 3)],
            component="behavior",
        )


def test_receipt_and_artifact_corruption_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_runtime_receipts(monkeypatch)
    spec = shards.ShardSpec(
        target_analysis="calibration",
        component="selector",
        profile="smoke",
        shard_index=0,
        num_shards=2,
        base_seed=7,
    )
    valid_dir = shards.run_shard_to_directory(spec, tmp_path / "valid" / "shard")
    verified = shards.discover_verified_shards(
        valid_dir.parent,
        target_analysis="calibration",
        profile="smoke",
        base_seed=7,
    )
    assert len(verified) == 1

    artifact_dir = tmp_path / "artifact-corruption" / "shard"
    shutil.copytree(valid_dir, artifact_dir)
    with (artifact_dir / "selector_null_raw.parquet").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="byte count differs|artifact hash differs"):
        shards.discover_verified_shards(
            artifact_dir.parent,
            target_analysis="calibration",
            profile="smoke",
            base_seed=7,
        )

    source_dir = tmp_path / "source-corruption" / "shard"
    shutil.copytree(valid_dir, source_dir)
    source_receipt_path = source_dir / "receipt.json"
    source_receipt = json.loads(source_receipt_path.read_text(encoding="ascii"))
    source_receipt["source_sha256"].pop(next(iter(source_receipt["source_sha256"])))
    source_receipt_path.write_text(
        json.dumps(source_receipt, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="source hash inventory"):
        shards.discover_verified_shards(
            source_dir.parent,
            target_analysis="calibration",
            profile="smoke",
            base_seed=7,
        )

    table_dir = tmp_path / "table-corruption" / "shard"
    shutil.copytree(valid_dir, table_dir)
    table_receipt_path = table_dir / "receipt.json"
    table_receipt = json.loads(table_receipt_path.read_text(encoding="ascii"))
    table_receipt["tables"]["selector_null_raw"]["rows"] += 1
    table_receipt_path.write_text(
        json.dumps(table_receipt, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="table metadata differs"):
        shards.discover_verified_shards(
            table_dir.parent,
            target_analysis="calibration",
            profile="smoke",
            base_seed=7,
        )

    spec_dir = tmp_path / "spec-corruption" / "shard"
    shutil.copytree(valid_dir, spec_dir)
    spec_receipt_path = spec_dir / "receipt.json"
    spec_receipt = json.loads(spec_receipt_path.read_text(encoding="ascii"))
    spec_receipt["spec"]["shard_index"] = True
    spec_receipt_path.write_text(
        json.dumps(spec_receipt, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    with pytest.raises(TypeError, match="shard_index must be an integer"):
        shards.discover_verified_shards(
            spec_dir.parent,
            target_analysis="calibration",
            profile="smoke",
            base_seed=7,
        )


def test_calibration_raw_corruption_and_missing_rows_are_rejected(
    calibration_smoke_raw: dict[str, pd.DataFrame],
) -> None:
    selector = calibration_smoke_raw["selector_null_raw"].copy(deep=True)
    selector.loc[0, "p_value"] = 2.0
    with pytest.raises(ValueError, match="permutation support"):
        calibration.validate_selector_null_raw(
            selector,
            "smoke",
            base_seed=7,
        )

    selector = calibration_smoke_raw["selector_null_raw"].copy(deep=True)
    selector.loc[0, "rejected"] = not bool(selector.loc[0, "rejected"])
    with pytest.raises(ValueError, match="rejection flags"):
        calibration.validate_selector_null_raw(
            selector,
            "smoke",
            base_seed=7,
        )

    root = calibration_smoke_raw["root_null_raw"].copy(deep=True)
    root.loc[0, "model_seed"] += 1
    with pytest.raises(ValueError, match="seeds differ"):
        calibration.validate_root_null_raw(
            root,
            "smoke",
            base_seed=7,
        )

    selector = calibration_smoke_raw["selector_null_raw"].iloc[:-1].copy()
    with pytest.raises(ValueError, match="inventory differs"):
        calibration.assemble_calibration_results(
            selector,
            calibration_smoke_raw["root_null_raw"],
            calibration_smoke_raw["cardinality_tree_raw"],
            calibration_smoke_raw["cardinality_forest_raw"],
            profile="smoke",
            base_seed=7,
        )

    without_cart = calibration_smoke_raw["cardinality_tree_raw"].loc[
        lambda frame: frame["method"].ne("cart")
    ]
    with pytest.raises(ValueError, match="inventory differs"):
        calibration.assemble_calibration_results(
            calibration_smoke_raw["selector_null_raw"],
            calibration_smoke_raw["root_null_raw"],
            without_cart,
            calibration_smoke_raw["cardinality_forest_raw"],
            profile="smoke",
            base_seed=7,
        )

    incomplete_forest = calibration_smoke_raw["cardinality_forest_raw"].iloc[:-1].copy()
    with pytest.raises(ValueError, match="inventory differs|retain all"):
        calibration.assemble_calibration_results(
            calibration_smoke_raw["selector_null_raw"],
            calibration_smoke_raw["root_null_raw"],
            calibration_smoke_raw["cardinality_tree_raw"],
            incomplete_forest,
            profile="smoke",
            base_seed=7,
        )


def test_merge_rejects_mixed_execution_contexts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_runtime_receipts(monkeypatch)
    shard_root = tmp_path / "mixed-context"
    for shard_index in range(2):
        spec = shards.ShardSpec(
            target_analysis="calibration",
            component="selector",
            profile="smoke",
            shard_index=shard_index,
            num_shards=2,
            base_seed=7,
        )
        shards.run_shard_to_directory(
            spec,
            shard_root / f"{shard_index:03d}",
        )

    changed_receipt_path = shard_root / "001" / "receipt.json"
    changed_receipt = json.loads(changed_receipt_path.read_text(encoding="ascii"))
    changed_receipt["platform"] = "different-worker-platform"
    context_payload = {
        key: changed_receipt[key]
        for key in (
            "git_sha",
            "git_dirty",
            "python",
            "platform",
            "r_environment",
            "source_sha256",
            "input_sha256",
            "versions",
        )
    }
    changed_receipt["execution_context_sha256"] = shards._json_sha256(context_payload)
    changed_receipt_path.write_text(
        json.dumps(changed_receipt, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="different execution contexts"):
        shards.discover_verified_shards(
            shard_root,
            target_analysis="calibration",
            profile="smoke",
            base_seed=7,
        )


def test_calibration_smoke_direct_results_equal_merged_shards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    calibration_smoke_raw: dict[str, pd.DataFrame],
) -> None:
    _patch_runtime_receipts(monkeypatch)
    direct = calibration.assemble_calibration_results(
        calibration_smoke_raw["selector_null_raw"],
        calibration_smoke_raw["root_null_raw"],
        calibration_smoke_raw["cardinality_tree_raw"],
        calibration_smoke_raw["cardinality_forest_raw"],
        profile="smoke",
        base_seed=7,
    )
    direct_output = tmp_path / "calibration-direct"
    calibration.write_results(
        direct,
        direct_output,
        profile="smoke",
        base_seed=7,
        elapsed_seconds=1.0,
    )

    shard_root = tmp_path / "calibration-shards"
    shard_counts = {
        "selector": 2,
        "root": 3,
        "cardinality_tree": 1,
        "cardinality_forest": 1,
    }
    execution_context = shards.capture_execution_context("calibration")
    for raw_component, num_shards in shard_counts.items():
        component = cast(shards.CalibrationComponent, raw_component)
        table_name = shards.CALIBRATION_TABLES[component]
        for shard_index in range(num_shards):
            spec = shards.ShardSpec(
                target_analysis="calibration",
                component=component,
                profile="smoke",
                shard_index=shard_index,
                num_shards=num_shards,
                base_seed=7,
            )
            replicates = shards.calibration_shard_replicates(spec)
            table = direct[table_name]
            subset = table[table["replicate"].isin(replicates)].copy()
            shards.write_shard(
                spec,
                {table_name: subset},
                shard_root / component / f"{shard_index:03d}",
                elapsed_seconds=1.0,
                execution_context=execution_context,
            )

    merged, verified = shards.merge_calibration_shards(
        shard_root,
        profile="smoke",
        base_seed=7,
    )
    assert len(verified) == sum(shard_counts.values())
    _assert_result_sets_equal(direct, merged)

    output_dir = shards.merge_shards_to_directory(
        "calibration",
        shard_root,
        tmp_path / "calibration-merged",
        profile="smoke",
        base_seed=7,
    )
    persisted = _read_parquet_results(output_dir, set(direct))
    _assert_result_sets_equal(direct, persisted)
    direct_persisted = _read_parquet_results(direct_output, set(direct))
    _assert_persisted_results_equal(direct_persisted, persisted)
    _assert_table_artifacts_identical(direct_output, output_dir)
    receipt = json.loads((output_dir / "receipt.json").read_text(encoding="ascii"))
    replicate._verify_child_output(
        replicate.ANALYSES[0],
        output_dir,
        profile="smoke",
        expected_git_sha=replicate._git_sha(),
        require_clean=False,
    )
    distributed = receipt["distributed_execution"]
    assert distributed["n_shards"] == sum(shard_counts.values())
    assert "paper/jss/replication/shards.py" in receipt["source_sha256"]
    for shard in distributed["shards"]:
        receipt_path = shard_root / shard["source_receipt"]
        assert shard["receipt_sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        receipt_artifact = output_dir / shard["receipt_artifact"]
        assert receipt_artifact.read_bytes() == receipt_path.read_bytes()


def test_behavior_smoke_direct_results_equal_merged_shards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_runtime_receipts(monkeypatch)
    monkeypatch.setattr(behavior, "load_behavior_datasets", _mock_behavior_datasets)
    monkeypatch.setattr(behavior, "_fit_method", _mock_behavior_fit)
    direct = behavior.run_behavior("smoke", base_seed=7)
    direct_output = tmp_path / "behavior-direct"
    behavior.write_results(
        direct,
        direct_output,
        profile="smoke",
        base_seed=7,
        elapsed_seconds=1.0,
    )

    shard_root = tmp_path / "behavior-shards"
    num_shards = len(behavior.behavior_cell_inventory("smoke"))
    for shard_index in range(num_shards):
        spec = shards.ShardSpec(
            target_analysis="behavior",
            component="behavior",
            profile="smoke",
            shard_index=shard_index,
            num_shards=num_shards,
            base_seed=7,
        )
        shards.run_shard_to_directory(
            spec,
            shard_root / f"{shard_index:03d}",
        )

    merged, verified = shards.merge_behavior_shards(
        shard_root,
        profile="smoke",
        base_seed=7,
    )
    assert len(verified) == num_shards
    _assert_result_sets_equal(direct, merged)

    output_dir = shards.merge_shards_to_directory(
        "behavior",
        shard_root,
        tmp_path / "behavior-merged",
        profile="smoke",
        base_seed=7,
    )
    persisted = _read_parquet_results(output_dir, set(direct))
    _assert_result_sets_equal(direct, persisted)
    direct_persisted = _read_parquet_results(direct_output, set(direct))
    _assert_persisted_results_equal(direct_persisted, persisted)
    _assert_table_artifacts_identical(direct_output, output_dir)
    receipt = json.loads((output_dir / "receipt.json").read_text(encoding="ascii"))
    assert receipt["distributed_execution"]["n_shards"] == num_shards
