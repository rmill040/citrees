"""Build extension campaign manifests from a frozen benchmark inventory.

An extension campaign adds grid-alias cells (for example ``cit_maxt``) to a
completed benchmark without touching the completed cells. The output manifest
carries the full source inventory with both execution flags off, so the
reproducibility gate still sees its complete replacement scope, plus one new
cell per alias configuration, dataset, and seed whose base-method counterpart
completed. Counterparts listed in an exclusion table are skipped and recorded.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from paper.benchmark.pipeline.manifest import (
    ManifestCell,
    RerunManifest,
    compute_campaign_sha256,
    parse_rerun_manifest,
    partition_rerun_manifest_by_account,
    serialize_rerun_manifest,
    validate_canonical_campaign,
    verify_account_manifest_shards,
)
from paper.benchmark.pipeline.methods import base_method, get_full_method_configs
from paper.benchmark.pipeline.types import CellKey, ExperimentConfig, MethodConfig, TaskType

EXCLUSION_COLUMNS = ("task", "dataset", "method_id", "seed", "reason")
EXTENSION_RECEIPT_SCHEMA = "citrees-extension-manifest-v1"
ALIAS_ONLY_PARAMETERS = frozenset({"threshold_test"})


@dataclass(frozen=True)
class ExtensionExclusion:
    """One completed-benchmark cell whose extension counterpart must not run."""

    task: TaskType
    dataset: str
    method_id: str
    seed: int
    reason: str

    @property
    def key(self) -> CellKey:
        return (self.task, self.dataset, self.method_id, self.seed)


@dataclass(frozen=True)
class ExtensionBuild:
    """Cells of one extension campaign and how they were selected."""

    source_manifest_sha256: str
    methods: tuple[str, ...]
    rerun_reason: str
    cells: tuple[ManifestCell, ...]
    added: tuple[ManifestCell, ...]
    excluded: tuple[tuple[CellKey, str], ...]

    def added_counts(self) -> dict[str, dict[str, int]]:
        """Return added cell counts by task and method name."""
        counts: dict[str, Counter[str]] = {}
        for cell in self.added:
            counts.setdefault(cell.config.task, Counter())[cell.config.method.name] += 1
        return {task: dict(sorted(items.items())) for task, items in sorted(counts.items())}


def load_extension_exclusions(payload: bytes) -> tuple[ExtensionExclusion, ...]:
    """Parse the strict exclusion table."""
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
    if tuple(reader.fieldnames or ()) != EXCLUSION_COLUMNS:
        raise ValueError("exclusion columns must exactly match: " + ",".join(EXCLUSION_COLUMNS))
    rows: list[ExtensionExclusion] = []
    seen: set[CellKey] = set()
    for line_number, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"exclusion line {line_number}: malformed CSV row")
        task = row["task"]
        if task not in {"classification", "regression"}:
            raise ValueError(f"exclusion line {line_number}: invalid task {task!r}")
        try:
            seed = int(row["seed"])
        except ValueError as exc:
            raise ValueError(f"exclusion line {line_number}: seed must be an integer") from exc
        reason = row["reason"]
        if not reason or reason.strip() != reason:
            raise ValueError(f"exclusion line {line_number}: reason must be non-empty and trimmed")
        exclusion = ExtensionExclusion(
            task=task,  # type: ignore[arg-type]
            dataset=row["dataset"],
            method_id=row["method_id"],
            seed=seed,
            reason=reason,
        )
        if exclusion.key in seen:
            raise ValueError(f"exclusion line {line_number}: duplicate cell {exclusion.key!r}")
        seen.add(exclusion.key)
        rows.append(exclusion)
    return tuple(rows)


def counterpart_label(config: MethodConfig, base_name: str) -> str:
    """Return the identity of the base-method configuration an alias config mirrors."""
    params = {k: v for k, v in config.params_dict.items() if k not in ALIAS_ONLY_PARAMETERS}
    return MethodConfig(name=base_name, params=tuple(sorted(params.items()))).label


def build_extension_cells(
    source: RerunManifest,
    *,
    methods: Sequence[str],
    rerun_reason: str,
    target_aws_account_id: str,
    exclusions: Sequence[ExtensionExclusion] = (),
) -> ExtensionBuild:
    """Carry the source inventory and add one cell per alias configuration and seed."""
    method_names = tuple(dict.fromkeys(methods))
    if not method_names:
        raise ValueError("extension methods must not be empty")
    for name in method_names:
        if base_method(name) == name:
            raise ValueError(
                f"{name!r} is not a grid alias; extension methods must resolve to a base method"
            )
    if not rerun_reason or rerun_reason.strip() != rerun_reason:
        raise ValueError("rerun_reason must be non-empty and trimmed")

    carried = tuple(
        replace(cell, stage1_required=False, stage2_required=False) for cell in source.cells
    )
    carried_keys = {cell.identity for cell in carried}
    # Counterpart cells indexed by (task, base label): the alias cell inherits the
    # dataset, identity, source, and seed of exactly one completed counterpart.
    counterparts: dict[tuple[TaskType, str], list[ManifestCell]] = {}
    for cell in source.cells:
        counterparts.setdefault((cell.config.task, cell.config.method.label), []).append(cell)

    exclusion_reasons = {exclusion.key: exclusion.reason for exclusion in exclusions}
    consumed: set[CellKey] = set()
    added: list[ManifestCell] = []
    excluded: list[tuple[CellKey, str]] = []
    for task in ("classification", "regression"):
        if not any(cell.config.task == task for cell in source.cells):
            continue
        for name in method_names:
            base_name = base_method(name)
            for config in get_full_method_configs([name], task):
                counterpart = counterpart_label(config, base_name)
                paired = counterparts.get((task, counterpart))
                if not paired:
                    raise ValueError(
                        f"{task} {config.label} has no {base_name} counterpart {counterpart} "
                        "in the source manifest; the grids have drifted"
                    )
                for source_cell in sorted(paired, key=lambda cell: cell.identity):
                    experiment = ExperimentConfig(
                        method=config,
                        dataset=source_cell.config.dataset,
                        seed=source_cell.config.seed,
                        task=task,
                        dataset_identity=source_cell.config.dataset_identity,
                    )
                    reason = exclusion_reasons.get(source_cell.identity)
                    if reason is not None:
                        consumed.add(source_cell.identity)
                        excluded.append((experiment.key, reason))
                        continue
                    if experiment.key in carried_keys:
                        raise ValueError(f"source manifest already contains {experiment.key!r}")
                    added.append(
                        ManifestCell(
                            config=experiment,
                            target_aws_account_id=target_aws_account_id,
                            dataset_source=source_cell.dataset_source,
                            rerun_reason=rerun_reason,
                            historically_omitted=False,
                            stage1_required=True,
                            stage2_required=True,
                        )
                    )
    unused = sorted(set(exclusion_reasons) - consumed)
    if unused:
        raise ValueError(f"exclusions did not match any extension cell: {unused[:5]}")
    if not added:
        raise ValueError("extension produced no cells")

    cells = tuple(sorted((*carried, *added), key=lambda cell: cell.identity))
    return ExtensionBuild(
        source_manifest_sha256=source.sha256,
        methods=method_names,
        rerun_reason=rerun_reason,
        cells=cells,
        added=tuple(sorted(added, key=lambda cell: cell.identity)),
        excluded=tuple(sorted(excluded)),
    )


def write_extension_campaign(
    build: ExtensionBuild,
    *,
    runtime_contract_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Serialize, re-validate, shard, and write one extension campaign."""
    campaign_sha256 = compute_campaign_sha256(
        build.cells,
        runtime_contract_sha256=runtime_contract_sha256,
    )
    payload = serialize_rerun_manifest(
        build.cells,
        campaign_sha256=campaign_sha256,
        runtime_contract_sha256=runtime_contract_sha256,
    )
    manifest = parse_rerun_manifest(payload)
    validate_canonical_campaign(manifest)
    shards = partition_rerun_manifest_by_account(manifest)
    shard_counts = verify_account_manifest_shards(manifest, shards)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.csv").write_bytes(payload)
    for account_id, shard_payload in shards.items():
        (output_dir / f"account-{account_id}.csv").write_bytes(shard_payload)
    receipt: dict[str, Any] = {
        "schema": EXTENSION_RECEIPT_SCHEMA,
        "source_manifest_sha256": build.source_manifest_sha256,
        "runtime_contract_sha256": runtime_contract_sha256,
        "campaign_sha256": manifest.campaign_sha256,
        "manifest_sha256": manifest.sha256,
        "methods": list(build.methods),
        "rerun_reason": build.rerun_reason,
        "cells": len(manifest.cells),
        "added": len(build.added),
        "added_counts": build.added_counts(),
        "excluded": [
            {
                "task": key[0],
                "dataset": key[1],
                "method_id": key[2],
                "seed": key[3],
                "reason": reason,
            }
            for key, reason in build.excluded
        ],
        "stage1_required": sum(cell.stage1_required for cell in manifest.cells),
        "stage2_required": sum(cell.stage2_required for cell in manifest.cells),
        "shard_counts": shard_counts,
    }
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
