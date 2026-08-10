"""Strict parsing and partitioning for corrected benchmark manifests."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from paper.benchmark.pipeline.runtime_contract import (
    validate_runtime_contract_sha256,
)
from paper.benchmark.pipeline.types import (
    CellKey,
    DatasetIdentity,
    ExperimentConfig,
    MethodConfig,
    StageType,
    TaskType,
)

DataSource = Literal["real", "synthetic"]

MANIFEST_COLUMNS = (
    "task",
    "target_aws_account_id",
    "campaign_sha256",
    "runtime_contract_sha256",
    "dataset_source",
    "dataset",
    "dataset_sha256",
    "dataset_n_samples",
    "dataset_n_features",
    "method_base",
    "method_id",
    "method_params_json",
    "seed",
    "rerun_reason",
    "historically_omitted",
    "stage1_required",
    "stage2_required",
    "status",
)
MANIFEST_S3_PREFIX = "rerun-manifests"
CANONICAL_MANIFEST_S3_PREFIX = "canonical-rerun-manifests"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_AWS_ACCOUNT_ID_PATTERN = re.compile(r"^[0-9]{12}$")
_DATASET_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SCALAR_TYPES = (str, int, float, bool, type(None))


@dataclass(frozen=True)
class ManifestCell:
    """One approved benchmark cell and its execution requirements."""

    config: ExperimentConfig
    target_aws_account_id: str
    dataset_source: DataSource
    rerun_reason: str
    historically_omitted: bool
    stage1_required: bool
    stage2_required: bool

    @property
    def identity(self) -> CellKey:
        """Return the globally unique cell identity."""
        return self.config.key

    def required_for(self, stage: StageType) -> bool:
        """Return whether this cell is required for the requested stage."""
        return self.stage1_required if stage == "rankings" else self.stage2_required


@dataclass(frozen=True)
class RerunManifest:
    """Validated immutable benchmark inventory."""

    sha256: str
    campaign_sha256: str
    runtime_contract_sha256: str
    cells: tuple[ManifestCell, ...]

    def configs_for(self, task: TaskType, stage: StageType) -> tuple[ExperimentConfig, ...]:
        """Return approved configurations for one task and stage."""
        return tuple(
            cell.config
            for cell in self.cells
            if cell.config.task == task and cell.required_for(stage)
        )

    @property
    def method_names(self) -> tuple[str, ...]:
        """Return method names in first-occurrence order."""
        return tuple(dict.fromkeys(cell.config.method.name for cell in self.cells))

    @property
    def account_ids(self) -> tuple[str, ...]:
        """Return the AWS accounts targeted by this manifest."""
        return tuple(sorted({cell.target_aws_account_id for cell in self.cells}))

    def method_counts(self, stage: StageType) -> dict[str, int]:
        """Return required cell counts by method for one stage."""
        counts = Counter(cell.config.method.name for cell in self.cells if cell.required_for(stage))
        return dict(sorted(counts.items()))


def validate_manifest_sha256(value: str) -> str:
    """Validate a lowercase hexadecimal SHA-256 digest."""
    normalized = value.strip()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError("manifest SHA-256 must be 64 lowercase hexadecimal characters")
    return normalized


def manifest_s3_key(sha256: str) -> str:
    """Return the content-addressed private S3 key for an account manifest."""
    return f"{MANIFEST_S3_PREFIX}/{validate_manifest_sha256(sha256)}.csv"


def canonical_manifest_s3_key(sha256: str) -> str:
    """Return the content-addressed private S3 key for a canonical manifest."""
    return f"{CANONICAL_MANIFEST_S3_PREFIX}/{validate_manifest_sha256(sha256)}.csv"


def compute_campaign_sha256(
    cells: tuple[ManifestCell, ...],
    *,
    runtime_contract_sha256: str,
) -> str:
    """Hash the complete canonical campaign independently of shard files."""
    if not cells:
        raise ValueError("campaign must contain at least one cell")
    runtime_contract_sha256 = validate_runtime_contract_sha256(runtime_contract_sha256)
    records: list[dict[str, Any]] = []
    for cell in sorted(cells, key=lambda value: value.identity):
        config = cell.config
        records.append(
            {
                "dataset": config.dataset,
                "dataset_n_features": config.dataset_identity.n_features,
                "dataset_n_samples": config.dataset_identity.n_samples,
                "dataset_sha256": config.dataset_identity.sha256,
                "dataset_source": cell.dataset_source,
                "historically_omitted": cell.historically_omitted,
                "method_base": config.method.name,
                "method_id": config.method.label,
                "method_params_json": json.dumps(
                    config.method.params_dict,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                "rerun_reason": cell.rerun_reason,
                "seed": config.seed,
                "stage1_required": cell.stage1_required,
                "stage2_required": cell.stage2_required,
                "target_aws_account_id": cell.target_aws_account_id,
                "task": config.task,
            }
        )
    payload = json.dumps(
        {
            "runtime_contract_sha256": runtime_contract_sha256,
            "cells": records,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_canonical_campaign(manifest: RerunManifest) -> None:
    """Require a full manifest's campaign digest to match all of its cells."""
    observed = compute_campaign_sha256(
        manifest.cells,
        runtime_contract_sha256=manifest.runtime_contract_sha256,
    )
    if manifest.campaign_sha256 != observed:
        raise ValueError(
            "canonical campaign SHA-256 mismatch: "
            f"declared {manifest.campaign_sha256}, observed {observed}"
        )


def _parse_bool(value: str, *, field: str, line_number: int) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"manifest line {line_number}: {field} must be True or False")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r} is not allowed")


def _params_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, value in pairs:
        if key in params:
            raise ValueError(f"duplicate method parameter {key!r}")
        if not isinstance(value, _SCALAR_TYPES):
            raise ValueError(f"method parameter {key!r} must be a JSON scalar")
        params[key] = value
    return params


def _parse_params(value: str, *, line_number: int) -> dict[str, Any]:
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_params_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"manifest line {line_number}: invalid method_params_json: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"manifest line {line_number}: method_params_json must be an object")
    canonical = json.dumps(
        parsed,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if value != canonical:
        raise ValueError(f"manifest line {line_number}: method_params_json must use canonical JSON")
    return parsed


def parse_rerun_manifest(
    payload: bytes,
    *,
    expected_sha256: str | None = None,
) -> RerunManifest:
    """Parse and fully validate one private corrected-benchmark manifest."""
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != validate_manifest_sha256(expected_sha256):
        raise ValueError(
            f"manifest SHA-256 mismatch: expected {expected_sha256}, observed {digest}"
        )

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("manifest must be UTF-8 encoded") from exc
    if text.startswith("\ufeff"):
        raise ValueError("manifest must not contain a UTF-8 byte-order mark")

    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
        raise ValueError("manifest columns must exactly match: " + ",".join(MANIFEST_COLUMNS))

    from paper.benchmark.config.constants import N_SEEDS
    from paper.benchmark.pipeline.methods import get_full_method_configs, get_methods

    methods_by_scope: dict[tuple[TaskType, str], dict[str, MethodConfig]] = {}
    identities: set[CellKey] = set()
    dataset_identities: dict[tuple[TaskType, DataSource, str], DatasetIdentity] = {}
    campaign_sha256: str | None = None
    runtime_contract_sha256: str | None = None
    cells: list[ManifestCell] = []

    for line_number, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"manifest line {line_number}: malformed CSV row")

        task_raw = row["task"]
        if task_raw not in {"classification", "regression"}:
            raise ValueError(f"manifest line {line_number}: invalid task {task_raw!r}")
        task = cast(TaskType, task_raw)

        target_aws_account_id = row["target_aws_account_id"]
        if not _AWS_ACCOUNT_ID_PATTERN.fullmatch(target_aws_account_id):
            raise ValueError(
                f"manifest line {line_number}: target_aws_account_id must contain 12 decimal digits"
            )

        row_campaign_sha256 = validate_manifest_sha256(row["campaign_sha256"])
        if campaign_sha256 is None:
            campaign_sha256 = row_campaign_sha256
        elif campaign_sha256 != row_campaign_sha256:
            raise ValueError(
                f"manifest line {line_number}: campaign_sha256 conflicts with an earlier row"
            )

        row_runtime_contract_sha256 = validate_runtime_contract_sha256(
            row["runtime_contract_sha256"]
        )
        if runtime_contract_sha256 is None:
            runtime_contract_sha256 = row_runtime_contract_sha256
        elif runtime_contract_sha256 != row_runtime_contract_sha256:
            raise ValueError(
                f"manifest line {line_number}: runtime_contract_sha256 "
                "conflicts with an earlier row"
            )

        source_raw = row["dataset_source"]
        if source_raw not in {"real", "synthetic"}:
            raise ValueError(f"manifest line {line_number}: invalid dataset_source {source_raw!r}")
        source = cast(DataSource, source_raw)

        dataset = row["dataset"]
        if not _DATASET_PATTERN.fullmatch(dataset):
            raise ValueError(f"manifest line {line_number}: dataset must use a safe canonical name")
        expected_source: DataSource = "synthetic" if "synthetic" in dataset else "real"
        if source != expected_source:
            raise ValueError(
                f"manifest line {line_number}: dataset_source {source!r} "
                f"does not match dataset {dataset!r}"
            )
        try:
            dataset_identity = DatasetIdentity(
                sha256=row["dataset_sha256"],
                n_samples=int(row["dataset_n_samples"]),
                n_features=int(row["dataset_n_features"]),
            )
        except ValueError as exc:
            raise ValueError(
                f"manifest line {line_number}: invalid dataset identity: {exc}"
            ) from exc
        if str(dataset_identity.n_samples) != row["dataset_n_samples"]:
            raise ValueError(
                f"manifest line {line_number}: dataset_n_samples must be a canonical integer"
            )
        if str(dataset_identity.n_features) != row["dataset_n_features"]:
            raise ValueError(
                f"manifest line {line_number}: dataset_n_features must be a canonical integer"
            )
        dataset_identity_key = (task, source, dataset)
        previous_identity = dataset_identities.setdefault(dataset_identity_key, dataset_identity)
        if previous_identity != dataset_identity:
            raise ValueError(
                f"manifest line {line_number}: dataset identity conflicts with an earlier row"
            )

        method_name = row["method_base"]
        if method_name not in get_methods(task):
            raise ValueError(
                f"manifest line {line_number}: method {method_name!r} is unavailable for {task}"
            )
        method_scope = (task, method_name)
        if method_scope not in methods_by_scope:
            method_configs = get_full_method_configs([method_name], task)
            methods_by_scope[method_scope] = {
                method_config.label: method_config for method_config in method_configs
            }

        params = _parse_params(row["method_params_json"], line_number=line_number)
        method_id = row["method_id"]
        method = methods_by_scope[method_scope].get(method_id)
        if method is None or method.params_dict != params:
            raise ValueError(
                f"manifest line {line_number}: method identity is not in the current grid"
            )

        seed_raw = row["seed"]
        try:
            seed = int(seed_raw)
        except ValueError as exc:
            raise ValueError(f"manifest line {line_number}: seed must be an integer") from exc
        if str(seed) != seed_raw or seed < 0 or seed >= N_SEEDS:
            raise ValueError(f"manifest line {line_number}: seed must be in 0..{N_SEEDS - 1}")

        rerun_reason = row["rerun_reason"]
        if not rerun_reason or rerun_reason.strip() != rerun_reason:
            raise ValueError(
                f"manifest line {line_number}: rerun_reason must be non-empty and trimmed"
            )
        historically_omitted = _parse_bool(
            row["historically_omitted"],
            field="historically_omitted",
            line_number=line_number,
        )
        stage1_required = _parse_bool(
            row["stage1_required"],
            field="stage1_required",
            line_number=line_number,
        )
        stage2_required = _parse_bool(
            row["stage2_required"],
            field="stage2_required",
            line_number=line_number,
        )
        if row["status"] != "pending":
            raise ValueError(f"manifest line {line_number}: status must be pending")

        config = ExperimentConfig(
            method=method,
            dataset=dataset,
            seed=seed,
            task=task,
            dataset_identity=dataset_identity,
        )
        cell = ManifestCell(
            config=config,
            target_aws_account_id=target_aws_account_id,
            dataset_source=source,
            rerun_reason=rerun_reason,
            historically_omitted=historically_omitted,
            stage1_required=stage1_required,
            stage2_required=stage2_required,
        )
        if cell.identity in identities:
            raise ValueError(
                f"manifest line {line_number}: duplicate cell identity {cell.identity!r}"
            )
        identities.add(cell.identity)
        cells.append(cell)

    if not cells:
        raise ValueError("manifest must contain at least one cell")
    assert campaign_sha256 is not None
    assert runtime_contract_sha256 is not None
    return RerunManifest(
        sha256=digest,
        campaign_sha256=campaign_sha256,
        runtime_contract_sha256=runtime_contract_sha256,
        cells=tuple(cells),
    )


def serialize_rerun_manifest(
    cells: tuple[ManifestCell, ...],
    *,
    campaign_sha256: str,
    runtime_contract_sha256: str,
) -> bytes:
    """Serialize cells as a canonical manifest CSV."""
    if not cells:
        raise ValueError("manifest shard must contain at least one cell")
    campaign_sha256 = validate_manifest_sha256(campaign_sha256)
    runtime_contract_sha256 = validate_runtime_contract_sha256(runtime_contract_sha256)

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for cell in sorted(cells, key=lambda value: value.identity):
        config = cell.config
        writer.writerow(
            {
                "task": config.task,
                "target_aws_account_id": cell.target_aws_account_id,
                "campaign_sha256": campaign_sha256,
                "runtime_contract_sha256": runtime_contract_sha256,
                "dataset_source": cell.dataset_source,
                "dataset": config.dataset,
                "dataset_sha256": config.dataset_identity.sha256,
                "dataset_n_samples": config.dataset_identity.n_samples,
                "dataset_n_features": config.dataset_identity.n_features,
                "method_base": config.method.name,
                "method_id": config.method.label,
                "method_params_json": json.dumps(
                    config.method.params_dict,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                "seed": config.seed,
                "rerun_reason": cell.rerun_reason,
                "historically_omitted": str(cell.historically_omitted),
                "stage1_required": str(cell.stage1_required),
                "stage2_required": str(cell.stage2_required),
                "status": "pending",
            }
        )
    return output.getvalue().encode("utf-8")


def partition_rerun_manifest_by_account(
    manifest: RerunManifest,
) -> dict[str, bytes]:
    """Serialize one deterministic shard for each account bound in the manifest."""
    validate_canonical_campaign(manifest)
    cells_by_account: dict[str, list[ManifestCell]] = defaultdict(list)
    for cell in manifest.cells:
        cells_by_account[cell.target_aws_account_id].append(cell)
    payloads = {
        account_id: serialize_rerun_manifest(
            tuple(cells),
            campaign_sha256=manifest.campaign_sha256,
            runtime_contract_sha256=manifest.runtime_contract_sha256,
        )
        for account_id, cells in sorted(cells_by_account.items())
    }
    verify_account_manifest_shards(manifest, payloads)
    return payloads


def account_manifest_sha256_map(manifest: RerunManifest) -> dict[str, str]:
    """Return the exact content digest of each deterministic account shard."""
    return {
        account_id: hashlib.sha256(payload).hexdigest()
        for account_id, payload in partition_rerun_manifest_by_account(manifest).items()
    }


def verify_account_manifest_shard(
    canonical: RerunManifest,
    *,
    account_id: str,
    payload: bytes,
) -> RerunManifest:
    """Require one byte-exact deterministic shard of a canonical manifest."""
    expected_payload = partition_rerun_manifest_by_account(canonical).get(account_id)
    if expected_payload is None:
        raise ValueError(f"account {account_id!r} is outside the canonical manifest")
    if payload != expected_payload:
        expected_sha256 = hashlib.sha256(expected_payload).hexdigest()
        observed_sha256 = hashlib.sha256(payload).hexdigest()
        raise ValueError(
            f"account {account_id} manifest shard differs from the canonical "
            f"partition: expected {expected_sha256}, observed {observed_sha256}"
        )
    return parse_rerun_manifest(
        payload,
        expected_sha256=hashlib.sha256(expected_payload).hexdigest(),
    )


def verify_account_manifest_shards(
    canonical: RerunManifest,
    shard_payloads: Mapping[str, bytes],
) -> dict[str, int]:
    """Prove account shards form an exact disjoint copy of the canonical inventory."""
    validate_canonical_campaign(canonical)
    if set(shard_payloads) != set(canonical.account_ids):
        raise ValueError(
            "shard accounts do not match canonical accounts: "
            f"expected {list(canonical.account_ids)}, observed {sorted(shard_payloads)}"
        )

    expected = {cell.identity: cell for cell in canonical.cells}
    observed: dict[CellKey, ManifestCell] = {}
    counts: dict[str, int] = {}
    for account_id, payload in sorted(shard_payloads.items()):
        shard = parse_rerun_manifest(payload)
        if shard.campaign_sha256 != canonical.campaign_sha256:
            raise ValueError(
                f"shard {account_id} campaign SHA-256 differs from the canonical campaign"
            )
        if shard.runtime_contract_sha256 != canonical.runtime_contract_sha256:
            raise ValueError(
                f"shard {account_id} runtime contract SHA-256 differs from the canonical campaign"
            )
        if shard.account_ids != (account_id,):
            raise ValueError(
                f"shard {account_id} contains account bindings {list(shard.account_ids)}"
            )
        counts[account_id] = len(shard.cells)
        for cell in shard.cells:
            if cell.identity in observed:
                raise ValueError(f"manifest shards overlap at cell {cell.identity!r}")
            observed[cell.identity] = cell

    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        changed = sorted(
            identity
            for identity in set(expected) & set(observed)
            if expected[identity] != observed[identity]
        )
        raise ValueError(
            "manifest shards do not exactly reproduce the canonical inventory: "
            f"missing={missing[:5]}, extra={extra[:5]}, changed={changed[:5]}"
        )
    return counts
