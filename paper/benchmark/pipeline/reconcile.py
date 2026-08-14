"""Exact reconciliation between a rerun manifest and immutable artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import pandas as pd

from paper.benchmark.adapters.store import ArtifactReadError, LoadedArtifact
from paper.benchmark.pipeline.manifest import ManifestCell, RerunManifest
from paper.benchmark.pipeline.types import ExperimentConfig, StageType
from paper.benchmark.pipeline.validation import (
    ArtifactValidationError,
    validate_artifact_provenance,
    validate_expected_provenance,
    validate_metrics_artifact,
    validate_metrics_ranking_payload,
    validate_ranking_artifact,
)

_ARTIFACT_FILENAME_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]*__[0-9a-f]{16}_seed(?:0|[1-9][0-9]*)\.parquet$"
)
_STAGES: tuple[StageType, ...] = ("rankings", "metrics")
_TASKS = {"classification", "regression"}


class ReconciliationStore(Protocol):
    """Storage operations required by exact artifact reconciliation."""

    artifact_prefix: str

    def artifact_key(self, stage: StageType, config: ExperimentConfig) -> str:
        """Return the canonical key for one artifact."""
        ...

    def list_stage_keys(self, stage: StageType) -> tuple[str, ...]:
        """List every key below one artifact-stage prefix."""
        ...

    def load(self, stage: StageType, config: ExperimentConfig) -> pd.DataFrame:
        """Load one expected artifact."""
        ...

    def load_with_payload_sha256(
        self,
        stage: StageType,
        config: ExperimentConfig,
    ) -> LoadedArtifact:
        """Load one expected artifact and hash its exact stored bytes."""
        ...


@dataclass(frozen=True, order=True)
class ArtifactIssue:
    """One invalid expected artifact and its validation failure."""

    key: str
    detail: str


@dataclass(frozen=True)
class ReconciliationReport:
    """Deterministic result of exact manifest-to-storage reconciliation."""

    expected_keys: tuple[str, ...]
    valid_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    extra_keys: tuple[str, ...]
    malformed_keys: tuple[str, ...]
    invalid_artifacts: tuple[ArtifactIssue, ...]
    provenance_mismatches: tuple[ArtifactIssue, ...]

    @property
    def issue_count(self) -> int:
        """Return the number of non-valid reconciliation outcomes."""
        return (
            len(self.missing_keys)
            + len(self.extra_keys)
            + len(self.malformed_keys)
            + len(self.invalid_artifacts)
            + len(self.provenance_mismatches)
        )

    @property
    def is_complete(self) -> bool:
        """Return whether every expected artifact is valid and no extras exist."""
        return not self.issue_count and self.valid_keys == self.expected_keys

    @property
    def counts(self) -> dict[str, int]:
        """Return stable category counts for CLI and test reporting."""
        return {
            "expected": len(self.expected_keys),
            "valid": len(self.valid_keys),
            "missing": len(self.missing_keys),
            "extra": len(self.extra_keys),
            "malformed": len(self.malformed_keys),
            "invalid": len(self.invalid_artifacts),
            "provenance_mismatch": len(self.provenance_mismatches),
        }


def _normalize_stages(stages: Sequence[StageType] | None) -> tuple[StageType, ...]:
    if stages is None:
        return _STAGES
    selected = tuple(stages)
    if not selected:
        raise ValueError("at least one reconciliation stage is required")
    if len(set(selected)) != len(selected):
        raise ValueError("reconciliation stages must be unique")
    invalid = set(selected) - set(_STAGES)
    if invalid:
        raise ValueError(f"invalid reconciliation stages: {sorted(invalid)}")
    return tuple(stage for stage in _STAGES if stage in selected)


def _is_structurally_valid_key(
    key: str,
    *,
    artifact_prefix: str,
    stage: StageType,
) -> bool:
    prefix = f"{artifact_prefix}/{stage}/"
    if not key.startswith(prefix):
        return False
    parts = key[len(prefix) :].split("/")
    if len(parts) != 3:
        return False
    task, dataset, filename = parts
    if task not in _TASKS or not dataset or dataset in {".", ".."}:
        return False
    if any(ord(character) < 32 or ord(character) == 127 for character in dataset):
        return False
    return _ARTIFACT_FILENAME_PATTERN.fullmatch(filename) is not None


def _expected_artifacts(
    store: ReconciliationStore,
    manifest: RerunManifest,
    stages: tuple[StageType, ...],
    *,
    rankings_required_for: Literal["rankings", "metrics"],
) -> dict[str, tuple[StageType, ManifestCell]]:
    expected: dict[str, tuple[StageType, ManifestCell]] = {}
    for stage in stages:
        requirement_stage = rankings_required_for if stage == "rankings" else stage
        for cell in manifest.cells:
            if not cell.required_for(requirement_stage):
                continue
            key = store.artifact_key(stage, cell.config)
            if not _is_structurally_valid_key(
                key,
                artifact_prefix=store.artifact_prefix,
                stage=stage,
            ):
                raise ValueError(f"manifest produced a non-canonical artifact key: {key}")
            if key in expected:
                raise ValueError(f"manifest produced a duplicate artifact key: {key}")
            expected[key] = (stage, cell)
    return expected


def _validate_scope(
    store: ReconciliationStore,
    manifest: RerunManifest,
    expected_provenance: Mapping[str, str],
) -> dict[str, str]:
    provenance = validate_expected_provenance(expected_provenance)
    if provenance["artifact_prefix"] != store.artifact_prefix:
        raise ValueError(
            "store artifact prefix does not match expected provenance: "
            f"{store.artifact_prefix!r} != {provenance['artifact_prefix']!r}"
        )
    if provenance["manifest_sha256"] != manifest.sha256:
        raise ValueError(
            "manifest digest does not match expected provenance: "
            f"{manifest.sha256!r} != {provenance['manifest_sha256']!r}"
        )
    if provenance["campaign_sha256"] != manifest.campaign_sha256:
        raise ValueError(
            "campaign digest does not match expected provenance: "
            f"{manifest.campaign_sha256!r} != {provenance['campaign_sha256']!r}"
        )
    if provenance["runtime_contract_sha256"] != manifest.runtime_contract_sha256:
        raise ValueError(
            "runtime contract digest does not match expected provenance: "
            f"{manifest.runtime_contract_sha256!r} != "
            f"{provenance['runtime_contract_sha256']!r}"
        )
    if manifest.account_ids != (provenance["aws_account_id"],):
        raise ValueError(
            "manifest account binding does not match expected provenance: "
            f"{list(manifest.account_ids)} != {provenance['aws_account_id']!r}"
        )
    return provenance


def reconcile_manifest_artifacts(
    store: ReconciliationStore,
    manifest: RerunManifest,
    expected_provenance: Mapping[str, str],
    *,
    stages: Sequence[StageType] | None = None,
    rankings_required_for: Literal["rankings", "metrics"] = "rankings",
) -> ReconciliationReport:
    """Validate the exact artifact set and load only manifest-approved objects."""
    selected_stages = _normalize_stages(stages)
    provenance = _validate_scope(store, manifest, expected_provenance)
    expected = _expected_artifacts(
        store,
        manifest,
        selected_stages,
        rankings_required_for=rankings_required_for,
    )
    observed: set[str] = set()
    extra: set[str] = set()
    malformed: set[str] = set()

    for stage in selected_stages:
        for key in store.list_stage_keys(stage):
            if key in expected:
                observed.add(key)
            elif _is_structurally_valid_key(
                key,
                artifact_prefix=store.artifact_prefix,
                stage=stage,
            ):
                extra.add(key)
            else:
                malformed.add(key)

    missing = set(expected) - observed
    valid: list[str] = []
    invalid: list[ArtifactIssue] = []
    provenance_mismatches: list[ArtifactIssue] = []
    ranking_artifacts: dict[str, LoadedArtifact] = {}
    for key in sorted(observed):
        stage, cell = expected[key]
        config = cell.config
        ranking_key = store.artifact_key("rankings", config)
        ranking_frame: pd.DataFrame | None = None

        try:
            if stage == "rankings":
                loaded_rankings = ranking_artifacts.get(ranking_key)
                if loaded_rankings is None:
                    loaded_rankings = store.load_with_payload_sha256(
                        "rankings",
                        config,
                    )
                    ranking_artifacts[ranking_key] = loaded_rankings
                frame = loaded_rankings.frame
                validate_ranking_artifact(frame, config)
            else:
                frame = store.load("metrics", config)
                loaded_rankings = ranking_artifacts.get(ranking_key)
                if loaded_rankings is None:
                    loaded_rankings = store.load_with_payload_sha256(
                        "rankings",
                        config,
                    )
                    ranking_artifacts[ranking_key] = loaded_rankings
                ranking_frame = loaded_rankings.frame
                validate_ranking_artifact(ranking_frame, config)
                validate_metrics_artifact(frame, config)
                validate_metrics_ranking_payload(
                    frame,
                    loaded_rankings.payload_sha256,
                )
        except (ArtifactReadError, ArtifactValidationError, FileNotFoundError) as exc:
            invalid.append(ArtifactIssue(key=key, detail=str(exc)))
            continue

        try:
            if ranking_frame is not None:
                validate_artifact_provenance(
                    ranking_frame,
                    provenance,
                )
            validate_artifact_provenance(
                frame,
                provenance,
                include_ranking_reference=stage == "metrics",
            )
        except ArtifactValidationError as exc:
            provenance_mismatches.append(ArtifactIssue(key=key, detail=str(exc)))
            continue
        valid.append(key)

    return ReconciliationReport(
        expected_keys=tuple(sorted(expected)),
        valid_keys=tuple(valid),
        missing_keys=tuple(sorted(missing)),
        extra_keys=tuple(sorted(extra)),
        malformed_keys=tuple(sorted(malformed)),
        invalid_artifacts=tuple(sorted(invalid)),
        provenance_mismatches=tuple(sorted(provenance_mismatches)),
    )
