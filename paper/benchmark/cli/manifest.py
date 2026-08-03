"""Commands for creating and verifying account-bound rerun manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, cast

import typer

from paper.benchmark.cli.console_output import console, error, heading, success
from paper.benchmark.pipeline.manifest import (
    parse_rerun_manifest,
    partition_rerun_manifest_by_account,
    verify_account_manifest_shards,
)
from paper.benchmark.pipeline.types import StageType

app = typer.Typer(
    name="manifest",
    help="Create and verify account-bound rerun manifests",
    no_args_is_help=True,
)


@app.command("shard")
def shard_manifest(
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Canonical account-bound rerun manifest",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            file_okay=False,
            resolve_path=True,
            help="Private directory for generated account shards",
        ),
    ],
) -> None:
    """Write one content-addressable manifest shard per bound AWS account."""
    canonical = parse_rerun_manifest(manifest_path.read_bytes())
    shards = partition_rerun_manifest_by_account(canonical)
    output_dir.mkdir(parents=True, exist_ok=True)

    heading("Account Manifest Shards")
    for account_id, payload in shards.items():
        path = output_dir / f"benchmark-rerun-account-{account_id}.csv"
        path.write_bytes(payload)
        shard = parse_rerun_manifest(payload)
        success(f"{account_id}: {len(shard.cells)} cells, sha256={shard.sha256}, path={path}")


@app.command("verify")
def verify_manifest_shards(
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Canonical account-bound rerun manifest",
        ),
    ],
    shard_paths: Annotated[
        list[Path],
        typer.Option(
            "--shard",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Account shard; pass once per account",
        ),
    ],
) -> None:
    """Prove shards are account-bound, disjoint, and exactly complete."""
    canonical = parse_rerun_manifest(manifest_path.read_bytes())
    payloads: dict[str, bytes] = {}
    for path in shard_paths:
        payload = path.read_bytes()
        shard = parse_rerun_manifest(payload)
        if len(shard.account_ids) != 1:
            raise typer.BadParameter(f"shard contains multiple account bindings: {path}")
        account_id = shard.account_ids[0]
        if account_id in payloads:
            raise typer.BadParameter(f"duplicate shard for account {account_id}")
        payloads[account_id] = payload

    counts = verify_account_manifest_shards(canonical, payloads)
    success(f"Exact disjoint union verified: {sum(counts.values())} cells across {counts}")


@app.command("reconcile")
def reconcile_manifest(
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Account-bound manifest for this artifact namespace",
        ),
    ],
    stage: Annotated[
        Literal["all", "rankings", "metrics"],
        typer.Option(
            "--stage",
            help="Artifact stage to reconcile",
        ),
    ] = "all",
) -> None:
    """Fail unless the selected manifest artifacts are exact and valid."""
    from paper.benchmark.adapters.store import S3Store
    from paper.benchmark.pipeline.reconcile import reconcile_manifest_artifacts
    from paper.benchmark.utils.env import (
        get_benchmark_scope,
        get_container_image,
        get_git_sha,
    )

    manifest = parse_rerun_manifest(manifest_path.read_bytes())
    store = S3Store.from_env(validate_uploads=True)
    selected_stages = None if stage == "all" else (cast(StageType, stage),)
    report = reconcile_manifest_artifacts(
        store,
        manifest,
        {
            **get_benchmark_scope(),
            "container_image": get_container_image(),
            "git_sha": get_git_sha(),
        },
        stages=selected_stages,
    )

    heading("Manifest Artifact Reconciliation")
    console.print(f"  S3 namespace: s3://{store.bucket}/{store.artifact_prefix}")
    console.print(f"  Manifest: {manifest.sha256}")
    for category, count in report.counts.items():
        console.print(f"  {category}: {count}")

    issue_groups = (
        ("missing", report.missing_keys),
        ("extra", report.extra_keys),
        ("malformed", report.malformed_keys),
        (
            "invalid",
            tuple(f"{issue.key}: {issue.detail}" for issue in report.invalid_artifacts),
        ),
        (
            "provenance mismatch",
            tuple(f"{issue.key}: {issue.detail}" for issue in report.provenance_mismatches),
        ),
    )
    for label, issues in issue_groups:
        if not issues:
            continue
        console.print(f"  {label} examples:")
        for issue in issues[:10]:
            console.print(f"    {issue}")
        if len(issues) > 10:
            console.print(f"    ... {len(issues) - 10} more")

    if not report.is_complete:
        error(f"Reconciliation failed with {report.issue_count} issue(s)")
        raise typer.Exit(1)
    success(f"Exact reconciliation passed for {len(report.valid_keys)} artifacts")
