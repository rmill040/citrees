"""Commands for creating and verifying account-bound rerun manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, cast

import typer

from paper.benchmark.cli.console_output import console, error, heading, step, success
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


@app.command("extend")
def extend_manifest(
    source_path: Annotated[
        Path,
        typer.Option(
            "--source",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Completed canonical manifest whose inventory is carried forward",
        ),
    ],
    runtime_contract_path: Annotated[
        Path,
        typer.Option(
            "--runtime-contract",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Frozen runtime contract the new campaign binds",
        ),
    ],
    methods: Annotated[
        str,
        typer.Option(
            "--methods", help="Comma-separated grid aliases, for example cit_maxt,cif_maxt"
        ),
    ],
    rerun_reason: Annotated[
        str,
        typer.Option("--rerun-reason", help="Reason recorded on every added cell"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            file_okay=False,
            resolve_path=True,
            help="Directory for manifest.csv, account shards, and receipt.json",
        ),
    ],
    exclusions_path: Annotated[
        Path | None,
        typer.Option(
            "--exclusions",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Completed-benchmark cells whose extension counterparts are skipped",
        ),
    ] = None,
) -> None:
    """Add grid-alias cells to a completed inventory and write the campaign files."""
    from paper.benchmark.pipeline.extension_manifest import (
        build_extension_cells,
        load_extension_exclusions,
        write_extension_campaign,
    )
    from paper.benchmark.pipeline.runtime_contract import (
        parse_runtime_contract,
        runtime_contract_sha256,
    )

    source = parse_rerun_manifest(source_path.read_bytes())
    if len(source.account_ids) != 1:
        raise typer.BadParameter("source manifest must bind exactly one AWS account")
    contract_sha256 = runtime_contract_sha256(
        parse_runtime_contract(runtime_contract_path.read_bytes())
    )
    exclusions = (
        load_extension_exclusions(exclusions_path.read_bytes())
        if exclusions_path is not None
        else ()
    )
    build = build_extension_cells(
        source,
        methods=[m.strip() for m in methods.split(",") if m.strip()],
        rerun_reason=rerun_reason,
        target_aws_account_id=source.account_ids[0],
        exclusions=exclusions,
    )
    receipt = write_extension_campaign(
        build,
        runtime_contract_sha256=contract_sha256,
        output_dir=output_dir,
    )

    heading("Extension Campaign Manifest")
    step(f"Source inventory: {receipt['source_manifest_sha256']} ({len(source.cells)} cells)")
    step(f"Runtime contract: {receipt['runtime_contract_sha256']}")
    step(f"Added: {receipt['added']} cells {receipt['added_counts']}")
    step(f"Excluded: {len(receipt['excluded'])} cells")
    step(f"Campaign: {receipt['campaign_sha256']}")
    step(f"Manifest: {receipt['manifest_sha256']}")
    success(f"Wrote manifest.csv, account shards, and receipt.json to {output_dir}")


def _read_cell_lines(path: Path) -> set[tuple[str, str, str, int]]:
    """Parse ``task/dataset/method_id_seedN`` lines (one ranking artifact stem per line)."""
    cells: set[tuple[str, str, str, int]] = set()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        task, dataset, stem = line.split("/")
        method_id, seed = stem.rsplit("_seed", 1)
        cells.add((task, dataset, method_id, int(seed)))
    return cells


@app.command("stage2")
def stage2_manifest(
    source_path: Annotated[
        Path,
        typer.Option(
            "--source",
            exists=True,
            dir_okay=False,
            resolve_path=True,
            help="Stage 1 canonical manifest",
        ),
    ],
    completed_path: Annotated[
        Path,
        typer.Option(
            "--completed",
            exists=True,
            dir_okay=False,
            resolve_path=True,
            help="File of completed ranking cells, one task/dataset/method_id_seedN per line",
        ),
    ],
    runtime_contract_path: Annotated[
        Path,
        typer.Option("--runtime-contract", exists=True, dir_okay=False, resolve_path=True),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False, resolve_path=True)],
) -> None:
    """Write the Stage 2 campaign: Stage 1 off, metrics required only where a ranking exists."""
    from paper.benchmark.pipeline.extension_manifest import stage2_cells, write_stage2_campaign
    from paper.benchmark.pipeline.runtime_contract import (
        parse_runtime_contract,
        runtime_contract_sha256,
    )

    source = parse_rerun_manifest(source_path.read_bytes())
    completed = _read_cell_lines(completed_path)
    cells = stage2_cells(source, completed)  # type: ignore[arg-type]
    receipt = write_stage2_campaign(
        cells,
        source_manifest_sha256=source.sha256,
        runtime_contract_sha256=runtime_contract_sha256(
            parse_runtime_contract(runtime_contract_path.read_bytes())
        ),
        output_dir=output_dir,
    )
    heading("Stage 2 Campaign Manifest")
    step(f"Source: {receipt['source_manifest_sha256']} ({len(source.cells)} cells)")
    step(f"Stage 2 required: {receipt['stage2_required']} of {receipt['cells']} cells")
    step(f"Campaign: {receipt['campaign_sha256']}")
    step(f"Manifest: {receipt['manifest_sha256']}")
    success(f"Wrote manifest.csv, account shards, and receipt.json to {output_dir}")


@app.command("materialize-rankings")
def materialize_rankings(
    source_prefix: Annotated[str, typer.Option("--source-prefix", help="Stage 1 artifact prefix")],
    target_manifest_path: Annotated[
        Path, typer.Option("--target-manifest", exists=True, dir_okay=False, resolve_path=True)
    ],
    target_prefix: Annotated[str, typer.Option("--target-prefix", help="Stage 2 artifact prefix")],
    canonical_manifest_path: Annotated[
        Path, typer.Option("--canonical-manifest", exists=True, dir_okay=False, resolve_path=True)
    ],
    gate_receipt_sha256: Annotated[str, typer.Option("--gate-receipt-sha256")],
    image_uri: Annotated[str, typer.Option("--image-uri")],
    git_sha: Annotated[str, typer.Option("--git-sha")],
) -> None:
    """Copy Stage 1 rankings into the Stage 2 prefix with exact provenance rewriting."""
    import hashlib

    import boto3
    import pandas as pd

    from paper.benchmark.infra.aws import get_aws_account_id, get_resource_name
    from paper.benchmark.pipeline.materialize import (
        RankingSource,
        materialize_canonical_rankings,
    )

    account_id = get_aws_account_id()
    bucket = get_resource_name(account_id)
    s3 = boto3.client("s3")
    target = parse_rerun_manifest(target_manifest_path.read_bytes())
    canonical = parse_rerun_manifest(canonical_manifest_path.read_bytes())
    required = [cell for cell in target.cells if cell.stage2_required]
    heading("Materialize Stage 1 rankings for Stage 2")
    step(f"{len(required)} required cells; source prefix {source_prefix}")

    sources: list[RankingSource] = []
    source_provenance: dict[str, str] | None = None
    for index, cell in enumerate(required, start=1):
        config = cell.config
        key = (
            f"{source_prefix}/rankings/{config.task}/{config.dataset}/"
            f"{config.method.label}_seed{config.seed}.parquet"
        )
        head = s3.head_object(Bucket=bucket, Key=key)
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        if source_provenance is None:
            frame = pd.read_parquet(__import__("io").BytesIO(body))
            row = frame.iloc[0]
            source_provenance = {
                "artifact_prefix": str(row["artifact_prefix"]),
                "aws_account_id": account_id,
                "campaign_sha256": str(row["campaign_sha256"]),
                "canonical_manifest_sha256": str(row["canonical_manifest_sha256"]),
                "container_image": str(row["container_image"]),
                "gate_receipt_sha256": str(row["gate_receipt_sha256"]),
                "git_sha": str(row["git_sha"]),
                "manifest_sha256": str(row["manifest_sha256"]),
                "runtime_contract_sha256": str(row["runtime_contract_sha256"]),
            }
            step(f"Source provenance: campaign {source_provenance['campaign_sha256'][:12]}...")
        sources.append(
            RankingSource(
                cell_key=cell.identity,
                source_aws_account_id=account_id,
                bucket=bucket,
                key=key,
                version_id=head.get("VersionId")
                if head.get("VersionId") not in (None, "null")
                else None,
                payload_sha256=hashlib.sha256(body).hexdigest(),
                expected_provenance=source_provenance,
            )
        )
        if index % 200 == 0:
            step(f"  prepared {index}/{len(required)}")
    target_provenance = {
        "artifact_prefix": target_prefix,
        "aws_account_id": account_id,
        "campaign_sha256": target.campaign_sha256,
        "canonical_manifest_sha256": canonical.sha256,
        "container_image": image_uri,
        "gate_receipt_sha256": gate_receipt_sha256,
        "git_sha": git_sha,
        "manifest_sha256": target.sha256,
        "runtime_contract_sha256": target.runtime_contract_sha256,
    }
    result = materialize_canonical_rankings(
        sources=sources,
        target_manifest=target,
        target_provenance=target_provenance,
        target_bucket=bucket,
        s3_client=s3,
    )
    success(f"Materialized {len(required)} rankings; receipt {result}")
