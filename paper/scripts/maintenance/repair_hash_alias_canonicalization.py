"""Canonicalize old-hash raw artifacts to the current hash labels.

Operational repair helper; not part of the current paper-facing rebuild path.

This script uses the audited alias manifest from ``audit_hash_alias_manifest.py`` to
clean up semantically equivalent raw artifacts under a local ``data/``
directory. It is intentionally conservative:

1. If the canonical target file already exists, the script verifies that the
   old file matches after normalizing hash-bearing metadata.
2. Only then does it remove the old duplicate.
3. If the canonical target file does not exist, the script rewrites the old
   file with canonical ``method``/``method_id`` metadata and writes it to the
   new path before removing the old file.
4. If any payload conflict is found, both files are left untouched.

Outputs:
  - paper/results/tables/hash_alias_canonicalization_actions.csv
  - paper/results/tables/hash_alias_canonicalization_summary.json

Usage:
  UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/maintenance/repair_hash_alias_canonicalization.py \
      --local-dir ../data

  UV_CACHE_DIR=./scratch/.uv_cache uv run python paper/scripts/maintenance/repair_hash_alias_canonicalization.py \
      --local-dir ../data --apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TABLES_DIR = RESULTS_DIR / "tables"
DEFAULT_MANIFEST = TABLES_DIR / "grid_hash_aliases.csv"
VOLATILE_COLS = {
    "created_at_utc",
    "elapsed_seconds",
    "evaluation_cpus",
    "git_sha",
    "library_versions",
    "selection_cpus",
}
IDENTITY_COLS = ("method", "method_id")
Stage = Literal["rankings", "metrics"]


def _sort_columns(stage: Stage, columns: list[str]) -> list[str]:
    """Return a stable sort key for normalized equality checks."""
    if stage == "rankings":
        candidates = ["fold_idx"]
    else:
        candidates = ["fold_idx", "downstream_model", "k", "n_features_selected"]
    return [col for col in candidates if col in columns]


def _normalize_for_compare(df: pd.DataFrame, *, canonical_label: str, stage: Stage) -> pd.DataFrame:
    """Drop volatile metadata and canonicalize hash-bearing columns."""
    normalized = df.copy()
    for col in IDENTITY_COLS:
        if col in normalized.columns:
            normalized[col] = canonical_label
    keep_cols = [col for col in normalized.columns if col not in VOLATILE_COLS]
    normalized = normalized[keep_cols]
    sort_cols = _sort_columns(stage, keep_cols)
    if sort_cols:
        normalized = normalized.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return normalized


def _rewrite_with_canonical_label(df: pd.DataFrame, canonical_label: str) -> pd.DataFrame:
    """Rewrite method label columns in a parquet payload."""
    rewritten = df.copy()
    for col in IDENTITY_COLS:
        if col in rewritten.columns:
            rewritten[col] = canonical_label
    return rewritten


def _payloads_match(old_path: Path, new_path: Path, *, canonical_label: str, stage: Stage) -> bool:
    """Check whether two parquet files are equivalent modulo hash metadata."""
    old_df = pd.read_parquet(old_path)
    new_df = pd.read_parquet(new_path)
    old_norm = _normalize_for_compare(old_df, canonical_label=canonical_label, stage=stage)
    new_norm = _normalize_for_compare(new_df, canonical_label=canonical_label, stage=stage)
    if list(old_norm.columns) != list(new_norm.columns):
        return False
    return old_norm.equals(new_norm)


def _materialize_canonical_file(old_path: Path, new_path: Path, *, canonical_label: str) -> None:
    """Write a canonicalized copy of an old artifact to the new path."""
    payload = pd.read_parquet(old_path)
    rewritten = _rewrite_with_canonical_label(payload, canonical_label)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    rewritten.to_parquet(new_path, index=False)


def _iter_actions(manifest: pd.DataFrame, local_dir: Path) -> list[dict[str, object]]:
    """Build the per-file canonicalization action plan."""
    actions: list[dict[str, object]] = []

    for row in manifest.itertuples(index=False):
        task = str(row.task)
        old_label = str(row.old_label)
        new_label = str(row.new_label)

        for stage in ("rankings", "metrics"):
            root = local_dir / stage / task
            if not root.exists():
                continue

            for old_path in sorted(root.rglob(f"{old_label}_seed*.parquet")):
                seed = int(old_path.stem.rsplit("_seed", 1)[1])
                dataset = old_path.parent.name
                new_path = old_path.with_name(old_path.name.replace(old_label, new_label, 1))

                if new_path.exists():
                    matches = _payloads_match(old_path, new_path, canonical_label=new_label, stage=stage)
                    action = "delete_old_duplicate" if matches else "conflict_keep_both"
                else:
                    matches = True
                    action = "rewrite_to_canonical"

                actions.append(
                    {
                        "task": task,
                        "stage": stage,
                        "dataset": dataset,
                        "seed": seed,
                        "old_label": old_label,
                        "new_label": new_label,
                        "old_path": str(old_path),
                        "new_path": str(new_path),
                        "new_exists": new_path.exists(),
                        "payload_match": matches,
                        "action": action,
                    }
                )

    return actions


def _apply_action(action: dict[str, object]) -> str:
    """Apply one canonicalization action and return the final disposition."""
    old_path = Path(str(action["old_path"]))
    new_path = Path(str(action["new_path"]))
    new_label = str(action["new_label"])

    match action["action"]:
        case "delete_old_duplicate":
            old_path.unlink()
            return "deleted_old_duplicate"
        case "rewrite_to_canonical":
            _materialize_canonical_file(old_path, new_path, canonical_label=new_label)
            old_path.unlink()
            return "rewrote_and_deleted_old"
        case "conflict_keep_both":
            return "kept_conflict"
        case _:
            raise ValueError(f"Unknown action: {action['action']}")


def _build_summary(actions: pd.DataFrame, *, apply: bool) -> dict[str, object]:
    """Aggregate a compact summary for reporting and auditing."""
    summary: dict[str, object] = {
        "mode": "apply" if apply else "dry_run",
        "total_files": int(len(actions)),
        "delete_old_duplicate": int((actions["action"] == "delete_old_duplicate").sum()),
        "rewrite_to_canonical": int((actions["action"] == "rewrite_to_canonical").sum()),
        "conflict_keep_both": int((actions["action"] == "conflict_keep_both").sum()),
    }

    by_stage = (
        actions.groupby(["task", "stage", "action"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .to_dict(orient="records")
    )
    summary["by_task_stage_action"] = by_stage
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonicalize old-hash raw artifact files")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("../data"),
        help="Local data directory containing rankings/ and metrics/",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Alias manifest CSV generated by audit_hash_alias_manifest.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TABLES_DIR,
        help="Directory for action-plan outputs",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply canonicalization changes to local-dir. Default is dry-run.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest).sort_values(["task", "old_label"]).reset_index(drop=True)
    actions = pd.DataFrame(_iter_actions(manifest, args.local_dir.resolve()))
    if actions.empty:
        summary = {"mode": "apply" if args.apply else "dry_run", "total_files": 0}
    else:
        if args.apply:
            actions["applied_action"] = actions.apply(lambda row: _apply_action(row.to_dict()), axis=1)
        summary = _build_summary(actions, apply=args.apply)
        if args.apply:
            summary["applied_action_counts"] = (
                actions["applied_action"].value_counts(dropna=False).sort_index().to_dict()
            )

    csv_out = output_dir / "hash_alias_canonicalization_actions.csv"
    json_out = output_dir / "hash_alias_canonicalization_summary.json"

    actions.to_csv(csv_out, index=False)
    json_out.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Saved {csv_out}")
    print(f"Saved {json_out}")
    if actions.empty:
        print("No alias files found.")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
