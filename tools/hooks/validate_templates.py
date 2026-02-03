#!/usr/bin/env python3
"""Validate that AWS template files retain placeholder tokens."""

from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        sys.stderr.write(f"ERROR: Missing template file: {path}\n")
        raise


def _check_config_template(path: Path) -> list[str]:
    text = _read_text(path)
    errors = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("s3_bucket"):
            continue
        if ":" not in stripped:
            continue
        _, value = stripped.split(":", 1)
        value = value.strip().strip('"').strip("'")
        if value and value not in {"__S3_BUCKET__", "null", "None"}:
            errors.append(f"{path}: s3_bucket should be empty or a placeholder, got {value!r}")
        break
    return errors


def main() -> int:
    root = _repo_root()
    config_path = root / "paper" / "scripts" / "infra" / "config.example.yaml"

    errors: list[str] = []
    try:
        errors.extend(_check_config_template(config_path))
    except FileNotFoundError:
        return 1

    if errors:
        sys.stderr.write("ERROR: Template validation failed:\n")
        for err in errors:
            sys.stderr.write(f"  - {err}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
