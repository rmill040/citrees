#!/usr/bin/env python3
"""Fail if AWS-specific runtime configs are staged for commit."""

from __future__ import annotations

import sys
from pathlib import Path

BLOCKED = {
    "paper/scripts/infra/config.yaml",
}


def _normalize(path: str) -> str:
    try:
        return Path(path).as_posix()
    except Exception:
        return path.replace("\\", "/")


def main(argv: list[str]) -> int:
    if len(argv) <= 1:
        return 0

    blocked = []
    for arg in argv[1:]:
        norm = _normalize(arg)
        if norm in BLOCKED or norm.endswith("/paper/scripts/infra/config.yaml"):
            blocked.append(arg)

    if not blocked:
        return 0

    sys.stderr.write(
        "ERROR: Refusing to commit AWS-specific runtime configs:\n"
        + "".join(f"  - {path}\n" for path in blocked)
    )
    sys.stderr.write(
        "Use the template file instead:\n  - paper/scripts/infra/config.example.yaml\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
