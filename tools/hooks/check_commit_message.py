#!/usr/bin/env python3
"""Enforce the commit message convention (commit-msg hook).

Subject: ``area: what changed`` in the imperative, at most 72 characters, no
trailing period, no ``WIP``/``TODO`` placeholders. Details go in the body after a
blank line, wrapped at 72 characters. Merge and revert commits are exempt.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MAX_SUBJECT = 72
MAX_BODY = 72
SUBJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _./+-]{0,30}: \S.*$")
BANNED_PREFIXES = ("wip", "todo", "[wip]", "misc", "stuff", "minor")


def check(text: str) -> list[str]:
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ["empty commit message"]
    subject = lines[0].rstrip()
    if subject.startswith(("Merge ", "Revert ", "fixup! ", "squash! ")):
        return []
    errors: list[str] = []
    if len(subject) > MAX_SUBJECT:
        errors.append(f"subject is {len(subject)} characters; keep it at most {MAX_SUBJECT}")
    if subject.endswith("."):
        errors.append("subject must not end with a period")
    if not SUBJECT_RE.match(subject):
        errors.append(
            "subject must read 'area: what changed' (for example 'arXiv: ...', 'citrees: ...')"
        )
    if (
        subject.lower().startswith(BANNED_PREFIXES)
        or subject.split(":")[0].strip().lower() in BANNED_PREFIXES
    ):
        errors.append("subject must not be a WIP/TODO/misc placeholder; say what changed")
    if len(lines) > 1 and lines[1].strip():
        errors.append("leave a blank line between the subject and the body")
    for i, ln in enumerate(lines[2:], start=3):
        if (
            len(ln) > MAX_BODY
            and " " in ln.strip()
            and not ln.lstrip().startswith(("http", "s3://", "`"))
        ):
            errors.append(f"body line {i} is {len(ln)} characters; wrap at {MAX_BODY}")
            break
    return errors


def main() -> int:
    path = Path(sys.argv[1])
    errors = check(path.read_text(encoding="utf-8"))
    if errors:
        print("commit message rejected:")
        for e in errors:
            print(f"  - {e}")
        print(
            "convention: 'area: what changed' (<=72 chars), blank line, wrapped body with the details"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
