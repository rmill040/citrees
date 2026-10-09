"""Text audits for the arXiv manuscript.

Operational audit helper; not part of the paper rebuild path.

Three checks on paper/arxiv:

- ``numbers``: lists numbers in the body and appendices at a baseline commit that
  no longer appear anywhere in the current body and appendices, so a cut or a
  rewrite cannot silently drop a result.
- ``repeats``: lists 7-word phrases shared by two or more body sections.
- ``v1-overlap``: counts how many sentences of the July 2026 v1 body (commit
  d71b6e4) reappear verbatim in the current body.

Usage:
  uv run python paper/maintenance/audit_manuscript_text.py numbers --baseline 76abb3e
  uv run python paper/maintenance/audit_manuscript_text.py repeats
  uv run python paper/maintenance/audit_manuscript_text.py v1-overlap
"""

from __future__ import annotations

import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Final

ROOT: Final[Path] = Path(__file__).resolve().parents[2]
ARXIV: Final[Path] = ROOT / "paper" / "arxiv"
V1_COMMIT: Final[str] = "d71b6e4"
NUMBER = re.compile(r"(?<![A-Za-z_@\\])[-+]?\d[\d,]*(?:\.\d+)?")
FLOAT_ENVS = re.compile(r"\\begin\{(table|figure|algorithm)\*?\}.*?\\end\{\1\*?\}", re.S)
REF_CMDS = re.compile(
    r"\\(?:cite[tp]?|citealp|citeauthor|citeyear|[cC]ref|ref|eqref|label|url|href)\*?"
    r"(?:\[[^\]]*\])*\{[^}]*\}"
)


def _strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def _prose(text: str) -> str:
    """Return the prose of a LaTeX source without floats, references, or commands."""
    text = FLOAT_ENVS.sub(" ", _strip_comments(text))
    text = REF_CMDS.sub("", text).replace("{,}", ",").replace("~", " ")
    text = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?", "", text)
    text = re.sub(r"[{}$]", "", text)
    return " ".join(text.split())


def _git_show(rev: str, path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{rev}:{path}"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout


def _git_tex_files(rev: str, *dirs: str) -> dict[str, str]:
    names = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", rev, *dirs],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return {name: _git_show(rev, name) for name in names if name.endswith(".tex")}


def _body_files() -> dict[str, str]:
    return {path.name: path.read_text() for path in sorted((ARXIV / "sections").glob("*.tex"))}


def _numbers(text: str) -> dict[str, str]:
    """Map each numeric token (sign and thousands separators dropped) to one context."""
    text = _strip_comments(text).replace("{,}", ",").replace("$-$", "-")
    found: dict[str, str] = {}
    for match in NUMBER.finditer(text):
        token = match.group(0).lstrip("+-").replace(",", "")
        if token and token not in {"0", "1", "2"} and token not in found:
            start, end = max(0, match.start() - 70), match.end() + 70
            found[token] = " ".join(text[start:end].split())
    return found


def audit_numbers(baseline: str) -> None:
    current: set[str] = set()
    for pattern in ("sections/*.tex", "appendices/*.tex"):
        for path in ARXIV.glob(pattern):
            current.update(_numbers(path.read_text()))
    baseline_files = _git_tex_files(baseline, "paper/arxiv/sections", "paper/arxiv/appendices")
    missing = {
        token: (Path(name).name, context)
        for name, text in baseline_files.items()
        for token, context in _numbers(text).items()
        if token not in current
    }
    print(f"{len(missing)} numbers from {baseline} are absent from the current manuscript.")
    for token, (name, context) in sorted(missing.items()):
        print(f"{token}\t{name}\t{context}")


def audit_repeats(width: int = 7) -> None:
    owners: dict[str, set[str]] = defaultdict(set)
    for name, text in _body_files().items():
        words = re.findall(r"[a-z0-9.']+", _prose(text).lower())
        for i in range(len(words) - width + 1):
            owners[" ".join(words[i : i + width])].add(name)
    shared = sorted((sorted(files), phrase) for phrase, files in owners.items() if len(files) > 1)
    print(f"{len(shared)} {width}-word phrases are shared across body sections.")
    for files, phrase in shared:
        print(f"{' + '.join(files)}\t{phrase}")


def _sentences(text: str, min_words: int = 8) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", _prose(text))
    return [part for part in parts if len(part.split()) >= min_words]


def audit_v1_overlap(prefix: int = 60) -> None:
    current = {s[:prefix].lower() for text in _body_files().values() for s in _sentences(text)}
    v1 = _git_tex_files(V1_COMMIT, "paper/arxiv/sections")
    hit = total = 0
    for name, text in sorted(v1.items()):
        sentences = _sentences(text)
        kept = sum(s[:prefix].lower() in current for s in sentences)
        hit, total = hit + kept, total + len(sentences)
        print(f"{Path(name).stem:18s} {kept:4d} / {len(sentences):4d}")
    print(f"{'total':18s} {hit:4d} / {total:4d}  ({100 * hit / max(total, 1):.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="check", required=True)
    numbers = sub.add_parser("numbers", help="numbers dropped since a baseline commit")
    numbers.add_argument("--baseline", required=True, help="git revision to compare against")
    sub.add_parser("repeats", help="7-word phrases shared across body sections")
    sub.add_parser("v1-overlap", help="v1 sentences reproduced verbatim")
    args = parser.parse_args()
    if args.check == "numbers":
        audit_numbers(args.baseline)
    elif args.check == "repeats":
        audit_repeats()
    else:
        audit_v1_overlap()


if __name__ == "__main__":
    main()
