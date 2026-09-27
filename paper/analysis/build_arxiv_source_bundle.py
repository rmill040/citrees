"""Inspect or build a deterministic arXiv source bundle for the paper.

The bundle holds the TeX sources that ``main.tex`` inputs, the bibliography,
the figures those sources include, and the compiled supplement as the arXiv
ancillary file ``anc/supplement.pdf``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARXIV_DIR = ROOT / "paper" / "arxiv"
DEFAULT_OUT = ARXIV_DIR / "build" / "citrees-arxiv-source.zip"
MAIN_TEX = "main.tex"
STATIC_FILES = ("main.tex", "macros.tex", "references.bib")
SUPPLEMENT_PDF = "supplement.pdf"
SUPPLEMENT_ARCNAME = "anc/supplement.pdf"
INPUT_RE = re.compile(r"^[^%\n]*?\\input\{([^}]+)\}", re.MULTILINE)
INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_FILE_MODE = 0o644


def _resolve_tex(name: str) -> Path:
    """Return the arXiv-directory path of a TeX file named by an ``\\input``."""
    path = ARXIV_DIR / name
    if path.suffix != ".tex":
        path = path.with_name(path.name + ".tex")
    if not path.exists():
        raise FileNotFoundError(f"{MAIN_TEX} inputs missing file {path.relative_to(ROOT)}")
    return path


def tex_sources() -> list[Path]:
    """Return ``main.tex`` and every TeX file it inputs, in input order."""
    sources: list[Path] = []
    pending = [ARXIV_DIR / MAIN_TEX]
    while pending:
        source = pending.pop(0)
        if source in sources:
            continue
        sources.append(source)
        text = source.read_text(encoding="utf-8")
        pending.extend(_resolve_tex(name) for name in INPUT_RE.findall(text))
    return sources


def collect_referenced_figures() -> list[Path]:
    """Collect the figure files included by the sources of ``main.tex``."""
    figures: set[Path] = set()
    for source in tex_sources():
        text = source.read_text(encoding="utf-8")
        for match in INCLUDEGRAPHICS_RE.findall(text):
            rel = Path(match)
            candidates = [rel] if rel.suffix else [rel.with_suffix(".png"), rel.with_suffix(".pdf")]
            for candidate in candidates:
                path = ARXIV_DIR / candidate
                if path.exists():
                    figures.add(path)
                    break
            else:
                raise FileNotFoundError(
                    f"{source.relative_to(ROOT)} references missing figure {match}"
                )
    return sorted(figures)


def bundle_members() -> list[Path]:
    """Return the arXiv-directory source files included in the bundle."""
    members: set[Path] = set()
    for relname in STATIC_FILES:
        path = ARXIV_DIR / relname
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.relative_to(ROOT)}.")
        members.add(path)

    members.update(tex_sources())
    members.update(collect_referenced_figures())
    return sorted(members, key=lambda path: path.relative_to(ARXIV_DIR).as_posix())


def ancillary_members() -> dict[str, Path]:
    """Return the ancillary files of the bundle, keyed by archive name."""
    path = ARXIV_DIR / SUPPLEMENT_PDF
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.relative_to(ROOT)}; build it with latexmk in paper/arxiv "
            "or pass --build-pdf."
        )
    return {SUPPLEMENT_ARCNAME: path}


def archive_members() -> dict[str, Path]:
    """Return every archive name of the bundle mapped to its source file."""
    members = {path.relative_to(ARXIV_DIR).as_posix(): path for path in bundle_members()}
    members.update(ancillary_members())
    return dict(sorted(members.items()))


def build_pdf() -> None:
    """Run latexmk, which builds main.pdf and then supplement.pdf per latexmkrc."""
    subprocess.run(["latexmk"], cwd=ARXIV_DIR, check=True)


def write_bundle(out_path: Path) -> dict[str, Path]:
    """Write the source bundle and return its members keyed by archive name."""
    members = archive_members()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for arcname, path in members.items():
            info = zipfile.ZipInfo(arcname, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = ZIP_FILE_MODE << 16
            archive.writestr(info, path.read_bytes())
    return members


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Bundle path to write")
    parser.add_argument(
        "--build-pdf",
        action="store_true",
        help="Run latexmk before checking or writing. This mutates paper/arxiv build outputs.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the source zip. Without this flag, the command only lists bundle members.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate bundle membership without writing a zip. This is the default behavior.",
    )
    args = parser.parse_args()

    if args.build_pdf:
        build_pdf()

    members = archive_members()
    if args.check or not args.write:
        for arcname in members:
            print(arcname)
        return

    write_bundle(args.out)
    try:
        display_path = args.out.relative_to(ROOT)
    except ValueError:
        display_path = args.out
    print(f"Wrote {display_path} with {len(members)} files")


if __name__ == "__main__":
    main()
