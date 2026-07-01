"""Build a deterministic arXiv source bundle for the paper."""

from __future__ import annotations

import argparse
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARXIV_DIR = ROOT / "paper" / "arxiv"
DEFAULT_OUT = ARXIV_DIR / "build" / "citrees-arxiv-source.zip"
STATIC_FILES = ("main.tex", "macros.tex", "references.bib")
INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_FILE_MODE = 0o644


def tex_sources() -> list[Path]:
    """Return TeX files that define the paper source."""
    return [
        ARXIV_DIR / "main.tex",
        *sorted((ARXIV_DIR / "sections").glob("*.tex")),
        *sorted((ARXIV_DIR / "appendices").glob("*.tex")),
    ]


def collect_referenced_figures() -> list[Path]:
    """Collect figure paths referenced by the TeX source."""
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
    """Return arXiv-relative files included in the source bundle."""
    members: set[Path] = set()
    for relname in STATIC_FILES:
        path = ARXIV_DIR / relname
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.relative_to(ROOT)}.")
        members.add(path)

    members.update(tex_sources())
    members.update(collect_referenced_figures())
    return sorted(members, key=lambda path: path.relative_to(ARXIV_DIR).as_posix())


def build_pdf() -> None:
    """Run latexmk so main.bbl and cross references are current."""
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=ARXIV_DIR,
        check=True,
    )


def write_bundle(out_path: Path) -> list[Path]:
    """Write the source bundle and return included members."""
    members = bundle_members()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in members:
            arcname = path.relative_to(ARXIV_DIR).as_posix()
            info = zipfile.ZipInfo(arcname, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = ZIP_FILE_MODE << 16
            archive.writestr(info, path.read_bytes())
    return members


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Bundle path to write")
    parser.add_argument(
        "--skip-build", action="store_true", help="Validate and zip the existing arXiv build state"
    )
    parser.add_argument(
        "--check", action="store_true", help="Validate bundle membership without writing a zip"
    )
    args = parser.parse_args()

    if not args.skip_build:
        build_pdf()

    members = bundle_members()
    if args.check:
        for path in members:
            print(path.relative_to(ARXIV_DIR).as_posix())
        return

    write_bundle(args.out)
    try:
        display_path = args.out.relative_to(ROOT)
    except ValueError:
        display_path = args.out
    print(f"Wrote {display_path} with {len(members)} files")


if __name__ == "__main__":
    main()
