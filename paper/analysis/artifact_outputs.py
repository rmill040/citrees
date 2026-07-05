"""Output helpers for paper artifact builders."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_write_arxiv_argument(parser: argparse.ArgumentParser) -> None:
    """Add the explicit opt-in switch for updating frozen arXiv figure copies."""
    parser.add_argument(
        "--write-arxiv",
        action="store_true",
        help="also write the frozen paper/arxiv/figures copy",
    )


def figure_output_dirs(
    figures_dir: Path,
    arxiv_figures_dir: Path,
    *,
    write_arxiv: bool,
) -> tuple[Path, ...]:
    """Return output directories for figure builders."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not write_arxiv:
        return (figures_dir,)
    arxiv_figures_dir.mkdir(parents=True, exist_ok=True)
    return (figures_dir, arxiv_figures_dir)
