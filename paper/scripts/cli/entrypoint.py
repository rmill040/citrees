"""Console-script entrypoint for the citrees experiment CLI.

This module exists so `citrees-exp` can fail with a clear message when the
optional experiment dependencies (Typer/Rich/Ray/etc.) are not installed.
"""

from __future__ import annotations

import sys


def _print_install_help(*, missing: str | None) -> None:
    if missing:
        print(f"citrees-exp: missing dependency '{missing}'.", file=sys.stderr)

    print("citrees-exp requires the experiment CLI dependencies.", file=sys.stderr)
    print("Install with one of:", file=sys.stderr)
    print("  uv sync --group paper", file=sys.stderr)
    print("  pip install -e '.[paper]'", file=sys.stderr)


def main() -> None:
    try:
        from paper.scripts.cli.app import app
    except ModuleNotFoundError as e:
        _print_install_help(missing=e.name)
        raise SystemExit(1) from None
    except ImportError:
        _print_install_help(missing=None)
        raise SystemExit(1) from None

    app()
