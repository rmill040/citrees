"""citrees-exp CLI package.

This package provides a command-line interface for managing citrees experiments,
including running experiments, checking progress, and managing infrastructure.

Usage:
    citrees-exp --help
    citrees-exp run classification
    citrees-exp check --by-method
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "app":
        from paper.scripts.cli.app import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["app"]
