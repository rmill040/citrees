"""citrees-exp CLI package.

This package provides a command-line interface for managing citrees experiments,
including running experiments, checking progress, and managing infrastructure.

Usage:
    citrees-exp --help
    citrees-exp run classification
    citrees-exp check --by-method
"""

from paper.scripts.cli.app import app

__all__ = ["app"]
