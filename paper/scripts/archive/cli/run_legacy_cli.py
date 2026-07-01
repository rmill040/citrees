"""Script entrypoint for running the citrees-exp CLI.

Archived wrapper; the live entrypoint is `paper.scripts.cli.entrypoint`.

Example:
    python paper/scripts/archive/cli/run_legacy_cli.py run classification --stage stage1
"""

from __future__ import annotations

from paper.scripts.cli.entrypoint import main

if __name__ == "__main__":
    main()
