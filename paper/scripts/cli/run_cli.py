"""Script entrypoint for running the citrees-exp CLI under `ray submit`/Ray Jobs.

Example:
    citrees-exp cluster submit paper/scripts/cli/run_cli.py run classification --stage stage1
"""

from __future__ import annotations

from paper.scripts.cli.entrypoint import main

if __name__ == "__main__":
    main()
