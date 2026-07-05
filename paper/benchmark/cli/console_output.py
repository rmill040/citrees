"""Shared Rich console utilities for the CLI.

Provides a consistent theme, console instance, and helper functions
for formatted output across all CLI commands.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table
from rich.theme import Theme

# Consistent theme across the CLI
theme = Theme(
    {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red bold",
        "heading": "bold blue",
        "muted": "dim",
        "highlight": "bold magenta",
        "path": "italic cyan",
        "number": "bold",
    }
)

console = Console(theme=theme)


def success(msg: str) -> None:
    """Print a success message with a checkmark."""
    console.print(f"[success]\u2713[/] {msg}")


def error(msg: str) -> None:
    """Print an error message with an X."""
    console.print(f"[error]\u2717[/] {msg}")


def warn(msg: str) -> None:
    """Print a warning message."""
    console.print(f"[warning]\u26a0[/] {msg}")


def info(msg: str) -> None:
    """Print an info message."""
    console.print(f"[info]\u2139[/] {msg}")


def heading(msg: str) -> None:
    """Print a section heading."""
    console.print(f"\n[heading]{msg}[/]")


def step(msg: str) -> None:
    """Print an indented step/detail message."""
    console.print(f"  [muted]{msg}[/]")


def create_table(
    title: str | None = None,
    columns: list[tuple[str, str]] | None = None,
    show_header: bool = True,
    box_style: Any = None,
) -> Table:
    """Create a Rich table with consistent styling.

    Parameters
    ----------
    title : str, optional
        Table title.
    columns : list of (name, style) tuples, optional
        Column definitions. Style can be empty string for default.
    show_header : bool, default True
        Whether to show the header row.
    box_style : optional
        Rich box style to use.
    """
    from rich import box

    table = Table(
        title=title,
        show_header=show_header,
        header_style="bold",
        box=box_style or box.ROUNDED,
        expand=False,
    )

    if columns:
        for name, style in columns:
            table.add_column(name, style=style if style else None)

    return table


def format_number(n: int | float) -> str:
    """Format a number with thousands separators."""
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{n:,}"


def format_percent(value: float, total: float) -> str:
    """Format a percentage."""
    if total == 0:
        return "0.0%"
    return f"{100 * value / total:.1f}%"


def progress_bar(completed: int, total: int, width: int = 20) -> str:
    """Create a simple ASCII progress bar."""
    if total == 0:
        return "\u2591" * width
    pct = completed / total
    filled = int(pct * width)
    return "\u2588" * filled + "\u2591" * (width - filled)
