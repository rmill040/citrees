"""Configuration management commands.

Commands for viewing, initializing, and validating configuration files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.syntax import Syntax

from paper.scripts.cli._console import console, error, info, success

app = typer.Typer(
    name="config",
    help="Configuration management",
    no_args_is_help=True,
)

# Path constants
INFRA_DIR = Path(__file__).parent.parent / "infra"
CONFIG_YAML = INFRA_DIR / "config.yaml"
CONFIG_EXAMPLE_YAML = INFRA_DIR / "config.example.yaml"


@app.command()
def show(
    raw: Annotated[
        bool,
        typer.Option(
            "--raw",
            help="Print raw YAML without syntax highlighting",
        ),
    ] = False,
) -> None:
    """Display current configuration with syntax highlighting.

    Shows config.yaml if it exists, otherwise shows config.example.yaml.
    """
    # Find config file
    if CONFIG_YAML.exists():
        config_path = CONFIG_YAML
        info(f"Showing [path]{config_path}[/]")
    elif CONFIG_EXAMPLE_YAML.exists():
        config_path = CONFIG_EXAMPLE_YAML
        info(f"No config.yaml found, showing [path]{config_path}[/]")
    else:
        error("No configuration file found")
        raise typer.Exit(1)

    content = config_path.read_text()

    if raw:
        console.print(content, markup=False)
    else:
        syntax = Syntax(
            content,
            "yaml",
            theme="monokai",
            line_numbers=True,
            word_wrap=True,
        )
        console.print(syntax)


@app.command()
def init(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing config.yaml",
        ),
    ] = False,
) -> None:
    """Initialize config.yaml from the example template.

    Creates a new config.yaml by copying config.example.yaml.
    """
    if not CONFIG_EXAMPLE_YAML.exists():
        error(f"Template not found: [path]{CONFIG_EXAMPLE_YAML}[/]")
        raise typer.Exit(1)

    if CONFIG_YAML.exists() and not force:
        error(f"[path]{CONFIG_YAML}[/] already exists. Use --force to overwrite.")
        raise typer.Exit(1)

    content = CONFIG_EXAMPLE_YAML.read_text()
    CONFIG_YAML.write_text(content)

    success(f"Created [path]{CONFIG_YAML}[/]")
    info("Edit the file to customize your configuration.")


@app.command()
def validate() -> None:
    """Validate configuration against the schema.

    Checks that config.yaml is valid and all required fields are present.
    """
    if not CONFIG_YAML.exists():
        if CONFIG_EXAMPLE_YAML.exists():
            info(f"No config.yaml found, validating [path]{CONFIG_EXAMPLE_YAML}[/]")
            config_path = CONFIG_EXAMPLE_YAML
        else:
            error("No configuration file found")
            raise typer.Exit(1)
    else:
        config_path = CONFIG_YAML

    try:
        from paper.scripts.config import load_config

        config = load_config(config_path)

        # Display validated config
        console.print("\n[heading]Configuration validated successfully[/]\n")

        console.print(f"  [muted]AWS Region:[/] {config.aws_region}")
        console.print(f"  [muted]S3 Bucket:[/] {config.s3_bucket or '(auto-derived)'}")
        console.print(f"  [muted]Experiment Type:[/] {config.experiment.type}")
        console.print(f"  [muted]Number of Seeds:[/] {config.experiment.n_seeds}")
        console.print(f"  [muted]Stale Timeout:[/] {config.experiment.stale_timeout_minutes} min")

        console.print("\n  [heading]Stage 1 (Selection) Resources:[/]")
        console.print("    [muted]Tiers:[/] LIGHT=1cpu/2GB, STANDARD=8cpu/4GB, HEAVY=16cpu/8GB")

        console.print("\n  [heading]Stage 2 (Evaluation) Resources:[/]")
        console.print(f"    [muted]Default CPUs:[/] {config.experiment.evaluation_cpus_default}")
        console.print(
            f"    [muted]Default Memory:[/] {config.experiment.evaluation_memory_gb_default} GB"
        )

        if config.experiment.selection_cpus_overrides:
            console.print("\n  [heading]Selection CPU Overrides:[/]")
            for method, cpus in config.experiment.selection_cpus_overrides.items():
                console.print(f"    [muted]{method}:[/] {cpus}")

        if config.experiment.evaluation_cpus_overrides:
            console.print("\n  [heading]Evaluation CPU Overrides:[/]")
            for method, cpus in config.experiment.evaluation_cpus_overrides.items():
                console.print(f"    [muted]{method}:[/] {cpus}")

        console.print()
        success("Configuration is valid")

    except Exception as e:
        error(f"Configuration validation failed: {e}")
        raise typer.Exit(1) from e


@app.command()
def path() -> None:
    """Show the path to configuration files.

    Useful for finding where config files are located.
    """
    console.print("\n[heading]Configuration Paths[/]\n")
    console.print(f"  [muted]Config directory:[/] [path]{INFRA_DIR}[/]")
    console.print(f"  [muted]config.yaml:[/] [path]{CONFIG_YAML}[/]", end="")
    if CONFIG_YAML.exists():
        console.print(" [success](exists)[/]")
    else:
        console.print(" [muted](not found)[/]")
    console.print(f"  [muted]config.example.yaml:[/] [path]{CONFIG_EXAMPLE_YAML}[/]", end="")
    if CONFIG_EXAMPLE_YAML.exists():
        console.print(" [success](exists)[/]")
    else:
        console.print(" [muted](not found)[/]")
    console.print()
