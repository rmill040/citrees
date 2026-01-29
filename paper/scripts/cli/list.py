"""List commands for discovering datasets and methods.

Commands for exploring available datasets and feature selection methods.
"""

from __future__ import annotations

from typing import Annotated

import typer

from paper.scripts.cli._console import console, create_table, format_number, info

app = typer.Typer(
    name="list",
    help="List datasets and methods",
    no_args_is_help=True,
)


@app.command()
def datasets(
    task: Annotated[
        str | None,
        typer.Option(
            "--task",
            "-t",
            help="Filter by task type: classification or regression",
        ),
    ] = None,
    source: Annotated[
        str,
        typer.Option(
            "--source",
            "-s",
            help="Filter by source: real, synthetic, or all",
        ),
    ] = "all",
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show additional metadata",
        ),
    ] = False,
) -> None:
    """List available datasets with metadata.

    Shows dataset names, dimensions, and source information.

    \b
    Examples:
        citrees-exp list datasets
        citrees-exp list datasets --task classification
        citrees-exp list datasets --source synthetic --verbose
    """
    from paper.scripts.adapters import get_dataset_shape, get_datasets

    # Determine which task types to show
    tasks = [task] if task else ["classification", "regression"]

    for task in tasks:
        dataset_names = get_datasets(task, source=source)  # type: ignore

        if not dataset_names:
            info(f"No {task} datasets found for source={source}")
            continue

        # Build table
        columns = [
            ("Dataset", ""),
            ("Source", ""),
            ("Samples", "number"),
            ("Features", "number"),
        ]
        if verbose:
            columns.append(("Type", "muted"))

        table = create_table(
            title=f"{task.title()} Datasets ({len(dataset_names)})",
            columns=columns,
        )

        for name in dataset_names:
            try:
                n_samples, n_features = get_dataset_shape(name, task)  # type: ignore
                ds_source = "synthetic" if "synthetic" in name else "real"

                row = [
                    name,
                    ds_source,
                    format_number(n_samples),
                    format_number(n_features),
                ]

                if verbose:
                    # Infer dataset type from name
                    if "synthetic" in name:
                        if "bias" in name:
                            ds_type = "bias"
                        elif "nonlinear" in name:
                            ds_type = "nonlinear"
                        elif "corr" in name:
                            ds_type = "correlated"
                        else:
                            ds_type = "standard"
                    else:
                        ds_type = "real"
                    row.append(ds_type)

                table.add_row(*row)
            except Exception as e:
                if verbose:
                    table.add_row(name, "?", "?", "?", str(e))
                else:
                    table.add_row(name, "?", "?", "?")

        console.print(table)
        console.print()


@app.command()
def methods(
    task: Annotated[
        str | None,
        typer.Option(
            "--task",
            "-t",
            help="Filter by task type: classification or regression",
        ),
    ] = None,
    category: Annotated[
        str | None,
        typer.Option(
            "--category",
            "-c",
            help="Filter by category: filter, ptest, embedding, wrapper",
        ),
    ] = None,
    configs: Annotated[
        bool,
        typer.Option(
            "--configs",
            help="Show number of hyperparameter configurations",
        ),
    ] = False,
) -> None:
    """List available feature selection methods.

    Shows method names, descriptions, and categories.

    \b
    Examples:
        citrees-exp list methods
        citrees-exp list methods --task classification
        citrees-exp list methods --category embedding --configs
    """
    from paper.scripts.pipeline import (
        get_all_method_info,
    )
    from paper.scripts.pipeline.methods import get_method_config_count

    # Determine which task types to show
    tasks = [task] if task else ["classification", "regression"]

    for task in tasks:
        method_infos = get_all_method_info(task, category)
        method_names = [info.name for info in method_infos]
        config_counts = get_method_config_count(method_names, task) if configs else {}

        # Build table
        columns = [
            ("Method", ""),
            ("Name", ""),
            ("Description", "muted"),
            ("Category", ""),
        ]
        if configs:
            columns.append(("Configs", "number"))

        table = create_table(
            title=f"{task.title()} Methods ({len(method_infos)})",
            columns=columns,
        )

        for method_info in method_infos:
            row = [
                method_info.name,
                method_info.display_name,
                method_info.description,
                method_info.category,
            ]
            if configs:
                n_configs = config_counts.get(method_info.name, 1)
                row.append(format_number(n_configs))

            table.add_row(*row)

        console.print(table)

        # Show total configs if requested
        if configs:
            total_configs = sum(config_counts.get(name, 1) for name in method_names)
            console.print(f"  Total configurations: [number]{format_number(total_configs)}[/]")

        console.print()
