"""Checkpoint inspection command."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from fastdrs.checkpoints import inspect_checkpoint


console = Console()
error_console = Console(stderr=True)


def inspect_command(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to a .fastdrs checkpoint.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Output checkpoint information as JSON.",
    ),
) -> None:
    """Inspect checkpoint architecture and metadata."""

    try:
        info = inspect_checkpoint(checkpoint)
    except Exception as exc:
        error_console.print(
            f"[red]Checkpoint inspection failed:[/red] {exc}"
        )
        raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(
            json.dumps(info, indent=2, default=str)
        )
        return

    table = Table(
        title="FastDRS Checkpoint",
        show_header=False,
    )

    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("File", str(checkpoint))
    table.add_row("Architecture", info["architecture"])
    table.add_row("Classes", str(info["num_classes"]))
    table.add_row(
        "Parameters",
        f"{info['parameter_count']:,}",
    )
    table.add_row(
        "File size",
        f"{info['size_bytes'] / (1024**2):.2f} MiB",
    )
    table.add_row(
        "Schema version",
        str(info["schema_version"]),
    )

    console.print(table)

    classes_table = Table(title="Class Mapping")
    classes_table.add_column("ID", style="cyan")
    classes_table.add_column("Label")

    for index, label in enumerate(info["classes"]):
        classes_table.add_row(str(index), label)

    console.print(classes_table)

    preprocessing = info["preprocessing"]

    preprocessing_table = Table(
        title="Preprocessing",
        show_header=False,
    )
    preprocessing_table.add_column("Property", style="cyan")
    preprocessing_table.add_column("Value")

    preprocessing_table.add_row(
        "Image size",
        str(preprocessing["img_size"]),
    )
    preprocessing_table.add_row(
        "Ben Graham",
        str(preprocessing["ben_graham"]),
    )
    preprocessing_table.add_row(
        "Mean",
        str(preprocessing["mean"]),
    )
    preprocessing_table.add_row(
        "Std",
        str(preprocessing["std"]),
    )

    console.print(preprocessing_table)

    if info["metadata"]:
        metadata_table = Table(
            title="Metadata",
            show_header=False,
        )
        metadata_table.add_column("Key", style="cyan")
        metadata_table.add_column("Value")

        for key, value in info["metadata"].items():
            metadata_table.add_row(str(key), str(value))

        console.print(metadata_table)