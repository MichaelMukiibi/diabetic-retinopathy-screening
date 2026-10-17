"""Model export command."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console


console = Console()
error_console = Console(stderr=True)


def export_command(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to a .fastdrs or legacy PyTorch checkpoint.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_format: str = typer.Option(
        "litert",
        "--format",
        "-f",
        help="Export format. Currently supported: litert.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Destination model file.",
    ),
    architecture: Optional[str] = typer.Option(
        None,
        "--architecture",
        help="Architecture for legacy PyTorch checkpoints.",
    ),
    num_classes: int = typer.Option(
        5,
        "--num-classes",
        min=1,
        help="Class count for legacy PyTorch checkpoints.",
    ),
    img_size: int = typer.Option(
        224,
        "--img-size",
        min=1,
        help="Input size for legacy PyTorch checkpoints.",
    ),
    dropout_rate: float = typer.Option(
        0.2,
        "--dropout-rate",
        min=0.0,
        max=0.999,
        help="Dropout rate for legacy PyTorch checkpoints.",
    ),
) -> None:
    """Export a FastDRS model for deployment."""

    if output_format.lower() != "litert":
        error_console.print(
            f"[red]Unsupported export format:[/red] "
            f"{output_format}"
        )
        raise typer.Exit(code=2)

    if output is None:
        output = checkpoint.with_suffix(".tflite")

    # Legacy checkpoint reconstruction requires architecture metadata.
    if (
        checkpoint.suffix.lower() != ".fastdrs"
        and architecture is None
    ):
        error_console.print(
            "[red]Legacy checkpoints require "
            "--architecture.[/red]"
        )
        raise typer.Exit(code=2)

    try:
        # Lazy import: LiteRT export is not needed by other commands.
        from fastdrs.export.litert import export_litert

        result = export_litert(
            checkpoint=checkpoint,
            output=output,
            architecture=architecture,
            num_classes=num_classes,
            img_size=img_size,
            dropout_rate=dropout_rate,
        )

    except ImportError as exc:
        error_console.print(
            f"[red]Export dependency error:[/red] {exc}"
        )
        error_console.print(
            "Install the export extra in this project "
            "with: uv sync --extra export"
        )
        raise typer.Exit(code=1) from exc

    except Exception as exc:
        error_console.print(
            f"[red]Export failed:[/red] {exc}"
        )
        raise typer.Exit(code=1) from exc

    console.print("[green]Export successful.[/green]")
    console.print(f"Model saved to: {result}")