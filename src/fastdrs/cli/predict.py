"""Image inference command."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import typer
from rich.console import Console
from rich.table import Table

from fastdrs.checkpoints import load_checkpoint


console = Console()
error_console = Console(stderr=True)


def predict_command(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to a self-defining .fastdrs checkpoint.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    image: Path = typer.Argument(
        ...,
        help="Path to a retinal fundus image.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Inference device, e.g. cpu, cuda, cuda:0.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Output prediction as JSON.",
    ),
) -> None:
    """Predict diabetic retinopathy severity."""

    try:
        target_device = torch.device(device)

        if target_device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(
                    "CUDA was requested but is unavailable."
                )

        loaded = load_checkpoint(
            checkpoint,
            device=target_device,
        )

        prediction = loaded.predict(image)

    except Exception as exc:
        error_console.print(
            f"[red]Prediction failed:[/red] {exc}"
        )
        raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(
            json.dumps(
                prediction.to_dict(),
                indent=2,
            )
        )
        return

    table = Table(
        title="FastDRS Prediction",
        show_header=False,
    )

    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Image", str(image))
    table.add_row("Architecture", loaded.architecture)
    table.add_row("Class ID", str(prediction.class_id))
    table.add_row("Label", prediction.label)
    table.add_row(
        "Confidence",
        f"{prediction.confidence:.2%}",
    )

    console.print(table)

    probabilities = Table(title="Class Probabilities")
    probabilities.add_column("Class", style="cyan")
    probabilities.add_column(
        "Probability",
        justify="right",
    )

    for label, probability in prediction.probabilities.items():
        probabilities.add_row(
            label,
            f"{probability:.2%}",
        )

    console.print(probabilities)