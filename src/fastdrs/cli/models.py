"""Model architecture listing command."""

import typer
from rich.console import Console
from rich.table import Table

from fastdrs.models import SUPPORTED_ARCHITECTURES


console = Console()


def models_command(
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Output supported architectures as JSON.",
    ),
) -> None:
    """List architectures supported by FastDRS."""
    if as_json:
        import json

        typer.echo(
            json.dumps(
                list(SUPPORTED_ARCHITECTURES),
                indent=2,
            )
        )
        return

    table = Table(
        title="Supported FastDRS Models",
        show_header=True,
    )

    table.add_column("Architecture", style="cyan")
    table.add_column("Family", style="green")

    families = {
        "resnet18": "ResNet",
        "resnet50": "ResNet",
        "densenet121": "DenseNet",
        "convnext_tiny": "ConvNeXt",
        "swin_t": "Swin Transformer",
        "mobilenet_v2": "MobileNet",
    }

    for architecture in SUPPORTED_ARCHITECTURES:
        table.add_row(
            architecture,
            families.get(architecture, "Other"),
        )

    console.print(table)