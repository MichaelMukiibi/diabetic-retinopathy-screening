"""FastDRS command-line interface."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import typer
from rich.console import Console

from .export import export_command
from .inspect import inspect_command
from .models import models_command
from .predict import predict_command


app = typer.Typer(
    name="fastdrs",
    help=(
        "FastDRS — tools for diabetic retinopathy "
        "model inspection, inference, and deployment."
    ),
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

console = Console(stderr=True)


def get_version() -> str:
    """Return the installed FastDRS version."""
    try:
        return version("fastdrs")
    except PackageNotFoundError:
        return "unknown"


def version_callback(value: bool) -> None:
    """Display the package version and exit."""
    if value:
        typer.echo(f"FastDRS {get_version()}")
        raise typer.Exit()


@app.callback()
def main(
    show_version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show the installed FastDRS version.",
    ),
) -> None:
    """FastDRS command-line utilities."""


app.command(
    name="models",
    help="List supported model architectures.",
)(models_command)

app.command(
    name="inspect",
    help="Inspect a self-defining FastDRS checkpoint.",
)(inspect_command)

app.command(
    name="predict",
    help="Run inference on a retinal image.",
)(predict_command)

app.command(
    name="export",
    help="Export a model for deployment.",
)(export_command)


if __name__ == "__main__":
    app()