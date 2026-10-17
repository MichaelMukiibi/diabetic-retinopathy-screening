"""
Checkpoint loading utilities.
"""

from __future__ import annotations

from pathlib import Path

import torch

from fastdrs.models import (
    build_model,
)

from .checkpoint import (
    FastDRSCheckpoint,
)
from .exceptions import (
    InvalidCheckpointError,
    UnsupportedArchitectureError,
)
from .schema import (
    ModelConfig,
    PreprocessingConfig,
)
from .validation import (
    validate_checkpoint,
)


def _create_model(
    config: ModelConfig,
):
    """
    Reconstruct a model from FastDRS checkpoint metadata.

    Pretrained weights are explicitly disabled because the
    checkpoint already contains the complete trained state dict.
    """
    try:
        return build_model(
            architecture=(
                config.architecture
            ),
            num_classes=(
                config.num_classes
            ),
            pretrained=False,
            **config.kwargs,
        )

    except ValueError as exc:
        raise UnsupportedArchitectureError(
            "Unable to reconstruct architecture "
            f"{config.architecture!r}."
        ) from exc


def load_checkpoint(
    path: str | Path,
    *,
    device: (
        str | torch.device
    ) = "cpu",
) -> FastDRSCheckpoint:
    """
    Load and reconstruct a self-defining FastDRS checkpoint.

    Args:
        path:
            Path to the checkpoint.

        device:
            Device onto which the model should be loaded.

    Returns:
        Loaded FastDRSCheckpoint.

    Raises:
        FileNotFoundError:
            If the file does not exist.

        InvalidCheckpointError:
            If the checkpoint is malformed or incompatible.
    """
    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint not found: "
            f"{checkpoint_path}"
        )

    if not checkpoint_path.is_file():
        raise InvalidCheckpointError(
            "Checkpoint path is not a file: "
            f"{checkpoint_path}"
        )

    try:
        raw = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    except Exception as exc:
        raise InvalidCheckpointError(
            "Could not deserialize checkpoint: "
            f"{checkpoint_path}"
        ) from exc

    validate_checkpoint(
        raw
    )

    model_config = (
        ModelConfig.from_dict(
            raw["model"]
        )
    )

    preprocessing = (
        PreprocessingConfig.from_dict(
            raw["preprocessing"]
        )
    )

    model = _create_model(
        model_config
    )

    try:
        model.load_state_dict(
            raw["state_dict"],
            strict=True,
        )

    except RuntimeError as exc:
        raise InvalidCheckpointError(
            "Checkpoint weights are incompatible "
            "with architecture "
            f"{model_config.architecture!r}."
        ) from exc

    model.to(device)
    model.eval()

    return FastDRSCheckpoint(
        model=model,
        model_config=model_config,
        classes=list(
            raw["classes"]
        ),
        preprocessing=preprocessing,
        metadata=dict(
            raw["metadata"]
        ),
        path=checkpoint_path,
    )