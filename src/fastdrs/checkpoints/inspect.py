"""
Checkpoint inspection utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .exceptions import (
    InvalidCheckpointError,
)
from .schema import (
    ModelConfig,
    PreprocessingConfig,
)
from .validation import (
    validate_checkpoint,
)


def inspect_checkpoint(
    path: str | Path,
) -> dict[str, Any]:
    """
    Inspect checkpoint metadata without reconstructing the model.

    Args:
        path:
            FastDRS checkpoint path.

    Returns:
        JSON-friendly checkpoint information.
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

    parameter_count = sum(
        tensor.numel()
        for tensor
        in raw["state_dict"].values()
        if isinstance(
            tensor,
            torch.Tensor,
        )
    )

    return {
        "path": str(
            checkpoint_path
        ),
        "format": raw[
            "format"
        ],
        "schema_version": raw[
            "schema_version"
        ],
        "architecture": (
            model_config.architecture
        ),
        "num_classes": (
            model_config.num_classes
        ),
        "model_kwargs": dict(
            model_config.kwargs
        ),
        "classes": list(
            raw["classes"]
        ),
        "preprocessing": (
            preprocessing.to_dict()
        ),
        "metadata": dict(
            raw["metadata"]
        ),
        "parameter_count": (
            parameter_count
        ),
        "size_bytes": (
            checkpoint_path
            .stat()
            .st_size
        ),
    }