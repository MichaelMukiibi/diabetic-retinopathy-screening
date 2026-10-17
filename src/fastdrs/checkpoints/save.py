"""
Checkpoint saving utilities.
"""

from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .schema import (
    FASTDRS_CHECKPOINT_FORMAT,
    FASTDRS_CHECKPOINT_SCHEMA_VERSION,
    ModelConfig,
    PreprocessingConfig,
)
from .validation import (
    validate_checkpoint,
)


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    classes: list[str],
    preprocessing: (
        PreprocessingConfig | None
    ) = None,
    metadata: (
        dict[str, Any] | None
    ) = None,
) -> Path:
    """
    Save a self-defining FastDRS checkpoint.

    The model must have been constructed through FastDRS
    ``build_model()`` so that reconstruction metadata is
    available through ``model.fastdrs_config``.

    Args:
        model:
            FastDRS model.

        path:
            Destination path.

        classes:
            Ordered output class labels.

        preprocessing:
            Input preprocessing configuration.

        metadata:
            Optional training, dataset, metrics, or experiment
            metadata.

    Returns:
        Path to the saved checkpoint.

    Raises:
        TypeError:
            If model is not a torch.nn.Module.

        ValueError:
            If reconstruction metadata is unavailable or class
            labels do not match the model output count.
    """
    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "model must be an instance "
            "of torch.nn.Module."
        )

    if not hasattr(
        model,
        "fastdrs_config",
    ):
        raise ValueError(
            "Model does not contain FastDRS "
            "reconstruction metadata. "
            "Create it using fastdrs.build_model()."
        )

    if not classes:
        raise ValueError(
            "classes must contain at least "
            "one class label."
        )

    raw_config = (
        model.fastdrs_config
    )

    if not isinstance(
        raw_config,
        dict,
    ):
        raise ValueError(
            "model.fastdrs_config must be "
            "a dictionary."
        )

    model_config = ModelConfig(
        architecture=raw_config[
            "architecture"
        ],
        num_classes=raw_config[
            "num_classes"
        ],
        kwargs=dict(
            raw_config.get(
                "kwargs",
                {},
            )
        ),
    )

    if (
        len(classes)
        != model_config.num_classes
    ):
        raise ValueError(
            "Number of class labels does "
            "not match model num_classes "
            f"({model_config.num_classes})."
        )

    preprocessing = (
        preprocessing
        or PreprocessingConfig()
    )

    checkpoint_metadata = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        **(metadata or {}),
    }

    checkpoint: dict[str, Any] = {
        "format": (
            FASTDRS_CHECKPOINT_FORMAT
        ),
        "schema_version": (
            FASTDRS_CHECKPOINT_SCHEMA_VERSION
        ),
        "model": (
            model_config.to_dict()
        ),
        "classes": list(classes),
        "preprocessing": (
            preprocessing.to_dict()
        ),
        "state_dict": (
            model.state_dict()
        ),
        "metadata": (
            checkpoint_metadata
        ),
    }

    validate_checkpoint(
        checkpoint
    )

    destination = Path(path)

    if destination.suffix == "":
        destination = (
            destination.with_suffix(
                ".fastdrs"
            )
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        destination,
    )

    return destination