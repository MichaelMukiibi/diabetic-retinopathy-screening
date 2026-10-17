"""
Validation utilities for FastDRS checkpoints.
"""

from __future__ import annotations

from typing import Any

from .exceptions import (
    InvalidCheckpointError,
    UnsupportedCheckpointVersionError,
)
from .schema import (
    FASTDRS_CHECKPOINT_FORMAT,
    FASTDRS_CHECKPOINT_SCHEMA_VERSION,
) 


_REQUIRED_TOP_LEVEL_FIELDS = {
    "format",
    "schema_version",
    "model",
    "classes",
    "preprocessing",
    "state_dict",
    "metadata",
}


def validate_checkpoint(
    checkpoint: dict[str, Any],
) -> None:
    """
    Validate a deserialized FastDRS checkpoint.

    Args:
        checkpoint:
            Deserialized checkpoint dictionary.

    Raises:
        InvalidCheckpointError:
            If the checkpoint structure is invalid.

        UnsupportedCheckpointVersionError:
            If its schema version is unsupported.
    """
    if not isinstance(checkpoint, dict):
        raise InvalidCheckpointError(
            "Checkpoint must deserialize to a dictionary."
        )

    missing = (
        _REQUIRED_TOP_LEVEL_FIELDS
        - checkpoint.keys()
    )

    if missing:
        fields = ", ".join(
            sorted(missing)
        )

        raise InvalidCheckpointError(
            "Checkpoint is missing required fields: "
            f"{fields}"
        )

    if (
        checkpoint["format"]
        != FASTDRS_CHECKPOINT_FORMAT
    ):
        raise InvalidCheckpointError(
            "File is not a FastDRS checkpoint. "
            f"Expected format "
            f"{FASTDRS_CHECKPOINT_FORMAT!r}, "
            f"got {checkpoint['format']!r}."
        )

    schema_version = checkpoint[
        "schema_version"
    ]

    if (
        schema_version
        != FASTDRS_CHECKPOINT_SCHEMA_VERSION
    ):
        raise UnsupportedCheckpointVersionError(
            "Unsupported FastDRS checkpoint "
            f"schema version {schema_version!r}. "
            "This installation supports schema "
            f"version "
            f"{FASTDRS_CHECKPOINT_SCHEMA_VERSION}."
        )

    _validate_model(
        checkpoint["model"]
    )

    _validate_classes(
        checkpoint["classes"]
    )

    _validate_preprocessing(
        checkpoint["preprocessing"]
    )

    if (
        len(checkpoint["classes"])
        != checkpoint["model"]["num_classes"]
    ):
        raise InvalidCheckpointError(
            "Number of class labels does not match "
            "'model.num_classes'."
        )

    if not isinstance(
        checkpoint["state_dict"],
        dict,
    ):
        raise InvalidCheckpointError(
            "'state_dict' must be a dictionary."
        )

    if not checkpoint["state_dict"]:
        raise InvalidCheckpointError(
            "'state_dict' must not be empty."
        )

    if not isinstance(
        checkpoint["metadata"],
        dict,
    ):
        raise InvalidCheckpointError(
            "'metadata' must be a dictionary."
        )


def _validate_model(
    model: Any,
) -> None:
    if not isinstance(
        model,
        dict,
    ):
        raise InvalidCheckpointError(
            "'model' must be a dictionary."
        )

    required = {
        "architecture",
        "num_classes",
        "kwargs",
    }

    missing = (
        required
        - model.keys()
    )

    if missing:
        fields = ", ".join(
            sorted(missing)
        )

        raise InvalidCheckpointError(
            "Model configuration is missing: "
            f"{fields}"
        )

    architecture = model[
        "architecture"
    ]

    if (
        not isinstance(
            architecture,
            str,
        )
        or not architecture.strip()
    ):
        raise InvalidCheckpointError(
            "'model.architecture' must be "
            "a non-empty string."
        )

    num_classes = model[
        "num_classes"
    ]

    if (
        not isinstance(
            num_classes,
            int,
        )
        or num_classes <= 0
    ):
        raise InvalidCheckpointError(
            "'model.num_classes' must be "
            "a positive integer."
        )

    if not isinstance(
        model["kwargs"],
        dict,
    ):
        raise InvalidCheckpointError(
            "'model.kwargs' must be a dictionary."
        )


def _validate_classes(
    classes: Any,
) -> None:
    if not isinstance(
        classes,
        list,
    ):
        raise InvalidCheckpointError(
            "'classes' must be a list."
        )

    if not classes:
        raise InvalidCheckpointError(
            "'classes' must not be empty."
        )

    if not all(
        isinstance(
            label,
            str,
        )
        and label.strip()
        for label in classes
    ):
        raise InvalidCheckpointError(
            "Every class label must be "
            "a non-empty string."
        )

    if len(
        set(classes)
    ) != len(classes):
        raise InvalidCheckpointError(
            "Class labels must be unique."
        )


def _validate_preprocessing(
    config: Any,
) -> None:
    if not isinstance(
        config,
        dict,
    ):
        raise InvalidCheckpointError(
            "'preprocessing' must be a dictionary."
        )

    required = {
        "img_size",
        "mean",
        "std",
        "ben_graham",
    }

    missing = (
        required
        - config.keys()
    )

    if missing:
        fields = ", ".join(
            sorted(missing)
        )

        raise InvalidCheckpointError(
            "Preprocessing configuration "
            f"is missing: {fields}"
        )

    image_size = config[
        "image_size"
    ]

    if (
        not isinstance(
            image_size,
            int,
        )
        or image_size <= 0
    ):
        raise InvalidCheckpointError(
            "'preprocessing.image_size' must "
            "be a positive integer."
        )

    for field in (
        "mean",
        "std",
    ):
        value = config[field]

        if (
            not isinstance(
                value,
                (list, tuple),
            )
            or len(value) != 3
            or not all(
                isinstance(
                    item,
                    (int, float),
                )
                for item in value
            )
        ):
            raise InvalidCheckpointError(
                f"'preprocessing.{field}' "
                "must contain three numeric values."
            )

    if not all(
        value > 0
        for value in config["std"]
    ):
        raise InvalidCheckpointError(
            "'preprocessing.std' values must "
            "be greater than zero."
        )

    if not isinstance(
        config["ben_graham"],
        bool,
    ):
        raise InvalidCheckpointError(
            "'preprocessing.ben_graham' "
            "must be a boolean."
        )