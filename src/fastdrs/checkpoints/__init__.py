"""
FastDRS self-defining checkpoint API.
"""

from .checkpoint import (
    FastDRSCheckpoint,
)
from .exceptions import (
    FastDRSCheckpointError,
    InvalidCheckpointError,
    UnsupportedArchitectureError,
    UnsupportedCheckpointVersionError,
)
from .inspect import (
    inspect_checkpoint,
)
from .load import (
    load_checkpoint,
)
from .save import (
    save_checkpoint,
)
from .schema import (
    FASTDRS_CHECKPOINT_FORMAT,
    FASTDRS_CHECKPOINT_SCHEMA_VERSION,
    ModelConfig,
    Prediction,
    PreprocessingConfig,
)


__all__ = [
    "FASTDRS_CHECKPOINT_FORMAT",
    "FASTDRS_CHECKPOINT_SCHEMA_VERSION",
    "FastDRSCheckpoint",
    "FastDRSCheckpointError",
    "InvalidCheckpointError",
    "ModelConfig",
    "Prediction",
    "PreprocessingConfig",
    "UnsupportedArchitectureError",
    "UnsupportedCheckpointVersionError",
    "inspect_checkpoint",
    "load_checkpoint",
    "save_checkpoint",
]