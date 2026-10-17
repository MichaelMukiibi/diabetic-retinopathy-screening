"""
Exceptions used by the FastDRS checkpoint subsystem.
"""


class FastDRSCheckpointError(Exception):
    """Base exception for FastDRS checkpoint failures."""


class InvalidCheckpointError(FastDRSCheckpointError):
    """Raised when a checkpoint is malformed or invalid."""


class UnsupportedCheckpointVersionError(
    FastDRSCheckpointError
):
    """Raised when the checkpoint schema version is unsupported."""


class UnsupportedArchitectureError(
    FastDRSCheckpointError
):
    """Raised when a checkpoint references an unsupported model."""