"""
Schema definitions for FastDRS self-defining checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


FASTDRS_CHECKPOINT_FORMAT = "fastdrs-checkpoint"
FASTDRS_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration required to reconstruct a FastDRS model.
    """

    architecture: str
    num_classes: int
    kwargs: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration."""
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "ModelConfig":
        """Construct configuration from serialized data."""
        return cls(
            architecture=str(
                data["architecture"]
            ),
            num_classes=int(
                data["num_classes"]
            ),
            kwargs=dict(
                data.get("kwargs", {})
            ),
        )


@dataclass(frozen=True)
class PreprocessingConfig:
    """
    Image preprocessing configuration stored in a checkpoint.
    """

    img_size: int  = 224

    mean: tuple[
        float,
        float,
        float,
    ] = (
        0.485,
        0.456,
        0.406,
    )

    std: tuple[
        float,
        float,
        float,
    ] = (
        0.229,
        0.224,
        0.225,
    )

    ben_graham: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize preprocessing configuration."""
        return {
            "img_size": int(self.img_size),
            "mean": list(
                self.mean
            ),
            "std": list(
                self.std
            ),
            "ben_graham": self.ben_graham,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "PreprocessingConfig":
        """Construct preprocessing configuration."""
        return cls(
            img_size=int(data.get("img_size", 224)),
            mean=tuple(
                data.get(
                    "mean",
                    (
                        0.485,
                        0.456,
                        0.406,
                    ),
                )
            ),
            std=tuple(
                data.get(
                    "std",
                    (
                        0.229,
                        0.224,
                        0.225,
                    ),
                )
            ),
            ben_graham=bool(
                data.get(
                    "ben_graham",
                    False,
                )
            ),
        )


@dataclass(frozen=True)
class Prediction:
    """
    Inference result produced by a FastDRS checkpoint.
    """

    class_id: int
    label: str
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction to a JSON-friendly dictionary."""
        return asdict(self)
