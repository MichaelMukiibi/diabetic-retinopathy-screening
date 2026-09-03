from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative",
]


@dataclass
class Prediction:
    """
    Result of a diabetic retinopathy prediction.
    """

    class_id: int
    class_name: str
    confidence: float
    probabilities: np.ndarray

    @property
    def probabilities_dict(self) -> dict[str, float]:
        """Return probabilities mapped to class names."""
        return {
            name: float(prob)
            for name, prob in zip(
                DEFAULT_CLASS_NAMES,
                self.probabilities,
            )
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction to a JSON-serializable dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "probabilities": self.probabilities_dict,
        }

    def __repr__(self) -> str:
        return (
            f"Prediction("
            f"class_id={self.class_id}, "
            f"class_name='{self.class_name}', "
            f"confidence={self.confidence:.4f}"
            f")"
        )