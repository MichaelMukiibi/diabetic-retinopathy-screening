from .base import BasePredictor
from .litert import LiteRTPredictor
from .prediction import Prediction
from .pytorch import PyTorchPredictor


Predictor = PyTorchPredictor


__all__ = [
    "BasePredictor",
    "Prediction",
    "Predictor",
    "PyTorchPredictor",
    "LiteRTPredictor",
]