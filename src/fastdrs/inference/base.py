from abc import ABC, abstractmethod
from typing import Any


from .prediction import Prediction


class BasePredictor(ABC):
    """
    Base interface for fastDRS inference backends.
    """

    @abstractmethod
    def predict(self, image: Any) -> Prediction:
        """
        Predict diabetic retinopathy severity for one image.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, images: list[Any]) -> list[Prediction]:
        """
        Predict diabetic retinopathy severity for multiple images.
        """
        raise NotImplementedError