from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..preprocessing import get_transforms
from .base import BasePredictor
from .prediction import DEFAULT_CLASS_NAMES, Prediction


class LiteRTPredictor(BasePredictor):
    """
    LiteRT/TFLite inference backend.

    The runtime is imported lazily so that LiteRT dependencies remain
    optional.
    """

    def __init__(
        self,
        model_path: str | Path,
        img_size: int = 224,
        use_ben_graham: bool = False,
        class_names: list[str] | None = None,
    ):
        self.model_path = Path(model_path)
        self.img_size = img_size
        self.use_ben_graham = use_ben_graham
        self.class_names = class_names or DEFAULT_CLASS_NAMES

        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as exc:
            raise ImportError(
                "LiteRT inference requires the optional LiteRT dependency. "
                "Install it with the fastDRS export/runtime extra."
            ) from exc

        self.interpreter = Interpreter(
            model_path=str(self.model_path)
        )

        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.transform = get_transforms(
            img_size=img_size,
            is_train=False,
            use_ben_graham=use_ben_graham,
        )

    def _load_image(self, image: Any) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")

        raise TypeError(
            "Unsupported image type. "
            "Expected a file path, PIL.Image.Image, or numpy.ndarray."
        )

    def _preprocess(self, image: Any) -> np.ndarray:
        pil_image = self._load_image(image)

        tensor = self.transform(pil_image)

        array = tensor.numpy()

        # PyTorch transform gives:
        # (C, H, W)
        #
        # Most LiteRT image models expect:
        # (H, W, C)
        array = np.transpose(array, (1, 2, 0))

        return array.astype(np.float32)

    def predict(self, image: Any) -> Prediction:
        input_data = self._preprocess(image)

        input_shape = self.input_details[0]["shape"]

        # Add batch dimension.
        input_data = np.expand_dims(input_data, axis=0)

        # Handle models whose input shape is dynamically defined.
        if tuple(input_shape) != tuple(input_data.shape):
            try:
                self.interpreter.resize_tensor_input(
                    self.input_details[0]["index"],
                    input_data.shape,
                )
                self.interpreter.allocate_tensors()

                self.input_details = (
                    self.interpreter.get_input_details()
                )
                self.output_details = (
                    self.interpreter.get_output_details()
                )
            except Exception:
                pass

        input_index = self.input_details[0]["index"]

        self.interpreter.set_tensor(
            input_index,
            input_data,
        )

        self.interpreter.invoke()

        output_index = self.output_details[0]["index"]

        output = self.interpreter.get_tensor(output_index)

        probabilities = np.asarray(output[0], dtype=np.float32)

        # Some exported models return logits rather than probabilities.
        # Normalize if necessary.
        if not np.isclose(
            probabilities.sum(),
            1.0,
            atol=1e-3,
        ) or np.any(probabilities < 0):
            exp = np.exp(
                probabilities - np.max(probabilities)
            )
            probabilities = exp / exp.sum()

        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])

        return Prediction(
            class_id=class_id,
            class_name=self.class_names[class_id],
            confidence=confidence,
            probabilities=probabilities,
        )

    def predict_batch(self, images: list[Any]) -> list[Prediction]:
        return [self.predict(image) for image in images]

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        version: str = "v0.1.0",
        **kwargs,
    ):
        from ..weights import (
            download_model,
            get_model_artifact,
        )

        model_path = download_model(
            model_name=model_name,
            backend="litert",
            version=version,
        )

        artifact = get_model_artifact(
            model_name=model_name,
            backend="litert",
            version=version,
        )

        return cls(
            model_path=model_path,
            img_size=artifact.img_size,
            **kwargs,
        )