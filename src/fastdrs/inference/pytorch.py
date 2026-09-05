from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..models import build_model
from ..preprocessing import get_transforms
from .base import BasePredictor
from .prediction import DEFAULT_CLASS_NAMES, Prediction


class PyTorchPredictor(BasePredictor):
    """
    PyTorch inference backend for fastDRS.

    Parameters
    ----------
    model:
        Loaded PyTorch model.
    device:
        Device used for inference.
    img_size:
        Input image size.
    use_ben_graham:
        Whether to apply Ben Graham preprocessing.
    class_names:
        Names corresponding to model output classes.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device = "cpu",
        img_size: int = 224,
        use_ben_graham: bool = False,
        class_names: list[str] | None = None,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.img_size = img_size
        self.use_ben_graham = use_ben_graham
        self.class_names = class_names or DEFAULT_CLASS_NAMES

        self.transform = get_transforms(
            img_size=img_size,
            is_train=False,
            use_ben_graham=use_ben_graham,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        architecture: str = "resnet50",
        num_classes: int = 5,
        pretrained: bool = False,
        device: str | torch.device | None = None,
        img_size: int = 224,
        use_ben_graham: bool = False,
        class_names: list[str] | None = None,
    ) -> "PyTorchPredictor":
        """
        Load a predictor from a fastDRS PyTorch checkpoint.

        Parameters
        ----------
        checkpoint_path:
            Path to the .pth checkpoint.
        architecture:
            Model architecture used during training.
        num_classes:
            Number of output classes.
        pretrained:
            Whether to initialize the base architecture with pretrained
            weights before loading the checkpoint.
        device:
            Inference device. Defaults to CUDA when available.
        """

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = build_model(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        # Current fastDRS training pipeline saves model.state_dict().
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.eval()

        return cls(
            model=model,
            device=device,
            img_size=img_size,
            use_ben_graham=use_ben_graham,
            class_names=class_names,
        )

    def _load_image(self, image: Any) -> Image.Image:
        """
        Convert supported image inputs into a PIL image.
        """

        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return Image.fromarray(image).convert("RGB")

            if image.ndim == 3:
                return Image.fromarray(image).convert("RGB")

            raise ValueError(
                "NumPy image must have shape (H, W) or (H, W, C)."
            )

        raise TypeError(
            "Unsupported image type. "
            "Expected a file path, PIL.Image.Image, or numpy.ndarray."
        )

    def _preprocess(self, image: Any) -> torch.Tensor:
        pil_image = self._load_image(image)

        tensor = self.transform(pil_image)

        return tensor.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict(self, image: Any) -> Prediction:
        """
        Run inference on a single image.
        """

        tensor = self._preprocess(image)

        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        class_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[class_id].item())

        probabilities_np = probabilities.cpu().numpy()

        class_name = self.class_names[class_id]

        return Prediction(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            probabilities=probabilities_np,
        )

    @torch.inference_mode()
    def predict_batch(self, images: list[Any]) -> list[Prediction]:
        """
        Run inference on multiple images.
        """

        if not images:
            return []

        tensors = [
            self._preprocess(image).squeeze(0)
            for image in images
        ]

        batch = torch.stack(tensors).to(self.device)

        logits = self.model(batch)
        probabilities = torch.softmax(logits, dim=1)

        predictions = []

        for probs in probabilities:
            class_id = int(torch.argmax(probs).item())
            confidence = float(probs[class_id].item())

            probs_np = probs.cpu().numpy()

            predictions.append(
                Prediction(
                    class_id=class_id,
                    class_name=self.class_names[class_id],
                    confidence=confidence,
                    probabilities=probs_np,
                )
            )

        return predictions

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device=None,
        version: str = "v0.1.0",
        **kwargs,
    ):
        from ..weights import (
            download_model,
            get_model_artifact,
        )

        checkpoint = download_model(
            model_name=model_name,
            backend="pytorch",
            version=version,
        )

        artifact = get_model_artifact(
            model_name=model_name,
            backend="pytorch",
            version=version,
        )

        return cls.from_checkpoint(
            checkpoint_path=checkpoint,
            architecture=artifact.architecture,
            num_classes=artifact.num_classes,
            device=device,
            img_size=artifact.img_size,
            **kwargs,
        )