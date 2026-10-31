"""
Runtime object for loaded FastDRS checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms


from fastdrs.preprocessing import get_transforms

from .schema import (
    ModelConfig,
    Prediction,
    PreprocessingConfig,
)


@dataclass
class FastDRSCheckpoint:
    """
    Loaded FastDRS checkpoint.

    Bundles the reconstructed model together with its class labels,
    preprocessing configuration, and metadata.
    """

    model: nn.Module
    model_config: ModelConfig
    classes: list[str]
    preprocessing: PreprocessingConfig
    metadata: dict[str, Any]
    path: Path | None = None

    def __post_init__(
        self,
    ) -> None:
        self.model.eval()

    @property
    def architecture(
        self,
    ) -> str:
        """Return checkpoint architecture."""
        return (
            self.model_config.architecture
        )

    @property
    def num_classes(
        self,
    ) -> int:
        """Return number of output classes."""
        return (
            self.model_config.num_classes
        )

    def build_transform(
        self,
    ):
        """
        Build the evaluation transform encoded by this checkpoint.

        Uses the project's canonical preprocessing pipeline with
        training augmentations disabled.
        """

        return get_transforms(
            img_size=self.preprocessing.img_size,
            is_train=False,
            mean=self.preprocessing.mean,
            std=self.preprocessing.std,
            use_ben_graham=self.preprocessing.ben_graham,
        )


    def preprocess(
        self,
        image: (
            str
            | Path
            | Image.Image
        ),
    ) -> Tensor:
        """
        Preprocess an image for inference.

        Args:
            image:
                Image path or PIL image.

        Returns:
            CHW image tensor.
        """
        if isinstance(
            image,
            (str, Path),
        ):
            with Image.open(
                image
            ) as opened:
                image = opened.convert(
                    "RGB"
                )

        elif isinstance(
            image,
            Image.Image,
        ):
            image = image.convert(
                "RGB"
            )

        else:
            raise TypeError(
                "image must be a file path "
                "or PIL.Image.Image."
            )

        transform = (
            self.build_transform()
        )

        return transform(
            image
        )

    @torch.inference_mode()
    def predict(
        self,
        image: (
            str
            | Path
            | Image.Image
        ),
        *,
        device: (
            str
            | torch.device
            | None
        ) = None,
    ) -> Prediction:
        """
        Run inference using this checkpoint.

        Args:
            image:
                Image path or PIL image.

            device:
                Optional inference device. If omitted, uses the
                model's current device.

        Returns:
            Prediction containing class ID, label, confidence,
            and all class probabilities.
        """
        if device is None:
            try:
                device = next(
                    self.model.parameters()
                ).device
            except StopIteration:
                device = torch.device(
                    "cpu"
                )

        else:
            device = torch.device(
                device
            )

            self.model.to(
                device
            )

        tensor = self.preprocess(
            image
        )

        tensor = (
            tensor
            .unsqueeze(0)
            .to(device)
        )

        logits = self.model(
            tensor
        )

        if isinstance(
            logits,
            tuple,
        ):
            logits = logits[0]

        if logits.ndim != 2:
            raise RuntimeError(
                "Expected model output with shape "
                "[batch_size, num_classes], "
                f"got {tuple(logits.shape)}."
            )

        if (
            logits.shape[-1]
            != self.num_classes
        ):
            raise RuntimeError(
                "Model output dimension does not "
                "match checkpoint class count."
            )

        probabilities_tensor = (
            torch.softmax(
                logits,
                dim=-1,
            )[0]
        )

        class_id = int(
            probabilities_tensor
            .argmax()
            .item()
        )

        confidence = float(
            probabilities_tensor[
                class_id
            ].item()
        )

        probabilities = {
            label: float(
                probability.item()
            )
            for label, probability
            in zip(
                self.classes,
                probabilities_tensor,
                strict=True,
            )
        }

        return Prediction(
            class_id=class_id,
            label=(
                self.classes[
                    class_id
                ]
            ),
            confidence=confidence,
            probabilities=probabilities,
        )
