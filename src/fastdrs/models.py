"""
Model architecture factory for Diabetic Retinopathy classification.

Supports building and benchmarking multiple vision architectures:
- ResNet (e.g. resnet50, resnet18)
- DenseNet (e.g. densenet121)
- ConvNeXt (e.g. convnext_tiny)
- Swin Transformer (e.g. swin_t)
- MobileNetV2 (efficient baseline)
"""

from typing import Any, Dict

import torch.nn as nn
import torchvision.models as models


SUPPORTED_ARCHITECTURES = (
    "resnet50",
    "resnet18",
    "densenet121",
    "convnext_tiny",
    "swin_t",
    "mobilenet_v2",
)


def _canonicalize_architecture(architecture: str) -> str:
    """
    Convert architecture aliases into canonical FastDRS names.

    Args:
        architecture:
            Architecture name or alias.

    Returns:
        Canonical FastDRS architecture name.
    """
    arch = architecture.lower().strip()

    aliases = {
        "convnext": "convnext_tiny",
        "swin": "swin_t",
        "mobilenet": "mobilenet_v2",
    }

    return aliases.get(arch, arch)


def build_model(
    architecture: str = "resnet50",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
) -> nn.Module:
    """
    Construct a vision model with a customized classification head.

    Models created through this function carry FastDRS reconstruction
    metadata in ``model.fastdrs_config``. This allows checkpoints to
    reconstruct the model architecture automatically.

    Args:
        architecture:
            Choice of resnet50, resnet18, densenet121,
            convnext_tiny, swin_t, or mobilenet_v2.

        num_classes:
            Number of output severity classes.

        pretrained:
            Whether to initialize the backbone with ImageNet weights.

        dropout_rate:
            Dropout probability before the final classification layer.

    Returns:
        Constructed PyTorch model.

    Raises:
        ValueError:
            If the architecture is unsupported.
    """
    if num_classes <= 0:
        raise ValueError(
            "num_classes must be greater than zero."
        )

    if not 0 <= dropout_rate < 1:
        raise ValueError(
            "dropout_rate must satisfy 0 <= dropout_rate < 1."
        )

    arch = _canonicalize_architecture(architecture)

    if arch == "resnet50":
        weights = (
            models.ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.resnet50(weights=weights)

        in_features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    elif arch == "resnet18":
        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.resnet18(weights=weights)

        in_features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    elif arch == "densenet121":
        weights = (
            models.DenseNet121_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.densenet121(weights=weights)

        in_features = model.classifier.in_features

        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    elif arch == "convnext_tiny":
        weights = (
            models.ConvNeXt_Tiny_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.convnext_tiny(weights=weights)

        in_features = model.classifier[2].in_features

        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    elif arch == "swin_t":
        weights = (
            models.Swin_T_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.swin_t(weights=weights)

        in_features = model.head.in_features

        model.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    elif arch == "mobilenet_v2":
        weights = (
            models.MobileNet_V2_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.mobilenet_v2(weights=weights)

        in_features = model.classifier[1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    else:
        supported = ", ".join(SUPPORTED_ARCHITECTURES)

        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Supported: {supported}"
        )

    # Store only configuration required to reconstruct the network.
    #
    # `pretrained` is intentionally excluded because checkpoint loading
    # restores the complete state_dict and should never need to download
    # ImageNet weights.
    model.fastdrs_config = {
        "architecture": arch,
        "num_classes": num_classes,
        "kwargs": {
            "dropout_rate": dropout_rate,
        },
    }

    return model


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Calculate parameter count and estimated model memory size in MB.

    Args:
        model:
            PyTorch model.

    Returns:
        Dictionary containing total parameters, trainable parameters,
        and estimated float32 model size.
    """
    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    param_size_mb = (
        total_params * 4
    ) / (1024 * 1024)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(param_size_mb, 2),
    }