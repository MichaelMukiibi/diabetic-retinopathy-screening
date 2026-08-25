"""
Model architecture factory for Diabetic Retinopathy classification.

Supports building and benchmarking multiple vision architectures:
- ResNet (e.g. resnet50, resnet18)
- DenseNet (e.g. densenet121)
- ConvNeXt (e.g. convnext_tiny)
- Swin Transformer (e.g. swin_t)
- MobileNetV2 (efficient baseline)
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torchvision.models as models


def build_model(
    architecture: str = "resnet50",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout_rate: float = 0.2
) -> nn.Module:
    """
    Constructs a vision model with customized classification head.
    
    Args:
        architecture: Choice of 'resnet50', 'resnet18', 'densenet121', 'convnext_tiny', 'swin_t', 'mobilenet_v2'.
        num_classes: Number of output severity classes (default: 5).
        pretrained: Whether to use ImageNet pre-trained weights.
        dropout_rate: Dropout rate before classification layer.
    """
    arch = architecture.lower()
    
    if arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    elif arch == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    elif arch in ("convnext_tiny", "convnext"):
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    elif arch in ("swin_t", "swin"):
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    elif arch in ("mobilenet_v2", "mobilenet"):
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError(f"Unsupported architecture '{architecture}'. "
                         f"Supported: resnet50, resnet18, densenet121, convnext_tiny, swin_t, mobilenet_v2")

    return model


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Calculates parameter count and estimated model memory size in MB.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate size assuming float32 (4 bytes per param)
    param_size_mb = (total_params * 4) / (1024 * 1024)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(param_size_mb, 2)
    }
