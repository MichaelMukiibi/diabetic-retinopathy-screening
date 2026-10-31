import torch
from pathlib import Path
from typing import Optional

from fastdrs.checkpoints import load_checkpoint
from fastdrs.models import build_model

def export_litert(
    checkpoint: str | Path,
    output: str | Path,
    *,
    architecture: str | None = None,
    img_size: int | None = None,
    num_classes: int = 5,
    dropout_rate: float = 0.2,
) -> str:
    """
    Exports a trained PyTorch model to LiteRT (.tflite) format.
    Supports:
        - Self-defining .fastdrs checkpoints
        - Legacy PyTorch state_dict checkpoints

    For self-defining checkpoints, architecture, preprocessing parameters are extracted automatically

    Legacy checkpoints require an explicit architecture and preprocessing parameters to be provided.
    Requires fastdrs[export] dependencies.
    """
    try:
        import litert_torch
    except ImportError:
        raise ImportError(
            "LiteRT export requires additional dependencies. "
            "Install them with: uv add 'fastdrs[export]'"
        )

    checkpoint_path = Path(checkpoint)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint}")

    output_path = Path(output)

    if output_path.suffix.lower() != ".tflite":
        raise ValueError("LiteRT output must have .tflite extension")

    # 1. Load model architecture and preprocessing parameters
    if checkpoint_path.suffix.lower() == ".fastdrs":
        loaded = load_checkpoint(checkpoint_path, device="cpu")

        model = loaded.model
        resolved_img_size = loaded.preprocessing.img_size
    else:
        # Legacy raw PyTorch state_dict
        if architecture is None:
            raise ValueError(
                "Legacy PyTorch checkpoints require an explicit architecture."
            )

        model = build_model(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=False,
            dropout_rate=dropout_rate,
        )

        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        model.eval()
        resolved_img_size = img_size or 224

    model.eval()

    sample_inputs = (
        torch.randn(
            1,
            3,
            resolved_img_size,
            resolved_img_size,
        ),
    )

    edge_model = litert_torch.convert(
        model,
        sample_inputs,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    edge_model.export(
        str(output_path)
    )

    if not output_path.is_file():
        raise RuntimeError(
            "LiteRT export finished but no output file "
            f"was created at {output_path}."
        )

    return str(output_path)
