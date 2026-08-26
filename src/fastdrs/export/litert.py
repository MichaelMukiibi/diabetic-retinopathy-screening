import torch
from pathlib import Path
from typing import Optional
from fastdrs.models import build_model

def export_litert(
    checkpoint: str,
    architecture: str,
    output: str,
    img_size: int = 224,
    num_classes: int = 5,
    dropout_rate: float = 0.2,
) -> str:
    """
    Exports a trained PyTorch model to LiteRT (.tflite) format.
    Requires fastdrs[export] dependencies.
    """
    try:
        import litert_torch
    except ImportError:
        raise ImportError(
            "LiteRT export requires additional dependencies. "
            "Install them with: pip install 'fastdrs[export]'"
        )

    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint}")

    # 1. Reconstruct model architecture
    model = build_model(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=False,
        dropout_rate=dropout_rate
    )

    # 2. Load state dict
    device = torch.device('cpu')
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Prepare dummy input (Batch, Channel, Height, Width)
    sample_inputs = (torch.randn(1, 3, img_size, img_size), )

    # 4. Perform conversion using ai-edge-torch
    edge_model = litert_torch.convert(model, sample_inputs)

    # 5. Export to file
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    edge_model.export(str(output_path))

    # Confirm the export was successful
    if not output_path.is_file():
        raise RuntimeError(
            f"LiteRT export completed but no model was created at {output_path}"
        )

    return str(output_path)
