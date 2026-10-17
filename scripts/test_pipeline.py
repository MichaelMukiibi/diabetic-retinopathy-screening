import torch

from fastdrs import (
    PreprocessingConfig,
    build_model,
    load_checkpoint,
    save_checkpoint,
)


classes = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]


torch.manual_seed(42)

model = build_model(
    architecture="resnet18",
    num_classes=5,
    pretrained=False,
    dropout_rate=0.2,
)

model.eval()

x = torch.randn(
    1,
    3,
    224,
    224,
)

with torch.inference_mode():
    expected = model(x)


path = save_checkpoint(
    model,
    "test-model.fastdrs",
    classes=classes,
    preprocessing=PreprocessingConfig(
        img_size=224,
    ),
    metadata={
        "dataset": "APTOS 2019",
        "purpose": "checkpoint roundtrip test",
    },
)

print(
    f"Saved checkpoint to: {path}"
)


del model


checkpoint = load_checkpoint(
    path
)

with torch.inference_mode():
    actual = checkpoint.model(
        x
    )


torch.testing.assert_close(
    expected,
    actual,
)


print("Checkpoint roundtrip passed.")
print(
    "Architecture:",
    checkpoint.architecture,
)
print(
    "Classes:",
    checkpoint.classes,
)
print(
    "Metadata:",
    checkpoint.metadata,
)