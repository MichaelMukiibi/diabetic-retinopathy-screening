# fastdrs

**Fast diabetic retinopathy screening and model development toolkit for PyTorch.**

`fastdrs` provides reusable components for building, training, evaluating, and eventually deploying diabetic retinopathy classification models. It is designed to separate the reusable ML functionality from the research and experimentation code in the `diabetic-retinopathy-screening` project.

> **Status:** Early development (`0.1.0`)

---

## Features

* 🧠 **Multiple vision architectures**

  * ResNet18
  * ResNet50
  * DenseNet121
  * ConvNeXt-Tiny
  * Swin Transformer-Tiny
  * MobileNetV2

* 🖼️ **Fundus image preprocessing**

  * Image resizing
  * ImageNet normalization
  * Data augmentation
  * Fundus boundary cropping
  * Ben Graham preprocessing

* 📊 **Dataset handling**

  * Hugging Face datasets
  * EYEPACS-compatible datasets
  * Reproducible train/validation/test splitting
  * PyTorch `Dataset` and `DataLoader` integration

* 🚂 **Model training**

  * PyTorch training loops
  * AdamW optimization
  * Cosine annealing learning-rate scheduling
  * Best-checkpoint saving
  * Optional Weights & Biases integration

* 📈 **Evaluation**

  * Accuracy
  * Precision
  * Sensitivity/Recall
  * Specificity
  * Macro F1
  * Weighted F1
  * Confusion matrices
  * Per-class metrics
  * Inference latency
  * Model parameter and size estimates

* 📱 **Deployment**

  * LiteRT export support is being developed for Android/on-device inference.

---

## Installation

### Basic installation

```bash
pip install fastdrs
```

Or with `uv`:

```bash
uv add fastdrs
```

### Training dependencies

Training functionality requires the training dependencies:

```bash
pip install "fastdrs[training]"
```

or:

```bash
uv add "fastdrs[training]"
```

### Development installation

Clone the repository and install the development environment with `uv`:

```bash
git clone https://github.com/MichaelMukiibi/diabetic-retinopathy-screening.git
cd diabetic-retinopathy-screening

uv sync
```

---

## Quick Start

### Build a model

```python
from fastdrs.models import build_model, get_model_info

model = build_model(
    architecture="mobilenet_v2",
    num_classes=5,
    pretrained=True,
)

info = get_model_info(model)

print(info)
```

Example output:

```text
{
    'total_parameters': ...,
    'trainable_parameters': ...,
    'model_size_mb': ...
}
```

---

## Supported architectures

The model factory currently supports:

```python
from fastdrs.models import build_model
```

| Architecture          | Identifier      |
| --------------------- | --------------- |
| ResNet-50             | `resnet50`      |
| ResNet-18             | `resnet18`      |
| DenseNet-121          | `densenet121`   |
| ConvNeXt-Tiny         | `convnext_tiny` |
| Swin Transformer-Tiny | `swin_t`        |
| MobileNetV2           | `mobilenet_v2`  |

For example:

```python
model = build_model(
    architecture="resnet50",
    num_classes=5,
    pretrained=True,
)
```

---

## Diabetic Retinopathy Classes

The classification pipeline uses five diabetic retinopathy severity classes:

| Label | Class         |
| ----: | ------------- |
|     0 | No DR         |
|     1 | Mild          |
|     2 | Moderate      |
|     3 | Severe        |
|     4 | Proliferative |

The model outputs five logits corresponding to these classes.

---

## Preprocessing

`fastdrs` provides preprocessing utilities specifically for retinal fundus images.

```python
from fastdrs.preprocessing import get_transforms

train_transforms = get_transforms(
    img_size=224,
    is_train=True,
)

eval_transforms = get_transforms(
    img_size=224,
    is_train=False,
)
```

Ben Graham preprocessing can also be enabled:

```python
train_transforms = get_transforms(
    img_size=224,
    is_train=True,
    use_ben_graham=True,
)
```

Individual preprocessing can be used directly:

```python
from fastdrs.preprocessing import ben_graham_preprocessing

processed_image = ben_graham_preprocessing(
    image,
    img_size=224,
)
```

---

## Dataset Loading

`fastdrs` supports Hugging Face datasets through its dataset utilities.

```python
from fastdrs.data import create_dataloaders

train_loader, val_loader, test_loader, dataset_info = create_dataloaders(
    dataset_name="sngsfydy/aptos",
    batch_size=32,
    img_size=224,
)
```

The loader creation utility handles:

* Dataset loading
* Train/validation/test splitting
* Image transformations
* PyTorch `DataLoader` construction

---

## Training

Training functionality is available through:

```python
from fastdrs.training import train_model

results = train_model(
    architecture="resnet18",
    dataset_name="sngsfydy/aptos",
    epochs=10,
    batch_size=32,
    lr=1e-4,
    img_size=224,
)
```

The training pipeline provides:

* Training and validation loops
* Cross-entropy loss
* AdamW optimization
* Cosine annealing
* Best validation-loss checkpointing
* Final test-set evaluation

A trained checkpoint is saved to the configured model directory.

---

## Evaluation

Models can be evaluated using:

```python
from fastdrs.evaluation import evaluate_model

results = evaluate_model(
    model,
    test_loader,
    device,
)
```

The returned results include:

```text
accuracy
precision_macro
sensitivity_macro
specificity_macro
f1_macro
f1_weighted
inference_latency_ms
model_size_mb
total_parameters
confusion_matrix
per_class
```

This makes the evaluation output suitable for both ML benchmarking and deployment-oriented model comparison.

---

## Project Architecture

`fastdrs` is developed as part of the diabetic retinopathy screening research project.

```text
diabetic-retinopathy-screening/
│
├── src/
│   └── fastdrs/
│       ├── data.py
│       ├── preprocessing.py
│       ├── models.py
│       ├── training.py
│       ├── evaluation.py
│       └── deployment/
│
├── notebooks/
├── data/
├── models/
├── results/
│
└── deployment/
    └── android/
```

The `fastdrs` package contains the reusable ML functionality, while notebooks and project-level scripts are used for experimentation and research.

---

## Deployment

On-device deployment is a major goal of the project.

The intended deployment pipeline is:

```text
PyTorch Model
     │
     ▼
Trained Checkpoint
     │
     ▼
Model Export
     │
     ▼
LiteRT
     │
     ▼
.tflite Model
     │
     ▼
Android Application
```

LiteRT export functionality is being developed as an optional component so that users who only need the core ML functionality do not have to install the additional conversion stack.

The planned installation interface is:

```bash
pip install "fastdrs[export]"
```

---

## Development

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency and environment management.

Install dependencies:

```bash
uv sync
```

Install training dependencies:

```bash
uv sync --extra training
```

Install all optional dependencies:

```bash
uv sync --all-extras
```

Run tests:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

Build the package:

```bash
uv build
```

---

## PyPI

The package is distributed through PyPI:

```bash
pip install fastdrs
```

The package can therefore be tested independently of the source repository, including in environments such as Google Colab.

For example:

```python
!pip install fastdrs
```

Then:

```python
from fastdrs.models import build_model

model = build_model(
    architecture="mobilenet_v2",
    num_classes=5,
)

print(model)
```

---

## Research Project

`fastdrs` is being developed alongside a diabetic retinopathy screening research project using retinal fundus images.

The broader project investigates:

* Diabetic retinopathy classification
* Transfer learning
* Vision architecture benchmarking
* Resource-constrained inference
* Model efficiency
* On-device screening

The package is intended to make the resulting ML pipeline reusable beyond the original research repository.

---

## License

This project is licensed under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

## Author

**Michael Mukiibi**

GitHub: [@MichaelMukiibi](https://github.com/MichaelMukiibi)

---

## Disclaimer

`fastdrs` is a research and engineering toolkit. It is **not a medical device** and its predictions should not be used as a substitute for professional medical diagnosis or clinical decision-making.
