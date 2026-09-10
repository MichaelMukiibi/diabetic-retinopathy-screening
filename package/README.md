# fastdrs

**Fast diabetic retinopathy screening and model development toolkit for PyTorch.**

`fastdrs` is a reusable machine learning toolkit for developing diabetic retinopathy classification systems. It provides components for model construction, fundus-image preprocessing, dataset loading, training, evaluation, checkpoint management, and deployment-oriented inference.

The package is designed to separate reusable ML functionality from the experimentation and research code in the `diabetic-retinopathy-screening` project.

> **Status:** Active development (`0.1.x`)

---

## Features

### 🧠 Vision architectures

`fastdrs` provides a common model factory for several image-classification architectures:

* ResNet18
* ResNet50
* DenseNet121
* ConvNeXt-Tiny
* Swin Transformer-Tiny
* MobileNetV2

All architectures expose a consistent interface through `build_model()`.

### 🖼️ Fundus image preprocessing

* Image resizing
* ImageNet normalization
* Training-time augmentation
* Fundus boundary cropping
* Ben Graham preprocessing

Preprocessing can be configured consistently for training and evaluation.

### 📊 Dataset handling

* Hugging Face datasets
* APTOS-compatible datasets
* EYEPACS-compatible dataset workflows
* Reproducible train/validation/test splitting
* PyTorch `Dataset` and `DataLoader` integration

### 🚂 Model training

The training pipeline provides:

* PyTorch training loops
* Cross-entropy loss
* AdamW optimization
* Cosine annealing learning-rate scheduling
* Validation monitoring
* Macro-F1-based early stopping
* Configurable patience and minimum improvement
* Best-model checkpointing
* Optional Weights & Biases integration
* Final test-set evaluation

### 💾 Metadata-aware checkpoints

Training checkpoints are self-describing rather than containing only a raw PyTorch `state_dict`.

A checkpoint can contain information such as:

* Model architecture
* Number of classes
* Input image size
* Class names
* Training epoch
* Monitored validation metric
* Best metric value
* Dataset information
* Checkpoint format version
* Model weights

This allows downstream inference code to determine how a model was trained without requiring the architecture to be manually specified.

Legacy state-dict-only checkpoints can also be loaded for backwards compatibility.

### 📈 Evaluation

The evaluation pipeline reports:

* Accuracy
* Macro precision
* Macro sensitivity/recall
* Macro specificity
* Macro F1
* Weighted F1
* Confusion matrix
* Per-class metrics
* Inference latency
* Parameter count
* Estimated model size

This makes the evaluation output suitable for both research benchmarking and deployment-oriented model comparison.

### 📱 Deployment

The project is being developed with resource-constrained and on-device inference in mind.

Deployment components include:

* PyTorch inference
* LiteRT inference
* LiteRT export for Android deployment
* Offline/on-device inference workflows

LiteRT functionality is optional so users who only need the core ML toolkit do not need to install the additional runtime or conversion dependencies.

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

The basic installation contains the lightweight core functionality.

---

### Training dependencies

Install the training stack with:

```bash
pip install "fastdrs[training]"
```

Or:

```bash
uv add "fastdrs[training]"
```

This installs the dependencies required for:

* PyTorch training
* TorchVision models
* Dataset loading
* Computer-vision preprocessing
* Evaluation
* Weights & Biases integration

---

### PyTorch inference

If PyTorch inference is being used independently of the training pipeline:

```bash
pip install "fastdrs[inference]"
```

---

### LiteRT inference

LiteRT inference dependencies can be installed separately:

```bash
pip install "fastdrs[litert]"
```

This keeps the core package lightweight while allowing deployment environments to install only the runtime they require.

---

### Development installation

Clone the research repository:

```bash
git clone https://github.com/MichaelMukiibi/diabetic-retinopathy-screening.git

cd diabetic-retinopathy-screening
```

Install the development environment:

```bash
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
    'estimated_size_mb': ...
}
```

---

## Supported Architectures

The model factory currently supports:

| Architecture          | Identifier      |
| --------------------- | --------------- |
| ResNet-50             | `resnet50`      |
| ResNet-18             | `resnet18`      |
| DenseNet-121          | `densenet121`   |
| ConvNeXt-Tiny         | `convnext_tiny` |
| Swin Transformer-Tiny | `swin_t`        |
| MobileNetV2           | `mobilenet_v2`  |

Example:

```python
from fastdrs.models import build_model

model = build_model(
    architecture="resnet50",
    num_classes=5,
    pretrained=True,
)
```

The same interface can therefore be used to benchmark different architectures without changing the surrounding training pipeline.

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

The preprocessing function can also be used directly:

```python
from fastdrs.preprocessing import ben_graham_preprocessing

processed_image = ben_graham_preprocessing(
    image,
    img_size=224,
)
```

---

## Dataset Loading

`fastdrs` provides utilities for loading and preparing retinal-image datasets.

```python
from fastdrs.data import create_dataloaders

train_loader, val_loader, test_loader, dataset_info = create_dataloaders(
    dataset_name="sngsfydy/aptos",
    batch_size=32,
    img_size=224,
)
```

The dataset utilities handle:

* Dataset loading
* Train/validation/test splitting
* Image transformations
* PyTorch `DataLoader` construction

Dataset splitting is configurable and can be made reproducible through the provided seed configuration.

---

## Training

Training is available through `train_model()`:

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
* Validation monitoring
* Early stopping
* Best-model checkpointing
* Final test-set evaluation

### Early stopping

By default, training can monitor validation macro F1:

```python
results = train_model(
    architecture="convnext_tiny",
    dataset_name="sngsfydy/aptos",
    epochs=30,
    batch_size=32,
    monitor="val_f1_macro",
    patience=7,
    min_delta=0.001,
)
```

Other validation metrics can be used as the stopping criterion where supported.

The best checkpoint is restored before final evaluation.

---

## Checkpoints

A trained model is saved as a metadata-aware checkpoint in the configured model directory.

A checkpoint contains the model weights together with information needed to identify the model configuration.

Conceptually:

```text
checkpoint.pth
│
├── model_state_dict
├── architecture
├── num_classes
├── img_size
├── class_names
├── dataset_name
├── epoch
├── monitor
├── best_metric
└── checkpoint_version
```

This is important for deployment because inference code does not have to rely on undocumented assumptions about which architecture or class mapping was used to produce a model.

### Loading a checkpoint

```python
from fastdrs.training import load_checkpoint

checkpoint = load_checkpoint(
    "models/resnet18_best.pth",
)

print(checkpoint["architecture"])
print(checkpoint["class_names"])
print(checkpoint["img_size"])
```

The checkpoint loader also supports legacy checkpoints containing only a PyTorch `state_dict`.

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

Example:

```python
print(results["f1_macro"])
print(results["sensitivity_macro"])
print(results["specificity_macro"])
```

The evaluation output is designed to make candidate architectures comparable across both predictive performance and deployment constraints.

---

## Weights & Biases

Weights & Biases integration is optional.

Enable it during training:

```python
results = train_model(
    architecture="resnet50",
    dataset_name="sngsfydy/aptos",
    epochs=10,
    use_wandb=True,
)
```

Training metrics and final evaluation metrics can then be tracked alongside the experiment configuration.

If W&B is disabled, the training pipeline runs without requiring a W&B run.

---

## Project Architecture

`fastdrs` is developed alongside the diabetic retinopathy screening research project.

```text
diabetic-retinopathy-screening/
│
├── src/
│   └── fastdrs/
│       ├── __init__.py
│       ├── data.py
│       ├── preprocessing.py
│       ├── models.py
│       ├── training.py
│       ├── evaluation.py
│       │
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── prediction.py
│       │   ├── pytorch.py
│       │   └── litert.py
│       │
│       └── export/
│           ├── __init__.py
│           └── litert.py
│
├── notebooks/
├── data/
├── models/
├── results/
│
└── deployment/
    └── android/
```

The package contains reusable ML functionality, while notebooks and project-level scripts are used for experimentation, research, benchmarking, and application development.

---

## Deployment

On-device deployment is a major goal of the project.

The intended pipeline is:

```text
                 Training
                    │
                    ▼
              PyTorch Model
                    │
                    ▼
          Metadata-aware Checkpoint
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

The package separates model development, inference, and export so that deployment dependencies do not unnecessarily become dependencies of the core package.

### LiteRT export

LiteRT export functionality is available as an optional component:

```bash
pip install "fastdrs[export]"
```

The resulting `.tflite` model can then be integrated into an Android application for on-device inference.

### LiteRT inference

LiteRT runtime support is kept separate from the export stack:

```bash
pip install "fastdrs[litert]"
```

This distinction allows a deployment environment to install the runtime without installing the model-conversion toolchain.

---

## Android Deployment

The research project includes an Android application intended to consume exported models.

The deployment architecture is designed around:

```text
Fundus Image
     │
     ▼
Image Preprocessing
     │
     ▼
LiteRT Model
     │
     ▼
5-Class Prediction
     │
     ▼
Screening Result
```

The goal is to support inference locally on the device rather than requiring a continuous network connection.

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

Install inference dependencies:

```bash
uv sync --extra inference
```

Install LiteRT dependencies:

```bash
uv sync --extra litert
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

`fastdrs` is distributed through PyPI:

```bash
pip install fastdrs
```

This allows the reusable ML components to be installed independently of the research repository.

For example, in Google Colab:

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

This separation makes it possible to use the package in notebooks, experiments, training environments, and downstream applications without cloning the complete research repository.

---

## Research Project

`fastdrs` is being developed alongside a diabetic retinopathy screening research project using retinal fundus images.

The broader research investigates:

* Diabetic retinopathy classification
* Transfer learning
* Vision architecture benchmarking
* Model efficiency
* Resource-constrained inference
* On-device screening
* Deployment-oriented evaluation

Candidate architectures are evaluated not only on classification performance but also on practical deployment characteristics such as parameter count, model size, and inference latency.

The package is intended to make the resulting ML pipeline reusable beyond the original research repository.

---

## Roadmap

Planned and ongoing development includes:

* [x] Multi-architecture model factory
* [x] Fundus preprocessing utilities
* [x] Dataset and DataLoader utilities
* [x] Training pipeline
* [x] Validation-based checkpointing
* [x] Early stopping
* [x] Metadata-aware checkpoints
* [x] Model evaluation metrics
* [x] PyPI distribution
* [ ] Expanded pretrained model distribution
* [ ] Checkpoint registry / remote model loading
* [ ] Stable PyTorch inference API
* [ ] LiteRT export pipeline
* [ ] LiteRT inference API
* [ ] Improved model/version metadata
* [ ] Android deployment tooling

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
