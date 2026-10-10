# Architecture

This document describes the design of the AI-assisted diabetic-retinopathy screening and triage project and its reusable `fastdrs` Python package.

> **Implementation status:** Some components below describe the intended architecture. Consult the repository's current status and tests to distinguish implemented, tested functionality from planned deployment work.

## End-to-end system

The intended pipeline is:

```text
Retinal fundus image
        |
        v
Exploratory data analysis and quality checks
        |
        v
Preprocessing
        |
        v
Model fine-tuning
        |
        +-------------------------------+
        |               |               |
        v               v               v
      ResNet          DenseNet       ConvNeXt / Swin
        |               |               |
        +---------------+---------------+
                        |
                        v
                 Model evaluation
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
     Classification  Efficiency   Deployment
        metrics       metrics      analysis
                        |
                        v
             Metadata-aware checkpoint
                        |
                        v
                 PyTorch / LiteRT
                        |
                        v
                  Android / API
```

The target role of the model is decision support, not autonomous diagnosis.

## Model benchmarking

The `fastdrs` model factory supports the following architectures, according to the project documentation:

| Architecture | Identifier |
|---|---|
| ResNet-18 | `resnet18` |
| ResNet-50 | `resnet50` |
| DenseNet-121 | `densenet121` |
| ConvNeXt-Tiny | `convnext_tiny` |
| Swin Transformer-Tiny | `swin_t` |
| MobileNetV2 | `mobilenet_v2` |

Architecture notes:

- **ResNet:** Residual / skip connections support training deeper convolutional networks.
- **DenseNet:** Dense connectivity encourages feature reuse.
- **ConvNeXt:** A modern convolutional architecture incorporating design principles developed alongside vision transformers.
- **Swin Transformer:** A hierarchical vision transformer using shifted-window attention.
- **MobileNetV2:** A lightweight architecture useful for studying resource-constrained inference.

The objective is not simply to choose the architecture with the highest raw accuracy. The comparison considers sensitivity, specificity, macro F1, generalization, model size, parameter count, latency, and deployment compatibility. Final model selection should be based on recorded experimental results, not assumptions about an architecture.

## Package structure

The reusable package is organized as follows:

```text
src/fastdrs/
├── __init__.py
├── data.py
├── preprocessing.py
├── models.py
├── training.py
├── evaluation.py
├── inference/
│   ├── __init__.py
│   ├── base.py
│   ├── prediction.py
│   ├── pytorch.py
│   └── litert.py
└── export/
    ├── __init__.py
    └── litert.py
```

The package is intended to hold reusable ML functionality. Notebooks, results, datasets, and application code belong in their own project directories.

## Training pipeline

Training functionality is implemented in `fastdrs`. The documented training pipeline includes:

- Training and validation loops
- Cross-entropy loss
- AdamW optimization
- Cosine-annealing learning-rate scheduling
- Validation monitoring
- Configurable early stopping
- Best-model restoration
- Metadata-aware checkpointing
- Final test-set evaluation
- Optional Weights & Biases tracking

Example from the project documentation:

```python
from fastdrs.training import train_model

results = train_model(
    architecture="convnext_tiny",
    dataset_name="sngsfydy/aptos",
    epochs=10,
    batch_size=32,
    lr=1e-4,
    img_size=224,
)
```

Early stopping can monitor validation macro F1:

```python
results = train_model(
    architecture="convnext_tiny",
    dataset_name="sngsfydy/aptos",
    epochs=30,
    monitor="val_f1_macro",
    patience=7,
    min_delta=0.001,
)
```

These examples reflect the source README. Verify their signatures against the current package before using them in a live demo or treating them as tested examples.

## Preprocessing

The documented preprocessing utilities include:

- Image resizing
- ImageNet normalization
- Data augmentation
- Fundus-boundary cropping
- Ben Graham preprocessing

Example:

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

Ben Graham preprocessing can be enabled when required:

```python
train_transforms = get_transforms(
    img_size=224,
    is_train=True,
    use_ben_graham=True,
)
```

Training and evaluation transforms should be configured consistently with the experiment and the preprocessing used by the trained model at inference time.

## Metadata-aware checkpoints

Rather than storing only a raw `state_dict`, the intended checkpoint format records the model state and the configuration required to reconstruct it.

```text
checkpoint.pth
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

This metadata helps inference code reconstruct the appropriate architecture and preprocessing configuration. The documented loader also supports legacy state-dict-only checkpoints.

The checkpoint format is intended to support reproducible experiments and future model distribution. Remote model access or a model registry should be considered planned until implemented and tested.

## Experiment tracking

Experiments are tracked using [Weights & Biases](https://wandb.ai/growsafe/fastdrs). The project documentation lists these tracked fields:

- Architecture and dataset
- Learning rate, batch size, and number of epochs
- Training and validation loss
- Validation accuracy and macro F1
- Classification metrics
- Inference latency
- Model size and parameter count

W&B is described as optional; training is intended to run locally without online tracking.

## Deployment architecture

The intended model deployment path is:

```text
Training
   |
   v
PyTorch model
   |
   v
Metadata-aware checkpoint
   |
   v
Model export
   |
   v
LiteRT / .tflite model
   |
   v
Android application
```

### Android: local inference

The intended local-inference flow is:

```text
Fundus image
    |
    v
Preprocessing
    |
    v
LiteRT model
    |
    v
Five-class prediction
    |
    v
Severity estimate / screening-support output
```

The goal is to investigate whether inference can run locally without continuous network connectivity.

### API: server-side inference

A server-side API is an alternative planned deployment path:

```text
Client
  |
  v
HTTP POST /predict
  |
  v
FastAPI
  |
  v
Preprocessing
  |
  v
Model inference
  |
  v
JSON response
```

The final deployment configuration depends on model size, inference performance, conversion reliability, and application requirements. Do not imply that Android integration, a stable API, or LiteRT export is complete unless the current implementation has been verified.

## Project layout

The source README describes the broader repository layout as:

```text
diabetic-retinopathy-screening/
├── README.md
├── ARCHITECTURE.md
├── pyproject.toml
├── uv.lock
├── src/
│   └── fastdrs/
├── notebooks/
├── data/
├── models/
├── results/
├── docs/
└── deployment/
    └── android/
```

Adjust this tree to match the actual repository. Include only files and directories that exist or clearly label planned components.

## Model selection criteria

The intended comparison weights the following criteria:

| Criterion | Stated importance |
|---|---|
| Sensitivity | High |
| Specificity | High |
| Macro F1 | High |
| Generalization | High |
| Model size | Medium–high |
| Parameter count | Medium |
| Inference latency | Medium–high |
| Deployment compatibility | High |

These are selection priorities, not evidence that a model has met clinical requirements. A benchmark on one dataset is not external or clinical validation.

## Future engineering work

The source project notes identify the following as ongoing or future work:

- Expanded pretrained-model distribution
- Remote model registry
- Stable inference API
- LiteRT export and inference
- Android integration
- External validation
- Prospective clinical validation
- Confidence calibration
- Explainability methods such as Grad-CAM
- Model quantization and on-device optimization
- Offline-first workflows
- Human-in-the-loop referral workflows
- Health-information-system integration
- Health-economic evaluation
- Field testing in resource-constrained healthcare settings

## Safety and limitations

The system remains a research and engineering prototype. Dataset performance does not establish clinical effectiveness. The dataset may not represent the intended population, labels may be uncertain, and image quality can vary.

Any clinical deployment would require appropriate external validation, clinical evaluation, regulatory review, and integration with qualified healthcare professionals. The model is intended for research into decision support, not autonomous diagnosis.
