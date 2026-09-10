# AI-Assisted Diabetic Retinopathy Screening & Triage

**Marconi Research & Innovations Lab — Internship Capstone Project**

An economically motivated computer-vision system for **AI-assisted diabetic retinopathy screening and clinical triage**, designed with resource-constrained healthcare environments in mind.

The project investigates whether modern deep-learning vision architectures can classify diabetic retinopathy severity from retinal fundus photographs while balancing **clinical sensitivity, predictive performance, model size, inference latency, and deployment feasibility**.

> **Important:** This project is a research and engineering prototype. It is **not a medical device and does not provide an autonomous clinical diagnosis**. Predictions are intended to demonstrate AI-assisted screening and referral prioritization.

---

## Overview

Diabetic retinopathy (DR) is a major complication of diabetes and an important cause of preventable vision loss. Early identification can enable timely clinical assessment and treatment.

In resource-constrained healthcare environments, screening can be limited by:

* Shortage of ophthalmologists and trained specialists
* Geographic barriers to specialist care
* Cost of specialist examinations
* Increasing numbers of patients requiring screening
* Limited access to regular retinal imaging assessment

This project explores whether an AI-assisted system can help healthcare workers identify retinal images that may require **higher-priority specialist review**.

The objective is not to replace clinicians. Instead, the system is designed around a **screening and triage workflow** in which machine learning can potentially help allocate limited clinical resources more efficiently.

---

# Research Question

> **Can modern computer-vision architectures provide sufficiently sensitive diabetic-retinopathy screening while remaining computationally efficient enough for deployment in resource-constrained healthcare environments?**

This makes model selection a multi-objective engineering problem:

```text
Clinical Performance
        +
Computational Efficiency
        +
Deployment Feasibility
        =
Practical Model
```

The project therefore evaluates models beyond raw accuracy.

---

# System

The current system is organized as an end-to-end ML pipeline:

```text
                    Retinal Fundus Image
                            │
                            ▼
                           EDA
                            │
                            ▼
                    Preprocessing
                            │
                            ▼
                 Model Fine-Tuning
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
          ResNet         DenseNet       ConvNeXt
                                           │
                                           ▼
                                      Swin Transformer
                            │
                            ▼
                     Model Evaluation
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
        Classification   Efficiency    Deployment
          Metrics         Metrics        Analysis
                            │
                            ▼
                  Metadata-Aware Checkpoint
                            │
                            ▼
                   PyTorch / LiteRT
                            │
                            ▼
                   Android / API
```

---

# Objectives

## Primary objective

Develop and evaluate a deployable AI-assisted diabetic-retinopathy screening prototype using retinal fundus photographs.

## Specific objectives

1. Perform exploratory data analysis on the APTOS 2019 dataset.
2. Develop a standardized image preprocessing and training pipeline.
3. Fine-tune modern computer-vision architectures for diabetic-retinopathy classification.
4. Compare multiple CNN and vision-transformer architectures.
5. Evaluate models using clinically relevant classification metrics.
6. Investigate the relationship between sensitivity, model size, and inference latency.
7. Select a model based on predictive performance and deployment feasibility.
8. Package reusable ML functionality through `fastdrs`.
9. Export and run selected models in deployment-oriented environments.
10. Integrate the trained model into an Android application or inference API.
11. Demonstrate an end-to-end screening workflow.

---

# Dataset

## APTOS 2019 Blindness Detection

The primary dataset is the **APTOS 2019 Blindness Detection** dataset.

It contains approximately **3,662 labeled retinal fundus photographs** across five diabetic-retinopathy severity grades.

Dataset source:

https://www.kaggle.com/competitions/aptos2019-blindness-detection

## Classes

| Label | Severity                           |
| ----: | ---------------------------------- |
|     0 | No diabetic retinopathy            |
|     1 | Mild                               |
|     2 | Moderate                           |
|     3 | Severe                             |
|     4 | Proliferative diabetic retinopathy |

Because the dataset is imbalanced, **accuracy is not treated as the sole measure of model quality**.

The project also considers sensitivity, specificity, precision, macro F1, per-class performance, and confusion matrices.

---

# Evaluation

The project evaluates models across both **clinical/research metrics** and **deployment metrics**.

## Classification metrics

* Accuracy
* Macro precision
* Macro sensitivity / recall
* Macro specificity
* Macro F1
* Weighted F1
* Per-class precision
* Per-class sensitivity
* Per-class specificity
* Confusion matrix

## Deployment metrics

* Inference latency
* Parameter count
* Estimated model size
* Deployment compatibility

The central comparison is:

> **Predictive performance vs. computational cost vs. deployment feasibility.**

---

# Model Benchmarking

The project currently supports the following architectures through the `fastdrs` model factory:

| Architecture          | Identifier      |
| --------------------- | --------------- |
| ResNet-18             | `resnet18`      |
| ResNet-50             | `resnet50`      |
| DenseNet-121          | `densenet121`   |
| ConvNeXt-Tiny         | `convnext_tiny` |
| Swin Transformer-Tiny | `swin_t`        |
| MobileNetV2           | `mobilenet_v2`  |

### ResNet

Residual/skip connections allow deep convolutional networks to be trained effectively.

### DenseNet

Dense connectivity promotes feature reuse between layers.

### ConvNeXt

A modern convolutional architecture incorporating design principles developed alongside the rise of vision transformers.

### Swin Transformer

A hierarchical vision-transformer architecture using shifted-window attention.

### MobileNetV2

A lightweight architecture useful for investigating resource-constrained inference.

The goal is not simply to find the model with the highest classification score.

The project asks:

> **Which model provides the best practical trade-off between screening performance and deployment requirements?**

---

# `fastdrs`

The reusable machine-learning components have been separated into a Python package called **`fastdrs`**.

`fastdrs` provides:

* Model construction
* Fundus-image preprocessing
* Dataset loading
* PyTorch training
* Early stopping
* Metadata-aware checkpoints
* Model evaluation
* PyTorch inference
* LiteRT inference
* LiteRT export functionality

The package is distributed through PyPI.

```bash
pip install fastdrs
```

Repository/package development is kept separate from experiment-specific notebooks and application code.

## Package architecture

```text
src/fastdrs/
│
├── __init__.py
├── data.py
├── preprocessing.py
├── models.py
├── training.py
├── evaluation.py
│
├── inference/
│   ├── __init__.py
│   ├── base.py
│   ├── prediction.py
│   ├── pytorch.py
│   └── litert.py
│
└── export/
    ├── __init__.py
    └── litert.py
```

This separation allows the ML pipeline to be reused independently of the research repository.

---

# Training

The training pipeline is implemented in `fastdrs`.

Example:

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

The pipeline provides:

* Training and validation loops
* Cross-entropy loss
* AdamW optimization
* Cosine annealing learning-rate scheduling
* Validation monitoring
* Configurable early stopping
* Best-model restoration
* Metadata-aware checkpointing
* Final test-set evaluation
* Optional W&B tracking

## Early stopping

Training can monitor validation macro F1:

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

The best model is restored before final evaluation.

---

# Metadata-Aware Checkpoints

A trained model is stored as a self-describing checkpoint rather than as a raw `state_dict` alone.

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

This makes the trained model easier to reproduce and deploy.

Instead of requiring inference code to know beforehand which architecture was used, the checkpoint itself contains the model configuration required to reconstruct it.

The loader also maintains compatibility with legacy state-dict-only checkpoints.

This is particularly important for the longer-term model-distribution architecture, where trained models may be remotely accessible and consumed by downstream inference clients.

---

# Preprocessing

The package provides retinal fundus preprocessing utilities including:

* Image resizing
* ImageNet normalization
* Data augmentation
* Fundus boundary cropping
* Ben Graham preprocessing

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

---

# Experiment Tracking

Experiments are tracked using **Weights & Biases**.

W&B workspace:

https://wandb.ai/growsafe/fastdrs

Tracked information includes:

* Model architecture
* Dataset
* Learning rate
* Batch size
* Number of epochs
* Training loss
* Validation loss
* Validation accuracy
* Validation macro F1
* Classification metrics
* Inference latency
* Model size
* Parameter count

The purpose is to make architecture comparisons reproducible rather than relying on isolated notebook outputs.

W&B is optional; training can also run locally without online tracking.

---

# Quick Start

## Install

For the core package:

```bash
pip install fastdrs
```

For training:

```bash
pip install "fastdrs[training]"
```

For PyTorch inference:

```bash
pip install "fastdrs[inference]"
```

For LiteRT inference:

```bash
pip install "fastdrs[litert]"
```

For model export:

```bash
pip install "fastdrs[export]"
```

## Build a model

```python
from fastdrs.models import build_model

model = build_model(
    architecture="mobilenet_v2",
    num_classes=5,
    pretrained=True,
)

print(model)
```

---

# Deployment

A trained model is not considered the final deliverable by itself.

The project is designed to move from research experimentation to a usable inference system.

The target deployment architecture is:

```text
                    Training
                       │
                       ▼
                 PyTorch Model
                       │
                       ▼
            Metadata-Aware Checkpoint
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

## Android

The Android deployment path is designed around local inference:

```text
Fundus Image
     │
     ▼
Preprocessing
     │
     ▼
LiteRT Model
     │
     ▼
5-Class Prediction
     │
     ▼
Severity / Screening Result
```

The objective is to investigate whether inference can be performed locally without requiring continuous network connectivity.

## API

A server-side inference API remains an alternative deployment path:

```text
Client
  │
  ▼
HTTP POST /predict
  │
  ▼
FastAPI
  │
  ▼
Preprocessing
  │
  ▼
Model
  │
  ▼
Prediction
  │
  ▼
JSON Response
```

The final deployment configuration depends on model size, inference performance, conversion reliability, and application requirements.

---

# Deployment-Oriented Model Selection

The final model will not necessarily be the model with the highest raw accuracy.

| Criterion                | Importance  |
| ------------------------ | ----------- |
| Sensitivity              | High        |
| Specificity              | High        |
| Macro F1                 | High        |
| Generalization           | High        |
| Model size               | Medium–High |
| Parameter count          | Medium      |
| Inference latency        | Medium–High |
| Deployment compatibility | High        |

The goal is to identify a model that provides an appropriate balance between:

```text
Clinical Relevance
        +
Predictive Performance
        +
Computational Efficiency
        +
Deployment Feasibility
```

---

# System Architecture

The target end-to-end system is:

```text
                    ┌────────────────────┐
                    │   Fundus Image     │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   Preprocessing    │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   Trained Model    │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │    Prediction      │
                    └─────────┬──────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             Severity Estimate     Confidence
                    │
                    ▼
             Screening / Triage
               Recommendation
```

The intended role of the model is **decision support**, not autonomous diagnosis.

---

# Project Structure

The repository is organized around reusable ML infrastructure, research experiments, results, and deployment.

```text
diabetic-retinopathy-screening/
│
├── README.md
├── pyproject.toml
├── uv.lock
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
│
├── data/
│
├── models/
│
├── results/
│
├── docs/
│
└── deployment/
    └── android/
```

The package contains reusable ML functionality.

The notebooks, results, and deployment directories contain project-specific experimentation and application work.

---

# Research Background

The project builds on work completed during the Marconi internship.

## Deep-learning foundations

* Neural-network mechanics
* Forward propagation
* Backpropagation
* Gradient descent
* Keras
* PyTorch
* MNIST classification

## Transfer learning

* MobileNetV2
* Custom PyTorch datasets
* DataLoader pipelines
* Image transforms

## Architecture benchmarking

### ResNet vs DenseNet

https://github.com/MichaelMukiibi/resnet-v-densenet

### ConvNeXt vs Swin

https://github.com/MichaelMukiibi/convnext-v-swin

## Hugging Face vision pipeline

### Drone Landing Safety

https://github.com/MichaelMukiibi/drone-landing-safety

## Additional work

### Omniglot

https://github.com/MichaelMukiibi/omniglot

### Electricity Consumption

https://github.com/MichaelMukiibi/electricity-consumption

These projects informed the reusable training, dataset, evaluation, and experiment-tracking infrastructure now being consolidated into `fastdrs`.

---

# Economic Rationale

The project is motivated by an operational healthcare problem rather than model development for its own sake.

A potential AI-assisted screening system could help:

* Extend screening capacity
* Prioritize patients for specialist review
* Reduce unnecessary specialist referrals
* Support healthcare workers without specialist expertise
* Reduce geographic barriers to initial screening
* Enable lower-cost preliminary assessment

The economic hypothesis is:

> **If AI can reliably identify higher-risk retinal images for specialist review, limited ophthalmology resources can potentially be allocated more efficiently.**

This project does **not** claim quantified real-world cost savings without clinical and operational validation.

Economic impact remains a hypothesis requiring field evaluation.

---

# Current Status

## Completed

* Deep-learning foundation training
* PyTorch training
* Transfer learning
* Medical ML evaluation concepts
* ResNet/DenseNet research
* ConvNeXt/Swin research
* W&B experiment tracking
* Hugging Face vision workflows
* EDA workflow development
* Reusable `fastdrs` package
* Multi-architecture model factory
* Fundus preprocessing utilities
* Dataset/DataLoader pipeline
* Training pipeline
* Early stopping
* Metadata-aware model checkpoints
* Model evaluation pipeline
* PyPI package distribution
* Initial PyTorch/LiteRT deployment architecture

## In progress

* Architecture benchmarking on APTOS
* Final model selection
* Remote/pretrained model distribution
* Stable inference APIs
* LiteRT export and inference workflows
* Android integration
* End-to-end screening demonstration

## Pending

* Final validated APTOS model
* Final benchmark results
* External validation
* Complete Android deployment
* End-to-end demonstration
* Clinical/operational validation

---

# Roadmap

```text
[x] Deep-learning foundation
[x] Transfer learning
[x] Architecture benchmarking infrastructure
[x] Dataset pipeline
[x] Preprocessing pipeline
[x] Training pipeline
[x] Evaluation pipeline
[x] Experiment tracking
[x] PyPI package
[x] Metadata-aware checkpoints
[x] Early stopping
[ ] Expanded pretrained model distribution
[ ] Remote model registry
[ ] Stable inference API
[ ] LiteRT export
[ ] LiteRT inference
[ ] Android integration
[ ] External validation
[ ] Prospective clinical validation
```

Future research may include:

* External validation on additional retinal datasets
* Model confidence calibration
* Explainability methods such as Grad-CAM
* Model quantization
* On-device inference optimization
* Offline-first deployment
* Human-in-the-loop referral workflows
* Integration with health-information systems
* Health-economic evaluation
* Field testing in resource-constrained healthcare settings

---

# Limitations

This remains a research prototype with important limitations.

* APTOS is relatively small compared with large-scale medical-imaging datasets.
* Dataset distribution may not represent the target population.
* Dataset labels may contain uncertainty.
* Image quality can vary substantially.
* Model performance does not automatically translate to clinical effectiveness.
* External validation is required.
* Clinical workflow integration has not been established.
* Regulatory approval would be required for clinical deployment.
* Economic impact has not yet been empirically demonstrated.

Therefore, the system should be presented as an **AI-assisted research prototype**, not an autonomous diagnostic product.

---

# Expected Capstone Outcome

The project aims to demonstrate the complete ML lifecycle:

```text
REAL-WORLD PROBLEM
        ↓
      DATA
        ↓
       EDA
        ↓
MODEL DEVELOPMENT
        ↓
EXPERIMENT TRACKING
        ↓
ARCHITECTURE BENCHMARKING
        ↓
CLINICALLY RELEVANT EVALUATION
        ↓
MODEL SELECTION
        ↓
DEPLOYMENT
        ↓
WORKING DEMONSTRATION
```

The central objective is not merely to achieve a high classification score.

The project demonstrates the progression from:

> **research question → machine-learning experimentation → reusable engineering infrastructure → model evaluation → deployment-oriented prototype**

---

# License

This project is licensed under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

# Author

**Michael Mukiibi**

GitHub: [@MichaelMukiibi](https://github.com/MichaelMukiibi)

---

# Disclaimer

`fastdrs` and the surrounding diabetic-retinopathy screening system are research and engineering projects.

They are **not medical devices** and do not provide autonomous clinical diagnoses. Model predictions should not be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.

Any clinical deployment would require appropriate external validation, clinical evaluation, regulatory review, and integration with qualified healthcare professionals.
