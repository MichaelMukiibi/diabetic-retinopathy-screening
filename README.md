# AI-Assisted Diabetic Retinopathy Screening & Triage

> **Marconi Research & Innovations Lab — Internship Capstone Project**

An economically motivated computer-vision system for **AI-assisted diabetic retinopathy screening and clinical triage**, designed with resource-constrained healthcare environments in mind.

The project investigates whether modern deep-learning vision architectures can classify diabetic retinopathy severity from retinal fundus photographs while balancing **clinical sensitivity, predictive performance, model size, and inference latency** for practical deployment.

> **Important:** This project is a research and engineering prototype. It is **not a medical device and does not provide an autonomous clinical diagnosis**. Predictions are intended to demonstrate AI-assisted screening and referral prioritization.

---

## 1. Problem

Diabetic retinopathy (DR) is a major complication of diabetes and an important cause of preventable vision loss. Early identification can enable timely clinical assessment and treatment.

In resource-constrained healthcare environments, however, screening can be limited by:

* Shortage of ophthalmologists and trained specialists
* Geographic barriers to specialist care
* Cost of specialist examinations
* Increasing numbers of patients requiring screening
* Limited access to regular retinal imaging assessment

This creates an opportunity for **AI-assisted screening and triage**.

Rather than replacing clinicians, an AI system can potentially help healthcare workers identify images that require **higher-priority specialist review**, allowing limited clinical resources to be allocated more efficiently.

---

## 2. Proposed Solution

This project develops a computer-vision pipeline that analyzes retinal fundus photographs and estimates the severity of diabetic retinopathy.

The system investigates multiple modern vision architectures and evaluates them not only by predictive performance, but also by their suitability for deployment.

### Core workflow

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
        ├── ResNet
        ├── DenseNet
        ├── ConvNeXt
        └── Swin Transformer
        │
        ▼
Model Evaluation
        │
        ├── Accuracy
        ├── Precision
        ├── Recall / Sensitivity
        ├── Specificity
        └── F1 Score
        │
        ▼
Deployment Analysis
        │
        ├── Model Size
        ├── Parameter Count
        └── Inference Latency
        │
        ▼
Final Model
        │
        ▼
Android Application / API
```

---

## 3. Capstone Research Question

> **Can modern computer-vision architectures provide sufficiently sensitive diabetic-retinopathy screening while remaining computationally efficient enough for deployment in resource-constrained healthcare environments?**

The project therefore treats model selection as a multi-objective engineering problem:

```text
Clinical Performance
        +
Computational Efficiency
        +
Deployment Feasibility
        =
Practical Model
```

---

## 4. Objectives

### Primary objective

Develop and evaluate a deployable AI-assisted diabetic-retinopathy screening prototype using retinal fundus photographs.

### Specific objectives

1. Perform exploratory data analysis on the APTOS 2019 dataset.
2. Develop a standardized image preprocessing and training pipeline.
3. Fine-tune modern computer-vision architectures for diabetic-retinopathy classification.
4. Compare ResNet, DenseNet, ConvNeXt, and Swin Transformer architectures.
5. Evaluate models using clinically relevant classification metrics.
6. Investigate the relationship between sensitivity, model size, and inference latency.
7. Select a final model based on predictive performance and deployment feasibility.
8. Integrate the trained model into an **Android application or inference API**.
9. Demonstrate an end-to-end screening workflow.

---

## 5. Dataset

### APTOS 2019 Blindness Detection

The primary dataset is the **APTOS 2019 Blindness Detection** dataset.

It contains approximately **3,662 labeled retinal fundus photographs** across five diabetic-retinopathy severity grades.

### Classes

| Label | Severity                           |
| ----: | ---------------------------------- |
|     0 | No diabetic retinopathy            |
|     1 | Mild                               |
|     2 | Moderate                           |
|     3 | Severe                             |
|     4 | Proliferative diabetic retinopathy |

The dataset was originally released through the APTOS 2019 Blindness Detection Kaggle competition.

Dataset source:

[https://www.kaggle.com/competitions/aptos2019-blindness-detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)

### Data considerations

The project will investigate:

* Class distribution
* Image dimensions
* Image quality
* Color/channel characteristics
* Class imbalance
* Training/validation/test splitting
* Appropriate augmentation
* Image normalization

Because the dataset is imbalanced, **accuracy will not be treated as the sole measure of model quality**.

---

## 6. Evaluation Strategy

The project evaluates models using multiple metrics.

### Accuracy

Measures the proportion of correctly classified samples.

### Precision

Measures how many predicted positive cases are actually positive.

### Recall / Sensitivity

Measures the proportion of relevant positive cases detected by the model.

For a screening application, sensitivity is particularly important because missed cases may require further clinical assessment.

### Specificity

Measures the proportion of negative cases correctly identified.

### F1 Score

Balances precision and recall.

### Confusion Matrix

Used to examine errors across the five diabetic-retinopathy severity classes.

---

## 7. Model Benchmarking

The project will investigate four major vision architectures.

### ResNet

Uses residual/skip connections to facilitate the training of deep convolutional networks.

### DenseNet

Uses dense connectivity and feature reuse between layers.

### ConvNeXt

A modern convolutional architecture incorporating design principles developed alongside the rise of vision transformers.

### Swin Transformer

Uses hierarchical vision-transformer representations with shifted-window attention.

The objective is not simply to identify the model with the highest accuracy.

The benchmark will investigate:

> **Predictive performance vs. computational cost vs. deployment feasibility.**

---

## 8. Experimental Tracking

Experiments will be tracked using **Weights & Biases (W&B)**.

Potentially tracked parameters include:

* Model architecture
* Learning rate
* Batch size
* Number of epochs
* Training loss
* Validation loss
* Accuracy
* Precision
* Recall
* Sensitivity
* Specificity
* F1 score
* Training duration

This enables systematic comparison between experiments instead of relying on isolated notebook outputs.

---

## 9. Training Framework

The project builds on technologies explored during the Marconi internship:

* Python
* PyTorch
* TensorFlow
* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face Trainer
* NumPy
* Matplotlib
* scikit-learn
* Weights & Biases

Where practical, the final implementation will favor a **standardized and reproducible Hugging Face/PyTorch workflow**.

---

## 10. Deployment

A trained model is not considered the final deliverable by itself.

The selected model will be integrated into a usable inference system.

Two deployment paths are being considered.

### Option A — Android

```text
Android Application
        │
        ▼
Fundus Image
        │
        ▼
Preprocessing
        │
        ▼
ML Model
        │
        ▼
Prediction
        │
        ▼
Severity / Referral Risk
```

Potential technologies:

* Kotlin
* Jetpack Compose
* ONNX Runtime
* TensorFlow Lite

### Option B — API

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
Trained Model
  │
  ▼
Prediction
  │
  ▼
JSON Response
```

Potential API stack:

* Python
* FastAPI
* Uvicorn
* PyTorch / Hugging Face

The final deployment path will be selected after evaluating the trained model's framework, size, conversion requirements, and inference performance.

---

## 11. Deployment-Oriented Model Selection

The final model will not necessarily be the model with the highest raw accuracy.

A model-selection framework will consider:

| Criterion                | Importance  |
| ------------------------ | ----------- |
| Sensitivity              | High        |
| Specificity              | High        |
| F1 score                 | High        |
| Generalization           | High        |
| Model size               | Medium–High |
| Parameter count          | Medium      |
| Inference latency        | Medium–High |
| Deployment compatibility | High        |

The goal is to identify a model that provides an appropriate balance between **screening performance and practical deployment requirements**.

---

## 12. Economic Rationale

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

This project does **not** attempt to quantify the real-world cost savings without clinical and operational validation. Economic impact remains a hypothesis requiring field evaluation.

---

## 13. System Architecture

The target system is:

```text
                   ┌─────────────────┐
                   │  Fundus Image   │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Preprocessing  │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Trained Model  │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   Prediction    │
                   └────────┬────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      Severity Estimate             Confidence
              │
              ▼
      Screening / Triage
      Recommendation
```

---

## 14. Project Structure

The final repository is expected to evolve toward:

```text
marconi-diabetic-retinopathy/
│
├── README.md
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_final_model.ipynb
│
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   └── README.md
│
├── results/
│   ├── metrics.csv
│   ├── model_comparison.csv
│   └── figures/
│
├── deployment/
│   ├── api/
│   └── android/
│
├── requirements.txt
│
└── presentation/
    └── README.md
```

The exact structure will be adapted from the existing Marconi repositories rather than unnecessarily rebuilding working code.

---

## 15. Existing Marconi Work

This project builds upon work completed during the internship.

### Deep-learning foundations

* Neural-network mechanics
* Forward propagation
* Backpropagation
* Gradient descent
* Keras
* PyTorch
* MNIST classification

### Transfer learning

* MobileNetV2
* Custom PyTorch datasets
* DataLoader pipelines
* Image transforms

### Architecture benchmarking

Existing repositories:

**ResNet vs DenseNet**

[https://github.com/MichaelMukiibi/resnet-v-densenet.git](https://github.com/MichaelMukiibi/resnet-v-densenet.git)

**ConvNeXt vs Swin**

[https://github.com/MichaelMukiibi/convnext-v-swin.git](https://github.com/MichaelMukiibi/convnext-v-swin.git)

### Hugging Face vision pipeline

Existing repository:

**Drone Landing Safety**

[https://github.com/MichaelMukiibi/drone-landing-safety.git](https://github.com/MichaelMukiibi/drone-landing-safety.git)

### Additional repositories

**Omniglot**

[https://github.com/MichaelMukiibi/omniglot.git](https://github.com/MichaelMukiibi/omniglot.git)

**Electricity Consumption**

[https://github.com/MichaelMukiibi/electricity-consumption.git](https://github.com/MichaelMukiibi/electricity-consumption.git)

The first development task is to inspect these repositories and reuse the most mature training/evaluation infrastructure.

---

## 16. Immediate Execution Plan

### Phase 1 — Repository inspection

Inspect:

1. `convnext-v-swin`
2. `resnet-v-densenet`
3. `drone-landing-safety`

Identify:

* Training scripts
* Dataset loaders
* Hugging Face integration
* W&B integration
* Evaluation functions
* Existing checkpoints
* Preprocessing functions
* Existing plots

### Phase 2 — APTOS integration

Adapt the strongest existing pipeline to APTOS.

Perform:

* Dataset download
* Data validation
* EDA
* Train/validation split
* Preprocessing
* Baseline training

### Phase 3 — Model comparison

Train feasible candidates:

* ResNet
* DenseNet
* ConvNeXt
* Swin

Do not attempt unnecessarily large experiments given the closeout deadline.

### Phase 4 — Evaluation

Generate:

* Confusion matrices
* Accuracy
* Precision
* Recall
* Sensitivity
* Specificity
* F1
* Model size
* Inference latency

### Phase 5 — Final model

Select a model using:

```text
Clinical relevance
+
Predictive performance
+
Computational efficiency
+
Deployment feasibility
```

### Phase 6 — Deployment

Implement either:

* Android inference

or

* FastAPI inference service

### Phase 7 — Closeout

Prepare:

* Working demo
* Final repository
* Results
* Presentation
* Technical Q&A preparation

---

## 17. Current Status

### Completed

* Deep-learning foundation training
* PyTorch training
* Transfer learning
* Medical ML evaluation concepts
* ResNet/DenseNet research
* ConvNeXt/Swin research
* W&B experiment tracking
* Hugging Face vision workflows
* EDA workflow development

### In progress

* Selecting the final economically meaningful capstone
* Adapting existing vision pipelines to APTOS 2019
* Model benchmarking
* Final model selection
* Deployment

### Pending

* Final trained APTOS model
* Final evaluation results
* Deployment implementation
* End-to-end demo
* Closeout presentation

---

## 18. Limitations

This prototype has important limitations.

* APTOS is a relatively small dataset compared with large-scale medical imaging datasets.
* Dataset distribution may not represent the target population.
* Dataset labels may contain uncertainty.
* Image quality can vary substantially.
* Model performance does not automatically translate to clinical effectiveness.
* External validation is required.
* Clinical workflow integration has not been established.
* Regulatory approval would be required for clinical deployment.
* Economic impact has not yet been empirically demonstrated.

Therefore, the system should be presented as an **AI-assisted research prototype**, not as an autonomous diagnostic product.

---

## 19. Future Work

Potential future development includes:

* External validation on additional retinal datasets
* Calibration of model confidence
* Explainability methods such as Grad-CAM
* Model quantization
* ONNX/TensorFlow Lite conversion
* On-device inference
* Offline-first deployment
* Prospective clinical validation
* Integration with existing health-information systems
* Human-in-the-loop referral workflows
* Health-economic evaluation
* Field testing in resource-constrained healthcare settings

---

## 20. Expected Capstone Outcome

The final deliverable should demonstrate the complete ML lifecycle:

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

The project aims to demonstrate that the intern can take a meaningful problem from **research question → machine-learning experimentation → engineering implementation → deployable prototype**.

---

## 21. Project Status

**Status:** Active development
**Intern:** MUKIIBI MICHAEL KIRINNYA
**Capstone:** AI-Assisted Diabetic Retinopathy Screening & Triage
**Dataset:** APTOS 2019 Blindness Detection
**Primary ML Task:** Five-class diabetic-retinopathy severity classification
**Deployment Requirement:** Android application **or** inference API