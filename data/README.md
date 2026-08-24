# Data

This directory contains the dataset resources and data-processing artifacts used for the diabetic retinopathy classification project.

## Dataset

The project uses the **EYEPACS** dataset available through Hugging Face:

**Dataset:** [`ctmedtech/EYEPACS`](https://huggingface.co/datasets/ctmedtech/EYEPACS)

The dataset is intended for **multi-class image classification** of diabetic retinopathy from retinal fundus images. The Hugging Face dataset card identifies it as an ophthalmology/medical dataset with tags including `fundus-image`, `diabetic-retinopathy`, and `dr-grading`.

### Dataset characteristics

| Property        | Value                            |
| --------------- | -------------------------------- |
| Task            | Image classification             |
| Sub-task        | Multi-class image classification |
| Modality        | Image                            |
| Format          | ImageFolder                      |
| Domain          | Ophthalmology                    |
| Image type      | Retinal fundus images            |
| Language        | English                          |
| Available split | `train`                          |
| Approx. size    | 35.1k rows                       |

The Hugging Face viewer currently exposes a single `train` split containing approximately **35.1k rows**.

## Directory Structure

```text
data/
├── README.md
├── raw/
│   └── eyepacs/
├── processed/
│   └── ...
└── splits/
    └── ...
```

### `raw/`

Contains the original dataset or an unmodified local copy of the downloaded dataset.

**Do not modify files in this directory.**

### `processed/`

Contains data generated during preprocessing, such as:

* resized images
* normalized images
* transformed datasets
* cached data
* preprocessing metadata

### `splits/`

Contains information required to reproduce the training, validation, and test splits created from the available dataset.

## Data Pipeline

The expected pipeline is:

```text
EYEPACS
   │
   ▼
Raw fundus images
   │
   ▼
Quality checks / preprocessing
   │
   ▼
Train / validation / test split
   │
   ▼
Image transformations
   │
   ▼
Model training
   │
   ▼
Evaluation
```

All preprocessing and splitting operations should be performed **after downloading the raw dataset**, rather than modifying the original data.

## Reproducibility

Data-processing scripts should record:

* dataset version or commit used
* random seed
* train/validation/test split proportions
* preprocessing configuration
* image resolution
* normalization parameters
* augmentation configuration

This allows experiments to be reproduced consistently.

## Data Integrity

The raw dataset should remain immutable. Generated files should be placed under `processed/` or another explicitly designated directory.

Avoid committing the full image dataset to Git. Large datasets should instead be downloaded from the original source or managed using an appropriate dataset/versioning system.

## Source

Hugging Face dataset:

https://huggingface.co/datasets/ctmedtech/EYEPACS

The dataset page identifies the resource as an image-classification dataset and provides links to its dataset card, viewer, and files/versions.

## Notes

This README documents the **data layer only**. Model architecture, training configuration, experiments, and evaluation procedures are documented elsewhere in the repository.
