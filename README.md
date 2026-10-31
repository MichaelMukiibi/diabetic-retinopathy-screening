# AI-Assisted Diabetic Retinopathy Screening & Triage

An economically motivated computer-vision project exploring **AI-assisted diabetic retinopathy (DR) screening and clinical triage** in resource-constrained healthcare environments.

The project investigates whether modern deep-learning vision architectures can classify DR severity from retinal fundus photographs while balancing predictive performance, sensitivity, model size, inference latency, and deployment feasibility.

> **Research prototype:** This project is not a medical device and does not provide an autonomous clinical diagnosis. Predictions are for research and engineering demonstrations only; they must not replace professional medical assessment.

## Overview

Diabetic retinopathy is a complication of diabetes and an important cause of preventable vision loss. Early identification can enable timely clinical assessment and treatment.

In resource-constrained healthcare environments, screening can be limited by specialist shortages, geographic barriers, examination costs, growing patient volumes, and limited access to retinal imaging assessment.

This project explores whether an AI-assisted system could help healthcare workers identify retinal images that may require higher-priority specialist review. The goal is decision support—not replacing clinicians.

## Research question

**Can modern computer-vision architectures provide sufficiently sensitive diabetic-retinopathy screening while remaining computationally efficient enough for deployment in resource-constrained healthcare environments?**

Model selection is treated as a multi-objective engineering problem involving:

- Classification performance and sensitivity
- Computational efficiency
- Deployment feasibility

The project therefore evaluates models beyond raw accuracy.

## Objectives

### Primary objective

Develop and evaluate a deployable AI-assisted diabetic-retinopathy screening prototype using retinal fundus photographs.

### Specific objectives

1. Perform exploratory data analysis on the APTOS 2019 dataset.
2. Develop a standardized image-preprocessing and training pipeline.
3. Fine-tune modern computer-vision architectures for DR classification.
4. Compare multiple CNN and vision-transformer architectures.
5. Evaluate models using clinically relevant classification metrics.
6. Investigate the relationship between sensitivity, model size, and inference latency.
7. Select a model based on predictive performance and deployment feasibility.
8. Package reusable ML functionality through `fastdrs`.
9. Investigate export and inference in deployment-oriented environments.
10. Work toward integrating a trained model into an Android application or inference API.
11. Work toward an end-to-end screening workflow demonstration.

## Dataset

### APTOS 2019 Blindness Detection

The project README identifies the **APTOS 2019 Blindness Detection** dataset as the primary dataset for initial experiments. It contains approximately 3,662 labeled retinal fundus photographs across five DR severity grades.

Source: [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)

| Label | Severity |
|---:|---|
| 0 | No diabetic retinopathy |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative diabetic retinopathy |

The dataset is imbalanced, so accuracy is not treated as the sole measure of model quality.

> Record the exact dataset source and revision used for each experiment. If other datasets, such as EYEPACS, are introduced, document them separately rather than treating them as interchangeable with APTOS.

## Evaluation

The project considers both classification and deployment metrics.

**Classification metrics**
- Accuracy
- Macro precision
- Macro sensitivity / recall
- Macro specificity
- Macro F1 and weighted F1
- Per-class precision, sensitivity, and specificity
- Confusion matrix

**Deployment metrics**
- Inference latency
- Parameter count
- Estimated model size
- Deployment compatibility

The central comparison is **predictive performance vs. computational cost vs. deployment feasibility**. These metrics are evaluation targets; do not interpret them as completed benchmark results unless supported by recorded experiments.

## `fastdrs`

Reusable machine-learning components are packaged in **`fastdrs`**, a Python package intended to separate common ML functionality from experiment-specific notebooks and application code.

The package includes functionality for:
- Model construction
- Fundus-image preprocessing
- Dataset loading
- PyTorch training and early stopping
- Metadata-aware checkpoints
- Model evaluation
- PyTorch inference
- LiteRT inference and export workflows, where implemented and tested

Install the package:

```bash
pip install fastdrs
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the package structure, training and checkpoint design, preprocessing, and intended deployment architecture.

## Quick start

Install the core package:

```bash
pip install fastdrs
```

Optional dependency groups are intended for specific workflows:

```bash
pip install "fastdrs[training]"
pip install "fastdrs[inference]"
pip install "fastdrs[litert]"
```

Confirm the currently published package extras and implementation status before relying on a particular workflow.

### Command-line usage

The CLI supports model listing, checkpoint inspection, prediction, and LiteRT export:

```bash
fastdrs --help
fastdrs models
fastdrs inspect model.fastdrs
fastdrs predict model.fastdrs retina.jpg --json
fastdrs export model.fastdrs --output model.tflite
```

You can also run the CLI through the Python interpreter used to install the package:

```bash
python -m fastdrs --help
```

LiteRT export requires `pip install "fastdrs[litert]"` in a compatible environment.
Self-defining `.fastdrs` checkpoints supply their architecture and input image size;
legacy PyTorch checkpoints require `--architecture` and accept `--img-size`.

Build a model:

```python
from fastdrs.models import build_model

model = build_model(
    architecture="mobilenet_v2",
    num_classes=5,
    pretrained=True,
)

print(model)
```

## Current status

### Completed according to the project notes

- Reusable `fastdrs` package and multi-architecture model factory
- Fundus preprocessing utilities
- Dataset / DataLoader pipeline
- Training and evaluation pipeline
- Early stopping and metadata-aware checkpoints
- Experiment tracking integration
- PyPI package distribution

### In progress

- Architecture benchmarking on APTOS
- Final model selection
- Remote or pretrained model distribution
- Stable inference APIs
- LiteRT export and inference workflows
- Android integration
- End-to-end screening demonstration

### Pending

- Final validated APTOS model and final benchmark results
- External validation
- Complete Android deployment and end-to-end demonstration
- Clinical and operational validation

Treat this status list as a release checklist: update it to reflect the code and experiments that are actually available at the time of publication.

## Limitations

This is a research prototype with important limitations:

- APTOS is relatively small compared with large-scale medical-imaging datasets.
- Dataset distribution may not represent the intended target population.
- Dataset labels may contain uncertainty.
- Retinal image quality can vary substantially.
- Model performance does not automatically translate to clinical effectiveness.
- External validation is required.
- Clinical workflow integration has not been established.
- Clinical deployment would require appropriate clinical evaluation and regulatory review.
- Economic impact has not been empirically demonstrated.

The economic rationale is a hypothesis: if a system reliably identifies higher-risk retinal images for specialist review, it may help allocate limited ophthalmology resources more efficiently. This project does not claim quantified real-world cost savings.

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system design, package modules, training, preprocessing, checkpoints, and deployment plans.
- [`data/README.md`](data/README.md) — dataset provenance, data organization, and reproducibility, if present.

## License

Check the repository's `LICENSE` file and package metadata for the applicable terms. The source README previously stated MIT; if the project is intended to be research-only, reconcile the license file, README, and published package metadata before release. **Do not describe the project as research-only while distributing it under an MIT license**, which permits commercial use.

## Author

**Michael Mukiibi**  
GitHub: [@MichaelMukiibi](https://github.com/MichaelMukiibi)

## Disclaimer

`fastdrs` and the surrounding diabetic-retinopathy screening system are research and engineering projects. They are not medical devices and do not provide autonomous clinical diagnoses. Model predictions must not be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.

Any clinical deployment would require appropriate external validation, clinical evaluation, regulatory review, and integration with qualified healthcare professionals.
