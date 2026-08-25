"""
Evaluation and benchmarking module for Diabetic Retinopathy screening models.

Calculates clinical metrics:
- Accuracy, Precision, Sensitivity/Recall, Specificity, F1 Score
- Per-class classification report and Confusion Matrix
- Inference latency and deployment feasibility metrics
"""

from typing import Dict, Any, List, Tuple, Optional
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.models import get_model_info


def compute_specificity(cm: np.ndarray) -> np.ndarray:
    """
    Computes per-class specificity from confusion matrix.
    Specificity = TN / (TN + FP)
    """
    num_classes = cm.shape[0]
    specificities = []
    total = np.sum(cm)
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = total - (tp + fp + fn)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    return np.array(specificities)


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluates a model on a dataloader, measuring clinical performance and inference latency.
    """
    if class_names is None:
        class_names = ["0 - No DR", "1 - Mild", "2 - Moderate", "3 - Severe", "4 - Proliferative"]

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    inference_times = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            
            start_time = time.perf_counter()
            outputs = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            batch_latency = (end_time - start_time) / inputs.size(0)
            inference_times.append(batch_latency)
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
    
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(class_names))))
    per_class_spec = compute_specificity(cm)
    per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average=None, labels=list(range(len(class_names))), zero_division=0
    )

    avg_latency_ms = float(np.mean(inference_times) * 1000.0)
    model_info = get_model_info(model)

    results = {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(precision), 4),
        "sensitivity_macro": round(float(recall), 4),
        "specificity_macro": round(float(np.mean(per_class_spec)), 4),
        "f1_macro": round(float(f1), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "inference_latency_ms": round(avg_latency_ms, 2),
        "model_size_mb": model_info["model_size_mb"],
        "total_parameters": model_info["total_parameters"],
        "confusion_matrix": cm.tolist(),
        "per_class": {
            class_names[i]: {
                "precision": round(float(per_class_prec[i]), 4),
                "sensitivity_recall": round(float(per_class_rec[i]), 4),
                "specificity": round(float(per_class_spec[i]), 4),
                "f1": round(float(per_class_f1[i]), 4),
            } for i in range(len(class_names))
        }
    }

    return results
