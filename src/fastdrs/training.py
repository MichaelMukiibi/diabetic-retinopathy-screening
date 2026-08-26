"""
Training script and loop for Diabetic Retinopathy classification models.

Features:
- PyTorch modular training loop with validation
- Support for class-weighted loss (for imbalanced DR classes)
- Learning rate schedulers (CosineAnnealing / ReduceLROnPlateau)
- Best model checkpointing
- Weights & Biases (wandb) logging integration
"""

from typing import Dict, Any, Optional, Tuple
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from fastdrs.data import create_dataloaders
from fastdrs.models import build_model, get_model_info
from fastdrs.evaluation import evaluate_model


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Returns:
        Tuple containing epoch loss and accuracy.
    """
     
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model for one epoch.

    Returns:
        Tuple containing validation loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def _initialize_wandb(
    config: Dict[str, Any],
):
    """
    Initialize a W&B run using WANDB_API_KEY from the environment.

    No W&B project, entity, or other credentials are required here.
    W&B will use its standard defaults/environment configuration.

    Returns:
        W&B run object.
    """
    api_key = os.getenv("WANDB_API_KEY")

    if not api_key:
        raise RuntimeError(
            "WANDB_API_KEY environment variable is not set. "
            "Set it before using use_wandb=True."
        )

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B is required when use_wandb=True. "
            "Install it with: pip install wandb"
        ) from exc

    wandb.login(key=api_key)

    run = wandb.init(
        project="fastdrs",
        config=config,
    )

    return run


def _log_model_artifact(
    run,
    checkpoint_path: str,
    architecture: str,
):
    """
    Upload the best model checkpoint to W&B as an artifact.
    """
    import wandb

    artifact = wandb.Artifact(
        name=f"fastdrs-{architecture}",
        type="model",
        description=(
            f"Best trained FastDRS {architecture} checkpoint."
        ),
        metadata={
            "architecture": architecture,
            "checkpoint": os.path.basename(checkpoint_path),
        },
    )

    artifact.add_file(checkpoint_path)

    run.log_artifact(
        artifact,
        aliases=["best"],
    )

def train_model(
    architecture: str = "resnet50",
    dataset_name: str = "sngsfydy/aptos",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    img_size: int = 224,
    save_dir: str = "models",
    use_wandb: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """
    Train a Diabetic Retinopathy classification model.

    Args:
        architecture:
            Model architecture supported by fastdrs.

        dataset_name:
            Hugging Face dataset identifier.

        epochs:
            Number of training epochs.

        batch_size:
            Training and evaluation batch size.

        lr:
            Initial learning rate.

        weight_decay:
            AdamW weight decay.

        img_size:
            Input image size.

        save_dir:
            Directory for local model checkpoints.

        use_wandb:
            Whether to enable Weights & Biases tracking.

            When True, WANDB_API_KEY must be present in the environment.
 
        device:
            PyTorch device.

    Returns:
        Dictionary containing training history, test results, and checkpoint information.
    """
    os.makedirs(save_dir, exist_ok=True)

    device_obj = torch.device(device)
    
    print(f"--- Training {architecture} on {device} ---")

    # W&B Initialisation
    if use_wandb:
        wandb_config = {
            "architecture": architecture,
            "dataset": dataset_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "image_size": img_size,
            "device": str(device_obj),
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "num_classes": 5,
        }

        wandb_run = _initialize_wandb(wandb_config)

    try:
        # 1. DataLoaders
        train_loader, val_loader, test_loader, ds_info = create_dataloaders(
            dataset_name=dataset_name,
            batch_size=batch_size,
            img_size=img_size
        )

        # 2. Build Model
        model = build_model(architecture=architecture, num_classes=5, pretrained=True)
        model.to(device_obj)
        
        model_info = get_model_info(model)

        print(
            f"Model parameters: "
            f"{model_info['total_parameters']:,} "
            f"({model_info['model_size_mb']} MB)"
        )

        # Log model/dataset information once.
        if wandb_run is not None:
            wandb_run.config.update(
                {
                    "total_parameters": model_info["total_parameters"],
                    "model_size_mb": model_info["model_size_mb"],
                    **ds_info,
                },
                allow_val_change=True,
            )

        # 3. Optimization Setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")

        best_checkpoint_path = os.path.join(save_dir, f"{architecture}_best.pth")

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        # 4. Training Loop
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device_obj,
            )

            val_loss, val_acc = validate_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device_obj
            )

            scheduler.step()

            elapsed = time.time() - t0

            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch:02d}/{epochs:02d} "
                f"[{elapsed:.1f}s] | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )

            # W&B Epoch Logging
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/accuracy": train_acc,
                        "validation/loss": val_loss,
                        "validation/accuracy": val_acc,
                        "learning_rate": current_lr,
                        "epoch_time_seconds": elapsed,
                    },
                    step=epoch,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save(
                    model.state_dict(),
                    best_checkpoint_path
                )

                print(
                    f" Saved new best model to "
                    f"{best_checkpoint_path}"
                )

        # 5. Load best checkpoint
        if os.path.exists(best_checkpoint_path):
            model.load_state_dict(
                torch.load(
                    best_checkpoint_path, 
                    map_location=device_obj,
                    )
                )

        # 6. Final Evaluation on Test Set
        print("\n--- Final Test Set Evaluation ---")

        test_results = evaluate_model(
            model, 
            test_loader, 
            device_obj
        )

        print(
            f"Test Accuracy: " 
            f"{test_results['accuracy']}"
        )

        print(
            f"Test Macro F1: " 
            f"{test_results['f1_macro']}"
        )

        print(
            f"Test Macro Sensitivity: " 
            f"{test_results['sensitivity_macro']}"
        )

        print(
            f"Test Latency: " 
            f"{test_results['inference_latency_ms']} ms/sample"
        )

        # 7. Log final evaluation to W&B
        if wandb_run is not None:
            wandb_run.log(
                {
                    "test/accuracy": test_results["accuracy"],
                    "test/precision_macro": test_results[
                        "precision_macro"
                    ],
                    "test/sensitivity_macro": test_results[
                        "sensitivity_macro"
                    ],
                    "test/specificity_macro": test_results[
                        "specificity_macro"
                    ],
                    "test/f1_macro": test_results["f1_macro"],
                    "test/f1_weighted": test_results[
                        "f1_weighted"
                    ],
                    "test/inference_latency_ms": test_results[
                        "inference_latency_ms"
                    ],
                    "test/model_size_mb": test_results[
                        "model_size_mb"
                    ],
                    "test/total_parameters": test_results[
                        "total_parameters"
                    ],
                }
            )

            # Upload best checkpoint to W&B as an artifact
            _log_model_artifact(
                run=wandb_run,
                checkpoint_path=best_checkpoint_path,
                architecture=architecture,
            )
    
        return {
            "architecture": architecture,
            "history": history,
            "test_results": test_results,
            "best_checkpoint": best_checkpoint_path
        }

    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    train_model(
        architecture="resnet50",
        epochs=5,
    )
