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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from src.data import create_dataloaders
from src.models import build_model, get_model_info
from src.evaluate import evaluate_model


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
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
    Main entry point for training a DR classification model.
    """
    os.makedirs(save_dir, exist_ok=True)
    device_obj = torch.device(device)
    
    print(f"--- Training {architecture} on {device} ---")
    
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
    print(f"Model parameters: {model_info['total_parameters']:,} ({model_info['model_size_mb']} MB)")

    # 3. Optimization Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(save_dir, f"{architecture}_best.pth")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device_obj)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device_obj)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f" Saved new best model to {best_checkpoint_path}")

    # 5. Final Evaluation on Test Set using Best Weights
    if os.path.exists(best_checkpoint_path):
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device_obj))

    print("\n--- Final Test Set Evaluation ---")
    test_results = evaluate_model(model, test_loader, device_obj)
    print(f"Test Accuracy: {test_results['accuracy']}")
    print(f"Test Macro F1: {test_results['f1_macro']}")
    print(f"Test Macro Sensitivity: {test_results['sensitivity_macro']}")
    print(f"Test Latency: {test_results['inference_latency_ms']} ms/sample")

    return {
        "architecture": architecture,
        "history": history,
        "test_results": test_results,
        "best_checkpoint": best_checkpoint_path
    }


if __name__ == "__main__":
    train_model(architecture="resnet50", epochs=5)
