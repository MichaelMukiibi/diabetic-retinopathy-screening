#!/usr/bin/env python3
"""
Top-level Training Script for Diabetic Retinopathy Screening & Triage.

Designed for single-command execution on local machines (PC/workstation),
featuring:
- Hugging Face Datasets integration (e.g. `sngsfydy/aptos` or `ctmedtech/EYEPACS`)
- Weights & Biases (W&B) live experiment tracking
- Multi-architecture support (ResNet, DenseNet, ConvNeXt, Swin, MobileNet)
- Clinical evaluation metrics & confusion matrix logging
- Automated checkpointing and results export to CSV

Usage:
    python train.py
    python train.py --arch densenet121 --epochs 15 --batch-size 16 --ben-graham
    python train.py --arch convnext_tiny --no-wandb
"""

import os
import sys
import time
import argparse
import random
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Import modular project components
try:
    from src.data import create_dataloaders
    from src.models import build_model, get_model_info
    from src.evaluate import evaluate_model
except ImportError:
    # Ensure current directory is in sys.path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.data import create_dataloaders
    from src.models import build_model, get_model_info
    from src.evaluate import evaluate_model


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Vision Models for Diabetic Retinopathy Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Model architecture
    parser.add_argument(
        "--arch", "-a", type=str, default="resnet50",
        choices=["resnet50", "resnet18", "densenet121", "convnext_tiny", "swin_t", "mobilenet_v2"],
        help="Vision architecture to fine-tune"
    )
    parser.add_argument(
        "--num-classes", type=int, default=5,
        help="Number of DR severity grades (0: No DR -> 4: Proliferative)"
    )
    # Dataset & Preprocessing
    parser.add_argument(
        "--dataset", "-d", type=str, default="sngsfydy/aptos",
        help="Hugging Face dataset identifier"
    )
    parser.add_argument(
        "--img-size", type=int, default=224,
        help="Image resolution (height and width)"
    )
    parser.add_argument(
        "--ben-graham", action="store_true", default=False,
        help="Enable Ben Graham color preprocessing (local color subtraction)"
    )
    # Training Hyperparameters
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data split & training")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader subprocess workers")
    
    # Device & Paths
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Compute device (auto picks CUDA if available, then MPS, else CPU)"
    )
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save model weights")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save metric tables")
    
    # Weights & Biases (W&B)
    parser.add_argument("--wandb-project", type=str, default="diabetic-retinopathy-screening", help="W&B Project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B Entity (username or team)")
    parser.add_argument("--no-wandb", action="store_true", default=False, help="Disable W&B tracking")

    return parser.parse_args()


def select_device(requested_device: str) -> torch.device:
    """Selects the execution device with informative diagnostics."""
    if requested_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
            print(f"[Device] Using CUDA GPU: {device_name}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("[Device] Using Apple Silicon MPS")
        else:
            device = torch.device("cpu")
            print("[Device] CUDA/MPS not available. Falling back to CPU.")
    else:
        device = torch.device(requested_device)
        print(f"[Device] Manually selected device: {device}")
    return device


def init_wandb(args: argparse.Namespace) -> Optional[Any]:
    """Initializes Weights & Biases if enabled and installed."""
    if args.no_wandb:
        print("[W&B] Tracking disabled via --no-wandb flag.")
        return None
    try:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.arch}_{args.dataset.split('/')[-1]}_{int(time.time())}",
            config=vars(args),
            reinit=True
        )
        print(f"[W&B] Initialized run: {wandb.run.name} ({wandb.run.url})")
        return run
    except ImportError:
        print("[W&B] wandb package not installed. Continuing without online tracking.")
        return None
    except Exception as e:
        print(f"[W&B] Failed to initialize W&B ({e}). Continuing locally.")
        return None


def run_training_loop(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    epochs: int,
    device: torch.device,
    save_path: str,
    wandb_run: Optional[Any] = None
) -> Dict[str, Any]:
    """Executes the full training loop with validation and best checkpointing."""
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

    print("\n" + "=" * 70)
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'Val F1':^8} | {'Time':^7}")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Training Step
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += inputs.size(0)

        epoch_train_loss = running_train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation Step
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += inputs.size(0)

                all_val_preds.extend(preds.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())

        epoch_val_loss = running_val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        # Calculate Validation Macro F1
        from sklearn.metrics import f1_score
        epoch_val_f1 = f1_score(all_val_targets, all_val_preds, average="macro", zero_division=0)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0

        # Print progress row
        print(f"{epoch:^7d} | {epoch_train_loss:^10.4f} | {epoch_train_acc:^10.4f} | {epoch_val_loss:^10.4f} | {epoch_val_acc:^10.4f} | {epoch_val_f1:^8.4f} | {elapsed:^6.1f}s")

        # Record history
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["val_f1"].append(epoch_val_f1)

        # Log to W&B
        if wandb_run is not None:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": epoch_train_loss,
                "train/accuracy": epoch_train_acc,
                "val/loss": epoch_val_loss,
                "val/accuracy": epoch_val_acc,
                "val/f1_macro": epoch_val_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_time_seconds": elapsed
            })

        # Save Best Model Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_f1 = epoch_val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_f1": best_val_f1
            }, save_path)
            print(f"        └─ Saved new best checkpoint (Val Loss: {best_val_loss:.4f}, Val F1: {best_val_f1:.4f})")

    print("=" * 70)
    return history


def save_summary_metrics(
    results: Dict[str, Any],
    args: argparse.Namespace,
    model_info: Dict[str, Any],
    results_dir: str
) -> str:
    """Appends final benchmark metrics to CSV for multi-model comparison."""
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "model_comparison.csv")

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": args.arch,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "ben_graham": args.ben_graham,
        "total_parameters": model_info["total_parameters"],
        "model_size_mb": model_info["model_size_mb"],
        "test_accuracy": results["accuracy"],
        "test_f1_macro": results["f1_macro"],
        "test_sensitivity_macro": results["sensitivity_macro"],
        "test_specificity_macro": results["specificity_macro"],
        "inference_latency_ms": results["inference_latency_ms"]
    }

    df_new = pd.DataFrame([record])
    if os.path.exists(summary_path):
        df_existing = pd.read_csv(summary_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("\n" + "#" * 70)
    print(f"# AI-Assisted Diabetic Retinopathy Screening — Training Pipeline")
    print(f"# Model Architecture : {args.arch}")
    print(f"# Dataset Source     : {args.dataset}")
    print(f"# Epochs / Batch Size: {args.epochs} / {args.batch_size}")
    print(f"# Image Resolution   : {args.img_size}x{args.img_size} (Ben Graham: {args.ben_graham})")
    print("#" * 70 + "\n")

    # 1. Device Setup
    device = select_device(args.device)

    # 2. Weights & Biases Init
    wandb_run = init_wandb(args)

    # 3. Create DataLoaders via Hugging Face
    print("\n[Data] Preparing Hugging Face Dataset & DataLoaders...")
    train_loader, val_loader, test_loader, ds_info = create_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        img_size=args.img_size,
        seed=args.seed,
        num_workers=args.num_workers,
        use_ben_graham=args.ben_graham
    )
    print(f"[Data] Train: {ds_info['train_samples']} | Val: {ds_info['val_samples']} | Test: {ds_info['test_samples']} samples")

    # 4. Model Construction
    print(f"\n[Model] Instantiating {args.arch} with {args.num_classes} output classes...")
    model = build_model(architecture=args.arch, num_classes=args.num_classes, pretrained=True)
    model.to(device)
    
    model_info = get_model_info(model)
    print(f"[Model] Parameters: {model_info['total_parameters']:,} | Model Size: {model_info['model_size_mb']} MB")

    if wandb_run is not None:
        import wandb
        wandb.config.update(model_info)
        wandb.watch(model, log="all", log_freq=100)

    # 5. Loss, Optimizer, and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 6. Checkpoint Path
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_file = f"{args.arch}_{args.dataset.split('/')[-1]}_best.pth"
    best_checkpoint_path = os.path.join(args.save_dir, checkpoint_file)

    # 7. Execute Training Loop
    history = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        save_path=best_checkpoint_path,
        wandb_run=wandb_run
    )

    # 8. Load Best Checkpoint for Final Test Evaluation
    if os.path.exists(best_checkpoint_path):
        print(f"\n[Evaluation] Loading best checkpoint from {best_checkpoint_path}...")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    print("\n[Evaluation] Running comprehensive clinical benchmark on test dataset...")
    test_results = evaluate_model(model, test_loader, device=device)

    # 9. Print Benchmark Summary
    print("\n" + "=" * 70)
    print(f"{'FINAL TEST BENCHMARK RESULTS':^70}")
    print("=" * 70)
    print(f"  • Test Accuracy             : {test_results['accuracy'] * 100:.2f}%")
    print(f"  • Macro F1 Score            : {test_results['f1_macro']:.4f}")
    print(f"  • Macro Sensitivity (Recall): {test_results['sensitivity_macro']:.4f}")
    print(f"  • Macro Specificity         : {test_results['specificity_macro']:.4f}")
    print(f"  • Inference Latency         : {test_results['inference_latency_ms']:.2f} ms / image")
    print(f"  • Total Parameters          : {test_results['total_parameters']:,}")
    print(f"  • Model Size                : {test_results['model_size_mb']} MB")
    print("=" * 70)

    # Print Per-Class Breakdown
    print(f"\n{'Per-Class Clinical Breakdown':^70}")
    print("-" * 70)
    print(f"{'Class':<22} | {'Precision':^10} | {'Sensitivity':^12} | {'Specificity':^12} | {'F1':^8}")
    print("-" * 70)
    for class_name, metrics in test_results["per_class"].items():
        print(f"{class_name:<22} | {metrics['precision']:^10.4f} | {metrics['sensitivity_recall']:^12.4f} | {metrics['specificity']:^12.4f} | {metrics['f1']:^8.4f}")
    print("-" * 70)

    # 10. Save Metrics to CSV
    csv_file = save_summary_metrics(test_results, args, model_info, args.results_dir)
    print(f"\n[Artifacts] Benchmark metrics appended to: {csv_file}")
    print(f"[Artifacts] Best model checkpoint saved to: {best_checkpoint_path}")

    # 11. Log Final Artifacts to W&B
    if wandb_run is not None:
        import wandb
        wandb.log({
            "test/accuracy": test_results["accuracy"],
            "test/f1_macro": test_results["f1_macro"],
            "test/sensitivity_macro": test_results["sensitivity_macro"],
            "test/specificity_macro": test_results["specificity_macro"],
            "test/inference_latency_ms": test_results["inference_latency_ms"]
        })
        
        # Save confusion matrix to W&B
        try:
            class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
            cm = np.array(test_results["confusion_matrix"])
            wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=None,
                preds=None,
                class_names=class_names,
                matrix_values=cm
            )})
        except Exception as e:
            print(f"[W&B] Note: Confusion matrix plot skipped ({e})")

        wandb.finish()
        print("[W&B] Run finished and synced successfully.")


if __name__ == "__main__":
    main()
