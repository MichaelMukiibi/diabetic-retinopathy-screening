"""
Data loading and dataset handling for Diabetic Retinopathy screening.

Supports Hugging Face datasets (e.g. `ctmedtech/EYEPACS`) as well as standard PyTorch datasets,
with reproducible train/validation/test splitting and DataLoader construction.
"""

from typing import Tuple, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from PIL import Image
import numpy as np

from src.preprocessing import get_transforms


class EyepacsDataset(Dataset):
    """
    PyTorch Wrapper for Hugging Face EYEPACS dataset or image list.
    """
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.dataset[int(idx)]
        image = item["image"]
        
        # Ensure image is PIL Image in RGB mode
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.uint8(image))
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        label = item.get("label", item.get("dr_level", item.get("level", 0)))
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, int(label)


def create_dataloaders(
    dataset_name: str = "sngsfydy/aptos",
    batch_size: int = 32,
    img_size: int = 224,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    num_workers: int = 2,
    use_ben_graham: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Loads dataset, splits into train/val/test, applies transforms, and returns PyTorch DataLoaders.
    
    Returns:
        train_loader, val_loader, test_loader, dataset_info metadata dict
    """
    # Load raw dataset from Hugging Face
    raw_dataset = load_dataset(dataset_name, split="train")
    total_size = len(raw_dataset)
    indices = np.arange(total_size)
    
    # Get labels if available for stratified splitting
    try:
        labels = [raw_dataset[i].get("label", raw_dataset[i].get("dr_level", 0)) for i in range(min(total_size, 1000))]
        stratify = None # Avoid full scan if dataset is large, or use stratified split on subset
    except Exception:
        labels = None
    
    # Train / Val / Test Split
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_split, random_state=seed, shuffle=True
    )
    relative_val_size = val_split / (1.0 - test_split)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val_size, random_state=seed, shuffle=True
    )
    
    train_idx = [int(i) for i in train_idx]
    val_idx = [int(i) for i in val_idx]
    test_idx = [int(i) for i in test_idx]
    
    train_transform = get_transforms(img_size=img_size, is_train=True, use_ben_graham=use_ben_graham)
    eval_transform = get_transforms(img_size=img_size, is_train=False, use_ben_graham=use_ben_graham)
    
    full_train_ds = EyepacsDataset(raw_dataset, transform=train_transform)
    full_eval_ds = EyepacsDataset(raw_dataset, transform=eval_transform)
    
    train_ds = Subset(full_train_ds, train_idx)
    val_ds = Subset(full_eval_ds, val_idx)
    test_ds = Subset(full_eval_ds, test_idx)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    info = {
        "dataset_name": dataset_name,
        "total_samples": total_size,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "batch_size": batch_size,
        "img_size": img_size
    }
    
    return train_loader, val_loader, test_loader, info
