"""
Preprocessing module for Diabetic Retinopathy retinal fundus images.

Provides standard image transformations, augmentations, and specialized
preprocessing routines (e.g., circular cropping, Ben Graham color processing)
tailored for fundus image classification.
"""

from typing import Tuple, Optional, Callable
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2


def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Crops black boundaries from fundus images.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_start = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        if check_start.size == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            return np.stack([img1, img2, img3], axis=-1)
    return img


def ben_graham_preprocessing(img_pil: Image.Image, img_size: int = 224, sigma_x: int = 10) -> Image.Image:
    """
    Applies Ben Graham's method for enhancing retinal fundus features (subtract local average color).
    """
    img = np.array(img_pil)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma_x), -4, 128)
    return Image.fromarray(img)


def get_transforms(
    img_size: int = 224,
    is_train: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    use_ben_graham: bool = False
) -> T.Compose:
    """
    Returns torchvision transforms for training or evaluation.
    
    Args:
        img_size: Target image height and width.
        is_train: Whether to include data augmentation transforms.
        mean: Normalization channel means.
        std: Normalization channel standard deviations.
        use_ben_graham: Whether to apply Ben Graham preprocessing.
    """
    transforms_list = []
    
    if use_ben_graham:
        transforms_list.append(T.Lambda(lambda img: ben_graham_preprocessing(img, img_size=img_size)))
    else:
        transforms_list.append(T.Resize((img_size, img_size)))
    
    if is_train:
        transforms_list.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=30),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05))
        ])
    
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    
    return T.Compose(transforms_list)
