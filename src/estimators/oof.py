"""Out-Of-Fold (OOF) prediction generation for literature-faithful transition matrix estimation (Northcutt et al., JAIR 2021)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Optional

from PIL import Image
from ..data.transforms import get_cifar_transforms


class FoldDataset(Dataset):
    """Dataset wrapper for a subset of samples during cross-validation."""

    def __init__(self, data: np.ndarray, labels: np.ndarray, transform=None):
        self.data = data
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img = self.data[idx]
        if self.transform is not None:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = self.transform(img)
        return img, int(self.labels[idx]), idx


def generate_deterministic_folds(
    num_samples: int,
    n_splits: int = 3,
    seed: int = 42,
) -> list:
    """Generate deterministic, disjoint K-fold index partitions.
    
    Args:
        num_samples: Total number of samples N.
        n_splits: Number of cross-validation folds K (default: 3).
        seed: Random seed for deterministic shuffling.
        
    Returns:
        List of tuples: (train_indices, val_indices) for each fold.
    """
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(num_samples)
    
    fold_sizes = np.full(n_splits, num_samples // n_splits, dtype=int)
    fold_sizes[: num_samples % n_splits] += 1
    
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = shuffled_indices[start:stop]
        train_idx = np.concatenate([shuffled_indices[:start], shuffled_indices[stop:]])
        folds.append((train_idx, val_idx))
        current = stop
        
    return folds


def compute_oof_predicted_probabilities(
    data: np.ndarray,
    noisy_labels: np.ndarray,
    num_classes: int,
    model_fn: Callable[[], nn.Module],
    n_splits: int = 3,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 0.05,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Compute out-of-fold (OOF) predicted probabilities across training dataset via K-fold CV.
    
    Guarantees:
    1. Every sample i is predicted by a model trained exclusively on data \\ S_k (where i in S_k).
    2. Only noisy labels are used (clean labels are strictly inoperable).
    3. Outputs an (N, num_classes) row-stochastic probability matrix.
    
    Args:
        data: (N, H, W, C) raw image array.
        noisy_labels: (N,) observed noisy training labels.
        num_classes: Number of classes K.
        model_fn: Factory function instantiating a fresh model architecture.
        n_splits: Number of CV folds (default: 3).
        epochs: Number of warm-up training epochs per fold (default: 5).
        batch_size: Mini-batch size.
        lr: Learning rate for warm-up SGD optimizer.
        seed: Random seed.
        device: PyTorch device.
        
    Returns:
        (N, num_classes) array of out-of-fold predicted probabilities.
    """
    num_samples = len(data)
    oof_probs = np.zeros((num_samples, num_classes), dtype=np.float32)
    
    train_transform = get_cifar_transforms(train=True)
    eval_transform = get_cifar_transforms(train=False)
    
    folds = generate_deterministic_folds(num_samples, n_splits=n_splits, seed=seed)
    
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        # 1. Instantiate fresh model for this fold
        fold_model = model_fn().to(device)
        fold_model.train()
        
        # 2. Build fold datasets
        train_ds = FoldDataset(data[train_indices], noisy_labels[train_indices], transform=train_transform)
        val_ds = FoldDataset(data[val_indices], noisy_labels[val_indices], transform=eval_transform)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        optimizer = torch.optim.SGD(fold_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # 3. Train fold model for warm-up epochs
        for epoch in range(epochs):
            for imgs, targets, _ in train_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad()
                logits = fold_model(imgs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                
        # 4. Predict probabilities on out-of-fold holdout partition
        fold_model.eval()
        fold_probs = []
        with torch.no_grad():
            for imgs, _, _ in val_loader:
                imgs = imgs.to(device)
                logits = fold_model(imgs)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                fold_probs.append(probs)
                
        fold_probs = np.concatenate(fold_probs, axis=0)
        oof_probs[val_indices] = fold_probs
        
    # Ensure numerical normalization
    row_sums = oof_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    oof_probs = oof_probs / row_sums
    
    return oof_probs
