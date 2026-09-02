"""Dataset wrappers and reproducible DataLoader builders."""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.datasets as datasets
import numpy as np
from typing import Tuple, Optional, Dict, Any
from .transforms import get_cifar_transforms, get_mnist_transforms
from ..noise.synthetic import generate_synthetic_noisy_labels
from ..noise.matrix_utils import build_symmetric_transition_matrix


class NoisyDataset(Dataset):
    """PyTorch Dataset wrapper attaching corrupted labels to base dataset."""

    def __init__(
        self,
        base_dataset: Dataset,
        noisy_labels: np.ndarray,
        clean_labels: np.ndarray,
        transform=None,
    ):
        self.base_dataset = base_dataset
        self.noisy_labels = np.asarray(noisy_labels, dtype=np.int64)
        self.clean_labels = np.asarray(clean_labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int, int]:
        # base_dataset item can be (image, label) or (data, label)
        data = self.base_dataset[index]
        if isinstance(data, tuple) or isinstance(data, list):
            img = data[0]
        else:
            img = data

        if self.transform is not None and not isinstance(img, torch.Tensor):
            img = self.transform(img)

        noisy_target = int(self.noisy_labels[index])
        clean_target = int(self.clean_labels[index])
        return img, noisy_target, clean_target, index


def build_cifar10_loaders(
    data_dir: str = "./data",
    noise_type: str = "symmetric",
    noise_rate: float = 0.2,
    batch_size: int = 128,
    num_workers: int = 2,
    val_size: int = 5000,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """Build train (noisy), val (clean/noisy), and test (clean) DataLoaders for CIFAR-10.
    
    Returns:
        train_loader, val_loader, test_loader, true_transition_matrix
    """
    train_transform = get_cifar_transforms(train=True)
    test_transform = get_cifar_transforms(train=False)

    raw_train_set = datasets.CIFAR10(root=data_dir, train=True, download=True)
    raw_test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    clean_targets = np.array(raw_train_set.targets, dtype=np.int64)
    num_classes = 10

    # Split train indices into train and val
    rng = np.random.default_rng(seed)
    indices = np.arange(len(clean_targets))
    rng.shuffle(indices)

    train_idx = indices[:-val_size]
    val_idx = indices[-val_size:]

    # Inject noise into train partition
    train_clean_targets = clean_targets[train_idx]
    train_noisy_targets, T, _, _ = generate_synthetic_noisy_labels(
        clean_labels=train_clean_targets,
        num_classes=num_classes,
        noise_type=noise_type,
        noise_rate=noise_rate,
        seed=seed,
    )

    # Base subsets
    train_subset = torch.utils.data.Subset(raw_train_set, train_idx)
    val_subset = torch.utils.data.Subset(raw_train_set, val_idx)

    train_dataset = NoisyDataset(
        base_dataset=train_subset,
        noisy_labels=train_noisy_targets,
        clean_labels=train_clean_targets,
        transform=train_transform,
    )

    val_dataset = NoisyDataset(
        base_dataset=val_subset,
        noisy_labels=clean_targets[val_idx],
        clean_labels=clean_targets[val_idx],
        transform=test_transform,
    )

    test_dataset = NoisyDataset(
        base_dataset=raw_test_set,
        noisy_labels=np.array(raw_test_set.targets),
        clean_labels=np.array(raw_test_set.targets),
        transform=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, T
