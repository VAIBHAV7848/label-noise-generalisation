"""Real-world human noise dataset utilities and loaders."""

import os
import numpy as np
from typing import Dict, Tuple, Optional


def load_cifar10n_noise(
    cifar10n_path: str,
    regime: str = "worst",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load human-annotated noisy labels from CIFAR-10N (.pt / .npy dictionary).
    
    Regimes supported:
      - 'clean': Original ground truth labels
      - 'aggre': Aggregated majority vote across 3 human annotators (~9.03% noise)
      - 'random1': Annotator 1 noisy labels (~17.2% noise)
      - 'random2': Annotator 2 noisy labels (~18.1% noise)
      - 'random3': Annotator 3 noisy labels (~17.6% noise)
      - 'worst': Annotator with lowest accuracy (~40.21% noise)
      
    Args:
        cifar10n_path: Filepath to CIFAR-10N annotation file (CIFAR-10_human.pt or .npy).
        regime: Label noise regime key.
        
    Returns:
        noisy_labels: 1D array of 50,000 human-annotated labels.
        clean_labels: 1D array of 50,000 ground-truth labels.
        actual_noise_rate: Empirical noise rate.
    """
    if not os.path.exists(cifar10n_path):
        raise FileNotFoundError(
            f"CIFAR-10N file not found at: {cifar10n_path}. "
            "Please download CIFAR-10_human.pt from the official CIFAR-10N repository."
        )

    # CIFAR-10N provides a dictionary with keys: 'clean_label', 'aggre_label', 'random_label1', 'random_label2', 'random_label3', 'worse_label'
    regime_map = {
        "clean": "clean_label",
        "aggre": "aggre_label",
        "random1": "random_label1",
        "random2": "random_label2",
        "random3": "random_label3",
        "worst": "worse_label",
    }

    regime_key = regime.lower()
    if regime_key not in regime_map:
        raise ValueError(f"Unknown CIFAR-10N regime: {regime}. Supported: {list(regime_map.keys())}")

    target_key = regime_map[regime_key]

    if cifar10n_path.endswith(".pt"):
        import torch
        data = torch.load(cifar10n_path, map_location="cpu")
    else:
        data = np.load(cifar10n_path, allow_pickle=True).item()

    clean_labels = np.asarray(data["clean_label"], dtype=np.int64)
    noisy_labels = np.asarray(data[target_key], dtype=np.int64)

    actual_noise_mask = (noisy_labels != clean_labels)
    actual_noise_rate = float(np.mean(actual_noise_mask))

    return noisy_labels, clean_labels, actual_noise_rate
