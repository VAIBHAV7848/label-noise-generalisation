"""Synthetic label noise injection algorithms."""

import numpy as np
from typing import Tuple, Optional
from .matrix_utils import (
    build_symmetric_transition_matrix,
    build_asymmetric_cifar10_transition_matrix,
    validate_transition_matrix,
)


def inject_noise_with_transition_matrix(
    clean_labels: np.ndarray,
    T: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Corrupt clean labels according to a given transition matrix T.
    
    For each sample with true label i, its noisy label j is drawn from Categorical(T_i).
    
    Args:
        clean_labels: 1D array of true integer labels in {0, ..., K-1}.
        T: K x K row-stochastic transition matrix where T_ij = P(Y_tilde = j | Y = i).
        seed: Optional random seed for reproducible sampling.
        
    Returns:
        noisy_labels: 1D array of corrupted integer labels.
        actual_noise_mask: Boolean array where True indicates flipped labels.
        empirical_noise_rate: Actual proportion of flipped labels.
    """
    validate_transition_matrix(T)
    clean_labels = np.asarray(clean_labels, dtype=np.int64)
    num_samples = len(clean_labels)
    num_classes = T.shape[0]

    if np.any(clean_labels < 0) or np.any(clean_labels >= num_classes):
        raise ValueError(f"Clean labels must be in range [0, {num_classes - 1}]")

    rng = np.random.default_rng(seed)
    noisy_labels = np.zeros(num_samples, dtype=np.int64)

    for c in range(num_classes):
        idx = np.where(clean_labels == c)[0]
        if len(idx) > 0:
            noisy_labels[idx] = rng.choice(num_classes, size=len(idx), p=T[c])

    actual_noise_mask = (noisy_labels != clean_labels)
    empirical_noise_rate = float(np.mean(actual_noise_mask))

    return noisy_labels, actual_noise_mask, empirical_noise_rate


def generate_synthetic_noisy_labels(
    clean_labels: np.ndarray,
    num_classes: int,
    noise_type: str,
    noise_rate: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate synthetic noisy labels given noise type and corruption rate.
    
    Args:
        clean_labels: 1D array of true integer labels.
        num_classes: Number of distinct classes K.
        noise_type: 'symmetric' or 'asymmetric'.
        noise_rate: Noise rate eta in [0, 1).
        seed: Random seed.
        
    Returns:
        noisy_labels: 1D array of corrupted labels.
        T: The theoretical K x K transition matrix used.
        actual_noise_mask: Boolean array of flipped instances.
        empirical_noise_rate: Proportion of flipped labels.
    """
    noise_type = noise_type.lower()
    if noise_type == "symmetric":
        T = build_symmetric_transition_matrix(num_classes, noise_rate)
    elif noise_type == "asymmetric":
        if num_classes == 10:
            T = build_asymmetric_cifar10_transition_matrix(noise_rate)
        else:
            # General pair-flip: class i -> class (i + 1) % K
            T = np.eye(num_classes, dtype=np.float64) * (1.0 - noise_rate)
            for i in range(num_classes):
                T[i, (i + 1) % num_classes] += noise_rate
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}. Must be 'symmetric' or 'asymmetric'")

    noisy_labels, actual_noise_mask, empirical_rate = inject_noise_with_transition_matrix(
        clean_labels=clean_labels,
        T=T,
        seed=seed,
    )
    return noisy_labels, T, actual_noise_mask, empirical_rate
