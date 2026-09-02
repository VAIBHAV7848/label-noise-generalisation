"""Anchor-point based transition matrix estimation (Patrini et al., CVPR 2017)."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Union
from ..noise.matrix_utils import normalize_transition_matrix


def estimate_transition_matrix_anchor_points(
    probs: Union[np.ndarray, torch.Tensor],
    noisy_labels: Union[np.ndarray, torch.Tensor],
    num_classes: int,
    percentile: float = 97.0,
) -> np.ndarray:
    """Estimate noise transition matrix T via anchor points.
    
    Under the anchor point assumption, for each class i, the instances x in class i
    maximizing predicted probability p_i(x) satisfy p_j(x) approx T_ij.
    
    Args:
        probs: (N, K) predicted posterior probabilities on noisy dataset.
        noisy_labels: (N,) observed noisy labels.
        num_classes: Number of classes K.
        percentile: Confidence percentile (default 97.0%) to select robust anchor points.
        
    Returns:
        K x K estimated transition matrix T_hat.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(noisy_labels, torch.Tensor):
        noisy_labels = noisy_labels.detach().cpu().numpy()

    T_hat = np.zeros((num_classes, num_classes), dtype=np.float64)

    for i in range(num_classes):
        class_indices = np.where(noisy_labels == i)[0]
        if len(class_indices) == 0:
            # Fallback if no samples in class
            T_hat[i, i] = 1.0
            continue

        class_probs = probs[class_indices, i]
        # Find threshold at the specified percentile
        threshold = np.percentile(class_probs, percentile)
        anchor_mask = class_probs >= threshold
        anchor_indices = class_indices[anchor_mask]

        if len(anchor_indices) > 0:
            T_hat[i, :] = np.mean(probs[anchor_indices, :], axis=0)
        else:
            max_idx = class_indices[np.argmax(class_probs)]
            T_hat[i, :] = probs[max_idx, :]

    return normalize_transition_matrix(T_hat)
