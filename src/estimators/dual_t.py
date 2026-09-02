"""Non-anchor Dual-T transition matrix estimator (Xia et al., NeurIPS 2019)."""

import numpy as np
import torch
from typing import Union
from ..noise.matrix_utils import normalize_transition_matrix


def estimate_transition_matrix_dual_t(
    probs: Union[np.ndarray, torch.Tensor],
    noisy_labels: Union[np.ndarray, torch.Tensor],
    num_classes: int,
    slack_scale: float = 0.95,
) -> np.ndarray:
    """Estimate transition matrix T without strict anchor points using extreme value statistics.
    
    When anchor points are absent, max_{x} p(Y_tilde = i | x) is an upper bound on T_ii.
    We estimate the diagonal scaling factor and renormalize.
    
    Args:
        probs: (N, K) predicted posterior probabilities on noisy data.
        noisy_labels: (N,) observed noisy labels.
        num_classes: Number of classes K.
        slack_scale: Slack multiplier for non-anchor relaxation.
        
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
            T_hat[i, i] = 1.0
            continue

        # Top 5 most confident instances for class i
        top_k = min(5, len(class_indices))
        sorted_idx = class_indices[np.argsort(-probs[class_indices, i])[:top_k]]
        
        avg_prob_vec = np.mean(probs[sorted_idx, :], axis=0)
        # Apply slack relaxation
        avg_prob_vec[i] *= slack_scale
        T_hat[i, :] = avg_prob_vec

    return normalize_transition_matrix(T_hat)
