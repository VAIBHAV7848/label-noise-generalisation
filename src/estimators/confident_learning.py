"""Confident Learning joint distribution and transition matrix estimator (Northcutt et al., JAIR 2021)."""

import numpy as np
import torch
from typing import Tuple, Union
from ..noise.matrix_utils import normalize_transition_matrix


def compute_class_thresholds(
    probs: np.ndarray,
    noisy_labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute per-class self-confidence thresholds t_j = E_{x in X_j}[p(Y_tilde=j | x)].
    
    Args:
        probs: (N, K) predicted probabilities.
        noisy_labels: (N,) noisy labels.
        num_classes: Number of classes K.
        
    Returns:
        (K,) vector of class thresholds.
    """
    thresholds = np.zeros(num_classes, dtype=np.float64)
    for j in range(num_classes):
        idx = np.where(noisy_labels == j)[0]
        if len(idx) > 0:
            thresholds[j] = np.mean(probs[idx, j])
        else:
            thresholds[j] = 1.0 / num_classes
    return thresholds


def estimate_transition_matrix_confident_learning(
    probs: Union[np.ndarray, torch.Tensor],
    noisy_labels: Union[np.ndarray, torch.Tensor],
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate joint distribution Q_{Y_tilde, Y*} and transition matrix T via Confident Learning.
    
    Args:
        probs: (N, K) predicted out-of-sample probabilities.
        noisy_labels: (N,) observed noisy labels.
        num_classes: Number of classes K.
        
    Returns:
        T_hat: (K, K) estimated transition matrix P(Y_tilde | Y*).
        joint_Q: (K, K) estimated joint distribution P(Y_tilde, Y*).
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(noisy_labels, torch.Tensor):
        noisy_labels = noisy_labels.detach().cpu().numpy()

    num_samples = len(noisy_labels)
    thresholds = compute_class_thresholds(probs, noisy_labels, num_classes)
    
    # Construct unnormalized confusion counting matrix C_jk
    # Each sample x is assigned to class k with highest margin p_k - t_k among classes exceeding threshold
    C = np.zeros((num_classes, num_classes), dtype=np.float64)

    for idx in range(num_samples):
        j = noisy_labels[idx]
        p_vec = probs[idx]
        
        # Check candidate classes exceeding threshold
        valid_candidates = np.where(p_vec >= thresholds)[0]
        if len(valid_candidates) > 0:
            # Pick class maximizing margin p_k - t_k
            best_k = valid_candidates[np.argmax(p_vec[valid_candidates] - thresholds[valid_candidates])]
        else:
            best_k = np.argmax(p_vec)
            
        C[j, best_k] += 1.0

    # Normalize C to obtain joint distribution Q(Y_tilde = j, Y* = k)
    total_count = np.sum(C)
    if total_count > 0:
        joint_Q = C / total_count
    else:
        joint_Q = np.eye(num_classes) / num_classes

    # Derive transition matrix T_kj = P(Y_tilde = j | Y* = k) = Q(j, k) / sum_j' Q(j', k)
    true_class_marginals = joint_Q.sum(axis=0, keepdims=True)  # (1, K)
    true_class_marginals = np.where(true_class_marginals == 0, 1.0, true_class_marginals)
    
    # Transpose so T_kj is row k (true), column j (noisy)
    T_hat = (joint_Q / true_class_marginals).T

    return normalize_transition_matrix(T_hat), joint_Q
