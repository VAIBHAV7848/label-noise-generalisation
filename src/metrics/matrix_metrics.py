"""Metrics for evaluating estimated noise transition matrices."""

import numpy as np


def compute_frobenius_error(T_hat: np.ndarray, T_true: np.ndarray) -> float:
    """Compute Frobenius norm error: ||T_hat - T_true||_F = sqrt(sum_ij (T_hat_ij - T_true_ij)^2).
    
    Args:
        T_hat: Estimated transition matrix.
        T_true: Ground-truth transition matrix.
        
    Returns:
        Frobenius norm error as float.
    """
    if T_hat.shape != T_true.shape:
        raise ValueError(f"Shape mismatch: {T_hat.shape} vs {T_true.shape}")
    return float(np.linalg.norm(T_hat - T_true, ord="fro"))


def compute_spectral_error(T_hat: np.ndarray, T_true: np.ndarray) -> float:
    """Compute Spectral (matrix 2-norm) error: ||T_hat - T_true||_2 = sigma_max(T_hat - T_true).
    
    Args:
        T_hat: Estimated transition matrix.
        T_true: Ground-truth transition matrix.
        
    Returns:
        Spectral norm error as float.
    """
    if T_hat.shape != T_true.shape:
        raise ValueError(f"Shape mismatch: {T_hat.shape} vs {T_true.shape}")
    return float(np.linalg.norm(T_hat - T_true, ord=2))


def compute_total_variation_error(T_hat: np.ndarray, T_true: np.ndarray) -> float:
    """Compute average Total Variation distance across all rows:
    
    TV(T_hat, T_true) = (1 / (2 * K)) * sum_i sum_j |T_hat_ij - T_true_ij|
    
    Args:
        T_hat: Estimated transition matrix.
        T_true: Ground-truth transition matrix.
        
    Returns:
        Total variation error as float in [0, 1].
    """
    if T_hat.shape != T_true.shape:
        raise ValueError(f"Shape mismatch: {T_hat.shape} vs {T_true.shape}")
    num_classes = T_hat.shape[0]
    return float((1.0 / (2.0 * num_classes)) * np.sum(np.abs(T_hat - T_true)))
