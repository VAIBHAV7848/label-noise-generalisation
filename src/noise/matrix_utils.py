"""Mathematical utilities for transition matrix generation, validation, and analysis."""

import numpy as np
from typing import Tuple


def build_symmetric_transition_matrix(num_classes: int, noise_rate: float) -> np.ndarray:
    """Construct a symmetric (uniform) noise transition matrix T in [0, 1]^(K x K).
    
    T_ij = 1 - eta if i == j else eta / (K - 1)
    
    Args:
        num_classes: Number of classes K >= 2.
        noise_rate: Overall corruption rate eta in [0, 1).
        
    Returns:
        K x K transition matrix as a float64 numpy array.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")
    if not (0.0 <= noise_rate < 1.0):
        raise ValueError(f"noise_rate must be in [0, 1), got {noise_rate}")

    if noise_rate == 0.0:
        return np.eye(num_classes, dtype=np.float64)

    off_diag = noise_rate / (num_classes - 1)
    diag = 1.0 - noise_rate
    T = np.full((num_classes, num_classes), off_diag, dtype=np.float64)
    np.fill_diagonal(T, diag)
    return T


def build_asymmetric_cifar10_transition_matrix(noise_rate: float) -> np.ndarray:
    """Construct asymmetric (pair-flip) noise transition matrix for CIFAR-10.
    
    Class mappings (Patrini et al. 2017):
      - TRUCK (9) -> AUTOMOBILE (1)
      - BIRD (2) -> AIRPLANE (0)
      - DEER (4) -> HORSE (7)
      - CAT (3) -> DOG (5)
      - DOG (5) -> CAT (3)
      
    Args:
        noise_rate: Flipping probability eta in [0, 0.5).
        
    Returns:
        10 x 10 transition matrix.
    """
    if not (0.0 <= noise_rate < 0.5):
        raise ValueError(f"Asymmetric pair-flip noise_rate must be in [0, 0.5), got {noise_rate}")

    T = np.eye(10, dtype=np.float64)
    if noise_rate == 0.0:
        return T

    # Pair flips
    flips = {
        9: 1,  # TRUCK -> AUTOMOBILE
        2: 0,  # BIRD -> AIRPLANE
        4: 7,  # DEER -> HORSE
        3: 5,  # CAT -> DOG
        5: 3,  # DOG -> CAT
    }

    for src, dst in flips.items():
        T[src, src] = 1.0 - noise_rate
        T[src, dst] = noise_rate

    return T


def validate_transition_matrix(T: np.ndarray, tol: float = 1e-6) -> bool:
    """Verify that T is a valid row-stochastic transition matrix.
    
    Args:
        T: K x K matrix.
        tol: Tolerance for row sum checks.
        
    Returns:
        True if valid, raises ValueError otherwise.
    """
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError(f"T must be a square 2D matrix, got shape {T.shape}")
    if np.any(T < -tol) or np.any(T > 1.0 + tol):
        raise ValueError("T entries must be probabilities in [0, 1]")
    row_sums = T.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        raise ValueError(f"Rows of T must sum to 1.0, got row sums: {row_sums}")
    return True


def normalize_transition_matrix(T: np.ndarray) -> np.ndarray:
    """Project and normalize an estimated matrix to the set of row-stochastic matrices.
    
    Args:
        T: K x K matrix (possibly unnormalized or containing small negative values).
        
    Returns:
        Valid row-stochastic transition matrix.
    """
    T_clipped = np.maximum(T, 0.0)
    row_sums = T_clipped.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return T_clipped / row_sums


def compute_matrix_condition_number(T: np.ndarray) -> float:
    """Compute the 2-norm condition number kappa(T) = ||T||_2 * ||T^-1||_2.
    
    Args:
        T: K x K matrix.
        
    Returns:
        Condition number as float.
    """
    return float(np.linalg.cond(T, p=2))
