"""Noise generation and matrix manipulation module."""

from .matrix_utils import (
    build_symmetric_transition_matrix,
    build_asymmetric_cifar10_transition_matrix,
    validate_transition_matrix,
    normalize_transition_matrix,
    compute_matrix_condition_number,
)
from .synthetic import (
    inject_noise_with_transition_matrix,
    generate_synthetic_noisy_labels,
)
from .real_world import load_cifar10n_noise, compute_empirical_transition_matrix

__all__ = [
    "build_symmetric_transition_matrix",
    "build_asymmetric_cifar10_transition_matrix",
    "validate_transition_matrix",
    "normalize_transition_matrix",
    "compute_matrix_condition_number",
    "inject_noise_with_transition_matrix",
    "generate_synthetic_noisy_labels",
    "load_cifar10n_noise",
    "compute_empirical_transition_matrix",
]

