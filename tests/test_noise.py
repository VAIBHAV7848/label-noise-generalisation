"""Unit tests for noise transition matrix generation and synthetic noise injection."""

import unittest
import numpy as np
from src.noise.matrix_utils import (
    build_symmetric_transition_matrix,
    build_asymmetric_cifar10_transition_matrix,
    validate_transition_matrix,
    normalize_transition_matrix,
    compute_matrix_condition_number,
)
from src.noise.synthetic import (
    inject_noise_with_transition_matrix,
    generate_synthetic_noisy_labels,
)


class TestNoise(unittest.TestCase):
    def test_symmetric_transition_matrix(self):
        num_classes = 10
        noise_rate = 0.4
        T = build_symmetric_transition_matrix(num_classes, noise_rate)
        
        self.assertEqual(T.shape, (10, 10))
        self.assertTrue(validate_transition_matrix(T))
        self.assertTrue(np.allclose(np.diag(T), 1.0 - noise_rate))
        
        off_diag_val = noise_rate / (num_classes - 1)
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:
                    self.assertTrue(np.isclose(T[i, j], off_diag_val))

    def test_asymmetric_cifar10_transition_matrix(self):
        noise_rate = 0.3
        T = build_asymmetric_cifar10_transition_matrix(noise_rate)
        
        self.assertEqual(T.shape, (10, 10))
        self.assertTrue(validate_transition_matrix(T))
        self.assertTrue(np.isclose(T[9, 1], noise_rate))
        self.assertTrue(np.isclose(T[9, 9], 1.0 - noise_rate))
        self.assertTrue(np.isclose(T[2, 0], noise_rate))
        self.assertTrue(np.isclose(T[4, 7], noise_rate))

    def test_matrix_condition_number(self):
        T_clean = np.eye(5)
        self.assertTrue(np.isclose(compute_matrix_condition_number(T_clean), 1.0))
        
        T_noisy = build_symmetric_transition_matrix(5, 0.4)
        kappa = compute_matrix_condition_number(T_noisy)
        self.assertGreater(kappa, 1.0)

    def test_synthetic_noise_injection(self):
        clean_labels = np.array([0, 1, 2, 3, 4] * 2000)
        num_classes = 5
        noise_rate = 0.3
        
        noisy_labels, T, mask, emp_rate = generate_synthetic_noisy_labels(
            clean_labels=clean_labels,
            num_classes=num_classes,
            noise_type="symmetric",
            noise_rate=noise_rate,
            seed=42,
        )
        
        self.assertEqual(len(noisy_labels), len(clean_labels))
        self.assertTrue(np.isclose(emp_rate, noise_rate, atol=0.02))


    def test_dense_symmetric_vs_sparse_asymmetric_min_entries(self):
        """Mechanically verify that symmetric T has T_min > 0 while asymmetric pair-flip T has T_min == 0."""
        # Symmetric dense matrix
        for eta in [0.2, 0.5]:
            T_sym = build_symmetric_transition_matrix(10, eta)
            expected_min = eta / 9.0
            self.assertTrue(np.isclose(np.min(T_sym), expected_min), f"Symmetric T_min should be {expected_min}")
            self.assertGreater(np.min(T_sym), 0.0)

        # Asymmetric sparse pair-flip matrix
        T_asym = build_asymmetric_cifar10_transition_matrix(0.4)
        self.assertEqual(np.min(T_asym), 0.0, "Asymmetric pair-flip matrix MUST have T_min == 0.0")
        zero_count = np.sum(T_asym == 0.0)
        self.assertEqual(zero_count, 85, "CIFAR-10 pair-flip matrix must contain exactly 85 zero entries")


if __name__ == "__main__":
    unittest.main()
