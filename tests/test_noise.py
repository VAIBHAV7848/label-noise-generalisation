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


if __name__ == "__main__":
    unittest.main()
