"""Unit tests for noise transition matrix estimators."""

import unittest
import numpy as np
from src.estimators.anchor_point import estimate_transition_matrix_anchor_points
from src.estimators.dual_t import estimate_transition_matrix_dual_t
from src.estimators.confident_learning import estimate_transition_matrix_confident_learning
from src.noise.matrix_utils import build_symmetric_transition_matrix, validate_transition_matrix


class TestEstimators(unittest.TestCase):
    def test_anchor_point_estimator_recovery(self):
        num_classes = 3
        T_true = build_symmetric_transition_matrix(num_classes, 0.2)
        
        num_samples_per_class = 500
        probs_list = []
        noisy_labels_list = []
        
        rng = np.random.default_rng(42)
        for c in range(num_classes):
            class_probs = np.tile(T_true[c], (num_samples_per_class, 1))
            class_probs += rng.normal(0, 0.01, class_probs.shape)
            class_probs = np.clip(class_probs, 1e-4, 1.0)
            class_probs /= class_probs.sum(axis=1, keepdims=True)
            
            noisy_labels = rng.choice(num_classes, size=num_samples_per_class, p=T_true[c])
            probs_list.append(class_probs)
            noisy_labels_list.append(noisy_labels)
            
        all_probs = np.vstack(probs_list)
        all_noisy_labels = np.concatenate(noisy_labels_list)
        
        T_hat = estimate_transition_matrix_anchor_points(
            probs=all_probs,
            noisy_labels=all_noisy_labels,
            num_classes=num_classes,
            percentile=95.0,
        )
        
        self.assertTrue(validate_transition_matrix(T_hat))
        frobenius_error = np.linalg.norm(T_hat - T_true, ord="fro")
        self.assertLess(frobenius_error, 0.15)

    def test_confident_learning_estimator(self):
        num_classes = 3
        probs = np.array([
            [0.9, 0.05, 0.05],
            [0.85, 0.1, 0.05],
            [0.1, 0.85, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.05, 0.1, 0.85],
        ])
        noisy_labels = np.array([0, 0, 1, 1, 2, 2])
        
        T_hat, joint_Q = estimate_transition_matrix_confident_learning(
            probs=probs,
            noisy_labels=noisy_labels,
            num_classes=num_classes,
        )
        
        self.assertTrue(validate_transition_matrix(T_hat))
        self.assertEqual(joint_Q.shape, (num_classes, num_classes))
        self.assertTrue(np.isclose(np.sum(joint_Q), 1.0, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
