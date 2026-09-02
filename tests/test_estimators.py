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
            global_search=True,
        )
        
        self.assertTrue(validate_transition_matrix(T_hat))
        frobenius_error = np.linalg.norm(T_hat - T_true, ord="fro")
        self.assertLess(frobenius_error, 0.15)

    def test_anchor_point_global_search_under_corrupted_anchors(self):
        """Adversarial test: True anchor points for class 0 received noisy label 1.
        Global search must successfully recover row 0 from p_0(x) regardless of noisy_labels."""
        num_classes = 2
        T_true = np.array([[0.8, 0.2], [0.3, 0.7]])
        
        # 100 samples of class 0 anchor points, but corrupted to label 1
        anchor_probs = np.tile(np.array([0.8, 0.2]), (100, 1))
        anchor_noisy_labels = np.ones(100, dtype=np.int64)  # Corrupted to class 1
        
        # 100 samples of class 1 anchor points, with label 1
        class1_probs = np.tile(np.array([0.3, 0.7]), (100, 1))
        class1_noisy_labels = np.ones(100, dtype=np.int64)
        
        all_probs = np.vstack([anchor_probs, class1_probs])
        all_noisy_labels = np.concatenate([anchor_noisy_labels, class1_noisy_labels])
        
        T_hat = estimate_transition_matrix_anchor_points(
            probs=all_probs,
            noisy_labels=all_noisy_labels,
            num_classes=num_classes,
            percentile=90.0,
            global_search=True,
        )
        
        self.assertTrue(validate_transition_matrix(T_hat))
        self.assertTrue(np.allclose(T_hat[0], [0.8, 0.2], atol=1e-3))

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
