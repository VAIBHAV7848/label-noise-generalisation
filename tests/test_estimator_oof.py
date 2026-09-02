"""Unit and adversarial tests for Out-Of-Fold (OOF) cross-validation and estimator pipelines."""

import unittest
import numpy as np
import torch
import torch.nn as nn
from src.estimators.oof import generate_deterministic_folds, compute_oof_predicted_probabilities
from src.estimators.confident_learning import estimate_transition_matrix_confident_learning, compute_class_thresholds
from src.estimators.anchor_point import estimate_transition_matrix_anchor_points


class TestEstimatorOOF(unittest.TestCase):
    def test_deterministic_k_fold_partition(self):
        """Verify that K-fold partitions are strictly disjoint, exhaustive, and reproducible."""
        num_samples = 35000
        n_splits = 3
        seed = 42

        folds_1 = generate_deterministic_folds(num_samples, n_splits=n_splits, seed=seed)
        folds_2 = generate_deterministic_folds(num_samples, n_splits=n_splits, seed=seed)

        self.assertEqual(len(folds_1), n_splits)

        all_val_indices = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds_1):
            # Determinism check
            np.testing.assert_array_equal(train_idx, folds_2[fold_idx][0])
            np.testing.assert_array_equal(val_idx, folds_2[fold_idx][1])

            # Disjointness within fold
            self.assertEqual(len(set(train_idx).intersection(set(val_idx))), 0, f"Fold {fold_idx} has train-val overlap!")

            # Exhaustiveness of fold partition
            self.assertEqual(len(train_idx) + len(val_idx), num_samples)
            all_val_indices.extend(val_idx)

        # Exhaustiveness across all validation folds
        self.assertEqual(len(all_val_indices), num_samples)
        self.assertEqual(len(set(all_val_indices)), num_samples, "Validation folds have overlapping sample indices!")

    def test_oof_predicted_probabilities_synthetic(self):
        """Verify OOF probability generation using a lightweight model on synthetic data."""
        num_samples = 150
        num_classes = 3

        rng = np.random.default_rng(42)
        # Synthetic image-like array matching CIFAR-10 (N, 32, 32, 3)
        synthetic_data = rng.integers(0, 256, size=(num_samples, 32, 32, 3), dtype=np.uint8)
        noisy_labels = rng.integers(0, num_classes, size=num_samples)

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3072, num_classes)

            def forward(self, x):
                x = x.view(x.size(0), -1).float()
                return self.fc(x)

        oof_probs = compute_oof_predicted_probabilities(
            data=synthetic_data,
            noisy_labels=noisy_labels,
            num_classes=num_classes,
            model_fn=TinyModel,
            n_splits=3,
            epochs=2,
            batch_size=32,
            lr=0.01,
            seed=42,
            device=torch.device("cpu"),
        )

        self.assertEqual(oof_probs.shape, (num_samples, num_classes))
        self.assertTrue(np.all(oof_probs >= 0.0))
        self.assertTrue(np.all(oof_probs <= 1.0))
        # Check row stochasticity
        np.testing.assert_allclose(oof_probs.sum(axis=1), np.ones(num_samples), atol=1e-5)

    def test_confident_learning_with_oof_probabilities(self):
        """Verify that Confident Learning correctly processes OOF probabilities."""
        num_samples = 300
        num_classes = 4

        rng = np.random.default_rng(1337)
        noisy_labels = rng.integers(0, num_classes, size=num_samples)
        raw_probs = rng.uniform(0.1, 1.0, size=(num_samples, num_classes))
        oof_probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)

        thresholds = compute_class_thresholds(oof_probs, noisy_labels, num_classes)
        self.assertEqual(thresholds.shape, (num_classes,))
        self.assertTrue(np.all(thresholds >= 0.0) and np.all(thresholds <= 1.0))

        T_hat, joint_Q = estimate_transition_matrix_confident_learning(oof_probs, noisy_labels, num_classes)

        self.assertEqual(T_hat.shape, (num_classes, num_classes))
        self.assertEqual(joint_Q.shape, (num_classes, num_classes))
        self.assertTrue(np.all(T_hat >= 0.0))
        # Row stochasticity: sum over noisy classes (columns) equals 1.0 for each true class (row)
        np.testing.assert_allclose(T_hat.sum(axis=1), np.ones(num_classes), atol=1e-5)
        # Joint distribution sums to 1.0
        np.testing.assert_allclose(joint_Q.sum(), 1.0, atol=1e-5)

    def test_anchor_point_estimator_properties(self):
        """Verify that Anchor Point estimator satisfies Patrini et al. (2017) specifications."""
        num_samples = 200
        num_classes = 3

        rng = np.random.default_rng(2024)
        noisy_labels = rng.integers(0, num_classes, size=num_samples)
        probs = rng.dirichlet(np.ones(num_classes), size=num_samples)

        T_hat_global = estimate_transition_matrix_anchor_points(
            probs, noisy_labels, num_classes, percentile=95.0, global_search=True
        )
        self.assertEqual(T_hat_global.shape, (num_classes, num_classes))
        np.testing.assert_allclose(T_hat_global.sum(axis=1), np.ones(num_classes), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
