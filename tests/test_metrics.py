"""Unit tests for classification, calibration, and matrix evaluation metrics."""

import unittest
import numpy as np
import torch
from src.metrics.classification import compute_topk_accuracy
from src.metrics.calibration import (
    compute_ece,
    compute_adaptive_ece,
    compute_brier_score,
    compute_reliability_diagram_data,
)
from src.metrics.matrix_metrics import (
    compute_frobenius_error,
    compute_spectral_error,
    compute_total_variation_error,
)


class TestMetrics(unittest.TestCase):
    def test_topk_accuracy(self):
        probs = torch.tensor([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.3, 0.3, 0.4],
            [0.9, 0.05, 0.05],
        ])
        targets = torch.tensor([0, 1, 2, 1])
        
        top1, = compute_topk_accuracy(probs, targets, topk=(1,))
        self.assertTrue(np.isclose(top1, 75.0))

    def test_ece_perfect_vs_miscalibrated(self):
        perfect_probs = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        targets = np.array([0, 0, 1, 1])
        
        ece_perfect = compute_ece(perfect_probs, targets, num_bins=10)
        self.assertTrue(np.isclose(ece_perfect, 0.0, atol=1e-5))
        
        wrong_probs = np.array([
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ])
        ece_wrong = compute_ece(wrong_probs, targets, num_bins=10)
        self.assertGreater(ece_wrong, 0.9)

    def test_brier_score(self):
        perfect_probs = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        targets = np.array([0, 1])
        brier = compute_brier_score(perfect_probs, targets)
        self.assertTrue(np.isclose(brier, 0.0))

    def test_matrix_metrics(self):
        T1 = np.eye(3)
        T2 = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        frob = compute_frobenius_error(T1, T2)
        self.assertGreater(frob, 0.0)
        tv = compute_total_variation_error(T1, T2)
        self.assertTrue(0.0 < tv < 1.0)


if __name__ == "__main__":
    unittest.main()
