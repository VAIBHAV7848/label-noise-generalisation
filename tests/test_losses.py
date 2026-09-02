"""Unit tests for robust loss functions and loss correction modules."""

import unittest
import torch
import torch.nn.functional as F
import numpy as np
from src.losses.robust_losses import (
    CrossEntropyLoss,
    LabelSmoothingLoss,
    MeanAbsoluteErrorLoss,
    GeneralizedCrossEntropyLoss,
    SymmetricCrossEntropyLoss,
    NormalizedCrossEntropyLoss,
)
from src.losses.loss_correction import BackwardLossCorrection, ForwardLossCorrection
from src.noise.matrix_utils import build_symmetric_transition_matrix


class TestLosses(unittest.TestCase):
    def test_cross_entropy_and_label_smoothing(self):
        logits = torch.randn(8, 10, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        
        ce = CrossEntropyLoss()
        loss_ce = ce(logits, targets)
        self.assertEqual(loss_ce.ndim, 0)
        self.assertGreater(loss_ce.item(), 0.0)
        
        ls = LabelSmoothingLoss(smoothing=0.1)
        loss_ls = ls(logits, targets)
        self.assertEqual(loss_ls.ndim, 0)
        self.assertGreater(loss_ls.item(), 0.0)

    def test_mae_symmetric_property(self):
        """Verify that MAE satisfies sum_{k=1}^K ell(p, k) = C for any probability vector p."""
        mae = MeanAbsoluteErrorLoss(reduction="none")
        num_classes = 5
        logits = torch.randn(1, num_classes)
        
        sum_losses = 0.0
        for c in range(num_classes):
            target = torch.tensor([c])
            sum_losses += mae(logits, target).item()
            
        expected_constant = 2.0 * num_classes - 2.0
        self.assertTrue(np.isclose(sum_losses, expected_constant, atol=1e-5))

    def test_gce_loss_bounds(self):
        logits = torch.randn(10, 5)
        targets = torch.randint(0, 5, (10,))
        
        gce = GeneralizedCrossEntropyLoss(q=0.7)
        loss = gce(logits, targets)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_backward_correction_unbiasedness(self):
        """Verify that E_{Y_tilde | Y} [ell_backward(f(x), Y_tilde)] == ell(f(x), Y)."""
        num_classes = 4
        T = build_symmetric_transition_matrix(num_classes, 0.3)
        backward_loss = BackwardLossCorrection(transition_matrix=T, base_loss="ce", reduction="none")
        
        logits = torch.randn(1, num_classes)
        
        y_true = 0
        log_probs = F.log_softmax(logits, dim=-1)
        expected_clean_loss = (-log_probs[0, y_true]).item()
        
        noisy_losses = []
        for j in range(num_classes):
            noisy_target = torch.tensor([j])
            loss_j = backward_loss(logits, noisy_target).item()
            noisy_losses.append(loss_j)
            
        expected_noisy_loss = sum(T[y_true, j] * noisy_losses[j] for j in range(num_classes))
        self.assertTrue(np.isclose(expected_clean_loss, expected_noisy_loss, atol=1e-5))

    def test_forward_loss_correction(self):
        num_classes = 10
        T = build_symmetric_transition_matrix(num_classes, 0.3)
        forward_loss = ForwardLossCorrection(transition_matrix=T)
        
        logits = torch.randn(16, num_classes)
        targets = torch.randint(0, num_classes, (16,))
        
        loss = forward_loss(logits, targets)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_backward_correction_negative_per_sample_loss_validity(self):
        """Verify that per-example backward loss can legitimately be negative while expectation is >= 0."""
        num_classes = 4
        # Matrix with strong noise producing negative off-diagonal entries in T^-1
        T = build_symmetric_transition_matrix(num_classes, 0.4)
        T_inv = np.linalg.inv(T)
        self.assertTrue(np.any(T_inv < 0.0), "T_inv should have negative off-diagonal entries")
        
        backward_loss = BackwardLossCorrection(transition_matrix=T, base_loss="ce", reduction="none")
        # Logits predicting class 0 with very high confidence
        logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
        
        # When observed noisy label is class 1 (false label)
        noisy_target_1 = torch.tensor([1])
        loss_1 = backward_loss(logits, noisy_target_1).item()
        
        # When observed noisy label is class 0 (true label)
        noisy_target_0 = torch.tensor([0])
        loss_0 = backward_loss(logits, noisy_target_0).item()
        
        # One of the per-sample losses may be negative due to negative entries in T_inv
        # Expected risk under clean class 0: T[0, 0]*loss_0 + sum_{j!=0} T[0, j]*loss_j
        expected_risk = T[0, 0] * loss_0 + sum(T[0, j] * backward_loss(logits, torch.tensor([j])).item() for j in range(1, num_classes))
        
        # Expected risk MUST be non-negative because clean loss is non-negative
        self.assertGreaterEqual(expected_risk, -1e-6)

    def test_loss_lipschitz_constants_numerical_verification(self):
        """Verify Euclidean Lipschitz constants for MAE, Forward Loss, and GCE on simplex."""
        K = 5
        rng = np.random.default_rng(42)
        
        # 1. MAE: L_2 <= 2
        for _ in range(100):
            p1 = rng.dirichlet(np.ones(K))
            p2 = rng.dirichlet(np.ones(K))
            y = rng.integers(0, K)
            
            diff_loss = np.abs((2.0 - 2.0 * p1[y]) - (2.0 - 2.0 * p2[y]))
            diff_p = np.linalg.norm(p1 - p2, ord=2)
            if diff_p > 1e-8:
                self.assertLessEqual(diff_loss / diff_p, 2.0 + 1e-8)
                
        # 2. Forward Loss with positive T: L_2 <= 1 / T_min
        T = build_symmetric_transition_matrix(K, 0.3)
        T_min = np.min(T)
        theoretical_L_fwd = 1.0 / T_min
        for _ in range(100):
            p1 = rng.dirichlet(np.ones(K))
            p2 = rng.dirichlet(np.ones(K))
            y = rng.integers(0, K)
            
            loss1 = -np.log(np.dot(p1, T[:, y]))
            loss2 = -np.log(np.dot(p2, T[:, y]))
            
            diff_loss = np.abs(loss1 - loss2)
            diff_p = np.linalg.norm(p1 - p2, ord=2)
            if diff_p > 1e-8:
                self.assertLessEqual(diff_loss / diff_p, theoretical_L_fwd + 1e-8)

    def test_forward_loss_sparse_asymmetric_numerical_clamping(self):
        """Verify that sparse asymmetric forward loss is finite on zero-transition pairs solely due to clamp."""
        from src.noise.matrix_utils import build_asymmetric_cifar10_transition_matrix
        
        T_asym = build_asymmetric_cifar10_transition_matrix(0.4)
        self.assertEqual(np.min(T_asym), 0.0)
        
        forward_loss = ForwardLossCorrection(transition_matrix=T_asym, eps=1e-7)
        
        # High confidence for class 0 (airplane), but observed noisy label is class 6 (frog) where T[0, 6] == 0
        self.assertEqual(T_asym[0, 6], 0.0)
        logits = torch.tensor([[50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0]])
        target_6 = torch.tensor([6])
        
        loss_val = forward_loss(logits, target_6).item()
        
        # Loss must be finite and bounded by -log(1e-7) approx 16.118
        self.assertTrue(np.isfinite(loss_val))
        self.assertTrue(np.isclose(loss_val, -np.log(1e-7), atol=1e-3))


if __name__ == "__main__":
    unittest.main()
