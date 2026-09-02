"""Loss correction modules using noise transition matrices."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union
from .base import BaseClassificationLoss


class BackwardLossCorrection(BaseClassificationLoss):
    """Backward Loss Correction (Natarajan et al., NeurIPS 2013):
    
    ell_backward(f(x), y_tilde) = [T^(-1) * ell_vec(f(x))]_y_tilde
    
    Provably unbiased estimator of clean risk under class-conditional noise.
    """

    def __init__(
        self,
        transition_matrix: Union[np.ndarray, torch.Tensor],
        base_loss: str = "ce",
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)
        if isinstance(transition_matrix, np.ndarray):
            T_np = transition_matrix.astype(np.float32)
        else:
            T_np = transition_matrix.detach().cpu().numpy().astype(np.float32)

        # Invert transition matrix
        T_inv_np = np.linalg.inv(T_np)
        self.register_buffer("T_inv", torch.from_numpy(T_inv_np))
        self.base_loss = base_loss.lower()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        if self.base_loss == "ce":
            log_probs = F.log_softmax(logits, dim=-1)
            # Full loss vector across all classes: ell_k = -log(p_k)
            ell_vec = -log_probs  # (Batch, NumClasses)
        else:
            probs = F.softmax(logits, dim=-1)
            ell_vec = 2.0 - 2.0 * probs

        # Corrected loss matrix: ell_corr_ij = sum_k [T^-1]_ik * ell_kj
        # ell_corrected = ell_vec @ T_inv^T -> (Batch, NumClasses)
        ell_corrected = torch.matmul(ell_vec, self.T_inv.t())

        # Select corrected loss corresponding to observed noisy target
        loss_per_sample = ell_corrected.gather(1, targets.unsqueeze(1)).squeeze(1)
        return self.apply_reduction(loss_per_sample)


class ForwardLossCorrection(BaseClassificationLoss):
    """Forward Loss Correction (Patrini et al., CVPR 2017):
    
    ell_forward(f(x), y_tilde) = -log([T^T * f(x)]_y_tilde)
    
    Transforms the predicted probability vector by the transition matrix transpose.
    """

    def __init__(
        self,
        transition_matrix: Union[np.ndarray, torch.Tensor],
        eps: float = 1e-7,
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)
        if isinstance(transition_matrix, np.ndarray):
            T_tensor = torch.from_numpy(transition_matrix.astype(np.float32))
        else:
            T_tensor = transition_matrix.detach().clone().float()

        self.register_buffer("T", T_tensor)
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)  # (Batch, NumClasses)
        # Corrupted model posterior: p_tilde = probs @ T
        # (Batch, K) @ (K, K) -> (Batch, K)
        p_corrupted = torch.matmul(probs, self.T)
        p_corrupted = torch.clamp(p_corrupted, min=self.eps, max=1.0)
        
        log_p_corrupted = torch.log(p_corrupted)
        loss_per_sample = -log_p_corrupted.gather(1, targets.unsqueeze(1)).squeeze(1)
        return self.apply_reduction(loss_per_sample)
