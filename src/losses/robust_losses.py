"""Implementations of robust and symmetric multi-class loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseClassificationLoss


class CrossEntropyLoss(BaseClassificationLoss):
    """Standard Categorical Cross-Entropy Loss."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(logits, targets, reduction=self.reduction)
        return loss


class LabelSmoothingLoss(BaseClassificationLoss):
    """Cross-Entropy with uniform label smoothing regularization."""

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"Smoothing factor must be in [0, 1), got {smoothing}")
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        num_classes = logits.size(-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1) if num_classes > 1 else 0.0)
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss_per_sample = -(true_dist * log_probs).sum(dim=-1)
        return self.apply_reduction(loss_per_sample)


class MeanAbsoluteErrorLoss(BaseClassificationLoss):
    """Mean Absolute Error (MAE / L1 loss) for multi-class classification:
    
    ell_MAE(p, y) = ||p - e_y||_1 = 2 - 2 * p_y
    
    Theoretically noise-tolerant to symmetric label noise (Charoenphakdee et al. 2019).
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        p_target = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss_per_sample = 2.0 - 2.0 * p_target
        return self.apply_reduction(loss_per_sample)


class GeneralizedCrossEntropyLoss(BaseClassificationLoss):
    """Generalized Cross Entropy (GCE / L_q loss) (Zhang & Sabuncu, NeurIPS 2018):
    
    L_q(p, y) = (1 - p_y^q) / q,   for q in (0, 1]
    
    Interpolates between Cross-Entropy (q -> 0) and MAE (q = 1).
    """

    def __init__(self, q: float = 0.7, eps: float = 1e-7, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        if not (0.0 < q <= 1.0):
            raise ValueError(f"Parameter q must be in (0, 1], got {q}")
        self.q = q
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        p_target = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_target = torch.clamp(p_target, min=self.eps, max=1.0)
        loss_per_sample = (1.0 - torch.pow(p_target, self.q)) / self.q
        return self.apply_reduction(loss_per_sample)


class SymmetricCrossEntropyLoss(BaseClassificationLoss):
    """Symmetric Cross Entropy (SCE) (Wang et al., ICCV 2019):
    
    SCE = alpha * CE(p, y) + beta * RCE(p, y)
    where RCE(p, y) = - sum_{k=1}^K p_k log(e_{y, k})
    With numerical stability clipping: RCE = - sum_{k=1}^K p_k log(clamp(e_{y,k}, eps, 1))
    """

    def __init__(self, alpha: float = 0.1, beta: float = 1.0, eps: float = 1e-7, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Active cross entropy: -log(p_y)
        ce_loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Passive reverse cross entropy: - \sum_k p_k log(y_k)
        # One-hot target vector
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        one_hot = torch.clamp(one_hot, min=self.eps, max=1.0)

        rce_loss = -(probs * torch.log(one_hot)).sum(dim=-1)

        loss_per_sample = self.alpha * ce_loss + self.beta * rce_loss
        return self.apply_reduction(loss_per_sample)


class NormalizedCrossEntropyLoss(BaseClassificationLoss):
    r"""Normalized Cross Entropy (NCE) (Ma et al., ICML 2020):
    
    NCE = -log(p_y) / \sum_{k=1}^K (-log(p_k))
    
    Provably symmetric and robust to symmetric label noise.
    """

    def __init__(self, eps: float = 1e-7, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        ce_per_class = -log_probs  # (Batch, NumClasses)
        ce_target = ce_per_class.gather(1, targets.unsqueeze(1)).squeeze(1)
        sum_ce = ce_per_class.sum(dim=-1)
        loss_per_sample = ce_target / torch.clamp(sum_ce, min=self.eps)
        return self.apply_reduction(loss_per_sample)
