"""Base abstract class for classification loss functions."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseClassificationLoss(nn.Module, ABC):
    """Abstract base class for all standard, robust, and corrected classification losses."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction mode: {reduction}. Choose from 'mean', 'sum', 'none'")
        self.reduction = reduction

    @abstractmethod
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss between model logits and integer class targets.
        
        Args:
            logits: (Batch, NumClasses) unnormalized model logits.
            targets: (Batch,) integer class labels in {0, ..., NumClasses-1}.
            
        Returns:
            Scalar tensor if reduction in {'mean', 'sum'} else (Batch,) tensor.
        """
        pass

    def apply_reduction(self, loss_per_sample: torch.Tensor) -> torch.Tensor:
        """Apply configured reduction operation on 1D per-sample loss tensor."""
        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        return loss_per_sample
