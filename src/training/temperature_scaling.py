"""Post-hoc Temperature Scaling for probability calibration (Guo et al., ICML 2017)."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple


class ModelWithTemperature(nn.Module):
    """Decorator model that scales logits by a learnable temperature parameter T > 0:
    
    p_k = exp(z_k / T) / sum_j exp(z_j / T)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_tensor)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature parameter."""
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(
        self,
        val_logits: torch.Tensor,
        val_targets: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """Tune temperature parameter on validation set via NLL minimization.
        
        Args:
            val_logits: (N, K) raw unscaled validation logits.
            val_targets: (N,) validation class labels.
            lr: Learning rate for L-BFGS optimizer.
            max_iter: Max optimization iterations.
            
        Returns:
            Optimal temperature value as float.
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(val_logits), val_targets)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return float(self.temperature.item())
