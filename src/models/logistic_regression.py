"""Multi-Class Logistic Regression (Softmax Linear Classifier) in PyTorch."""

import torch
import torch.nn as nn


class MultiClassLogisticRegression(nn.Module):
    """Linear Softmax Classifier for multi-class classification."""

    def __init__(self, in_features: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features, num_classes, bias=bias)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten if 2D/3D image tensor
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)
