"""2-Layer Multi-Layer Perceptron (MLP) in PyTorch."""

import torch
import torch.nn as nn


class TwoLayerMLP(nn.Module):
    """2-Layer Multi-Layer Perceptron with Batch Normalization and Dropout."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 512,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out
