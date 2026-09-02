"""Standard classification performance metrics."""

import torch
import numpy as np
from typing import Tuple, Union


def compute_topk_accuracy(
    logits_or_probs: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    topk: Tuple[int, ...] = (1,),
) -> Tuple[float, ...]:
    """Compute Top-k classification accuracies.
    
    Args:
        logits_or_probs: (N, K) tensor or array.
        targets: (N,) ground-truth class labels.
        topk: Tuple of k values (e.g. (1, 5)).
        
    Returns:
        Tuple of accuracy percentages (0.0 to 100.0) for each specified k.
    """
    if isinstance(logits_or_probs, np.ndarray):
        logits_or_probs = torch.from_numpy(logits_or_probs)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        if batch_size == 0:
            return tuple(0.0 for _ in topk)

        _, pred = logits_or_probs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(float(correct_k.item() * (100.0 / batch_size)))
        return tuple(res)
