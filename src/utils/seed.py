"""Global utility functions for seeds, configuration, and logging."""

import os
import random
import logging
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested_device: Optional[str] = None) -> torch.device:
    """Determine compute device (CUDA, MPS, or CPU).
    
    Args:
        requested_device: Explicit device string if provided ('cuda', 'cpu', etc.).
        
    Returns:
        torch.device instance.
    """
    if requested_device is not None:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
