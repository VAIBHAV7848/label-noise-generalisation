"""Calibration and reliability metrics: ECE, Adaptive ECE, and Brier Score."""

import numpy as np
import torch
from typing import Dict, Tuple, Union, Any


def compute_ece(
    probs: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    num_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE) (Guo et al., ICML 2017) with equal-width bins.
    
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    
    Args:
        probs: (N, K) predicted softmax probability distribution.
        targets: (N,) true integer ground-truth class labels.
        num_bins: Number of equal-width bins in [0, 1] (default 15).
        
    Returns:
        ECE as a float in [0, 1].
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total_samples = len(targets)

    if total_samples == 0:
        return 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            ece += (bin_size / total_samples) * np.abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_adaptive_ece(
    probs: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    num_bins: int = 15,
) -> float:
    """Compute Adaptive Expected Calibration Error (AdaECE) using equal-frequency bins.
    
    Args:
        probs: (N, K) predicted probability distribution.
        targets: (N,) true integer ground-truth class labels.
        num_bins: Number of equal-frequency bins.
        
    Returns:
        AdaECE as a float in [0, 1].
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets).astype(np.float64)
    total_samples = len(targets)

    if total_samples == 0:
        return 0.0

    # Sort by confidence
    sort_idx = np.argsort(confidences)
    sorted_conf = confidences[sort_idx]
    sorted_acc = accuracies[sort_idx]

    # Split into equal-frequency chunks
    bin_chunks = np.array_split(np.arange(total_samples), num_bins)
    ada_ece = 0.0

    for chunk in bin_chunks:
        if len(chunk) > 0:
            avg_confidence = np.mean(sorted_conf[chunk])
            avg_accuracy = np.mean(sorted_acc[chunk])
            ada_ece += (len(chunk) / total_samples) * np.abs(avg_accuracy - avg_confidence)

    return float(ada_ece)


def compute_brier_score(
    probs: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute multi-class Brier Score (Mean Squared Probability Forecast Error):
    
    Brier = (1 / N) * sum_{n=1}^N sum_{k=1}^K (p_{n, k} - I(y_n == k))^2
    
    Args:
        probs: (N, K) predicted probability distribution.
        targets: (N,) true integer ground-truth class labels.
        
    Returns:
        Brier score as a float in [0, 2].
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    num_samples, num_classes = probs.shape
    if num_samples == 0:
        return 0.0

    one_hot = np.zeros((num_samples, num_classes), dtype=np.float64)
    one_hot[np.arange(num_samples), targets] = 1.0

    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return float(brier)


def compute_reliability_diagram_data(
    probs: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    num_bins: int = 15,
) -> Dict[str, Any]:
    """Compute binned accuracies, confidences, and counts for plotting reliability diagrams.
    
    Args:
        probs: (N, K) predicted probabilities.
        targets: (N,) ground-truth labels.
        num_bins: Number of bins.
        
    Returns:
        Dictionary containing bin boundaries, bin accuracies, confidences, and counts.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        count = int(np.sum(in_bin))
        bin_counts.append(count)
        if count > 0:
            bin_accs.append(float(np.mean(accuracies[in_bin])))
            bin_confs.append(float(np.mean(confidences[in_bin])))
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.5 * (bin_lower + bin_upper))

    return {
        "bin_boundaries": bin_boundaries.tolist(),
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
        "ece": compute_ece(probs, targets, num_bins),
        "ada_ece": compute_adaptive_ece(probs, targets, num_bins),
        "brier_score": compute_brier_score(probs, targets),
    }
