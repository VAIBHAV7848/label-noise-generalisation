"""General PyTorch training loop with comprehensive metric logging and calibration evaluation."""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
from ..metrics.classification import compute_topk_accuracy
from ..metrics.calibration import compute_ece, compute_brier_score


class ModelTrainer:
    """Trainer executing robust empirical risk minimization with per-epoch metric logging."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train model for one epoch on noisy data.
        
        Returns:
            Dictionary with training loss and training accuracy.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for batch_data in train_loader:
            images, noisy_targets, clean_targets, _ = batch_data
            images = images.to(self.device)
            noisy_targets = noisy_targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, noisy_targets)
            loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=-1)
            correct += (preds == noisy_targets).sum().item()
            total_samples += batch_size

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "train_loss": total_loss / total_samples,
            "train_noisy_acc": (correct / total_samples) * 100.0,
        }

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance on a dataset (val or test).
        
        Computes Top-1 accuracy, Cross-Entropy loss, 15-bin ECE, and Brier Score.
        """
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_targets = []

        for batch_data in eval_loader:
            images, noisy_targets, clean_targets, _ = batch_data
            images = images.to(self.device)
            # When evaluating clean test or val set, evaluate against clean_targets
            targets = clean_targets.to(self.device)

            logits = self.model(images)
            loss = F.cross_entropy(logits, targets)

            probs = F.softmax(logits, dim=-1)
            total_loss += loss.item() * images.size(0)
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

        all_probs = torch.cat(all_probs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        top1, = compute_topk_accuracy(all_probs, all_targets, topk=(1,))
        ece = compute_ece(all_probs, all_targets, num_bins=15)
        brier = compute_brier_score(all_probs, all_targets)

        return {
            "eval_loss": total_loss / len(all_targets),
            "top1_accuracy": top1,
            "ece": ece,
            "brier_score": brier,
        }

    @torch.no_grad()
    def extract_logits_and_targets(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract all logits and clean targets for post-hoc calibration."""
        self.model.eval()
        all_logits = []
        all_targets = []
        for batch_data in loader:
            images, _, clean_targets, _ = batch_data
            images = images.to(self.device)
            logits = self.model(images)
            all_logits.append(logits.cpu())
            all_targets.append(clean_targets)
        return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)
