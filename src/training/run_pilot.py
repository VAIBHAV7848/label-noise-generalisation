"""Phase 2 Pre-Registered Pilot Runner for Label Noise & Calibration Benchmarking.

Executes the registered 84-run diagnostic grid across CIFAR-10 with structured JSON provenance.
Supports single run selection, batch chunking, resumability, and manifest audit validation for Kaggle / cluster execution.
"""

import os
import sys
import json
import time
import argparse
import datetime
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets

from ..utils.seed import set_seed
from ..noise.synthetic import generate_synthetic_noisy_labels
from ..noise.matrix_utils import (
    build_symmetric_transition_matrix,
    build_asymmetric_cifar10_transition_matrix,
    compute_matrix_condition_number,
)
from ..losses.robust_losses import GeneralizedCrossEntropyLoss, SymmetricCrossEntropyLoss
from ..losses.loss_correction import ForwardLossCorrection
from ..estimators.anchor_point import estimate_transition_matrix_anchor_points
from ..estimators.confident_learning import estimate_transition_matrix_confident_learning
from ..estimators.oof import compute_oof_predicted_probabilities
from ..models.resnet import PreActResNet18
from ..models.mlp import TwoLayerMLP
from ..data.transforms import get_cifar_transforms
from ..metrics.classification import compute_topk_accuracy
from ..metrics.calibration import compute_ece, compute_adaptive_ece, compute_brier_score
from ..training.temperature_scaling import ModelWithTemperature

from PIL import Image


class IndexedNoisyDataset(Dataset):
    """Dataset with explicit noisy, clean labels and sample indices."""

    def __init__(self, data: np.ndarray, noisy_labels: np.ndarray, clean_labels: np.ndarray, transform=None):
        self.data = data
        self.noisy_labels = np.asarray(noisy_labels, dtype=np.int64)
        self.clean_labels = np.asarray(clean_labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img = self.data[idx]
        if self.transform is not None:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = self.transform(img)
        return img, int(self.noisy_labels[idx]), int(self.clean_labels[idx]), idx


def get_git_commit_sha() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def validate_provenance_record(record: dict) -> bool:
    """Validate that a provenance record is complete, valid, and uncorrupted."""
    if not isinstance(record, dict):
        return False
    if record.get("status") != "COMPLETED":
        return False

    required_keys = ["provenance", "configuration", "metrics", "history"]
    if not all(k in record for k in required_keys):
        return False

    prov = record["provenance"]
    if not all(k in prov for k in ["experiment_id", "timestamp_utc", "git_commit_sha"]):
        return False

    metrics = record["metrics"]
    required_metrics = [
        "test_top1_acc",
        "raw_test_ece",
        "raw_ada_ece",
        "raw_brier",
        "ts_clean_test_ece",
        "ts_corrupted_test_ece",
        "val_calibration_delta",
    ]
    for m in required_metrics:
        val = metrics.get(m)
        if val is None or not np.isfinite(val):
            return False

    return True


def check_run_already_completed(output_dir: str, experiment_id: str) -> bool:
    """Check if valid, non-corrupted results already exist for an experiment ID."""
    result_file = os.path.join(output_dir, f"{experiment_id}.json")
    if not os.path.exists(result_file):
        return False
    try:
        with open(result_file, "r") as f:
            data = json.load(f)
        return validate_provenance_record(data)
    except Exception:
        return False


def build_pilot_splits(
    cifar_train: datasets.CIFAR10,
    cifar_test: datasets.CIFAR10,
    noise_regime: str,
    seed: int,
    batch_size: int = 128,
    num_workers: int = 2,
):
    """Build train (35k noisy), clean val (5k), corrupted val (5k), and test (10k) loaders."""
    num_classes = 10
    raw_data = cifar_train.data  # (50000, 32, 32, 3)
    raw_targets = np.array(cifar_train.targets, dtype=np.int64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(raw_targets))

    train_idx = perm[:35000]
    val_clean_idx = perm[35000:40000]
    val_corrupted_idx = perm[40000:45000]
    # Remaining 5000 unused in pilot to maintain exact 35k/5k/5k balance

    train_clean_targets = raw_targets[train_idx]
    val_clean_targets = raw_targets[val_clean_idx]
    val_corrupted_clean_targets = raw_targets[val_corrupted_idx]

    # Generate transition matrix T and noisy labels
    if noise_regime == "clean":
        T = np.eye(num_classes, dtype=np.float64)
        train_noisy_targets = train_clean_targets.copy()
        val_corrupted_noisy_targets = val_corrupted_clean_targets.copy()
    elif noise_regime == "symmetric_0.2":
        T = build_symmetric_transition_matrix(num_classes, 0.2)
        train_noisy_targets, _, _, _ = generate_synthetic_noisy_labels(
            train_clean_targets, num_classes, "symmetric", 0.2, seed=seed
        )
        val_corrupted_noisy_targets, _, _, _ = generate_synthetic_noisy_labels(
            val_corrupted_clean_targets, num_classes, "symmetric", 0.2, seed=seed + 1000
        )
    elif noise_regime == "symmetric_0.5":
        T = build_symmetric_transition_matrix(num_classes, 0.5)
        train_noisy_targets, _, _, _ = generate_synthetic_noisy_labels(
            train_clean_targets, num_classes, "symmetric", 0.5, seed=seed
        )
        val_corrupted_noisy_targets, _, _, _ = generate_synthetic_noisy_labels(
            val_corrupted_clean_targets, num_classes, "symmetric", 0.5, seed=seed + 1000
        )
    elif noise_regime == "asymmetric_0.4":
        T = build_asymmetric_cifar10_transition_matrix(0.4)
        train_noisy_targets, _, _, _ = generate_synthetic_noisy_labels(
            train_clean_targets, num_classes, "asymmetric", 0.4, seed=seed
        )
        val_corrupted_noisy_targets, _, _, _ = generate_synthetic_noisy_labels(
            val_corrupted_clean_targets, num_classes, "asymmetric", 0.4, seed=seed + 1000
        )
    else:
        raise ValueError(f"Unknown noise regime: {noise_regime}")

    # Datasets and Loaders
    train_transform, test_transform = get_cifar_transforms()

    train_ds = IndexedNoisyDataset(
        raw_data[train_idx], train_noisy_targets, train_clean_targets, transform=train_transform
    )
    val_clean_ds = IndexedNoisyDataset(
        raw_data[val_clean_idx], val_clean_targets, val_clean_targets, transform=test_transform
    )
    val_corrupted_ds = IndexedNoisyDataset(
        raw_data[val_corrupted_idx], val_corrupted_noisy_targets, val_corrupted_clean_targets, transform=test_transform
    )
    test_ds = IndexedNoisyDataset(
        cifar_test.data, np.array(cifar_test.targets), np.array(cifar_test.targets), transform=test_transform
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_clean_loader = DataLoader(val_clean_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_corrupted_loader = DataLoader(val_corrupted_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_clean_loader, val_corrupted_loader, test_loader, T, train_noisy_targets


def evaluate_model_full(model: nn.Module, loader: DataLoader, device: torch.device):
    """Evaluate model and return probs, logits, targets, top1, ece, ada_ece, brier."""
    model.eval()
    all_logits = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, _, clean_targets, _ in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=-1)
            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_targets.append(clean_targets)

    all_logits = torch.cat(all_logits, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    top1, top5 = compute_topk_accuracy(all_probs, all_targets, topk=(1, 5))
    ece = compute_ece(all_probs, all_targets, num_bins=15)
    ada_ece = compute_adaptive_ece(all_probs, all_targets, num_bins=15)
    brier = compute_brier_score(all_probs, all_targets)

    return {
        "logits": all_logits,
        "probs": all_probs,
        "targets": all_targets,
        "top1": float(top1),
        "top5": float(top5),
        "ece": float(ece),
        "ada_ece": float(ada_ece),
        "brier": float(brier),
    }


def run_single_experiment(
    experiment_id: str,
    cifar_train: datasets.CIFAR10,
    cifar_test: datasets.CIFAR10,
    noise_regime: str,
    model_name: str,
    track_name: str,
    seed: int,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 0.05,
    device_str: str = "cuda",
    force: bool = False,
) -> dict:
    """Execute a single pilot training run and log structured JSON provenance.
    
    Skips execution if valid completed results already exist and force is False.
    """
    if not force and check_run_already_completed(output_dir, experiment_id):
        result_file = os.path.join(output_dir, f"{experiment_id}.json")
        with open(result_file, "r") as f:
            return json.load(f)

    set_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    try:
        # Build data splits
        train_loader, val_clean_loader, val_corrupted_loader, test_loader, true_T, train_noisy_labels = build_pilot_splits(
            cifar_train=cifar_train,
            cifar_test=cifar_test,
            noise_regime=noise_regime,
            seed=seed,
            batch_size=batch_size,
        )

        # Instantiate model (PreAct-ResNet18 for official 84-run pilot)
        if model_name == "PreActResNet18":
            model = PreActResNet18(num_classes=10)
        elif model_name == "TwoLayerMLP":
            model = TwoLayerMLP(input_dim=3072, hidden_dim=512, num_classes=10)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model = model.to(device)

        # Determine transition matrix and loss criterion
        T_hat = true_T.copy()
        frobenius_error = 0.0

        if track_name == "CE":
            criterion = nn.CrossEntropyLoss()
        elif track_name == "GCE":
            criterion = GeneralizedCrossEntropyLoss(q=0.7)
        elif track_name == "SCE":
            criterion = SymmetricCrossEntropyLoss(alpha=0.1, beta=1.0)
        elif track_name == "ForwardCorrection_TrueT":
            T_hat = true_T.copy()
            criterion = ForwardLossCorrection(transition_matrix=T_hat)
            frobenius_error = 0.0
        elif track_name == "ForwardCorrection_AnchorT":
            # Estimate T via anchor points using a quick 5-epoch warm-up model
            warmup_model = PreActResNet18(num_classes=10).to(device) if model_name == "PreActResNet18" else TwoLayerMLP(3072, 512, 10).to(device)
            warmup_opt = torch.optim.SGD(warmup_model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
            warmup_model.train()
            for _ in range(5):
                for imgs, targets, _, _ in train_loader:
                    imgs, targets = imgs.to(device), targets.to(device)
                    warmup_opt.zero_grad()
                    F.cross_entropy(warmup_model(imgs), targets).backward()
                    warmup_opt.step()
            warmup_model.eval()
            all_train_probs = []
            with torch.no_grad():
                for imgs, _, _, _ in DataLoader(train_loader.dataset, batch_size=256, shuffle=False):
                    all_train_probs.append(F.softmax(warmup_model(imgs.to(device)), dim=-1).cpu())
            all_train_probs = torch.cat(all_train_probs, dim=0).numpy()
            T_hat = estimate_transition_matrix_anchor_points(all_train_probs, train_noisy_labels, 10, percentile=97.0, global_search=True)
            frobenius_error = float(np.linalg.norm(T_hat - true_T, ord="fro"))
            criterion = ForwardLossCorrection(transition_matrix=T_hat)
        elif track_name == "ForwardCorrection_ConfidentLearningT":
            # Estimate T via Confident Learning using true out-of-fold (OOF) cross-validation (Northcutt et al., 2021)
            model_factory = (lambda: PreActResNet18(num_classes=10)) if model_name == "PreActResNet18" else (lambda: TwoLayerMLP(3072, 512, 10))
            raw_train_data = train_loader.dataset.data
            oof_train_probs = compute_oof_predicted_probabilities(
                data=raw_train_data,
                noisy_labels=train_noisy_labels,
                num_classes=10,
                model_fn=model_factory,
                n_splits=3,
                epochs=5,
                batch_size=batch_size,
                lr=0.05,
                seed=seed,
                device=device,
            )
            T_hat, _ = estimate_transition_matrix_confident_learning(oof_train_probs, train_noisy_labels, 10)
            frobenius_error = float(np.linalg.norm(T_hat - true_T, ord="fro"))
            criterion = ForwardLossCorrection(transition_matrix=T_hat)
        elif track_name == "ForwardCorrection_BadT":
            # Deliberately perturbed matrix T_bad = 0.5 T + 0.5 Uniform
            U = np.ones((10, 10), dtype=np.float64) / 10.0
            T_hat = 0.5 * true_T + 0.5 * U
            frobenius_error = float(np.linalg.norm(T_hat - true_T, ord="fro"))
            criterion = ForwardLossCorrection(transition_matrix=T_hat)
        else:
            raise ValueError(f"Unknown track: {track_name}")

        criterion = criterion.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Training loop
        history = {"train_loss": [], "train_acc": [], "clean_val_acc": []}
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss, total_correct, total_samples = 0.0, 0, 0
            for imgs, noisy_targets, _, _ in train_loader:
                imgs, noisy_targets = imgs.to(device), noisy_targets.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, noisy_targets)
                if torch.isnan(loss) or torch.isinf(loss):
                    raise FloatingPointError(f"Numerical explosion at epoch {epoch}: loss is {loss.item()}")
                loss.backward()
                optimizer.step()

                bs = imgs.size(0)
                total_loss += loss.item() * bs
                preds = logits.argmax(dim=-1)
                total_correct += (preds == noisy_targets).sum().item()
                total_samples += bs

            scheduler.step()
            epoch_loss = total_loss / total_samples
            epoch_acc = (total_correct / total_samples) * 100.0

            # Eval on clean val for tracking
            val_eval = evaluate_model_full(model, val_clean_loader, device)
            history["train_loss"].append(float(epoch_loss))
            history["train_acc"].append(float(epoch_acc))
            history["clean_val_acc"].append(float(val_eval["top1"]))

        training_duration = time.time() - start_time

        # Final evaluation on Clean Test
        test_metrics = evaluate_model_full(model, test_loader, device)
        val_clean_eval = evaluate_model_full(model, val_clean_loader, device)
        val_corrupted_eval = evaluate_model_full(model, val_corrupted_loader, device)

        # Post-hoc Temperature Scaling on Clean Val vs Corrupted Val
        # 1. Clean Val TS
        ts_clean_model = ModelWithTemperature(model)
        temp_clean = ts_clean_model.set_temperature(val_clean_eval["logits"], val_clean_eval["targets"])
        with torch.no_grad():
            ts_clean_test_logits = ts_clean_model.temperature_scale(test_metrics["logits"])
            ts_clean_test_probs = F.softmax(ts_clean_test_logits, dim=-1)
            ts_clean_test_ece = float(compute_ece(ts_clean_test_probs, test_metrics["targets"], num_bins=15))
            ts_clean_test_brier = float(compute_brier_score(ts_clean_test_probs, test_metrics["targets"]))

        # 2. Corrupted Val TS
        corrupted_val_noisy_targets = torch.tensor(val_corrupted_loader.dataset.noisy_labels, dtype=torch.long)
        ts_corrupted_model = ModelWithTemperature(model)
        temp_corrupted = ts_corrupted_model.set_temperature(val_corrupted_eval["logits"], corrupted_val_noisy_targets)
        with torch.no_grad():
            ts_corrupted_test_logits = ts_corrupted_model.temperature_scale(test_metrics["logits"])
            ts_corrupted_test_probs = F.softmax(ts_corrupted_test_logits, dim=-1)
            ts_corrupted_test_ece = float(compute_ece(ts_corrupted_test_probs, test_metrics["targets"], num_bins=15))
            ts_corrupted_test_brier = float(compute_brier_score(ts_corrupted_test_probs, test_metrics["targets"]))

        val_calibration_delta = float(ts_corrupted_test_ece - ts_clean_test_ece)

        cond_num = float(compute_matrix_condition_number(T_hat))
        try:
            inv_norm = float(np.linalg.norm(np.linalg.inv(T_hat), ord=2))
        except Exception:
            inv_norm = float("inf")

        # Record Provenance JSON
        provenance_record = {
            "status": "COMPLETED",
            "provenance": {
                "experiment_id": experiment_id,
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "git_commit_sha": get_git_commit_sha(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "training_duration_sec": training_duration,
            },
            "configuration": {
                "dataset": "cifar10",
                "noise_regime": noise_regime,
                "model": model_name,
                "track": track_name,
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "frobenius_error": frobenius_error,
                "condition_number": cond_num,
                "spectral_norm_inv": inv_norm,
            },
            "metrics": {
                "test_top1_acc": test_metrics["top1"],
                "test_top5_acc": test_metrics["top5"],
                "raw_test_ece": test_metrics["ece"],
                "raw_ada_ece": test_metrics["ada_ece"],
                "raw_brier": test_metrics["brier"],
                "temp_clean": float(temp_clean),
                "ts_clean_test_ece": ts_clean_test_ece,
                "ts_clean_test_brier": ts_clean_test_brier,
                "temp_corrupted": float(temp_corrupted),
                "ts_corrupted_test_ece": ts_corrupted_test_ece,
                "ts_corrupted_test_brier": ts_corrupted_test_brier,
                "val_calibration_delta": val_calibration_delta,
            },
            "history": history,
        }

        # Save checkpoint immediately
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_id}_final.pt")
        torch.save(model.state_dict(), checkpoint_path)

    except Exception as e:
        provenance_record = {
            "status": "FAILED",
            "error_message": str(e),
            "provenance": {
                "experiment_id": experiment_id,
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "git_commit_sha": get_git_commit_sha(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            },
            "configuration": {
                "dataset": "cifar10",
                "noise_regime": noise_regime,
                "model": model_name,
                "track": track_name,
                "seed": seed,
            },
        }

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{experiment_id}.json")
    with open(out_file, "w") as f:
        json.dump(provenance_record, f, indent=2)

    return provenance_record


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 Pilot Benchmark Runner for CIFAR-10")
    parser.add_argument("--manifest", type=str, default="04_EXPERIMENTS/pilot_run_manifest.json", help="Path to manifest JSON")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for datasets")
    parser.add_argument("--output-dir", type=str, default="05_RESULTS/pilot", help="Directory for output JSONs and checkpoints")
    parser.add_argument("--run-id", type=str, default=None, help="Execute a single specific experiment ID")
    parser.add_argument("--run-ids", type=str, default=None, help="Comma-separated list of specific experiment IDs")
    parser.add_argument("--batch-index", type=int, default=None, help="0-based batch index for parallel chunking")
    parser.add_argument("--num-batches", type=int, default=None, help="Total number of parallel batch chunks")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of runs to execute in this process")
    parser.add_argument("--validate-manifest", action="store_true", help="Audit manifest and check completed results without running")
    parser.add_argument("--force", action="store_true", help="Re-run experiments even if completed JSON exists")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device ('cuda' or 'cpu')")
    return parser.parse_args()


def audit_manifest_progress(manifest_path: str, output_dir: str) -> dict:
    """Audit all 84 official runs against output directory."""
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    runs = manifest["runs"]
    total = len(runs)
    completed = []
    failed = []
    missing = []

    for r in runs:
        exp_id = r["experiment_id"]
        res_file = os.path.join(output_dir, f"{exp_id}.json")
        if os.path.exists(res_file):
            try:
                with open(res_file, "r") as f:
                    data = json.load(f)
                if validate_provenance_record(data):
                    completed.append(exp_id)
                else:
                    failed.append(exp_id)
            except Exception:
                failed.append(exp_id)
        else:
            missing.append(exp_id)

    return {
        "total_manifest_runs": total,
        "completed_runs": len(completed),
        "failed_runs": len(failed),
        "missing_runs": len(missing),
        "completed_ids": completed,
        "failed_ids": failed,
        "missing_ids": missing,
    }


def main():
    args = parse_args()
    manifest_path = os.path.abspath(args.manifest)
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)

    if args.validate_manifest:
        audit = audit_manifest_progress(manifest_path, output_dir)
        print("=" * 70)
        print("PILOT BENCHMARK PROGRESS AUDIT")
        print("=" * 70)
        print(f"Total Official Manifest Runs : {audit['total_manifest_runs']}")
        print(f"Completed Valid Runs         : {audit['completed_runs']}")
        print(f"Failed / Corrupted Runs      : {audit['failed_runs']}")
        print(f"Pending / Missing Runs       : {audit['missing_runs']}")
        print("=" * 70)
        if audit["completed_runs"] == audit["total_manifest_runs"]:
            print("ALL 84 OFFICIAL RUNS COMPLETED AND VALIDATED.")
        else:
            pct = (audit["completed_runs"] / audit["total_manifest_runs"]) * 100.0
            print(f"Progress: {pct:.1f}% complete ({audit['missing_runs']} remaining).")
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    all_runs = manifest["runs"]
    selected_runs = []

    if args.run_id:
        selected_runs = [r for r in all_runs if r["experiment_id"] == args.run_id]
        if not selected_runs:
            print(f"Error: run-id '{args.run_id}' not found in manifest.")
            sys.exit(1)
    elif args.run_ids:
        target_ids = set([s.strip() for s in args.run_ids.split(",") if s.strip()])
        selected_runs = [r for r in all_runs if r["experiment_id"] in target_ids]
    elif args.batch_index is not None and args.num_batches is not None:
        if not (0 <= args.batch_index < args.num_batches):
            print(f"Error: batch-index must be in [0, {args.num_batches-1}]")
            sys.exit(1)
        chunk_size = int(np.ceil(len(all_runs) / args.num_batches))
        start_idx = args.batch_index * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_runs))
        selected_runs = all_runs[start_idx:end_idx]
        print(f"Executing Batch Chunk {args.batch_index+1}/{args.num_batches} (Runs {start_idx} to {end_idx-1}, Total: {len(selected_runs)})")
    else:
        selected_runs = all_runs

    if args.limit:
        selected_runs = selected_runs[:args.limit]

    print(f"Selected {len(selected_runs)} runs out of {len(all_runs)} manifest runs.")

    # Load CIFAR-10 datasets
    print("Loading datasets from data directory...")
    cifar_train = datasets.CIFAR10(root=data_dir, train=True, download=False)
    cifar_test = datasets.CIFAR10(root=data_dir, train=False, download=False)

    total_runs = len(selected_runs)
    for idx, r in enumerate(selected_runs, 1):
        exp_id = r["experiment_id"]
        if not args.force and check_run_already_completed(output_dir, exp_id):
            print(f"[{idx}/{total_runs}] Skipping already completed run: {exp_id}")
            continue

        print(f"\n[{idx}/{total_runs}] Starting run: {exp_id}")
        print(f"  - Noise: {r['noise_regime']} ({r.get('noise_rate', 0.0)}) | Track: {r['track']} | Seed: {r['seed']}")
        t0 = time.time()
        res = run_single_experiment(
            experiment_id=exp_id,
            cifar_train=cifar_train,
            cifar_test=cifar_test,
            noise_regime=r["noise_regime"],
            model_name=r["model"],
            track_name=r["track"],
            seed=r["seed"],
            output_dir=output_dir,
            epochs=r.get("epochs", 30),
            batch_size=r.get("batch_size", 128),
            lr=r.get("lr", 0.05),
            device_str=args.device,
            force=args.force,
        )
        dt = time.time() - t0
        print(f"  -> Finished {exp_id} in {dt:.1f}s with status: {res.get('status')}")


if __name__ == "__main__":
    main()
