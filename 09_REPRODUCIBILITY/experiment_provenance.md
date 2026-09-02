# Experiment Provenance & Audit Trail Protocol

This document defines the strict provenance standards required by the Academic Research Skills (ARS) framework for every pilot and benchmark experiment.

---

## 1. Mandatory Run Metadata Schema

Every training run must emit a structured JSON log containing the following immutable metadata fields:

```json
{
  "provenance": {
    "experiment_id": "EXP-PILOT-CIFAR10-SYM05-FWD_TRUE-S42",
    "timestamp_utc": "2026-09-02T16:15:00Z",
    "git_commit_sha": "dce57bf7a34e0c...",
    "git_branch": "main",
    "clean_working_tree": true,
    "user_id": "VAIBHAV7848",
    "hostname": "research-runner-01",
    "cuda_device": "NVIDIA GeForce RTX 3080",
    "python_version": "3.10.12",
    "torch_version": "2.2.0"
  },
  "configuration": {
    "dataset": "cifar10",
    "noise_regime": "symmetric",
    "noise_rate": 0.5,
    "true_transition_matrix": [[...]],
    "estimated_transition_matrix": [[...]],
    "frobenius_error": 0.0,
    "condition_number": 2.25,
    "model_architecture": "PreActResNet18",
    "loss_function": "ForwardCorrection",
    "base_loss": "ce",
    "optimizer": "SGD",
    "learning_rate": 0.05,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "epochs": 30,
    "batch_size": 128,
    "seed": 42
  },
  "metrics": {
    "test_top1_acc": 0.0,
    "test_top5_acc": 0.0,
    "raw_test_ece": 0.0,
    "raw_brier_score": 0.0,
    "ts_clean_test_ece": 0.0,
    "ts_corrupted_test_ece": 0.0,
    "val_calibration_delta": 0.0
  }
}
```

---

## 2. Storage & Reproducibility Policy

1. **Artifact Location**: All run logs must be written to `05_RESULTS/logs/<experiment_id>.json`.
2. **Deterministic Re-execution**: Given the recorded `git_commit_sha`, `seed`, and `configuration`, executing the training script must reproduce metric values within standard floating-point tolerance ($\le 10^{-4}$).
3. **No Phantom Runs**: No experimental finding or figure may be included in reports or manuscripts without a corresponding traceable provenance JSON artifact.
