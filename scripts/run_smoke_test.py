"""Engineering Smoke Test Runner for Label Noise & Calibration Pipeline.

Executes a 4-run minimal end-to-end smoke test (1 seed, 2 noise regimes, 2 tracks, 1 epoch)
to validate data partitioning, noise injection, model initialization, optimization,
loss/gradient finiteness, calibration metrics, temperature scaling, and JSON provenance.
"""

import os
import sys
import time
import json
import torch
import numpy as np
import torchvision.datasets as datasets

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.training.run_pilot import run_single_experiment


def main():
    print("=" * 70)
    print("STARTING END-TO-END PIPELINE SMOKE TEST")
    print("=" * 70)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Compute Device: {device_str}")
    if device_str == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    output_dir = os.path.abspath("05_RESULTS/smoke_test")
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.abspath("data")

    print("Loading CIFAR-10 datasets from data/...")
    cifar_train = datasets.CIFAR10(root=data_dir, train=True, download=False)
    cifar_test = datasets.CIFAR10(root=data_dir, train=False, download=False)
    print(f"Loaded train={len(cifar_train)}, test={len(cifar_test)}")

    # 4 Representative Configurations for Smoke Testing
    configs = [
        {"noise_regime": "symmetric_0.2", "track_name": "CE", "seed": 42},
        {"noise_regime": "symmetric_0.2", "track_name": "ForwardCorrection_TrueT", "seed": 42},
        {"noise_regime": "asymmetric_0.4", "track_name": "CE", "seed": 42},
        {"noise_regime": "asymmetric_0.4", "track_name": "ForwardCorrection_TrueT", "seed": 42},
    ]

    results = []
    t_start = time.time()

    for idx, cfg in enumerate(configs, 1):
        noise = cfg["noise_regime"]
        track = cfg["track_name"]
        seed = cfg["seed"]
        exp_id = f"smoke_cifar10_{noise}_preact_resnet18_{track}_s{seed}"

        print(f"\n[{idx}/4] Launching Run: {exp_id}")
        print(f"  - Noise Regime : {noise}")
        print(f"  - Track Name   : {track}")
        print(f"  - Seed         : {seed}")
        print(f"  - Epochs       : 1 (Smoke Test)")

        t0 = time.time()
        res = run_single_experiment(
            experiment_id=exp_id,
            cifar_train=cifar_train,
            cifar_test=cifar_test,
            noise_regime=noise,
            model_name="PreActResNet18",
            track_name=track,
            seed=seed,
            output_dir=output_dir,
            epochs=1,
            batch_size=128,
            lr=0.05,
            device_str=device_str,
        )
        dt = time.time() - t0

        status = res.get("status", "UNKNOWN")
        print(f"  -> Result Status: {status} in {dt:.1f}s")
        if status == "COMPLETED":
            top1 = res["metrics"]["test_top1_acc"]
            ece = res["metrics"]["raw_test_ece"]
            brier = res["metrics"]["raw_brier"]
            delta = res["metrics"]["val_calibration_delta"]
            print(f"     Metrics: Top-1 Acc={top1:.2f}%, ECE={ece:.4f}, Brier={brier:.4f}, Val Delta={delta:.4f}")
        else:
            print(f"     Error: {res.get('error_message')}")

        results.append(res)

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"SMOKE TEST COMPLETED IN {total_time:.1f}s")
    print(f"Provenance Logs Written to: {output_dir}")
    print("=" * 70)

    # Validate all 4 runs completed
    all_passed = all(r.get("status") == "COMPLETED" for r in results)
    if all_passed:
        print("\nALL 4 SMOKE TEST RUNS PASSED.")
        return 0
    else:
        print("\nSMOKE TEST FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
