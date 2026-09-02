# Phase 2 Pilot Protocol (Pre-Registered)

This document pre-registers the exact experimental protocol, dataset splits, hyperparameter values, and execution steps for the Phase 2 Pilot validation.

---

## 1. Experimental Grid (84 Total Pilot Runs)

- **Dataset**: CIFAR-10 (35,000 train, 5,000 clean val, 5,000 corrupted val, 10,000 test)
- **Seeds**: `[42, 1337, 2024]`
- **Model Backbone**: PreAct-ResNet18 (Primary) and TwoLayerMLP (Diagnostic)
- **Training Hyperparameters**:
  - Optimizer: SGD with momentum = 0.9, weight decay = 5e-4
  - Batch size: 128
  - Epochs: 30 epochs (pilot budget)
  - Learning rate schedule: Cosine Annealing, initial LR = 0.05
  - Data Augmentation: Standard random crop (32x32, padding 4) + random horizontal flip

### Track Breakdown per Noise Regime:
1. **Track 1**: Standard Cross-Entropy (Baseline)
2. **Track 2**: Generalized Cross-Entropy (GCE, $q=0.7$)
3. **Track 3**: Symmetric Cross-Entropy (SCE, $\alpha=0.1, \beta=1.0$)
4. **Track 4**: Forward Loss Correction (Known true $T$)
5. **Track 5**: Forward Loss Correction (Anchor Point $\hat{T}_{\text{anchor}}$)
6. **Track 6**: Forward Loss Correction (Confident Learning $\hat{T}_{\text{CL}}$)
7. **Track 7**: Forward Loss Correction (Perturbed $\hat{T}_{\text{bad}} = 0.5 T + 0.5 \mathbf{U}$)

### Noise Regimes:
1. Regime 1: Clean Baseline ($\eta = 0.0$)
2. Regime 2: Symmetric Moderate ($\eta = 0.2$)
3. Regime 3: Symmetric Heavy ($\eta = 0.5$)
4. Regime 4: Asymmetric Pair-flip ($\eta = 0.4$)

---

## 2. Validation & Calibration Procedure

For each trained model checkpoint:
1. Evaluate raw uncalibrated predictions on 10,000 clean test images $\to$ record `test_top1_acc`, `test_top5_acc`, `raw_test_ece`, `raw_brier`.
2. **Clean Val Calibration**: Fit temperature $T_{\text{clean}} > 0$ on 5,000 clean validation images $\to$ evaluate on test set $\to$ record `ts_clean_test_ece`.
3. **Corrupted Val Calibration**: Fit temperature $T_{\text{corrupted}} > 0$ on 5,000 noisy validation images $\to$ evaluate on test set $\to$ record `ts_corrupted_test_ece`.
4. Compute delta: $\Delta \text{ECE}_{\text{val}} = \text{ts\_corrupted\_test\_ece} - \text{ts\_clean\_test\_ece}$.

---

## 3. Pre-Registration Lock
Once approved by external review, the pilot will execute this exact configuration without ad-hoc parameter tuning. Any deviation must be recorded in `10_DECISIONS/decision_log.md`.
