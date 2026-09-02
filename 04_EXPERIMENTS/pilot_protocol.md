# Phase 2 Pilot Protocol (Pre-Registered)

This document pre-registers the exact experimental protocol, dataset splits, hyperparameter values, estimator protocols, and execution budget for the Phase 2 Pilot validation.

---

## 1. Experimental Grid (84 Official Benchmark Runs)

- **Primary Architecture**: PreAct-ResNet18 (Primary Deep Benchmark for all 84 official runs)
- **Dataset Partitioning (CIFAR-10, 60,000 Total Images)**:
  - 35,000 Noisy Training Instances (accessed strictly with noisy labels)
  - 5,000 Clean Validation Instances (used strictly for Clean Temperature Scaling)
  - 5,000 Corrupted Validation Instances (used strictly for Corrupted Temperature Scaling with noisy labels)
  - 5,000 Unused Buffer Instances (held out to maintain balanced partitions)
  - 10,000 Clean Test Instances (evaluated strictly in `torch.no_grad()` post-training)
- **Seeds**: `[42, 1337, 2024]`
- **Training Hyperparameters**:
  - Optimizer: SGD with momentum = 0.9, weight decay = 5e-4
  - Batch size: 128
  - Epochs: 30 epochs per official benchmark run
  - Learning rate schedule: Cosine Annealing, initial LR = 0.05
  - Data Augmentation: Standard random crop (32x32, padding 4) + random horizontal flip

### Diagnostic Tracks:
1. **Track 1**: Standard Cross-Entropy (Baseline)
2. **Track 2**: Generalized Cross-Entropy (GCE, $q=0.7$)
3. **Track 3**: Symmetric Cross-Entropy (SCE, $\alpha=0.1, \beta=1.0$)
4. **Track 4**: Forward Loss Correction (Known True $T$)
5. **Track 5**: Forward Loss Correction (Anchor Point $\hat{T}_{\text{anchor}}$, Patrini et al., 2017)
6. **Track 6**: Forward Loss Correction (Confident Learning $\hat{T}_{\text{CL}}$ with 3-Fold Out-of-Fold Cross-Validation, Northcutt et al., 2021)
7. **Track 7**: Forward Loss Correction (Perturbed $\hat{T}_{\text{bad}} = 0.5 T + 0.5 \mathbf{U}$)

### Noise Regimes:
1. Regime 1: Clean Baseline ($\eta = 0.0$)
2. Regime 2: Symmetric Moderate ($\eta = 0.2$)
3. Regime 3: Symmetric Heavy ($\eta = 0.5$)
4. Regime 4: Asymmetric Pair-flip ($\eta = 0.4$)

$$\text{Official Runs} = 4 \text{ Noise Regimes} \times 7 \text{ Diagnostic Tracks} \times 3 \text{ Seeds} = \mathbf{84 \text{ Runs}}$$

---

## 2. Computational Budget & Preprocessing Accounting

To ensure complete clarity between benchmark evaluations and estimator estimation:

| Category | Description | Models / Folds | Epochs | Total Epochs |
| :--- | :--- | :---: | :---: | :---: |
| **Official Benchmark Runs** | 84 Pre-registered 30-epoch runs | 84 | 30 | 2,520 |
| **Anchor Point Preprocessing** | 1 warm-up model per regime-seed (Track 5) | 12 | 5 | 60 |
| **Confident Learning OOF Preprocessing** | 3-Fold CV warm-up models per regime-seed (Track 6) | 36 | 5 | 180 |
| **Total Computation** | **132 discrete training jobs** | **132** | — | **2,760 epochs** |

---

## 3. Information-Access Matrix

| Track | Clean Train Labels? | Noisy Train Labels? | Clean Val Labels? | Corrupted Val Labels? | Test Labels? | OOF Required? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Track 1: CE** | $\times$ | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |
| **Track 2: GCE** | $\times$ | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |
| **Track 3: SCE** | $\times$ | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |
| **Track 4: Forward (True $T$)** | $\times$ | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |
| **Track 5: Forward (Anchor $\hat{T}$)** | $\times$ | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ (Base model candidates) |
| **Track 6: Forward (CL $\hat{T}$)** | $\times$ | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\checkmark$ (3-Fold CV OOF) |
| **Track 7: Forward (Bad $\hat{T}$)** | $\times$ | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |

---

## 4. Validation & Calibration Procedure

For each trained model checkpoint:
1. Evaluate raw uncalibrated predictions on 10,000 clean test images $\to$ record `test_top1_acc`, `test_top5_acc`, `raw_test_ece`, `raw_brier`.
2. **Clean Val Calibration**: Fit temperature $T_{\text{clean}} > 0$ on 5,000 clean validation images $\to$ evaluate on test set $\to$ record `ts_clean_test_ece`.
3. **Corrupted Val Calibration**: Fit temperature $T_{\text{corrupted}} > 0$ on 5,000 noisy validation images (using noisy labels) $\to$ evaluate on test set $\to$ record `ts_corrupted_test_ece`.
4. Compute calibration transfer delta: $\Delta \text{ECE}_{\text{val}} = \text{ts\_corrupted\_test\_ece} - \text{ts\_clean\_test\_ece}$.

---

## 5. Pre-Registration Lock
Once approved by external review, the pilot will execute this exact configuration without ad-hoc parameter tuning. Any deviation must be recorded in `10_DECISIONS/decision_log.md`.
