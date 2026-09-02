# Pilot Evidence Requirements & Experimental Design Specification

This document defines the exact minimum evidence requirements, controls, and calibration protocols necessary for Phase 2 Pilot validation.

---

## 1. Core Objectives of the Pilot Validation

The pilot is **NOT** a mini-benchmark to showcase positive results. It is an instrumented diagnostic test designed to answer 5 specific operational questions before investing computational resources in the 300-run MDES:

1. **Pipeline & Data Integrity**: Does the end-to-end training, noise injection, validation splitting, and metric logging pipeline execute without data leakage, memory leaks, or NaN exceptions?
2. **Ground-Truth Control Behavior**: Does loss correction with **known true $T$** perform as mathematically predicted (preventing performance collapse compared to uncorrected CE under heavy noise)?
3. **Estimation Error Degradation (Proposition 2 Validation)**: Does downstream test accuracy and excess risk degrade monotonically as the transition matrix estimation error $\epsilon = \|\hat{T} - T\|_F$ increases?
4. **Calibration Measurability (Hypothesis H3 Validation)**: Can 15-bin ECE and Brier score reliably capture the difference between Temperature Scaling tuned on clean vs corrupted validation splits without ceiling or floor effects?
5. **Numerical Stability**: Does backward loss correction maintain numerical stability (no gradient explosion or loss NaN) under condition numbers $\kappa(T) \approx 5.0$?

---

## 2. Minimum Decisive Pilot Configuration

| Dimension | Pilot Specification |
| :--- | :--- |
| **Dataset** | **CIFAR-10** (50,000 images, $K=10$) |
| **Noise Regimes (4 Conditions)** | 1. Clean Baseline ($\eta = 0.0$)<br>2. Symmetric Moderate ($\eta = 0.2$)<br>3. Symmetric Heavy ($\eta = 0.5$)<br>4. Asymmetric Pair-flip ($\eta = 0.4$) |
| **Model Backbones** | **PreAct-ResNet18** (Primary representation) + **TwoLayerMLP** (Diagnostic) |
| **Methods & Controls (7 Tracks)** | 1. **CE (Baseline / Negative Control)**: Standard Empirical Risk Minimization<br>2. **GCE ($q=0.7$)**: Robust bounded surrogate loss<br>3. **Forward Correction (Known $T$)**: Positive theoretical control<br>4. **Backward Correction (Known $T$)**: Positive theoretical control<br>5. **Forward Correction (Anchor $\hat{T}_{\text{anchor}}$)**: Empirical estimator track<br>6. **Forward Correction (Confident Learning $\hat{T}_{\text{CL}}$)**: Empirical joint estimator track<br>7. **Forward Correction (Perturbed $\hat{T}_{\text{bad}}$)**: Deliberate negative control ($\hat{T} = 0.5 T + 0.5 \text{Uniform}$, $\epsilon \approx 0.40$) |
| **Validation Splits (Strict Leakage Prevention)** | - **Clean Val Track**: 5,000 uncorrupted holdout images for oracle tuning<br>- **Corrupted Val Track**: 5,000 images corrupted with the identical noise transition matrix $T$ |
| **Pilot Seeds** | 3 fixed seeds: `[42, 1337, 2024]` |
| **Budget** | $4 \text{ noise regimes} \times 7 \text{ tracks} \times 3 \text{ seeds} = \mathbf{84 \text{ pilot runs}}$ (fast 30-epoch validation) |

---

## 3. Mandatory Experimental Controls

```
                        ┌───────────────────────────────┐
                        │   EXPERIMENTAL CONTROLS       │
                        └──────────────┬────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
┌───────────────────┐        ┌───────────────────┐        ┌───────────────────┐
│ Positive Control  │        │ Estimator Tracks  │        │ Negative Control  │
│                   │        │                   │        │                   │
│ Known Ground-     │        │ Anchor Points &   │        │ Deliberately      │
│ Truth Matrix T    │        │ Confident Learning│        │ Perturbed Matrix  │
│ (Forward/Backward)│        │ (||T_hat - T||_F) │        │ T_bad (eps >> 0)  │
└───────────────────┘        └───────────────────┘        └───────────────────┘
```

1. **Positive Control (Known $T$)**: Forward and Backward loss correction supplied with the exact ground-truth matrix $T$. If these fail to outperform standard CE under $\eta=0.5$, the loss implementation is mathematically defective.
2. **Negative Control (Perturbed $T_{\text{bad}}$)**: Forward correction supplied with $\hat{T}_{\text{bad}} = 0.5 T + 0.5 \mathbf{U}$. By Proposition 2, this must exhibit excess risk degradation proportional to $\epsilon = \|\hat{T}_{\text{bad}} - T\|_F$.
3. **No-Leakage Calibration Control**: Temperature Scaling is fitted independently on:
   - Split A: 100% clean validation set ($N=5000$)
   - Split B: 100% noisy validation set ($N=5000$)
   Both are evaluated on the identical unseen, clean test set ($N=10000$).

---

## 4. Required Observable Metrics per Run

1. `test_top1_acc`: Final clean test top-1 accuracy (%)
2. `test_top5_acc`: Final clean test top-5 accuracy (%)
3. `test_ece_15bins`: 15-bin Expected Calibration Error on clean test set
4. `test_ada_ece`: Adaptive 15-bin Expected Calibration Error
5. `test_brier_score`: Brier score on clean test set
6. `ts_clean_test_ece`: Test ECE after Temperature Scaling fitted on clean validation split
7. `ts_corrupted_test_ece`: Test ECE after Temperature Scaling fitted on corrupted validation split
8. `val_calibration_delta`: $\Delta \text{ECE}_{\text{val}} = \text{ts\_corrupted\_test\_ece} - \text{ts\_clean\_test\_ece}$
9. `T_frobenius_error`: $\|\hat{T} - T\|_F$
10. `T_spectral_norm_inv`: $\|\hat{T}^{-1}\|_2$
11. `T_condition_number`: $\kappa(\hat{T})$
