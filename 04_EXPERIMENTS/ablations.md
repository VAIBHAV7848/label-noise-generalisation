# Ablation Studies & Sensitivity Dimensions

To isolate the individual contribution of each theoretical component and algorithmic hyperparameter, we plan four systematic ablation axes:

---

## 1. Ablation Axis 1: Transition Matrix Estimation Error Sensitivity ($\|\hat{T} - T\|_F$)
- **Goal**: Empirically validate theoretical Proposition 2 (excess risk scaling with transition estimation error).
- **Protocol**: Artificially inject Gaussian perturbation noise into the true transition matrix:
  $$\hat{T}_{\sigma} = \text{Normalize}\left( \max\left(0, T + \mathcal{N}(0, \sigma^2 I)\right) \right)$$
  for $\sigma \in \{0.0, 0.05, 0.1, 0.2, 0.3, 0.5\}$.
- **Measured Response**: Plot Test Accuracy and ECE vs. $\|\hat{T}_{\sigma} - T\|_F$ across Forward vs Backward correction.

---

## 2. Ablation Axis 2: Hyperparameter Robustness Grid ($q$ in GCE, $\alpha/\beta$ in SCE)
- **Goal**: Quantify sensitivity to hyperparameter tuning under misspecified noise rates.
- **Protocol**:
  - GCE: Evaluate $q \in \{0.1, 0.3, 0.5, 0.7, 0.9, 1.0\}$.
  - SCE: Grid search $\alpha \in \{0.01, 0.1, 1.0, 2.0\}$ and $\beta \in \{0.1, 0.5, 1.0, 5.0\}$.
- **Measured Response**: Variance in test accuracy and calibration degradation across symmetric vs asymmetric noise regimes.

---

## 3. Ablation Axis 3: Anchor-Point Percentile Selection ($\alpha$)
- **Goal**: Measure how anchor-point selection percentile ($\alpha \in \{90\%, 95\%, 97\%, 99\%, 100\%\}$) affects $\hat{T}$ Frobenius error and subsequent model accuracy.
- **Protocol**: Compare transition matrix estimation error on CIFAR-10 ($K=10$) and CIFAR-100 ($K=100$).

---

## 4. Ablation Axis 4: Post-Hoc Temperature Scaling with Clean vs. Noisy Validation Splits
- **Goal**: Test whether temperature scaling calibration requires clean validation data or succeeds using noisy validation data.
- **Protocol**: Fit optimal temperature parameter $T_{\text{temp}}^*$ on:
  1. 100% clean validation split.
  2. Corrupted validation split matching training noise rate $\eta$.
- **Measured Response**: Post-scaling ECE and reliability diagrams evaluated on clean test partition.
