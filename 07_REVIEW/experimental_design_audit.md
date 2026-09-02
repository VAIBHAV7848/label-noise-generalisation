# Experimental Design Audit & Decisive Scope Reduction

This document audits the computational feasibility, redundancy, and efficiency of the planned experimental suite.

---

## 1. Audit of Original Experimental Budget

In the initial Phase 0 design:
- **Datasets**: MNIST, CIFAR-10, CIFAR-100, CIFAR-10N, Animal-10N, UCI Adult (6 datasets)
- **Noise Types**: Symmetric (5 rates), Asymmetric (4 rates) = 9 noise settings per synthetic dataset
- **Models**: Logistic Regression, Decision Tree, Random Forest, 2-layer MLP, PreAct-ResNet18, ResNet-50 (6 architectures)
- **Baselines**: 10 distinct methods
- **Seeds**: 5 seeds

### Total Combinatorial Run Count:
$$\text{Runs} \approx 6 \times 9 \times 6 \times 10 \times 5 = 16,200 \text{ training runs!}$$
- **Feasibility Verdict**: **CATASTROPHIC SCOPE BLOAT**.
- Executing 16,200 deep neural network training runs would require thousands of GPU-hours, generate massive redundant tables, and dilute the central scientific message of the research.

---

## 2. Redundancy Analysis

1. **MNIST and UCI Adult Redundancy**:
   - Running deep ResNets and complex baselines (DivideMix, SCE, GCE) on MNIST and UCI Adult provides near-zero discriminative value (accuracy saturates at >98% even with noise).
2. **ResNet-50 vs. PreAct-ResNet18**:
   - Training both ResNet-18 and ResNet-50 on CIFAR-10 is redundant. PreAct-ResNet18 is the standard universal benchmark architecture in all primary label noise literature (Patrini 2017, Li 2020, Wei 2022).
3. **Noise Rate Pruning**:
   - Testing 5 symmetric rates (0.0, 0.2, 0.4, 0.6, 0.8) and 4 asymmetric rates (0.1, 0.2, 0.3, 0.4) on every architecture is unnecessary. Three representative noise levels (Clean 0.0, Moderate 0.2, Heavy 0.5) capture the full non-linear degradation dynamics.

---

## 3. Minimum Decisive Experiment Set (MDES)

To maximize scientific signal-to-noise ratio while remaining 100% computationally feasible on standard hardware:

### Core Benchmark Configuration (MDES):
1. **Primary Dataset**: CIFAR-10 (Synthetic: Symmetric $\eta \in \{0.2, 0.5\}$, Asymmetric $\eta \in \{0.2, 0.4\}$) + CIFAR-10N (Human Noise: Aggregate, Worst).
2. **Scalability Dataset**: CIFAR-100 (Symmetric $\eta=0.4$).
3. **Core Model Architecture**: PreAct-ResNet18 (Deep representation standard) + 2-Layer MLP (Shallow diagnostic).
4. **Decisive Baselines (6 Focused Methods)**:
   - *Standard*: Cross-Entropy (CE)
   - *Regularized*: Label Smoothing (LS)
   - *Robust Loss*: Generalized Cross-Entropy (GCE, $q=0.7$) & Symmetric Cross-Entropy (SCE)
   - *Loss Correction*: Forward Correction ($T^\top$) & Backward Correction ($T^{-1}$)
   - *Dataset Cleaning*: Confident Learning (Cleanlab)
5. **Decisive Metrics**:
   - Top-1 Clean Test Accuracy (%)
   - 15-bin Expected Calibration Error (ECE) & Brier Score
   - Frobenius Transition Error $\|\hat{T} - T\|_F$
   - Clean vs. Corrupted Validation Set Calibration Delta ($\Delta \text{ECE}_{\text{val}}$)
6. **Seeds**: 5 fixed seeds (`[42, 1337, 2024, 7, 999]`).

### Total MDES Run Count:
$$\text{Runs} = 2 \text{ datasets} \times 5 \text{ noise regimes} \times 6 \text{ methods} \times 5 \text{ seeds} = 300 \text{ runs}$$
- **Feasibility**: Easily executable within hours on modern GPU hardware while delivering 100% of the required scientific evidence.
