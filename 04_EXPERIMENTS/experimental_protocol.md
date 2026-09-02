# Experimental Protocol & Minimum Decisive Experiment Set (MDES)

Following the Phase 1 Research Integrity Audit, the experimental scope is pruned to a **Minimum Decisive Experiment Set (MDES)** of exactly 300 runs to eliminate redundancy, avoid combinatorial scope bloat, and maximize statistical evidence.

---

## 1. Minimum Decisive Experiment Set (MDES) Configuration

| Dimension | Specification | Rationale |
| :--- | :--- | :--- |
| **Primary Vision Benchmark** | **CIFAR-10** (50,000 images, $K=10$) | Universal standard in all primary label noise literature. |
| **Scalability Benchmark** | **CIFAR-100** (50,000 images, $K=100$) | High class-count scalability testing for transition matrix estimators. |
| **Real Human Benchmark** | **CIFAR-10N** (50,000 images, $K=10$) | Real crowdsourced human annotator noise (Aggregate ~9%, Worst ~40%). |
| **Synthetic Noise Regimes** | - Symmetric $\eta \in \{0.2, 0.5\}$<br>- Asymmetric Pair-flip $\eta \in \{0.2, 0.4\}$ | Covers both moderate and severe corruption across symmetric and directional flips. |
| **Model Architectures** | **PreAct-ResNet18** (Primary Deep Backbone) + **TwoLayerMLP** (Diagnostic) | Compares overparameterized deep representation with shallow neural model. |
| **Decisive Baselines** | 1. Standard Cross-Entropy (CE)<br>2. Label Smoothing (LS, $\alpha=0.1$)<br>3. Generalized Cross-Entropy (GCE, $q=0.7$)<br>4. Symmetric Cross-Entropy (SCE, $\alpha=0.1, \beta=1.0$)<br>5. Forward Correction ($T^\top$)<br>6. Backward Correction ($T^{-1}$)<br>7. Confident Learning (Cleanlab) | Spans uncorrected, regularized, robust loss, matrix-corrected, and sample selection families. |
| **Random Seeds** | 5 fixed seeds: `[42, 1337, 2024, 7, 999]` | Enforces exact seed-by-seed pairing for Wilcoxon and paired t-testing. |

---

## 2. Decisive Evaluation Metrics

1. **Top-1 Clean Test Accuracy (%)**: Generalization performance on uncorrupted test set.
2. **Expected Calibration Error (ECE)**: 15-bin equal-width calibration error.
3. **Adaptive Expected Calibration Error (AdaECE)**: 15-bin equal-frequency calibration error.
4. **Brier Score**: Mean squared error of probability forecasts.
5. **Corrupted-Validation Calibration Delta ($\Delta \text{ECE}_{\text{val}}$)**: ECE difference when Temperature Scaling is tuned on corrupted vs. clean validation splits.
6. **Matrix Frobenius Error $\|\hat{T} - T\|_F$**: Estimation quality of Anchor Point vs Confident Learning estimators.

---

## 3. Statistical Testing Protocol

- All metrics reported as **Mean $\pm$ Standard Deviation** across the 5 paired seeds.
- Non-parametric **Wilcoxon Signed-Rank Test** computed for all paired differences.
- **Holm-Bonferroni correction** applied across baseline comparisons ($\alpha = 0.05$).
- **Hedges' $g$** effect sizes reported for all statistically significant differences.
