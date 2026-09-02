# Proposed Methodological Framework

## 1. Overview

Rather than proposing a purely ad-hoc loss heuristic without theoretical grounding, our methodological contribution consists of a **Unified Noise-Robust Evaluation & Calibration-Preserving Framework (UNREC)** that systematically bridges:
1. Exact and estimated noise transition matrix modeling ($T$).
2. Gradient-bounded robust loss interpolation.
3. Post-hoc and in-training probability calibration under noise.

---

## 2. Framework Architecture

The framework consists of four modular stages designed to be strictly evaluated against standard and state-of-the-art baselines:

```
+-----------------------------------------------------------------------------------+
| Stage 1: Noise Transition Estimation & Validation                                 |
| - Anchor-Point Estimation (Patrini et al., 2017)                                  |
| - Non-Anchor Dual-T Estimator (Xia et al., 2019)                                 |
| - Confident Learning Joint Matrix (Northcutt et al., 2021)                        |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
| Stage 2: Robust Loss Execution & Risk Correction                                  |
| - Backward Unbiased Correction: ell_backward = T^-1 ell                           |
| - Forward Correction: ell_forward = ell_CE(T^T f(x), y_tilde)                     |
| - Interpolated GCE (q=0.7) & Symmetric Cross Entropy (alpha*CE + beta*RCE)        |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
| Stage 3: Dynamic Memorisation Auditing & Early Stopping Gating                    |
| - Per-epoch Clean Validation Tracking                                             |
| - Tracking of Area Under Margin (AUM) and Loss Distribution Inflection Points     |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
| Stage 4: Calibration Restoration & Reliability Profiling                          |
| - 15-Bin Expected Calibration Error (ECE) & Adaptive ECE computation              |
| - Cross-Validated Noise-Aware Temperature Scaling                                 |
| - Brier Score and Negative Log-Likelihood Decomposition                           |
+-----------------------------------------------------------------------------------+
```

---

## 3. Methodological Innovations & Differentiators

1. **Simultaneous Accuracy and Calibration Tracking**: Unlike prior benchmarks that exclusively measure classification accuracy, our methodology treats **Expected Calibration Error (ECE)** and **Brier Score** as equal first-class citizens.
2. **Transition Matrix Sensitivity Profiling**: We explicitly measure how synthetic and estimated transition matrix error $\|\hat{T} - T\|_F$ impacts both accuracy and calibration.
3. **Cross-Architecture Generalisation Spectrum**: We evaluate identical loss formulations and noise structures across the full spectrum of classifier capacities (Logistic Regression $\to$ Decision Tree $\to$ 2-layer MLP $\to$ PreAct-ResNet18).
