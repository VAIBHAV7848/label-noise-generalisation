# Pilot Kill & Diagnostic Failure Criteria (Phase 2)

**Document Type**: Pre-Registered Failure Criteria & Methodological Standards  
**Framework**: Academic Research Skills (ARS) Falsification Standards  
**Single Source of Truth**: [`VAIBHAV7848/label-noise-generalisation`](https://github.com/VAIBHAV7848/label-noise-generalisation)

---

## 1. Revised Non-Negotiable Kill & Diagnostic Triggers

| Trigger ID | Diagnostic Failure Condition | Target Scope | Action & Scientific Protocol |
| :--- | :--- | :--- | :--- |
| **KILL-01** | **Loss Pipeline & Numerical Integrity**: Loss calculation produces `NaN`, `Inf`, or persistent arithmetic divergence ($> 10^6$) under known true $T$, or in controlled Monte Carlo synthetic verification tests the empirical mean of corrected loss deviates from the clean expectation beyond sampling error ($|\frac{1}{N}\sum \tilde{\ell} - \frac{1}{N}\sum \ell_{\text{clean}}| > 5 \cdot \text{SEM}$). *(Note: Negative individual per-sample loss values in Backward correction are mathematically valid due to negative entries in $T^{-1}$ and do NOT constitute failure; Forward correction in the pilot is strictly non-negative).* | Loss Correction Pipeline & Implementation Integrity | **HALT EXECUTION IMMEDIATELY**. Indicates catastrophic gradient explosion, numerical overflow, or algebraic indexing bug in loss correction implementation. |
| **KILL-02** | **Perturbation Sensitivity Collapse**: Deliberately corrupted matrix $\hat{T}_{\text{bad}} = 0.5 T + 0.5 \mathbf{U}$ ($\epsilon \approx 0.40$) systematically and statistically significantly outperforms the true matrix $T$ ($\epsilon = 0.0$) across all 3 seeds under identical training conditions. | Proposition 2 Empirical Sensitivity Check | **PIVOT MODEL / LOSS ANALYSIS**. Indicates that the model's loss landscape is insensitive to transition matrix geometry; investigate loss regularization. |
| **KILL-03** | **Numerical Instability / Divergence**: Model training produces `NaN` losses, infinite gradients, or numerical overflow during SGD optimization under non-singular transition matrices ($\kappa(T) \le 10.0$). | Optimization & Gradient Dynamics | **FIX OPTIMIZATION ARCHITECTURE**. Enforce gradient norm clipping ($\|\mathbf{g}\|_2 \le 5.0$) or adjust initial learning rate schedule. |
| **KILL-04** | **Validation Calibration Invariance**: Post-hoc Temperature Scaling tuned on corrupted validation sets achieves identical test ECE to Temperature Scaling tuned on clean validation sets across all noise regimes including $\eta=0.5$ and $\eta=0.4$ Asymmetric. | Hypothesis H3 & Research Gap 2 | **DOCUMENT EMPIRICAL INVARIANCE**. If validation corruption produces zero measurable calibration transfer penalty on CIFAR-10, report this finding and pivot focus to in-training loss calibration dynamics (GCE/SCE). |
| **KILL-05** | **Estimator Degeneracy**: Estimator produces a near-singular transition matrix ($\kappa(\hat{T}) > 10^4$) or Frobenius error $\|\hat{T} - T\|_F > 0.60$ under moderate noise ($\eta = 0.2$), performing worse than random uniform guessing. | Identifiability & Estimator Health | **DIAGNOSE ESTIMATOR**. Review warm-up feature representations and adjust cross-validation fold epochs or margin thresholds. |

---

## 2. Gate Passage Diagnostic Objectives (Pre-Phase 3 Review)

Before proposing progression to Phase 3 (300-run MDES benchmark), the pilot data must be reviewed against these diagnostic benchmarks:

1. **Numerical Sanity**: Zero `NaN` or `Inf` loss values, bounded risk, and 100% complete provenance JSON logs across all 84 runs.
2. **Perturbation Sensitivity**: Known True $T$ achieves higher test accuracy than the deliberately degraded $\hat{T}_{\text{bad}}$ control across seeds.
3. **Estimator Stability**: Both Anchor-Point and Confident Learning estimators produce non-singular matrices ($\kappa(\hat{T}) \le 10^4$) with Frobenius error $\|\hat{T} - T\|_F < 0.40$ on CIFAR-10 Symmetric $\eta=0.2$.
4. **Calibration Transfer Sensitivity**: $\Delta \text{ECE}_{\text{val}} = \text{ECE}(\text{TS}_{\text{corrupted}}) - \text{ECE}(\text{TS}_{\text{clean}})$ is empirically measured across all regimes to quantify the post-hoc calibration degradation.
