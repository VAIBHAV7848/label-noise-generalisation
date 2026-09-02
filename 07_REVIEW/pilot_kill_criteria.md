# Pilot Kill & Pivot Criteria (Phase 2)

This document establishes the explicit, non-negotiable quantitative criteria that dictate whether the pilot experiment fails, requires an architectural pivot, or permits proceeding to the full 300-run MDES benchmark.

---

## 1. Non-Negotiable Kill & Pivot Triggers

| Trigger ID | Failure Condition | Target Hypothesis / Proposition | Action & Pivot Protocol |
| :--- | :--- | :--- | :--- |
| **KILL-01** | **Ground-Truth Inversion Collapse**: Forward or Backward Loss Correction with **known true $T$** fails to outperform standard Cross-Entropy by at least $5.0\%$ top-1 test accuracy under heavy symmetric noise ($\eta=0.5$). | Proposition 1 & Loss Correction Pipeline | **HALT EXECUTION IMMEDIATELY**. Indicates mathematical implementation bug or catastrophic gradient failure in `src/losses/loss_correction.py`. |
| **KILL-02** | **Monotonicity Violation of Excess Risk**: Deliberately corrupted matrix $\hat{T}_{\text{bad}}$ ($\epsilon \approx 0.40$) achieves equal or better test accuracy than known true $T$ ($\epsilon = 0.0$) across all 3 seeds under identical training conditions. | Proposition 2 ($\mathcal{E}(\hat{f}) \propto \epsilon$) | **PIVOT THEORY**. The empirical risk does not track theoretical matrix perturbation bounds; theoretical bound is invalidated as an empirical predictive tool. |
| **KILL-03** | **Numerical NaN / Gradient Explosion**: Backward Loss Correction produces `NaN` loss or gradient overflow during training under non-singular transition matrices ($\kappa(T) \le 10.0$). | Numerical Stability & SGD Dynamics | **FIX ARCHITECTURE**. Enforce gradient norm clipping ($\|\mathbf{g}\|_2 \le 5.0$) or replace explicit matrix inversion with iterative linear solve. |
| **KILL-04** | **Calibration Invariance**: Post-hoc Temperature Scaling tuned on corrupted validation sets achieves identical test ECE ($|\Delta \text{ECE}_{\text{val}}| \le 0.005$) to Temperature Scaling tuned on clean validation sets across all noise regimes. | Hypothesis H3 & Research Gap 2 | **PIVOT RESEARCH SCOPE**. If validation corruption has zero effect on post-hoc calibration recovery, drop corrupted-validation calibration as a core contribution and focus exclusively on robust loss optimization. |
| **KILL-05** | **Estimator Degeneracy**: Both Anchor-Point and Confident Learning estimators produce transition matrices with Frobenius error $\|\hat{T} - T\|_F > 0.60$ under moderate noise ($\eta = 0.2$), performing worse than random uniform guessing. | Identifiability Assumptions | **REPLACE ESTIMATORS**. Retrain warm-up feature extractors with contrastive learning before estimating $T$. |

---

## 2. Gate Passage Thresholds (Proceeding to MDES)

To achieve a green light for Phase 3 (300-run MDES benchmark), the pilot **MUST** satisfy ALL of the following criteria across the 3 pilot seeds:

1. $\text{Acc}(\text{Forward}_{\text{known } T}) - \text{Acc}(\text{CE}) \ge +5.0\%$ on CIFAR-10 Symmetric $\eta=0.5$.
2. $\text{Acc}(\text{Forward}_{\text{known } T}) > \text{Acc}(\text{Forward}_{\hat{T}_{\text{estimated}}}) > \text{Acc}(\text{Forward}_{\hat{T}_{\text{bad}}})$.
3. $\|\hat{T}_{\text{anchor}} - T\|_F < 0.20$ and $\|\hat{T}_{\text{CL}} - T\|_F < 0.20$ on CIFAR-10 Symmetric $\eta=0.2$.
4. $\Delta \text{ECE}_{\text{val}} = \text{ECE}(\text{TS}_{\text{corrupted}}) - \text{ECE}(\text{TS}_{\text{clean}}) \ge +0.02$ on CIFAR-10 Asymmetric $\eta=0.4$ (demonstrating statistical measurability of validation corruption).
5. Zero unhandled runtime exceptions, zero NaN losses, and 100% experiment provenance logging.
