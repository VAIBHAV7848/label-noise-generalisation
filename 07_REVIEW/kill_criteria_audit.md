# Kill & Diagnostic Criteria Reformulation Audit (Phase 2)

**Document Type**: Pre-Registered Failure Criteria & Methodological Re-Audit  
**Framework**: Academic Research Skills (ARS) Falsification Standards  
**Single Source of Truth**: [`VAIBHAV7848/label-noise-generalisation`](https://github.com/VAIBHAV7848/label-noise-generalisation)

---

## 1. Critique of Previous Arbitrary Kill Criteria

An adversarial audit of the previous `KILL-01` through `KILL-05` criteria identified critical methodological flaws:

1. **Flaw in Previous KILL-01**: Demanded that Forward correction with known true $T$ outperform Cross-Entropy by $\ge +5.0\%$ under $\eta=0.5$.  
   *Correction*: Proposition 1 proves that the surrogate loss vector is unbiased in expectation. However, finite-sample SGD dynamics, implicit regularization, and network capacity mean that an exact $+5.0\%$ test accuracy margin is an empirical outcome, not a theoretical theorem entailment. Conflating a mathematical identity with an arbitrary $+5.0\%$ empirical threshold is scientifically unsound.
2. **Flaw in Previous KILL-02**: Demanded strict accuracy monotonicity $\text{Acc}(\text{True } T) > \text{Acc}(\text{Estimated } T) > \text{Acc}(\text{Bad } T)$ and interpreted any inversion as "falsifying Proposition 2".  
   *Correction*: Proposition 2 establishes an **upper bound** on excess risk. An upper bound does not imply that empirical test accuracy is a strictly monotonic bijection of $\|\hat{T}-T\|_F$. Stochastic variance across seeds can cause small fluctuations between Anchor $\hat{T}$ and Confident Learning $\hat{T}$.
3. **Flaw in Previous KILL-04**: Set an arbitrary threshold $|\Delta \text{ECE}_{\text{val}}| \le 0.005$ as a kill trigger.  
   *Correction*: The $0.005$ value was an ad-hoc engineering constant. It is now framed as a pre-registered diagnostic sensitivity metric.

---

## 2. Rigorous, Scientifically Justified Diagnostic Criteria

| Trigger ID | Revised Diagnostic Failure Condition | Methodological Scope | Scientific Consequence & Action |
| :--- | :--- | :--- | :--- |
| **KILL-01 (Revised)** | **Loss Pipeline & Numerical Integrity**: Forward or Backward Loss Correction produces `NaN`, `Inf`, or diverging negative loss under known true $T$, or the Monte Carlo empirical mean of the corrected loss deviates from the clean expectation on synthetic verification batches. | Mathematical Implementation Integrity | **HALT EXECUTION IMMEDIATELY**. Indicates catastrophic gradient explosion or algebraic index bug in loss correction implementation. |
| **KILL-02 (Revised)** | **Perturbation Sensitivity Collapse**: Deliberately corrupted matrix $\hat{T}_{\text{bad}} = 0.5 T + 0.5 \mathbf{U}$ ($\epsilon \approx 0.40$) systematically and statistically significantly outperforms the true matrix $T$ ($\epsilon = 0.0$) across all 3 seeds under identical training conditions. | Proposition 2 Empirical Sensitivity Check | **PIVOT MODEL / LOSS ANALYSIS**. Indicates that the model's loss landscape is insensitive to transition matrix geometry; investigate loss regularization. |
| **KILL-03 (Revised)** | **Numerical Instability / Divergence**: Model training produces `NaN` losses, infinite gradients, or numerical overflow during SGD optimization under non-singular transition matrices ($\kappa(T) \le 10.0$). | Optimization & Gradient Stability | **FIX OPTIMIZATION ARCHITECTURE**. Enforce gradient clipping ($\|\mathbf{g}\|_2 \le 5.0$) or adjust initial learning rate schedule. |
| **KILL-04 (Revised)** | **Validation Calibration Invariance**: Post-hoc Temperature Scaling tuned on corrupted validation sets achieves identical test ECE to Temperature Scaling tuned on clean validation sets across all noise regimes including $\eta=0.5$ and $\eta=0.4$ Asymmetric. | Hypothesis H3 & Research Gap 2 | **DOCUMENT EMPIRICAL INVARIANCE**. If validation corruption produces zero measurable calibration transfer penalty on CIFAR-10, report this finding and pivot focus to in-training loss calibration dynamics (GCE/SCE). |
| **KILL-05 (Revised)** | **Estimator Degeneracy**: Estimator produces a near-singular transition matrix ($\kappa(\hat{T}) > 10^4$) or Frobenius error $\|\hat{T} - T\|_F > 0.60$ under moderate noise ($\eta = 0.2$), performing worse than random uniform guessing. | Identifiability & Estimator Health | **DIAGNOSE ESTIMATOR**. Review warm-up feature representations and adjust cross-validation fold epochs or margin thresholds. |

---

## 3. Strict Information Separation for Evaluation

- **Oracle Metrics (Evaluation Only)**: True transition matrix $T$ and ground-truth clean test labels are utilized strictly for computing Frobenius error $\|\hat{T} - T\|_F$ and post-training test accuracy/ECE.
- **Estimator Inputs (No Oracle Information)**: Estimators (`src/estimators/anchor_point.py` and `src/estimators/oof.py`) receive strictly the 35,000 noisy training instances and noisy labels $\tilde{y}$.
