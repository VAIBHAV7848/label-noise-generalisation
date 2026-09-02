# ARS Pre-Pilot Gate Review Report (Phase 2 Readiness)

This document provides the final, fail-closed pre-pilot review conducted under the Academic Research Skills (ARS) methodology prior to initiating Phase 2 Pilot Experimentation.

---

## 1. Current State
The project has completed its Phase 0 research foundation and Phase 1 modular architecture. Mathematical proofs and estimator fidelity have been verified, resulting in the correction of Proposition 2's operator norm bound, the disproof and replacement of Proposition 4's inverted ECE lower bound with exact Theorems 4A & 4B, the replacement of the flawed condition-number hypothesis (H1), and the elimination of sample selection bias in the anchor-point estimator. Exactly 20 unit and adversarial tests pass. Exactly zero experimental training runs have occurred.

---

## 2. ARS Methodology Applied
The review applied core Academic Research Skills (ARS) principles:
- **Traceability Chain**: Explicit mapping from Primary RQ $\to$ Gap $\to$ Hypothesis $\to$ Prediction $\to$ Experiment $\to$ Metric $\to$ Falsification condition.
- **Fail-Closed Verification**: Mathematical claims were challenged with adversarial counterexamples (e.g. Jensen cancellation in ECE, symmetric vs asymmetric boundary shift) rather than trusted on existing pass statuses.
- **Provenance Protocol**: Formal pre-registration of run metadata and protocol locking before launching experiments.
- **Separation of Roles**: ARS acts as the methodological control layer; the GitHub repository [`VAIBHAV7848/label-noise-generalisation`](https://github.com/VAIBHAV7848/label-noise-generalisation) acts as the evidence source of truth.

---

## 3. Research Question Audit
- **Current Primary RQ**: *"Under what exact theoretical bounds and empirical conditions do loss-correction methods and robust losses maintain probability calibration (ECE) and excess risk guarantees when subject to transition matrix estimation error and corrupted validation distributions across varying classifier capacities?"*
- **Critique**:
  - The term "maintain" is operationally defined as bounding test ECE inflation ($\Delta \text{ECE} \le \text{Threshold}$) and bounding excess risk within theoretical perturbation envelopes.
  - Calibration and excess risk are intrinsically linked because noisy labels distort both the predicted class rankings (excess risk) and posterior probability magnitudes (ECE), both governed by the operator $T^\top$.
- **Verdict**: **[ALIGNED & OPERATIONALLY DEFINED]**.

---

## 4. Research Gap Audit
- **Gap 1 (Excess Risk under Finite-Sample Matrix Estimation Error)**: Confirmed. Derived non-asymptotic bound linear in $\sqrt{K} M \|T^{-1}\|_2 \|\hat{T} - T\|_F$.
- **Gap 2 (Calibration Recovery on Corrupted Validation Sets)**: Confirmed. Focuses on post-hoc Temperature Scaling fitted on corrupted vs clean validation splits.
- **Gaps 3 & 4 (Memorisation Discovery & Human Noise Failure)**: Correctly rejected as pre-solved by prior literature (Arpit 2017, Li 2020, Wei 2022) and demoted to baseline diagnostics.

---

## 5. Hypothesis $\to$ Experiment Traceability
All four revised hypotheses were audited in `07_REVIEW/hypothesis_experiment_traceability.md`:
- **H1 (Asymmetry-driven displacement)**: `[ALIGNED]` (Tested via linear models and cosine angle $\theta$).
- **H2 (Capacity & generalisation window)**: `[ALIGNED]` (Tested via MLP width scaling and gradient norm tracking).
- **H3 (Bidirectional miscalibration & validation split)**: `[ALIGNED]` (Tested via clean vs corrupted Temperature Scaling on PreAct-ResNet18).
- **H4 (Confounder-controlled human noise transfer)**: `[ALIGNED]` (Tested on CIFAR-10N controlling for MixUp augmentation).

---

## 6. Proposition Readiness
- **Proposition 1 (Unbiasedness)**: Analytically verified; exact match with PyTorch implementation.
- **Proposition 2 (Excess Risk Perturbation)**: Proved from first principles; verified across 10,000 numerical simulations with 0 violations.
- **Theorem 4A (Vector Calibration Distortion)**: Exact equality $\text{VCE} = \mathbb{E}[\|(T^\top - I)\mathbf{p}\|_1]$.
- **Theorem 4B (Symmetric Top-Label ECE)**: Exact equality $\text{ECE} = \frac{K \eta}{K-1} \mathbb{E}[\hat{P} - 1/K]$.
- **Theoretical Contribution Status**: **[SUPPORTING FRAMEWORK BOUNDS]** (solid unified operator-norm characterizations, honestly scoped without exaggerated novelty claims).

---

## 7. Estimator Readiness
- **Anchor-Point Estimator (`src/estimators/anchor_point.py`)**: Faithful to Patrini et al. (2017) with dataset-wide search.
- **Confident Learning (`src/estimators/confident_learning.py`)**: Faithful to Northcutt et al. (2021) with max-margin single assignment.
- **Dual-T (`src/estimators/dual_t.py`)**: Accurately scoped and renamed to `estimate_transition_matrix_heuristic_slack` to distinguish it from the full constrained optimization of Xia et al. (2019).

---

## 8. Pilot Design
- **Objective**: Diagnostic pipeline test to verify ground-truth inversion, estimation error monotonicity, and calibration measurability.
- **Scope**: CIFAR-10 across 4 noise regimes, PreAct-ResNet18 and TwoLayerMLP backbones, 7 method tracks, 3 seeds (`[42, 1337, 2024]`), totaling **84 runs**.
- **Controls Included**:
  - Positive Control: Ground-truth known $T$ (Forward/Backward).
  - Negative Control: Deliberately perturbed matrix $\hat{T}_{\text{bad}}$ ($\epsilon \approx 0.40$).
  - Baseline Control: Standard Cross-Entropy.

---

## 9. Calibration Design
- **No Data Leakage**: Evaluated across two independent validation splits:
  - Split A: 5,000 clean holdout images.
  - Split B: 5,000 corrupted holdout images (corrupted via identical transition matrix $T$).
- Measures raw uncalibrated test ECE, clean-tuned test ECE, corrupted-tuned test ECE, and $\Delta \text{ECE}_{\text{val}}$.

---

## 10. Transition Matrix Design
- Tracks Frobenius error $\|\hat{T} - T\|_F$, inverse spectral norm $\|\hat{T}^{-1}\|_2$, and condition number $\kappa(\hat{T})$ for every run, directly correlating matrix error with downstream test accuracy and ECE.

---

## 11. Statistical Design
- Pilot uses 3 fixed seeds to verify pipeline stability and detect gross failures.
- Final benchmark inference uses 5 paired seeds with Wilcoxon signed-rank tests, Holm-Bonferroni correction, and Hedges' $g$ effect sizes.

---

## 12. Provenance Requirements
- Enforced structured JSON provenance logging per run in `09_REPRODUCIBILITY/experiment_provenance.md` recording git SHA, exact hyperparameters, transition matrices, hardware, and environment versions.

---

## 13. Kill Criteria
- Documented in `07_REVIEW/pilot_kill_criteria.md`:
  - Immediate halt if known-$T$ correction fails to beat standard CE by $\ge 5\%$ on heavy noise ($\eta=0.5$).
  - Immediate pivot if deliberately corrupted matrix $\hat{T}_{\text{bad}}$ outperforms estimated or true $T$ (violating Proposition 2 monotonicity).
  - Immediate scope pivot if corrupted validation splits produce zero measurable change in test ECE ($|\Delta \text{ECE}_{\text{val}}| \le 0.005$).

---

## 14. Remaining Risks
- **Computational Run Time**: Full 30-epoch training on 84 runs requires modest GPU compute time (~1–2 hours on standard GPU hardware).
- **Noise Estimation Variance on Deep Features**: If initial feature representations under heavy noise are degenerate, anchor point estimation variance may be elevated.

---

## 15. Required Pre-Pilot Changes Completed
1. Pre-registered pilot protocol in `04_EXPERIMENTS/pilot_protocol.md`.
2. Documented provenance schema in `09_REPRODUCIBILITY/experiment_provenance.md`.
3. Created hypothesis-to-experiment traceability matrix in `07_REVIEW/hypothesis_experiment_traceability.md`.
4. Established non-negotiable pilot kill criteria in `07_REVIEW/pilot_kill_criteria.md`.

---

## 16. FINAL DECISION

**DECISION**: **CONDITIONAL GO TO PILOT**

**Conditions for Opening Phase 2 Execution**:
1. External human reviewer reviews and approves the pre-registered `04_EXPERIMENTS/pilot_protocol.md` and `07_REVIEW/pre_pilot_gate.md`.
2. The pilot execution proceeds strictly according to the pre-registered 84-run grid with structured JSON provenance logging and no ad-hoc parameter modifications.

---

## Hard Stop Enforced
No pilot experiments, training commands, or GPU executions have been initiated. Execution is halted awaiting external human authorization.
