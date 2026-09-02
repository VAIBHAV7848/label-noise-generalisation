# Research Gap Audit: Adversarial Evaluation

This document scrutinizes every candidate research gap claimed in Phase 0 and Phase 1, assessing whether each is genuinely unresolved or already solved by prior literature.

---

## 1. Audit of Candidate Gap A: Non-Asymptotic Excess Risk Bounds under Matrix Estimation Error ($\|\hat{T} - T\|_F$)

- **Stated Gap**: "Natarajan et al. (2013) assume exact knowledge of $T$. How does finite-sample matrix estimation error $\epsilon = \|\hat{T} - T\|_F$ propagate to excess risk $\mathcal{E}(\hat{f})$?"
- **Search for Prior Art**:
  - *Natarajan et al. (NeurIPS 2013)*: Section 4 discusses estimating $\rho_+$, but focuses on the asymptotic rate without explicit matrix norm operator perturbation.
  - *Scott (AISTATS 2015)*: "A Rate of Convergence for Mixture Proportion Estimation" derives minimax rates for transition estimates.
  - *Xia et al. (NeurIPS 2019)*: Analyzes estimation error bounds for non-anchor estimators.
  - *Zhu et al. (TMLR 2023)* / *Cheng et al. (2020)*: Analyzes sample complexity under instance-dependent noise.
- **Evaluation**:
  - **Verdict**: **[PARTIAL GAP]**.
  - **What is Solved**: The general matrix inverse perturbation $(T + E)^{-1} = T^{-1} - T^{-1} E \hat{T}^{-1}$ is standard linear algebra.
  - **What Remains Genuinely Unresolved**: A clean, non-asymptotic excess risk bound connecting empirical Rademacher complexity $\mathcal{R}_n(\mathcal{F})$ of deep networks with the condition number $\kappa(T)$ and finite-sample Frobenius estimation error $\|\hat{T} - T\|_F$ under multi-class loss correction.
  - **Required Evidence**: A mathematically rigorous, step-by-step proof without hand-waving constants (see `07_REVIEW/mathematical_audit.md`).

---

## 2. Audit of Candidate Gap B: Confidence Calibration under Noisy-Label Learning

- **Stated Gap**: "Prior works optimize exclusively for top-1 accuracy under label noise; the effect of loss correction and robust losses on Expected Calibration Error (ECE) and Brier score is unexplored."
- **Search for Prior Art**:
  - *Wang et al. (NeurIPS 2021)*: "On the Calibration of Noisy-Label Learning" explicitly investigates calibration under label noise and shows that robust losses alter overconfidence.
  - *Bai et al. (ICLR 2021)*: "How Does Mixup Help With Robustness and Calibration under Label Noise?" proves that Mixup calibrates noisy neural networks.
  - *Thulasidasan et al. (NeurIPS 2019)*: Demonstrates that label noise exacerbates miscalibration and MixUp repairs it.
- **Evaluation**:
  - **Verdict**: **[WEAK / PARTIALLY SOLVED]**.
  - **Critical Flaw in Original Claim**: Claiming that calibration under label noise is "unexplored" or "first study" is **CONTRADICTED BY LITERATURE**. Wang et al. (2021) and Bai et al. (2021) have already published on this exact topic.
  - **What Genuinely Remains Unresolved**: How loss correction methods (Backward vs. Forward) affect calibration under corrupted validation splits (where practitioners lack a clean validation set to tune Temperature Scaling).
  - **Required Evidence**: The research scope must pivot from a broad "we are the first to study calibration" claim to a focused, controlled analysis of **post-hoc calibration recovery on corrupted validation sets**.

---

## 3. Audit of Candidate Gap C: Capacity-Dependent Memorisation Dynamics

- **Stated Gap**: "Investigating whether overparameterized models exhibit a sharp bifurcation epoch $\tau_{\text{crit}}$ while underparameterized models degrade smoothly without an early peak."
- **Search for Prior Art**:
  - *Zhang et al. (ICLR 2017)*: Proved deep nets fit random labels with zero training error.
  - *Arpit et al. (ICML 2017)*: Established early learning of clean patterns before noise memorisation.
  - *Li et al. (ICML 2020)*: Formally proved the exact early-stopping inflection epoch $\tau_{\text{crit}} = \Theta(\log(1/\eta))$ on two-layer networks.
  - *Liu et al. (ICLR 2020)*: "Early-Learning Regularization" built algorithms based on this exact inflection.
- **Evaluation**:
  - **Verdict**: **[NOT A GAP (SOLVED)]**.
  - **Status**: The existence of early-learning and capacity-dependent memorisation in neural networks is thoroughly established. Presenting this as a novel gap or headline contribution would result in immediate reviewer rejection.
  - **Required Action**: Demote this from a primary research gap to a standard empirical diagnostic dimension.

---

## 4. Audit of Candidate Gap D: Synthetic vs. Real Human Noise Failure Modes

- **Stated Gap**: "Evaluating why transition-matrix-based methods fail on real human crowdsourced noise (CIFAR-10N)."
- **Search for Prior Art**:
  - *Wei et al. (ICLR 2022)*: "Learning with Noisy Labels Revisited: A Study on ImageNet and CIFAR-10N" already proved that synthetic class-conditional methods collapse on CIFAR-10N because human noise is instance-dependent ($P(\tilde{Y} \mid X, Y)$).
- **Evaluation**:
  - **Verdict**: **[NOT A NOVEL GAP (SOLVED BY WEI ET AL. 2022)]**.
  - **Status**: Wei et al. (2022) already thoroughly established this exact empirical observation and released the dataset for that purpose.
  - **Required Action**: Frame CIFAR-10N evaluation as a necessary external validation benchmark, not as a discovery of our own.

---

## 5. Summary Table of Gap Audits

| Candidate Gap | Initial Claim | Literature Verdict | Surviving Core (if any) |
| :--- | :--- | :--- | :--- |
| **Gap A (Excess Risk Bounds)** | Novel excess risk bounds under $\|\hat{T} - T\|_F$ | **PARTIAL GAP** | Non-asymptotic excess risk bound incorporating Rademacher complexity and condition number $\kappa(T)$. |
| **Gap B (Calibration under Noise)** | First study of calibration under label noise | **WEAK / OVERSTATED** | Refocus strictly on calibration repair on **corrupted validation sets**. |
| **Gap C (Memorisation Bifurcation)** | Novel capacity memorisation bifurcation | **NOT A GAP (Solved)** | Demoted to baseline diagnostic dimension. |
| **Gap D (Synthetic vs Real Noise)** | Novel discovery that matrix methods fail on human noise | **NOT A GAP (Solved)** | Demoted to standard validation benchmark. |
