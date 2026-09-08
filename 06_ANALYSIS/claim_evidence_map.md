# Claim-to-Evidence Verification Map

This document establishes the bidirectional traceability matrix mapping every paper claim to its theoretical proof or empirical validation experiment.

---

## 1. Claim-to-Evidence Matrix

| Claim ID | Core Scientific Claim | Theoretical Foundation / Proposition | Required Empirical Evidence | Falsification Criteria | Verification Status | Support Level |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CLM-01** | Backward/Forward loss correction is an unbiased estimator of clean risk under known invertible transition matrix $T$. | **Proposition 1** (`02_THEORY/propositions.md`), Proof in `02_THEORY/proofs/forward_backward_unbiasedness.md`. | Verified analytically and empirically via ForwardCorrection_TrueT (+6.19% gain on Asym 0.4, $p=0.0019$). | If $\mathbb{E}_{\tilde{\mathcal{D}}}[\tilde{\ell}] \ne \mathbb{E}_{\mathcal{D}}[\ell]$. | **Analytically Proven & Empirically Validated (Pilot)** | **PROVEN** |
| **CLM-02** | Excess risk under imperfect matrix $\hat{T}$ scales with condition number $\|T^{-1}\|_2^2$ and estimation error $\epsilon$. | **Proposition 2A & 2B** (`02_THEORY/propositions.md`). | Evaluated across TrueT, AnchorT, ConfidentLearningT, BadT. Holds for Prop 2B strictly under independent sample splitting ($S_T \perp S_R$). | If empirical risk degradation fails to correlate with Frobenius error or condition number $\kappa(T)$. | **Empirically Quantified in Pilot ($\kappa=34.66$ in Asym drives ECE to 0.1193; same-sample CL is heuristic)** | **PARTIALLY SUPPORTED (THEORY)** |
| **CLM-03** | Overparameterized models exhibit an early-learning peak followed by memorisation decay, arrested by loss correction. | Theoretical memorisation dynamics (Arpit et al. 2017). | Tracking validation dynamics: PreActResNet-18 exhibits 2.99% (Sym 0.5) and 2.29% (Asym 0.4) memorization decay, arrested by TrueT/GCE. | If ResNet does not memorize noise or if loss correction fails to arrest decay. | **Memorization Decay Arrest Confirmed in Pilot; Multi-Capacity Grid Pending Future Work** | **PARTIALLY SUPPORTED** |
| **CLM-04** | Corrupted validation sets trap post-hoc Temperature Scaling, degrading clean confidence calibration. | **Proposition 4** (`02_THEORY/propositions.md`). | ECE, AdaECE, Brier score, and post-hoc Temperature Scaling across clean vs corrupted validation sets. | If corrupted validation TS achieves equal or lower ECE than clean validation TS. | **Empirically Confirmed in Pilot (Corrupted TS degrades ECE by up to +16.63%; SCE ECE reaches 0.338)** | **EMPIRICALLY SUPPORTED** |
| **CLM-05** | Transition-matrix-based methods suffer excess degradation on human noise (CIFAR-10N). | Instance-dependent noise literature (Wei et al. 2022). | Empirical evaluation on CIFAR-10N human noise benchmark. | If matrix-corrected models match or outperform sample selection on CIFAR-10N Worst. | **Out of Scope / Future Work (Synthetic Class-Conditional Scope Preserved)** | **OUT OF SCOPE / FUTURE WORK** |

