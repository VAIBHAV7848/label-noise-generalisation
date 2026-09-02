# Claim-to-Evidence Verification Map

This document establishes the bidirectional traceability matrix mapping every paper claim to its theoretical proof or empirical validation experiment.

---

## 1. Claim-to-Evidence Matrix

| Claim ID | Core Scientific Claim | Theoretical Foundation / Proposition | Required Empirical Evidence | Falsification Criteria | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CLM-01** | Backward loss correction is an unbiased estimator of clean risk under known invertible transition matrix $T$. | **Proposition 1** (`02_THEORY/propositions.md`), Proof in `02_THEORY/proofs/forward_backward_unbiasedness.md`. | Verified analytically via Law of Total Probability and inner product expansion. | If $\mathbb{E}_{\tilde{\mathcal{D}}}[\tilde{\ell}] \ne \mathbb{E}_{\mathcal{D}}[\ell]$. | **Proven Analytically** |
| **CLM-02** | Excess risk under imperfect matrix $\hat{T}$ scales proportionally to $\|T^{-1}\|_2^2 \|\hat{T} - T\|_F$. | **Proposition 2** (`02_THEORY/propositions.md`). | Ablation Axis 1 (`04_EXPERIMENTS/ablations.md`): Perturbation experiment measuring accuracy vs $\|\hat{T} - T\|_F$. | If empirical risk degradation fails to correlate with Frobenius error or condition number $\kappa(T)$. | **Theoretical Derivation Complete; Empirical Test Pending Phase 2** |
| **CLM-03** | Overparameterized models exhibit an early-learning peak followed by memorisation, whereas underparameterized models degrade monotonically without early peak. | Theoretical memorisation dynamics (Arpit et al. 2017). | Multi-capacity benchmark (`04_EXPERIMENTS/experimental_protocol.md`): Epoch-by-epoch tracking on Logistic Regression vs MLP vs ResNet. | If Logistic Regression shows early validation accuracy peak or if ResNet does not memorize noise. | **Empirical Test Pending Phase 2** |
| **CLM-04** | Robust losses and loss correction systematically degrade confidence calibration (ECE) compared to clean data baseline. | **Proposition 4** (`02_THEORY/propositions.md`). | 15-bin Reliability Diagrams, ECE, and Brier Score evaluation on CIFAR-10/100 across 5 seeds. | If robust loss models achieve equal or lower ECE without post-hoc calibration intervention. | **Theoretical Lower Bound Established; Empirical Test Pending Phase 2** |
| **CLM-05** | Transition-matrix-based methods suffer a larger relative performance drop on human noise (CIFAR-10N) than sample-selection methods. | Instance-dependent noise literature (Wei et al. 2022). | Head-to-head comparison table on CIFAR-10N (Clean, Aggregate, Random, Worst) across all 10 baselines. | If matrix-corrected models outperform sample selection on CIFAR-10N Worst noise. | **Empirical Test Pending Phase 2** |
