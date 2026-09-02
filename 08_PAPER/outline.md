# Publication Manuscript Outline

**Target Venue**: NeurIPS / ICML / AISTATS / JMLR
**Format**: 9-page standard conference format (excluding references/appendices)

---

## Abstract (200–250 words)
- Problem statement & prevalence of label noise.
- Limitations of prior art (accuracy-only metrics, unquantified matrix misspecification, synthetic vs real gap).
- Key theoretical results: Excess risk bounds under $\|\hat{T} - T\|_F$.
- Key empirical findings: Multi-capacity generalisation spectrum, ECE calibration trade-offs, and CIFAR-10N human noise audit.
- Broader impact for reliable machine learning.

---

## 1. Introduction
- Motivation: Supervised learning with noisy annotations in real-world deployments.
- Core Challenges: Risk bias, memorisation dynamics, and confidence miscalibration.
- Summary of Contributions (Theoretical, Empirical, Diagnostic).

---

## 2. Related Work & Problem Formulation
- 2.1 Theoretical Foundations of Label Noise & Noise Tolerance.
- 2.2 Noise Transition Matrix Models ($T$) and Unbiased Risk Minimization.
- 2.3 Robust Losses, Sample Selection & Calibration under Corrupted Distributions.
- 2.4 Formal Mathematical Formulation: Clean vs. Corrupted Risk.

---

## 3. Theoretical Analysis: Generalisation & Risk Bounds
- 3.1 Unbiased Loss Correction Mechanics.
- 3.2 Proposition 1: Risk Unbiasedness under Invertible $T$.
- 3.3 Proposition 2: Excess Risk Bounds under Transition Matrix Misspecification ($\|\hat{T} - T\|_F$).
- 3.4 Proposition 3: Noise Tolerance Constraints of Convex Multi-Class Surrogates.
- 3.5 Proposition 4: Posterior Simplex Distortion and Calibration Lower Bounds.

---

## 4. Experimental Framework & Methodological Suite
- 4.1 Datasets: Synthetic (CIFAR-10/100, MNIST, Adult) and Real-World (CIFAR-10N, Animal-10N).
- 4.2 Model Capacity Spectrum: Logistic Regression, Decision Trees, 2-layer MLP, PreAct-ResNet18.
- 4.3 Baseline Taxonomy: CE, LS, MAE, GCE, SCE, Forward/Backward Correction, Dual-T, Cleanlab, DivideMix.
- 4.4 Evaluation Metrics: Clean Test Accuracy, ECE, AdaECE, Brier Score, $\|\hat{T} - T\|_F$.

---

## 5. Empirical Results & Analysis
- 5.1 Multi-Capacity Generalisation & Early Memorisation Dynamics.
- 5.2 Transition Matrix Estimation Quality vs. Downstream Accuracy.
- 5.3 The Accuracy-Calibration Trade-off across Noise-Mitigation Paradigms.
- 5.4 Synthetic-to-Real Transfer Gap: Why Class-Conditional Methods Struggle on Human Noise.

---

## 6. Ablation Studies & Sensitivity Analysis
- 6.1 Sensitivity to Matrix Perturbation $\sigma$.
- 6.2 Hyperparameter Grids ($q$ in GCE, $\alpha/\beta$ in SCE).
- 6.3 Post-hoc Temperature Scaling on Clean vs. Noisy Validation Sets.

---

## 7. Discussion, Limitations & Future Work
- Practical guidelines for practitioners.
- Limitations (instance-dependent noise theory, compute scaling).
- Conclusion.

---

## References & Appendices
- Full mathematical proofs.
- Complete 15-bin Reliability Diagrams.
- Extended hyperparameter tables and reproducibility protocol.
