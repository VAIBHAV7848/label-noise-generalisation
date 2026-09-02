# Hypotheses Audit: Critical Adversarial Analysis (H1–H4)

This document provides a line-by-line adversarial critique of each hypothesis formulated in Phase 0.

---

## 1. Audit of Hypothesis H1: Decision Boundary Angular Deviation vs Condition Number $\kappa(T)$

- **Original Statement**:
  > *"Standard Empirical Risk Minimization with cross-entropy loss under asymmetric label noise $T$ produces an asymptotic decision boundary shift whose angular deviation from the optimal Bayes decision boundary is strictly bounded by the condition number $\kappa(T)$ and the minimum class separation margin $\gamma$."*
- **Mathematical Critique & Counterexample**:
  1. *Under-specification*: "Angular deviation" is only defined for linear hyperplanes $w^\top x + b = 0$. For non-linear models (Decision Trees, MLPs, ResNets), decision boundaries are non-linear manifolds where a single angular normal vector does not exist.
  2. *Fatal Counterexample for Condition Number*:
     - Consider binary classification with balanced priors ($P(Y=1) = P(Y=2) = 0.5$).
     - Let $T_{\text{sym}} = \begin{bmatrix} 0.7 & 0.3 \\ 0.3 & 0.7 \end{bmatrix}$. The eigenvalues are $\lambda_1 = 1.0, \lambda_2 = 0.4$. The condition number is $\kappa(T_{\text{sym}}) = 1.0 / 0.4 = 2.5$.
     - Under $T_{\text{sym}}$, $P(\tilde{Y}=1 \mid x) = 0.7 P(Y=1 \mid x) + 0.3 P(Y=2 \mid x)$. The decision boundary where $P(\tilde{Y}=1 \mid x) = P(\tilde{Y}=2 \mid x)$ occurs exactly where $P(Y=1 \mid x) = P(Y=2 \mid x)$. **The decision boundary shift is EXACTLY ZERO degrees.**
     - Now let $T_{\text{asym}} = \begin{bmatrix} 1.0 & 0.0 \\ 0.6 & 0.4 \end{bmatrix}$. Here $\kappa(T_{\text{asym}}) = 2.5$.
     - Under $T_{\text{asym}}$, the decision boundary where $P(\tilde{Y}=1 \mid x) = P(\tilde{Y}=2 \mid x)$ shifts significantly into class 2.
     - Both matrices have identical condition number $\kappa(T) = 2.5$, but one produces 0 shift while the other produces a large shift!
  3. *Conclusion*: Condition number $\kappa(T)$ does **NOT** control decision boundary shift. Boundary shift is governed by **noise asymmetry** (e.g. $\|T - T^\top\|_F$ or $|T_{12} - T_{21}|$) and class imbalance, not spectral conditioning.
- **Verdict**: **[REJECT & REPLACE]**.
- **Replacement Hypothesis (H1-Revised)**:
  > *"Under linear surrogate risk minimization, the displacement of the empirical decision boundary from the clean Bayes optimal boundary is bounded by the noise asymmetry measure $\|T - T^\top\|_F / \min_i T_{ii}$ and the margin distribution, whereas symmetric noise ($\|T - T^\top\|_F = 0$) induces zero asymptotic boundary shift for balanced classes."*

---

## 2. Audit of Hypothesis H2: Capacity-Dependent Memorisation Bifurcation

- **Original Statement**:
  > *"Overparameterized models exhibit a sharp bifurcation epoch $\tau_{\text{crit}}$ before which clean patterns dominate empirical risk minimization and after which memorisation of noisy labels accelerates exponentially, whereas underparameterized models exhibit smooth, non-bifurcated performance degradation."*
- **Literature Critique**:
  1. *Known Art*: Arpit et al. (ICML 2017) and Li et al. (ICML 2020) already mathematically derived and empirically established the existence of $\tau_{\text{crit}}$. Presenting this as an unverified hypothesis is redundant.
  2. *Vagueness of "Sharp Bifurcation"*: "Bifurcation" is dynamical systems terminology that was used loosely without a formal bifurcation parameter or Lyapunov exponent.
- **Verdict**: **[MODIFY]**.
- **Revised Hypothesis (H2-Revised)**:
  > *"The duration of the clean generalisation window $\Delta \tau = \tau_{\text{memorize}} - \tau_{\text{learn}}$ shrinks inversely with parameter overparameterization ratio $p/N$, and loss correction methods preserve test accuracy by arresting the gradient norm on mislabeled samples rather than delaying $\tau_{\text{memorize}}$."*

---

## 3. Audit of Hypothesis H3: Calibration Degradation & Robust Losses

- **Original Statement**:
  > *"While robust losses and loss-correction methods improve top-1 classification accuracy on noisy datasets, they systematically inflate Expected Calibration Error (ECE) and produce over-confident incorrect predictions unless explicit confidence penalties or post-hoc temperature scaling are incorporated."*
- **Literature Critique**:
  1. *Directional Flaw*: Recent literature (Lukasik et al. 2020, Wang et al. 2021) shows that robust losses like GCE ($L_q$) often suffer from *underconfidence* (not overconfidence) due to gradient truncation on hard classes.
  2. *Validation Split Confounder*: Post-hoc temperature scaling requires a validation split. In noisy learning, whether the validation split is clean or corrupted fundamentally changes whether ECE can be repaired.
- **Verdict**: **[MODIFY]**.
- **Revised Hypothesis (H3-Revised)**:
  > *"Robust loss functions and uncorrected empirical risk minimization exhibit distinct miscalibration profiles under label noise: standard cross-entropy produces overconfident errors driven by noise memorisation, while bounded symmetric losses produce underconfident predictions on hard classes; furthermore, post-hoc Temperature Scaling fitted on corrupted validation sets achieves sub-optimal calibration recovery compared to in-training confidence regularization."*

---

## 4. Audit of Hypothesis H4: Synthetic vs Real Human Noise Generalisation Gap

- **Original Statement**:
  > *"Methods designed specifically for class-conditional transition matrix correction suffer a significantly larger relative performance drop when evaluated on real-world human label noise (CIFAR-10N) than sample-selection and semi-supervised approaches due to the presence of instance-dependent label ambiguity."*
- **Methodological Critique**:
  1. *Confounding Variables*: DivideMix employs MixUp data augmentation and dual networks. Comparing vanilla Forward Correction against DivideMix conflates semi-supervised data augmentation with noise handling.
  2. *Causality*: Without controlling for data augmentation and network architecture, performance differences cannot be solely attributed to "instance-dependent noise".
- **Verdict**: **[MODIFY]**.
- **Revised Hypothesis (H4-Revised)**:
  > *"When controlled for identical data augmentation (MixUp) and backbone capacity, class-conditional transition matrix methods experience an excess generalization degradation on CIFAR-10N compared to feature-cluster filtering methods, proportional to the degree of instance-dependent noise variance across the input feature manifold."*
