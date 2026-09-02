# Formal Research Hypotheses (Audited & Revised)

Each hypothesis is formulated to be strictly falsifiable and controlled for confounding variables.

---

## Hypothesis 1 (H1-Revised: Asymmetry-Driven Decision Boundary Displacement)
> **Statement**: Under linear surrogate empirical risk minimization, the angular and spatial displacement of the empirical decision boundary from the optimal Bayes decision boundary is strictly bounded by the noise asymmetry metric $\frac{\|T - T^\top\|_F}{\min_i T_{ii}}$ and the clean margin distribution $\gamma$, whereas symmetric noise ($\|T - T^\top\|_F = 0$) induces zero asymptotic boundary shift under balanced class priors.
> - **Test Mechanism**: Mathematical derivation of the hyperplane angle under linear models + empirical measurement of angular discrepancy between $\mathbf{w}^*_{\text{clean}}$ and $\mathbf{w}^*_{\text{noisy}}$ as a function of noise asymmetry $\|T - T^\top\|_F$.
> - **Falsification Condition**: If non-zero boundary displacement is observed under symmetric noise with balanced priors, or if displacement under asymmetric noise fails to correlate with $\|T - T^\top\|_F$.

---

## Hypothesis 2 (H2-Revised: Parametric Generalisation Window & Capacity)
> **Statement**: The duration of the clean generalisation window $\Delta \tau = \tau_{\text{memorize}} - \tau_{\text{learn}}$ (measured in training epochs) contracts inversely with the overparameterization ratio $p/N$, and loss correction methods preserve test accuracy by arresting the gradient residual norm on mislabeled instances rather than merely delaying the onset epoch $\tau_{\text{memorize}}$.
> - **Test Mechanism**: Epoch-by-epoch tracking of gradient norms on clean vs noisy instances across varying hidden layer widths ($p \in [10^3, 10^7]$) on 2-layer MLP and PreAct-ResNet18.
> - **Falsification Condition**: If $\Delta \tau$ does not contract as parameter count $p$ increases, or if loss-corrected models exhibit identical gradient norms on corrupted instances as uncorrected cross-entropy.

---

## Hypothesis 3 (H3-Revised: Divergent Miscalibration Profiles & Corrupted Validation Recovery)
> **Statement**: Robust loss functions and uncorrected empirical risk minimization exhibit opposite miscalibration profiles under label noise: standard cross-entropy produces overconfident errors driven by noise memorisation, while bounded symmetric losses produce underconfident predictions on hard clean classes; furthermore, post-hoc Temperature Scaling fitted on corrupted validation sets achieves sub-optimal calibration recovery compared to in-training confidence regularization.
> - **Test Mechanism**: Evaluate 15-bin Reliability Diagrams, ECE, AdaECE, and Brier score comparing Temperature Scaling fitted on 100% clean vs noise-corrupted validation splits.
> - **Falsification Condition**: If Temperature Scaling tuned on corrupted validation sets achieves equal or lower ECE than when tuned on clean validation sets.

---

## Hypothesis 4 (H4-Revised: Controlled Confounder Audit on Human Noise)
> **Statement**: When strictly controlled for identical data augmentation (MixUp) and backbone capacity, class-conditional transition matrix methods experience an excess generalization degradation on CIFAR-10N compared to feature-cluster filtering methods, proportional to the degree of instance-dependent noise variance across the input feature manifold.
> - **Test Mechanism**: Controlled head-to-head comparison on CIFAR-10N (Clean, Aggregate, Worst) comparing Forward Correction + MixUp vs Confident Learning + MixUp on identical PreAct-ResNet18 backbones.
> - **Falsification Condition**: If transition-matrix-based methods match or outperform sample-filtering methods on CIFAR-10N Worst noise when both utilize identical MixUp augmentation.
