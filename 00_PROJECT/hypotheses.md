# Formal Research Hypotheses

Each hypothesis is formulated to be statistically or mathematically falsifiable.

---

## Hypothesis 1 (H1: Risk Bias & Decision Boundary Shift)
> **Statement**: Standard Empirical Risk Minimization with cross-entropy loss under asymmetric label noise $T$ produces an asymptotic decision boundary shift whose angular deviation from the optimal Bayes decision boundary is strictly bounded by the condition number $\kappa(T)$ and the minimum class separation margin $\gamma$.
> - **Test Mechanism**: Mathematical derivation of the hyperplane angle under linear models + empirical measurement of angular discrepancy between $\mathbf{w}^*_{\text{clean}}$ and $\mathbf{w}^*_{\text{noisy}}$ as a function of noise asymmetry.
> - **Falsification Condition**: If angular deviation does not increase monotonically with $\kappa(T)$ or remains zero under non-diagonal asymmetric transition matrices.

---

## Hypothesis 2 (H2: Capacity-Dependent Memorisation Threshold)
> **Statement**: Overparameterized models (where parameter count $p \gg N$) exhibit a sharp bifurcation epoch $\tau_{\text{crit}}$ before which clean patterns dominate empirical risk minimization and after which memorisation of noisy labels accelerates exponentially, whereas underparameterized models (Logistic Regression, Decision Trees) exhibit smooth, non-bifurcated performance degradation without an early-learning peak.
> - **Test Mechanism**: Epoch-by-epoch tracking of clean validation accuracy vs noisy training accuracy across model capacity spectrum (Logistic Regression $\rightarrow$ 2-layer MLP $\rightarrow$ PreAct-ResNet18).
> - **Falsification Condition**: If overparameterized models do not show an early peak in validation accuracy prior to memorisation, or if underparameterized models show identical sharp epoch bifurcation dynamics.

---

## Hypothesis 3 (H3: Calibration Degradation & Restoration)
> **Statement**: While robust losses (e.g., GCE, SCE) and loss-correction methods (Backward/Forward) improve top-1 classification accuracy on noisy datasets, they systematically inflate Expected Calibration Error (ECE) and produce over-confident incorrect predictions unless explicit confidence penalties or post-hoc temperature scaling are incorporated.
> - **Test Mechanism**: Compute 15-bin Reliability Diagrams, ECE, AdaECE, and Brier Score for all baselines across noise rates $\eta \in [0.0, 0.8]$.
> - **Falsification Condition**: If robust loss functions or loss-corrected models maintain equal or lower ECE compared to clean baseline models without additional calibration intervention.

---

## Hypothesis 4 (H4: Synthetic-to-Real Generalisation Gap)
> **Statement**: Methods designed specifically for class-conditional transition matrix correction (e.g., Forward Correction, Dual-T) suffer a significantly larger relative performance drop when evaluated on real-world human label noise (CIFAR-10N) than sample-selection and semi-supervised approaches (e.g., DivideMix, Confident Learning) due to the presence of instance-dependent (feature-conditioned) label ambiguity.
> - **Test Mechanism**: Paired comparison of synthetic asymmetric noise vs CIFAR-10N (Worst/Random) test accuracy and ranking shifts across all baselines.
> - **Falsification Condition**: If transition-matrix-based methods achieve equivalent or superior performance on human noise compared to semi-supervised filtering methods.
