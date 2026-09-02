# Research Objectives

## 1. Primary Objective

To establish a unified theoretical and empirical framework that quantifies how label noise degrades classifier generalisation, empirical risk convergence, and probability calibration across classifier families, while rigorously auditing the robustness limits and failure modes of loss-correction, robust-loss, and sample-selection paradigms.

---

## 2. Specific Theoretical Objectives

- **TO 1 (Excess Risk Bounds under Transition Matrix Misspecification)**:
  Derive formal bounds on the excess risk $\mathcal{E}(f) = R(f) - R(f^*)$ under empirical risk minimization with estimated noise transition matrices $\hat{T}$, quantifying how estimation error $\| \hat{T} - T \|_F \le \epsilon$ translates into generalisation error.

- **TO 2 (Noise Tolerance Conditions for Multi-Class Loss Functions)**:
  Formalize the necessary and sufficient mathematical conditions under which surrogate loss functions $\ell$ are strictly noise-tolerant (classification-calibrated under symmetric vs asymmetric noise distributions).

- **TO 3 (Calibration Error Bounds under Label Corruption)**:
  Mathematically analyze how label flipping distorts the true posterior probability $P(Y=k \mid X=x)$ into corrupted posterior $P(\tilde{Y}=k \mid X=x) = \sum_j T_{jk} P(Y=j \mid X=x)$, and formalize the theoretical lower bound on Expected Calibration Error (ECE) for uncorrected empirical risk minimizers.

---

## 3. Specific Empirical Objectives

- **EO 1 (Multi-Family Model Capacity Spectrum Evaluation)**:
  Evaluate the empirical generalisation and memorisation curves across a spectrum of model complexities:
  - Linear/Convex: Multi-Class Logistic Regression
  - Non-parametric / Rule-based: Decision Trees / Random Forests
  - Shallow Neural: Multi-Layer Perceptron (2-layer MLP)
  - Deep / Overparameterized: ResNet-18 / ResNet-50 / PreAct-ResNet18

- **EO 2 (Unified Benchmark of Noise-Mitigation Paradigms)**:
  Execute a rigorous head-to-head empirical comparison across major noise-handling families:
  1. *Uncorrected Baselines*: Standard Cross-Entropy (CE), Label Smoothing (LS).
  2. *Robust Losses*: Mean Absolute Error (MAE), Generalized Cross-Entropy (GCE), Symmetric Cross-Entropy (SCE), Normalized Cross-Entropy + Reverse CE (NCE+RCE).
  3. *Loss Correction*: Forward Correction (Patrini et al., 2017), Backward Correction (Natarajan et al., 2013), Dual-T Estimator (Xia et al., 2019).
  4. *Sample Selection / Semi-Supervised*: Co-teaching (Han et al., 2018), Confident Learning (Northcutt et al., 2021), DivideMix (Li et al., 2020).

- **EO 3 (Comprehensive Multi-Metric Evaluation Matrix)**:
  Evaluate all methods not only on standard clean test classification accuracy, but simultaneously across:
  - Top-1 & Top-5 Clean Test Accuracy (%)
  - Expected Calibration Error (ECE) & Adaptive ECE (AdaECE)
  - Brier Score (mean squared error of probability forecasts)
  - Transition Matrix Estimation Frobenius Norm Error $\| \hat{T} - T \|_F$
  - Effective Training Stability / Convergence Speed (epochs to peak validation accuracy)

- **EO 4 (Synthetic vs Real-World Noise Transfer Audit)**:
  Measure the performance drop when transitioning from idealized class-conditional synthetic noise (CIFAR-10 / CIFAR-100 with known transition matrix $T$) to real-world human annotator noise (CIFAR-10N, Animal-10N, Clothing1M).

---

## 4. Definition of Done for Phase 0

- Complete literature matrix with 25+ peer-reviewed papers cataloged and analyzed.
- Full mathematical problem formulation with clean vs noisy risk, transition matrix derivations, and proof of unbiasedness.
- Adversarial novelty audit identifying non-gaps and genuine research gaps.
- Full repository architecture created and synchronized to GitHub (`VAIBHAV7848/label-noise-generalisation`).
- Comprehensive Phase 0 Research Blueprint delivered and ready for formal review.
