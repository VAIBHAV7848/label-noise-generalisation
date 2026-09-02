# Key Research Decisions (Phase 0)

This document formalizes foundational architectural and scientific decisions established during Phase 0.

---

### Decision 001: Expanded Model Capacity Spectrum
- **Date**: 2026-09-02
- **Decision**: Include deep convolutional architectures (PreAct-ResNet18, ResNet-50) alongside shallow classical models (Logistic Regression, Decision Tree, 2-layer MLP).
- **Rationale**: Prior literature (Zhang et al. 2017, Arpit et al. 2017, Song et al. 2022) demonstrates that overparameterized networks behave fundamentally differently under label noise (memorisation, early-learning phases) than shallow convex classifiers. Evaluating only classical models renders findings non-generalizable to modern deep learning.
- **Alternatives Considered**: Only evaluating Logistic Regression, Decision Tree, and 2-layer MLP.
- **Why Rejected**: Desk-rejection risk at Tier-1 venues due to outdated empirical baseline.

---

### Decision 002: Inclusion of Calibration as a Primary Evaluation Metric
- **Date**: 2026-09-02
- **Decision**: Make Expected Calibration Error (ECE), Adaptive ECE, and Brier Score core primary metrics alongside Top-1 Clean Test Accuracy.
- **Rationale**: The literature overwhelmingly focuses on top-1 accuracy under label noise, leaving confidence calibration largely neglected. Miscalibration under label noise is a major safety failure mode in deployed ML systems.
- **Alternatives Considered**: Reporting Top-1 and Top-5 accuracy only.
- **Why Rejected**: High competition and saturation on pure accuracy benchmarks; measuring calibration creates a distinct, high-impact scientific contribution.

---

### Decision 003: Inclusion of Real-World Human Noise Benchmark (CIFAR-10N / Animal-10N)
- **Date**: 2026-09-02
- **Decision**: Benchmark both synthetic (symmetric & class-conditional) and real-world crowdsourced human noise (CIFAR-10N, Animal-10N).
- **Rationale**: Synthetic noise models assume noise is conditionally independent of input features $X$ given true label $Y$ ($P(\tilde{Y} \mid X, Y) = P(\tilde{Y} \mid Y)$). Real-world noise is intrinsically instance-dependent ($P(\tilde{Y} \mid X, Y)$), as hard or ambiguous images have higher mislabeling probability (Wei et al., ICLR 2022).
- **Alternatives Considered**: Only synthetic random label flips.
- **Why Rejected**: Reviewers frequently penalize papers that only test synthetic uniform or class-conditional noise.

---

### Decision 004: Strict Reproducibility and Multi-Seed Statistical Protocol
- **Date**: 2026-09-02
- **Decision**: All experiments must use 5 independent random seeds with fixed random seeds for dataset corruptions, parameter initializations, and mini-batch sampling. All reported metrics must include mean $\pm$ standard deviation and two-tailed paired t-tests / Wilcoxon signed-rank tests ($p < 0.05$).
- **Rationale**: Machine learning papers with single-seed results or overlapping confidence intervals lack statistical rigor.
- **Alternatives Considered**: 3 seeds or single seed runs.
- **Why Rejected**: 3 seeds are insufficient for robust statistical significance testing.
