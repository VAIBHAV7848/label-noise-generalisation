# Research Roadmap

```mermaid
gantt
    title Phased Research Execution Roadmap
    dateFormat  YYYY-MM-DD
    section Phase 0: Foundation
    Literature Matrix & Gap Audit       :done, 2026-09-02, 1d
    Mathematical Problem Formulation   :done, 2026-09-02, 1d
    Research Blueprint & Architecture  :done, 2026-09-02, 1d
    section Phase 1: Core Framework
    Theoretical Derivations & Proofs   :active, 2026-09-03, 3d
    Data Pipeline & Noise Generators   :2026-09-06, 3d
    Model Architectures & Baselines    :2026-09-09, 4d
    section Phase 2: Pilot & Empirical
    Pilot Experiments & Sanity Checks  :2026-09-13, 3d
    Full Synthetic Benchmark Runs      :2026-09-16, 5d
    Real-World Human Noise Benchmark   :2026-09-21, 4d
    section Phase 3: Analysis & Calibration
    Calibration (ECE/Brier) Analysis   :2026-09-25, 3d
    Statistical Significance Auditing  :2026-09-28, 2d
    section Phase 4: Review & Manuscript
    Adversarial Novelty Review         :2026-09-30, 2d
    Manuscript Drafting (LaTeX)        :2026-10-02, 6d
    Camera-Ready & Code Release        :2026-10-08, 2d
```

---

## Phase Details

### Phase 0: Research Foundation & Repository Architecture (CURRENT)
- Exhaustive literature matrix & source notes.
- Theoretical formulation of clean vs noisy risk, transition matrix $T$, forward/backward correction.
- Adversarial novelty audit and risk analysis.
- Phase 0 Research Blueprint.

### Phase 1: Mathematical Foundations & Pipeline Implementation
- Formal verification of theoretical propositions.
- Modular Python source implementation:
  - Noise injection modules (Symmetric, Asymmetric, Pair-flip, Instance-dependent).
  - Transition matrix estimators (Anchor point, Dual-T, Confident Learning).
  - Robust loss function implementations (GCE, SCE, MAE, NCE+RCE).
  - Loss correction layers (Forward, Backward).
  - Sample-selection baselines (Co-teaching, DivideMix).

### Phase 2: Pilot Validation & Comprehensive Benchmarking
- Pilot sanity checks on toy synthetic 2D datasets and MNIST.
- Full experimental execution on CIFAR-10, CIFAR-100, CIFAR-10N, Animal-10N across 5 seeds.

### Phase 3: Calibration, Dynamics & Statistical Verification
- Deep analysis of memorisation curves, early learning inflection points, and transition matrix Frobenius error.
- Comprehensive reliability diagram plotting and ECE calculation.
- Wilcoxon signed-rank and paired t-tests.

### Phase 4: Peer-Review Simulation & Manuscript Preparation
- Simulated peer reviews (NeurIPS / ICML format).
- Complete LaTeX manuscript drafting with figures and tables.
- Reproducibility verification and open-source release.
