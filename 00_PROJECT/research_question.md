# Research Questions & Evaluation (Audited & Refined)

## 1. Context & Scope Refinement

Following the Phase 1 Research Integrity Audit, the research question is sharpened to focus on the verified research gaps:
1. **Mathematical generalisation guarantees under transition matrix estimation error $\|\hat{T} - T\|_F$**.
2. **Confidence calibration dynamics and post-hoc recovery when validation sets are corrupted**.
3. **Controlled synthetic-to-human noise generalisation gap**.

---

## 2. Refined Primary Research Question

> **Primary RQ:**
> *"Under what exact theoretical bounds and empirical conditions do loss-correction methods and robust losses maintain probability calibration (ECE) and excess risk guarantees when subject to transition matrix estimation error and corrupted validation distributions across varying classifier capacities?"*
>
> **Sub-Questions (SRQs):**
> - **SRQ 1 (Theoretical Excess Risk Bounds):** How does the finite-sample matrix estimation error $\epsilon = \|\hat{T} - T\|_F$ and transition condition number $\kappa(T)$ quantitatively bound the excess risk $\mathcal{E}(\hat{f}) = R(\hat{f}) - R(f^*)$ for multi-class backward loss correction?
> - **SRQ 2 (Calibration Recovery on Corrupted Validation Sets):** How severely does tuning post-hoc Temperature Scaling on corrupted validation sets degrade clean test ECE compared to in-training robust losses (GCE, SCE)?
> - **SRQ 3 (Parametric Generalisation Window):** How does the duration of the clean generalisation window $\Delta \tau$ scale with overparameterization ratio $p/N$ under asymmetric label noise?
> - **SRQ 4 (Controlled Human Noise Transfer Gap):** When controlling for data augmentation (MixUp) and model capacity, what is the exact performance delta between class-conditional transition matrix correction and sample-filtering methods on CIFAR-10N?

---

## 3. Narrower Alternative Research Questions (Ranked)

| Rank | Alternative RQ Candidate | Novelty | Theoretical Depth | Empirical Feasibility | Reviewer Defensibility | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Refined Primary Composite RQ (Above)** | **High** | **High** | **High (MDES 300 runs)** | **Very High** | **PRIMARY** |
| **2** | *Post-Hoc vs In-Training Calibration under Noisy Validation Sets* | Medium-High | Medium | Very High (120 runs) | High | Fallback A |
| **3** | *Non-Asymptotic Excess Risk Bounds under Operator Perturbations in Noisy ERM* | High | Very High | Low-Medium (Theory) | High | Fallback B |
| **4** | *Controlled Confounder Benchmark of Synthetic vs Real Human Noise on CIFAR-10N* | Medium | Low-Medium | High (150 runs) | Medium | Diagnostic |
