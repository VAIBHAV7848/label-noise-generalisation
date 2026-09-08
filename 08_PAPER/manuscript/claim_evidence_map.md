# Claim-to-Evidence Traceability Matrix (Manuscript Audit)

This document provides a strict, bidirectional mapping between every substantive scientific claim in the manuscript and its mathematical theorem or empirical evidence derived from the frozen 84-run benchmark.

---

## 1. Primary Theoretical Claims

| Claim ID | Formal Scientific Claim | Formal Theory / Proposition | Mathematical Scope & Assumptions | Verification Status | Support Level |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **THM-01** | Backward loss correction $\vec{\tilde{\ell}} = T^{-1} \vec{\ell}$ is an unbiased estimator of clean risk under known invertible transition matrix $T$. | **Proposition 1** (`02_THEORY/propositions.md`) | Row-stochastic $T \in [0, 1]^{K \times K}$, invertible ($\det(T) \ne 0$), finite base loss expectation. | Derived analytically in `proofs/forward_backward_unbiasedness.md`. | **PROVEN** |
| **THM-02** | Expected risk estimation bias under imperfect transition matrix $\hat{T}$ is bounded by $\frac{\sqrt{K} M \|T^{-1}\|_2^2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon}$. | **Proposition 2A** (`02_THEORY/propositions.md`) | $M$-bounded base loss ($0 \le \ell \le M$), $\|E\|_F = \|\hat{T} - T\|_F \le \epsilon < 1/\|T^{-1}\|_2$. | Proved via Neumann series inversion perturbation bound. | **PROVEN** |
| **THM-03** | Finite-sample clean excess risk of empirical risk minimizer $\hat{f}$ scales with matrix estimation error $\epsilon$, condition number $\kappa(T)$, and sample complexity. | **Proposition 2B** (`02_THEORY/propositions.md`) | $M$-bounded, $L_{\ell, 2}$-Lipschitz loss, strictly under independent sample splitting $S_T \perp S_R$. Applies to Category A losses (MAE, Dense Forward $T_{\min}>0$); excludes same-sample CL heuristics without sample splitting. | Proved via Rademacher symmetrization and bounded difference concentration. | **PARTIALLY SUPPORTED (THEORY)** |
| **THM-04** | Convex multi-class surrogate losses cannot be simultaneously classification-calibrated and symmetric ($K \ge 3$). | **Proposition 3** (`02_THEORY/propositions.md`) | $K \ge 3$, strictly convex surrogate loss on simplex. | Derived from Charoenphakdee et al. (ICML 2019) theorem. | **PROVEN (PRIOR WORK)** |
| **THM-05** | Vector Calibration Error (VCE) under transition noise is exactly governed by $\text{VCE}_{\tilde{\mathcal{D}}}(f) = \mathbb{E}_{\mathbf{p}}[\|(T^\top - I)\mathbf{p}\|_1]$. | **Theorem 4A & 4B** (`02_THEORY/propositions.md`) | Class-conditional transition matrix $T$, clean calibrated model. | Proved via $\ell_1$ norm expansion of corrupted posterior expectation. | **PROVEN** |

---

## 2. Empirical Benchmark Claims

| Claim ID | Empirical Claim | Evidence Source | Figures / Tables | Verified Quantitative Metric | Support Level |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EMP-01** | Exact transition matrix inversion substantially mitigates asymmetric noise degradation on PreActResNet-18. | Pilot Dataset (`frozen_pilot_results.csv`) | Table 1, Table 4, Figure 1 | Forward True $T$ achieves $87.80 \pm 0.25\%$ vs CE $81.62 \pm 0.71\%$ ($\Delta = +6.19\%$, $t=22.77$, uncorrected $p=0.0019$, Holm $p=0.0442$, Cohen's $d=13.15$). | **EMPIRICALLY SUPPORTED** |
| **EMP-02** | Robust loss without matrix inversion (GCE) outperforms matrix correction under symmetric noise via dynamic gradient trimming, but degrades under asymmetric noise. | Pilot Dataset (`frozen_pilot_results.csv`) | Table 1, Table 4, Figure 1 | In Symmetric 50%, GCE achieves $82.40 \pm 0.42\%$ vs True $T$ $79.17 \pm 0.72\%$; in Asymmetric 40%, GCE degrades to $78.73 \pm 0.48\%$ ($\Delta = -2.88\%$ vs CE). | **EMPIRICALLY SUPPORTED** |
| **EMP-03** | Transition matrix ill-conditioning ($\kappa(\hat{T}) \gg 1$) severely degrades accuracy and inflates calibration error. | Pilot Dataset (`frozen_pilot_results.csv`) | Table 3, Figure 6 | Confident Learning in Asymmetric 40% has condition number $\kappa = 34.66 \pm 3.47$ ($\|\hat{T}^{-1}\|_2 = 33.13$), dropping accuracy to $82.58\%$ and inflating raw ECE to $0.1193$. | **EMPIRICALLY SUPPORTED** |
| **EMP-04** | Overparameterized models exhibit an early generalisation peak followed by memorisation decay, which is arrested by loss correction and GCE. | Training History Logs (`frozen_pilot_results.json`) | Figure 9, Figure 10, Table 1 | In Symmetric 50%, CE clean val accuracy drops by $2.99\%$ from peak (epoch 23.3); True $T$ reduces drop to $1.00\%$, and GCE reduces it to $0.11\%$. | **EMPIRICALLY SUPPORTED** |
| **EMP-05** | Tuning post-hoc Temperature Scaling on corrupted validation sets degrades clean calibration across noise regimes. | Pilot Dataset (`frozen_pilot_results.csv`) | Table 2, Table 4, Figure 8 | Corrupted TS ECE is worse than Clean TS ECE in CE Sym 20% by $+16.63\%$ ($p=0.0276$) and in True $T$ Sym 50% by $+8.92\%$ ($p=0.0008$). | **EMPIRICALLY SUPPORTED** |
| **EMP-06** | Symmetric Cross-Entropy (SCE) exhibits severe probability underconfidence under label noise. | Pilot Dataset (`frozen_pilot_results.csv`) | Table 2, Figure 2, Figure 7 | Raw ECE of SCE reaches $0.3378 \pm 0.0190$ in Sym 50% and $0.2924 \pm 0.0349$ in Asym 40%; optimal temperature $T^* > 2.0$. | **EMPIRICALLY SUPPORTED** |

---

## 3. Pre-Registered Hypotheses & Research Questions

| ID | Stated Hypothesis / Question | Pre-Registered Falsification Criteria | Empirical Verdict | Exact Evidence / Rationale | Support Level |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **H1** | Asymmetry-driven decision boundary displacement: Noise asymmetry $\|T - T^\top\|_F$ drives boundary shift, requiring matrix inversion. | If matrix correction provides no benefit over symmetric robust losses on asymmetric noise. | **Partially Supported** | True $T$ gains $+6.19\%$ ($p=0.0019$) over CE on Asym 40%, whereas GCE drops $-2.88\%$. Linear hyperplane angular shift awaits 2D synthetic ablation. | **PARTIALLY SUPPORTED** |
| **H2** | Parametric generalisation window contracts with capacity $p/N$; loss correction arrests memorisation decay. | If loss correction exhibits identical memorisation drop as CE. | **Partially Supported** | Memorisation drop is arrested from $2.99\%$ (CE) to $1.00\%$ (True $T$) and $0.11\%$ (GCE). Contraction rate across variable capacity awaits multi-architecture grid. | **PARTIALLY SUPPORTED** |
| **H3** | Robust losses exhibit divergent miscalibration; corrupted-validation TS degrades clean reliability. | If corrupted-validation TS achieves equal or lower ECE than clean-validation TS. | **Supported** | Corrupted validation TS degrades clean ECE across noisy regimes (up to $+16.63\%$). Falsification condition is nowhere met. | **EMPIRICALLY SUPPORTED** |
| **H4** | Transition matrix methods suffer excess degradation on human noise (CIFAR-10N) vs sample filtering. | If transition matrix methods match or beat sample filtering on CIFAR-10N Worst. | **Out of Scope / Future Work** | Manuscript is strictly scoped to class-conditional synthetic noise benchmarks. Real-world instance-dependent noise evaluation deferred to future work. | **OUT OF SCOPE / FUTURE WORK** |
| **SRQ1** | Excess risk bounds vs Frobenius error $\|\hat{T} - T\|_F$ and condition number $\kappa(T)$. | Correlation analysis and empirical bounds. | **Empirically Quantified** | Validated Proposition 2: Ill-conditioning ($\kappa=34.66$) severely impairs downstream risk. | **EMPIRICALLY SUPPORTED** |
| **SRQ2** | Calibration recovery penalty from corrupted validation tuning. | $\Delta \text{ECE} = \text{ECE}_{\text{corr}} - \text{ECE}_{\text{clean}}$. | **Empirically Resolved** | Quantified significant ECE penalty ($+5.16\%$ to $+16.63\%$). | **EMPIRICALLY SUPPORTED** |
| **SRQ3** | Scaling of generalisation window $\Delta \tau$ with capacity ratio $p/N$. | Tracking epoch dynamics. | **Partially Resolved** | Peak identified at epoch $22.3 \pm 2.5$ for PreActResNet-18; multi-capacity grid pending. | **PARTIALLY SUPPORTED** |
| **SRQ4** | Controlled transfer gap to real human noise on CIFAR-10N. | Head-to-head comparison on CIFAR-10N. | **Out of Scope** | Synthetic benchmark focus; CIFAR-10N human noise deferred to future study. | **OUT OF SCOPE / FUTURE WORK** |

