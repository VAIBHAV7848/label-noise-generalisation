# Adversarial Novelty Audit

This document conducts an adversarial novelty review across all proposed concepts to ensure no solved problem or trivial variation is presented as novel.

---

## 1. Concept-by-Concept Novelty Categorization

| Proposed Concept / Claim | Direct Prior Art Search | Older & Adjacent Literature | Status Label | What is Genuinely Unresolved (Differentiator) |
| :--- | :--- | :--- | :--- | :--- |
| **Backward Loss Correction Unbiasedness** | Natarajan et al. (NeurIPS 2013), Patrini et al. (CVPR 2017) | Angluin & Laird (1988), Scott et al. (2013) | **Clearly Established (Solved)** | We treat this as an established theoretical theorem and analyze its perturbation sensitivity under finite-sample estimation error $\hat{T}$. |
| **Symmetric Loss Noise Tolerance Condition** | Charoenphakdee et al. (ICML 2019), van Rooyen et al. (NeurIPS 2015) | Manwani & Sastry (2013) | **Clearly Established (Solved)** | We adopt their definition to explain why MAE fails to optimize on complex datasets and why GCE/SCE trade convexity for drivability. |
| **Memorisation of Random Labels in Deep Nets** | Zhang et al. (ICLR 2017), Arpit et al. (ICML 2017) | Bartlett (1998), Neyshabur et al. (2015) | **Well Studied (Solved)** | We bridge this qualitative phenomenon with continuous capacity scaling ($p/N$ ratio) across classical vs deep models. |
| **Anchor-Point Transition Matrix Estimation** | Patrini et al. (CVPR 2017), Xia et al. (NeurIPS 2019) | Scott (2015), Blanchard et al. (2010) | **Well Studied (Solved)** | We benchmark anchor vs non-anchor estimators and measure their empirical Frobenius estimation error $\|\hat{T} - T\|_F$. |
| **Excess Risk Bounds under Matrix Misspecification ($\|\hat{T} - T\|_F > 0$)** | Natarajan et al. (2013) (assumes exact $T$ or asymptotic rates) | Bounded perturbation theory | **Plausible Research Gap** | Non-asymptotic excess risk bounds explicitly formulated as a function of the transition matrix condition number $\kappa(T)$ and Frobenius error. |
| **Probability Calibration Degradation under Noise-Mitigation Techniques** | Guo et al. (2017), Thulasidasan et al. (2019) | Platt (1999), Zadrozny & Elkan (2002) | **Plausible Research Gap (Underexplored)** | Comprehensive multi-metric calibration audit (ECE, AdaECE, Brier) across robust losses, loss correction, and sample selection under corrupted validation sets. |
| **Cross-Capacity Spectrum & Synthetic-to-Real Noise Transfer Gap** | Wei et al. (ICLR 2022), Song et al. (2022) | Xiao et al. (2015), Lee et al. (2018) | **Plausible Research Gap** | Systematic head-to-head empirical audit isolating why transition matrix methods fail on human noise compared to semi-supervised filtering. |

---

## 2. Explicit Rules for Manuscript Writing (Phase 4 Gate)

1. **Zero False Claims of Novelty**: The paper will never claim invention of loss correction, symmetric loss theory, early-learning dynamics, or anchor-point estimation.
2. **Explicit Attribution**: Every foundational equation (e.g. $\vec{\tilde{\ell}} = T^{-1} \vec{\ell}$, $\mathcal{L}_q = (1 - f_y^q)/q$, $\text{ECE}$) will cite its original primary author.
3. **Focus on Differentiators**: The paper’s claimed novel contributions are strictly restricted to:
   - Derivation of excess risk bounds under transition matrix misspecification (Proposition 2).
   - The first systematic calibration (ECE/Brier) and reliability audit across modern noise-mitigation paradigms.
   - Comprehensive multi-capacity empirical spectrum (Linear $\to$ Tree $\to$ MLP $\to$ Deep CNN) evaluating the synthetic-to-human noise generalisation gap.
