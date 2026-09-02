# Research Gaps: Genuine Gaps vs. Saturated Non-Gaps

An essential duty in Phase 0 is conducting an adversarial audit to separate **genuine, defensible research gaps** from **saturated or already-solved non-gaps**.

---

## 1. Non-Gaps & Saturated Domains (DO NOT PURSUE AS NOVEL)

| Apparent Gap Claimed | Why it is a NON-GAP (Prior Art Proof) | Literature Citation | Status |
| :--- | :--- | :--- | :--- |
| "Proving that backward loss correction is an unbiased estimator of clean risk." | **Completely Solved**. Natarajan et al. already proved the exact equality $\mathbb{E}_{\tilde{\mathcal{D}}}[\ell_{\text{backward}}] = \mathbb{E}_{\mathcal{D}}[\ell]$ for any invertible transition matrix $T$. | Natarajan et al. (NeurIPS 2013) | **Non-Gap (Solved)** |
| "Showing that symmetric losses like MAE are noise-tolerant under symmetric noise." | **Completely Solved**. Manwani & Sastry (2013), van Rooyen et al. (2015), and Charoenphakdee et al. (2019) fully proved the noise tolerance conditions. | Charoenphakdee et al. (ICML 2019) | **Non-Gap (Solved)** |
| "Demonstrating that deep neural networks can memorize random labels." | **Completely Solved**. Zhang et al. (2017) and Arpit et al. (2017) exhaustively proved and analyzed random label memorisation. | Zhang et al. (ICLR 2017), Arpit et al. (ICML 2017) | **Non-Gap (Solved)** |
| "Testing 2-layer MLP on synthetic uniform MNIST noise." | **Saturated & Trivial**. Tested in dozens of papers since 1990s; provides zero novel scientific insight. | Multiple sources (1990–2020) | **Non-Gap (Saturated)** |
| "Estimating transition matrix $T$ using anchor points on MNIST." | **Completely Solved**. Patrini et al. provided the full anchor-point estimation framework and code in 2017. | Patrini et al. (CVPR 2017) | **Non-Gap (Solved)** |

---

## 2. Genuine Unresolved Research Gaps

### Gap 1: Excess Risk Bounds under Transition Matrix Perturbation ($\| \hat{T} - T \|_F > 0$)
- **Problem**: Natarajan et al. (2013) and Patrini et al. (2017) assume either exact knowledge of $T$ or asymptotic convergence. In practice, on finite datasets with high noise, $\hat{T}$ is estimated with non-trivial error.
- **Unresolved Question**: How does the condition number $\kappa(T)$ and Frobenius estimation error $\epsilon = \| \hat{T} - T \|_F$ propagate to the excess risk bound $\mathcal{E}(f) = R(f) - R(f^*)$ for multi-class classifiers?
- **Research Opportunity**: Derive explicit non-asymptotic excess risk bounds as a function of $\|\hat{T} - T\|_F$.

### Gap 2: Confidence Calibration & Reliability under Noise-Mitigation Techniques
- **Problem**: Most modern noisy-label methods (GCE, SCE, Forward/Backward, DivideMix) are benchmarked exclusively on top-1 classification accuracy. However, loss modifications alter softmax logits and probability simplex geometry.
- **Unresolved Question**: Do loss correction and robust loss functions preserve, improve, or degrade predictive confidence calibration (Expected Calibration Error, Adaptive ECE, Brier Score)? Can standard post-hoc temperature scaling restore calibration when the validation set itself is corrupted?
- **Research Opportunity**: Establish the first unified empirical and theoretical audit of calibration degradation across noise-mitigation families on synthetic and real-world noisy benchmarks.

### Gap 3: Capacity-Dependent Decision Margin Distortion across Classifier Families
- **Problem**: Most theoretical studies examine linear models, while empirical studies examine deep ResNets, leaving a gap in understanding how model capacity (VC-dimension / parameter count) governs the transition from underfitting to noisy memorisation.
- **Unresolved Question**: How does the empirical risk landscape and decision margin distort as a continuous function of model capacity ($p/N$ ratio) under asymmetric label noise?
- **Research Opportunity**: Multi-capacity empirical and geometric margin analysis bridging linear models, decision trees, MLPs, and deep networks under identical controlled noise regimes.

### Gap 4: Theoretical vs Real-World Generalisation Gap (Synthetic vs Human Noise)
- **Problem**: Theoretical loss corrections assume class-conditional noise $P(\tilde{Y} \mid X, Y) = P(\tilde{Y} \mid Y)$. Human noise (CIFAR-10N) is inherently instance-dependent ($P(\tilde{Y} \mid X, Y)$).
- **Unresolved Question**: Why do transition-matrix-based methods fail on human noise benchmarks, and what is the exact performance gap between matrix-corrected ERM and sample-filtering SSL approaches on real crowdsourced noise?
- **Research Opportunity**: Formal empirical breakdown and feature-space margin analysis comparing synthetic $T$-corruptions with CIFAR-10N human annotations.
