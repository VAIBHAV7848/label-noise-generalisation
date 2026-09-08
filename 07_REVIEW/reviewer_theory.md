# REVIEWER 1: THEORY & STATISTICAL LEARNING
**Confidence**: 5/5 (Expert in statistical learning theory and risk bounds)
**Overall Score**: 4/10 (Weak Reject)

## Summary
The manuscript attempts to bridge the gap between asymptotic transition-matrix loss correction and finite-sample excess risk, explicitly bringing the matrix condition number $\kappa(T)$ into the risk bound. It also categorizes surrogate losses based on their mathematical eligibility for these bounds and provides an exact population identity for Vector Calibration Error (VCE). While the mathematical exposition is largely correct, there is a severe disconnect between the theoretical assumptions (e.g., sample-splitting, perfect clean calibration) and the empirical implementation, rendering the bounds practically inapplicable to the experiments performed.

## Strengths
1. **Explicit Role of Condition Number**: Highlighting $\kappa(T)$ in Proposition 2A/2B is a valuable theoretical contribution that explains why empirical estimation methods (like Confident Learning) collapse under certain noise regimes.
2. **Rigorous Loss Taxonomy**: The four-tier scope categorization (Category A-D) is intellectually honest. Acknowledging that standard unclipped Cross-Entropy (Category B/D) is not covered by the uniform Rademacher bounds without explicit numerical clamping (Category C) shows theoretical maturity.

## Fatal Concerns (Must Fix)
1. **The Sample-Splitting Disconnect**: Proposition 2B explicitly assumes that $\hat{T}$ is estimated from an independent training partition $S_T$. However, in the empirical protocol (e.g., Track 6: Forward ConfidentLearning), the transition matrix is estimated using cross-validation on the *exact same noisy training set* $S_R$ used for risk minimization. The remark in the theory section acknowledges this, but acknowledging a fatal gap does not fix it. You cannot claim that your theorem explains your empirical results when your empirical results violate the core independence assumption of the theorem.
2. **Perfect Calibration Assumption in Theorem 4A/4B**: The VCE population identity (Theorem 4A) assumes the classifier is perfectly calibrated on clean data ($\mathbb{E}[\mathbf{e}_Y \mid f(X) = \mathbf{p}] = \mathbf{p}$). No deep neural network satisfies this in practice; modern architectures are notoriously overconfident even on clean data (Guo et al. 2017). The theorem fails to separate the intrinsic miscalibration of the architecture from the induced miscalibration of the noise.

## Major Weaknesses
1. **Uninformative Bounds for Deep Networks**: The bound in Proposition 2B scales with the empirical Rademacher complexity $\mathcal{R}_{n_R}(\mathcal{F})$. For highly overparameterized models like PreActResNet-18, the Rademacher complexity is vacuous (often $>1$), rendering the finite-sample bound meaningless in the very regime the paper empirically tests.
2. **Lipschitz Constant of Clamped GCE**: The Lipschitz constant for GCE ($q=0.7$) under clamping $\epsilon_{\text{clamp}} = 10^{-7}$ is extremely large ($L \approx 125.89$). When plugged into Proposition 2B, the complexity term explodes, making the theoretical guarantee practically useless for GCE.
3. **Novelty of Unbiasedness**: Proposition 1 is a known result from Patrini et al. (2017) and Natarajan et al. (2013). Presenting it as a numbered proposition alongside novel contributions inflates the perceived theoretical novelty.

## Minor Concerns
- The assumption $\epsilon < 1/\|T^{-1}\|_2$ is restrictive. What if the estimation error exceeds this threshold, as is highly possible in high-noise regimes with small samples? The Neumann series expansion breaks.

## Recommendation
**WEAK REJECT**. The theoretical bounds are mathematically sound but suffer from a severe applicability gap with the empirical methods evaluated. The authors must either align the empirical protocol to match the sample-splitting assumption of the theory or derive a uniform stability bound that permits same-sample estimation.
