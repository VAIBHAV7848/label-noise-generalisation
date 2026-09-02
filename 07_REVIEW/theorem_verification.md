# Mathematical & Estimator Verification Gate Report

This report provides the exhaustive, independent mathematical audit of Propositions 1, 2, and 4, all loss correction modules, transition matrix estimators, and numerical stability bounds.

---

## 1. Mathematical Convention

To eliminate ambiguity across all theory documents and implementations:
- **Clean Label**: $Y \in \{1, 2, \dots, K\}$
- **Noisy Label**: $\tilde{Y} \in \{1, 2, \dots, K\}$
- **Feature Vector**: $X \in \mathcal{X} \subseteq \mathbb{R}^d$
- **Number of Classes**: $K \in \mathbb{N}_{\ge 2}$
- **Noise Transition Matrix**: $T \in [0, 1]^{K \times K}$ where $T_{ij} = P(\tilde{Y}=j \mid Y=i)$ is row-stochastic ($\sum_{j=1}^K T_{ij} = 1$).
- **Clean Posterior**: Column vector $\vec{\eta}(x) \in \Delta^{K-1}$ with $[\vec{\eta}(x)]_i = P(Y=i \mid X=x)$.
- **Corrupted Posterior**: Column vector $\vec{\tilde{\eta}}(x) = T^\top \vec{\eta}(x) \in \Delta^{K-1}$.
- **Loss Vector**: Column vector $\vec{\ell}(f(x)) \in [0, M]^K$ with $[\vec{\ell}(f(x))]_k = \ell(f(x), k)$.
- **Clean Expected Risk**: $R_{\mathcal{D}}(f) = \mathbb{E}_X [\vec{\eta}(X)^\top \vec{\ell}(f(X))]$.
- **Noisy Expected Risk**: $R_{\tilde{\mathcal{D}}}(f) = \mathbb{E}_X [\vec{\tilde{\eta}}(X)^\top \vec{\ell}(f(X))] = \mathbb{E}_X [\vec{\eta}(X)^\top T \vec{\ell}(f(X))]$.
- **Batch Tensor Multiplication**: For probability matrix $\mathbf{P} \in \mathbb{R}^{B \times K}$, corrupted probability is $\tilde{\mathbf{P}} = \mathbf{P} T$. For loss matrix $\mathbf{L} \in \mathbb{R}^{B \times K}$, backward corrected loss matrix is $\tilde{\mathbf{L}} = \mathbf{L} (T^{-1})^\top$.

---

## 2. Proposition 1 Verdict
**Classification**: **[VERIFIED]**  
Backward loss correction $\vec{\tilde{\ell}} = T^{-1} \vec{\ell}$ satisfies $\mathbb{E}_{\tilde{\mathcal{D}}}[\tilde{\ell}(f(X), \tilde{Y})] = \mathbb{E}_{\mathcal{D}}[\ell(f(X), Y)]$ for any invertible $T$ and bounded surrogate loss $\ell$. Boundedness is not required for the expectation identity itself (only for subsequent concentration bounds).

---

## 3. Proposition 2 Verdict
**Classification**: **[VERIFIED WITH CONSTANT CORRECTION AS RIGOROUS THEORETICAL FRAMEWORK BOUND]**  
The Phase 0 bound contained an un-cancelled $\|T^{-1}\|_2^2 \epsilon$ factor. The exact first-order operator-norm perturbation bound has been derived from first principles, verified via 10,000 numerical test cases with zero violations, and proven to be linear in $\|T^{-1}\|_2 \epsilon / (1 - \|T^{-1}\|_2 \epsilon)$.

---

## 4. Proposition 2 Independent Derivation
Let $\hat{T} = T + E$ with $\|E\|_F \le \epsilon < \frac{1}{\|T^{-1}\|_2}$. Pointwise risk bias is:
$$\text{Bias}(x) = \vec{\eta}(x)^\top (T \hat{T}^{-1} - I) \vec{\ell}(f(x)) = -\vec{\eta}(x)^\top E \hat{T}^{-1} \vec{\ell}(f(x))$$
Applying Cauchy-Schwarz and operator norms ($\|\vec{\eta}\|_2 \le 1$, $\|E\|_2 \le \epsilon$, $\|\vec{\ell}\|_2 \le \sqrt{K} M$, and $\|\hat{T}^{-1}\|_2 \le \frac{\|T^{-1}\|_2}{1 - \|T^{-1}\|_2 \epsilon}$):
$$|\text{Bias}(x)| \le \frac{\sqrt{K} M \|T^{-1}\|_2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon}$$
By uniform convergence decomposition:
$$\mathcal{E}(\hat{f}) \le \frac{2 \sqrt{K} M \|T^{-1}\|_2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon} + 4 \sqrt{2} L_{\ell} \|\hat{T}^{-1}\|_2 \mathcal{R}_n(\mathcal{F}) + 2 \sqrt{K} M \|\hat{T}^{-1}\|_2 \sqrt{\frac{\ln(2/\delta)}{2n}}$$

---

## 5. Proposition 2 Literature Overlap
- **Natarajan et al. (NeurIPS 2013)**: Proved binary excess risk bounds for known $T$; asymptotic rate for estimated $T$.
- **Patrini et al. (CVPR 2017)**: Proved multi-class consistency with known $T$.
- **Scott et al. (2015) / Xia et al. (2019)**: Proved convergence rates for estimators.
- **Verdict**: We do not claim Proposition 2 as an unprecedented mathematical theorem, but as a clean, unified operator-norm excess risk characterization.

---

## 6. Proposition 2 Counterexample Search
Tested across 10,000 random stochastic matrices (varying $K \in \{2, 3, 5, 10\}$, small and large $\epsilon$, random simplex posteriors $\vec{\eta}$, and random bounded losses $\vec{\ell}$). Observed empirical ratio $\frac{\text{Actual Bias}}{\text{Theoretical Bound}} \le 0.5001$. Exactly zero violations found.

---

## 7. Proposition 4 Verdict
**Classification**: **[INCORRECT IN ORIGINAL FORM $\to$ REFORMULATED & VERIFIED]**  
The Phase 0 lower bound claim $\text{ECE}_{\tilde{\mathcal{D}}}(f) \ge \mathbb{E}_{\mathbf{p}}[|\dots|]$ was mathematically false due to reversing the direction of Jensen's inequality across classes sharing the same confidence.

---

## 8. Proposition 4 Independent Derivation
Using the tower property:
$$\text{ECE}_{\tilde{\mathcal{D}}}(f) = \mathbb{E}_{\hat{P}} [ | \mathbb{E}_{\mathbf{p} \mid \hat{P}} [ [T^\top \mathbf{p}]_{\hat{Y}} - \hat{P} ] | ] \le \mathbb{E}_{\mathbf{p}} [ | [T^\top \mathbf{p}]_{\hat{Y}} - \max_k p_k | ]$$
For full Vector Calibration Error (VCE), the exact equality holds:
$$\text{VCE}_{\tilde{\mathcal{D}}}(f) = \mathbb{E}_{\mathbf{p}} [ \| (T^\top - I) \mathbf{p} \|_1 ]$$

---

## 9. Proposition 4 Counterexamples
Constructed an analytical 2-point counterexample where $T = \begin{bmatrix} 0.9 & 0.1 \\ 0.8 & 0.2 \end{bmatrix}$ and confidence $\hat{P}=0.7$:
- True Noisy ECE = $\mathbf{0.18}$
- Phase 0 Proposed Lower Bound = $\mathbf{0.35}$
Since $0.18 < 0.35$, the lower bound was disproven.

---

## 10. Corrected Calibration Results
1. **Theorem 4A (Exact Vector Calibration Distortion)**: $\text{VCE}_{\tilde{\mathcal{D}}}(f) = \mathbb{E}_{\mathbf{p}} [ \| (T^\top - I) \mathbf{p} \|_1 ] > 0$ for all non-stationary posterior distributions whenever $T \ne I$.
2. **Theorem 4B (Exact Symmetric Top-Label ECE)**: Under symmetric noise with rate $\eta$, $\text{ECE}_{\tilde{\mathcal{D}}}(f) = \frac{K \eta}{K-1} \mathbb{E}_{\hat{P}} [ \hat{P} - 1/K ] > 0$.

---

## 11. Backward Correction Implementation Verdict
**Classification**: **[VERIFIED]**  
`src/losses/loss_correction.py` exactly implements $\mathbf{L} (T^{-1})^\top$ and extracts row entries via `gather(1, targets)`. Hand calculations for $K=2$ and $K=3$ match tensor outputs to machine precision.

---

## 12. Forward Correction Implementation Verdict
**Classification**: **[VERIFIED]**  
`src/losses/loss_correction.py` exactly implements $\tilde{\mathbf{P}} = \mathbf{P} T$ and evaluates $-\log(\tilde{p}_{\tilde{y}})$.

---

## 13. Anchor Estimator Verdict
**Classification**: **[VERIFIED - FAITHFUL TO PATRINI ET AL. 2017]**  
Implements global 97th-percentile anchor candidate averaging across the entire dataset $\mathcal{S}$ with row-stochastic normalization.

---

## 14. Confident Learning Verdict
**Classification**: **[VERIFIED - FAITHFUL TO NORTHCUTT ET AL. 2021]**  
Implements self-confidence thresholding and single-assignment max-margin sample allocation.

---

## 15. Dual-T Verdict
**Classification**: **[SIMPLIFIED HEURISTIC $\to$ ACCURATELY RENAMED]**  
`src/estimators/dual_t.py` is accurately renamed to `NonAnchorHeuristicEstimator` (`estimate_transition_matrix_heuristic_slack`) to distinguish it from the full constrained factorization of Xia et al. (2019).

---

## 16. Numerical Conditioning Findings
- Symmetric noise becomes singular at $\eta = \frac{K-1}{K}$ ($0.90$ for $K=10$).
- Asymmetric pair flips become singular at $\eta = 0.50$.
- Invertibility check with $\kappa(T) \le 10^4$ and loss clipping added to prevent gradient overflow.

---

## 17. Required Code Changes
1. Added condition number validation in `BackwardLossCorrection` raising `ValueError` if $\kappa(T) > 10^4$.
2. Renamed heuristic in `src/estimators/dual_t.py` to `estimate_transition_matrix_heuristic_slack`.
3. Added 6 new adversarial mathematical tests in `tests/`.

---

## 18. Required Theory Changes
1. Updated `02_THEORY/problem_formulation.md` with unified mathematical convention.
2. Updated `02_THEORY/propositions.md` with corrected Proposition 2 bound and Proposition 4 Theorems 4A & 4B.
3. Updated `02_THEORY/assumptions.md` with noise rate singular limits.

---

## 19. Remaining Scientifically Defensible Contribution
A unified theoretical and empirical characterization connecting **finite-sample transition matrix estimation error $\|\hat{T} - T\|_F$** to **excess risk bounds** and **corrupted-validation calibration recovery (ECE)** under the 300-run Minimum Decisive Experiment Set (MDES).

---

## 20. FINAL GATE
**GATE DECISION**: **GO**  
All mathematical claims are rigorously verified, counterexamples resolved, code conventions confirmed, and estimators faithful or accurately scoped.

---

## Hard Stop Enforced
No large-scale experiments, benchmark runs, or manuscript writing have been executed.
