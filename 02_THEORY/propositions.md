# Formal Theoretical Propositions (Audited & Verified)

Here we state the mathematically verified theoretical propositions governing excess risk bounds, loss convexity constraints, and calibration distortion.

---

## Proposition 1 (Unbiasedness of Backward Loss Correction)
Let $T \in [0, 1]^{K \times K}$ be an invertible class-conditional noise transition matrix ($T_{ij} = P(\tilde{Y}=j \mid Y=i)$). Define the backward corrected loss vector $\vec{\tilde{\ell}}(f(x)) = T^{-1} \vec{\ell}(f(x))$. Then for any measurable hypothesis $f: \mathcal{X} \to \Delta^{K-1}$ and any surrogate loss $\ell$ with finite expectation:
$$\mathbb{E}_{(X, \tilde{Y}) \sim \tilde{\mathcal{D}}} [\tilde{\ell}(f(X), \tilde{Y})] = \mathbb{E}_{(X, Y) \sim \mathcal{D}} [\ell(f(X), Y)]$$
*(Proof provided in `proofs/forward_backward_unbiasedness.md`)*

---

## Proposition 2 (Non-Asymptotic Excess Risk under Perturbed $\hat{T}$)
Let $T$ be an invertible transition matrix and $\hat{T} = T + E$ be an estimated transition matrix with estimation error $\|E\|_F \le \epsilon < \frac{1}{\|T^{-1}\|_2}$. Let $\hat{\ell}_{\text{backward}}(f(x), \tilde{y}) = [\hat{T}^{-1} \vec{\ell}(f(x))]_{\tilde{y}}$. 
Assume the base loss $\ell$ is $M$-bounded ($0 \le \ell \le M$) and $L_{\ell}$-Lipschitz.
Then the pointwise expected risk estimation bias is bounded by:
$$\left| \mathbb{E}_{\tilde{\mathcal{D}}} [\hat{\ell}_{\text{backward}}(f(X), \tilde{Y})] - R_{\mathcal{D}}(f) \right| \le \frac{\sqrt{K} M \|T^{-1}\|_2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon}$$
Furthermore, the excess risk $\mathcal{E}(\hat{f}) = R_{\mathcal{D}}(\hat{f}) - \min_{f \in \mathcal{F}} R_{\mathcal{D}}(f)$ of the empirical risk minimizer $\hat{f} = \arg\min_{f \in \mathcal{F}} \hat{R}_{\tilde{S}, \hat{T}}(f)$ satisfies:
$$\mathcal{E}(\hat{f}) \le \frac{2 \sqrt{K} M \|T^{-1}\|_2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon} + 4 \sqrt{2} L_{\ell} \|\hat{T}^{-1}\|_2 \mathcal{R}_n(\mathcal{F}) + 2 \sqrt{K} M \|\hat{T}^{-1}\|_2 \sqrt{\frac{\ln(2/\delta)}{2n}}$$
with probability at least $1 - \delta$ over the draw of noisy sample $\tilde{S}$ of size $n$.
*(First-principles derivation documented in `07_REVIEW/proposition2_proof_audit.md`)*

---

## Proposition 3 (Noise Tolerance Barrier of Convex Multi-Class Losses)
Let $K \ge 3$. There does not exist any strictly convex, classification-calibrated surrogate loss function $\ell: \Delta^{K-1} \times \{1, \dots, K\} \to \mathbb{R}_+$ that is symmetric in the sense of $\sum_{k=1}^K \ell(\mathbf{p}, k) = C$ for all $\mathbf{p} \in \Delta^{K-1}$. (Charoenphakdee et al., ICML 2019).

---

## Proposition 4 (Posterior Calibration Distortion under Noise)

### Theorem 4A (Exact Vector Calibration Distortion)
Let $f: \mathcal{X} \to \Delta^{K-1}$ be perfectly calibrated on clean distribution $\mathcal{D}$ ($\mathbb{E}[\mathbf{e}_Y \mid f(X) = \mathbf{p}] = \mathbf{p}$). Under class-conditional noise with transition matrix $T$, the Vector Calibration Error on corrupted data $\tilde{\mathcal{D}}$ satisfies the exact identity:
$$\text{VCE}_{\tilde{\mathcal{D}}}(f) = \mathbb{E}_{\mathbf{p}} [ \| (T^\top - I) \mathbf{p} \|_1 ]$$
For any distribution whose posterior support is not confined to the stationary eigenspace of $T^\top$, $\text{VCE}_{\tilde{\mathcal{D}}}(f) > 0$ strictly holds whenever $T \ne I$.

### Theorem 4B (Top-Label ECE Upper Bound & Symmetric Exact Distortion)
1. For general transition matrices $T$, top-label confidence calibration is bounded by:
   $$\text{ECE}_{\tilde{\mathcal{D}}}(f) \le \mathbb{E}_{\mathbf{p}} [ | [T^\top \mathbf{p}]_{\hat{Y}} - \max_k p_k | ]$$
2. Under symmetric label noise with flip probability $\eta \in (0, \frac{K-1}{K})$, for every instance with confidence $\hat{P} > 1/K$, the calibration error has a uniform negative sign, yielding the exact equality:
   $$\text{ECE}_{\tilde{\mathcal{D}}}(f) = \frac{K \eta}{K-1} \mathbb{E}_{\hat{P}} \left[ \hat{P} - \frac{1}{K} \right] > 0$$
*(Counterexample and proof documented in `07_REVIEW/proposition4_calibration_audit.md`)*
