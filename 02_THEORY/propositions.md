# Formal Theoretical Propositions (Audited & Corrected)

Here we state the mathematically verified theoretical propositions governing excess risk bounds, loss convexity constraints, and calibration distortion.

---

## Proposition 1 (Unbiasedness of Backward Loss Correction)
Let $T \in [0, 1]^{K \times K}$ be an invertible class-conditional noise transition matrix. Define the backward corrected loss vector $\vec{\tilde{\ell}}(f(x)) = T^{-1} \vec{\ell}(f(x))$. Then for any measurable hypothesis $f: \mathcal{X} \to \Delta^{K-1}$ and any bounded surrogate loss $\ell$:
$$\mathbb{E}_{(X, \tilde{Y}) \sim \tilde{\mathcal{D}}} [\tilde{\ell}(f(X), \tilde{Y})] = \mathbb{E}_{(X, Y) \sim \mathcal{D}} [\ell(f(X), Y)]$$
*(Proof provided in `proofs/forward_backward_unbiasedness.md`)*

---

## Proposition 2 (Excess Risk under Imperfect Transition Matrix $\hat{T}$ — Corrected)
Let $T$ be an invertible transition matrix and $\hat{T} = T + E$ be an estimated transition matrix with estimation error $\|E\|_F \le \epsilon < \frac{1}{\|T^{-1}\|_2}$. Let $\hat{\ell}_{\text{backward}}(f(x), \tilde{y}) = [\hat{T}^{-1} \vec{\ell}(f(x))]_{\tilde{y}}$. 
Assume the base loss $\ell$ is $M$-bounded ($0 \le \ell \le M$) and $L_{\ell}$-Lipschitz.
Then the pointwise expected risk estimation bias is bounded by:
$$\left| \mathbb{E}_{\tilde{\mathcal{D}}} [\hat{\ell}_{\text{backward}}(f(X), \tilde{Y})] - R_{\mathcal{D}}(f) \right| \le \frac{\sqrt{K} M \|T^{-1}\|_2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon}$$
Furthermore, the excess risk $\mathcal{E}(\hat{f}) = R_{\mathcal{D}}(\hat{f}) - \min_{f \in \mathcal{F}} R_{\mathcal{D}}(f)$ of the empirical risk minimizer $\hat{f} = \arg\min_{f \in \mathcal{F}} \hat{R}_{\tilde{S}, \hat{T}}(f)$ satisfies:
$$\mathcal{E}(\hat{f}) \le \frac{2 \sqrt{K} M \|T^{-1}\|_2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon} + \frac{4 L_{\ell} \|\hat{T}^{-1}\|_2}{\sqrt{n}} \mathcal{R}_n(\mathcal{F}) + 2 M \|\hat{T}^{-1}\|_2 \sqrt{\frac{\ln(2/\delta)}{2n}}$$
with probability at least $1 - \delta$ over the draw of noisy sample $\tilde{S}$ of size $n$.
*(First-principles derivation documented in `07_REVIEW/mathematical_audit.md`)*

---

## Proposition 3 (Noise Tolerance Barrier of Convex Multi-Class Losses)
Let $K \ge 3$. There does not exist any strictly convex, classification-calibrated surrogate loss function $\ell: \Delta^{K-1} \times \{1, \dots, K\} \to \mathbb{R}_+$ that is symmetric in the sense of $\sum_{k=1}^K \ell(\mathbf{p}, k) = C$ for all $\mathbf{p} \in \Delta^{K-1}$. (Charoenphakdee et al., ICML 2019).

---

## Proposition 4 (Posterior Calibration Distortion under Asymmetric Noise)
Let the clean classifier be perfectly calibrated on $\mathcal{D}$ such that $P(Y = k \mid f(X) = \mathbf{p}) = p_k$. Under class-conditional noise with transition matrix $T \ne I$, the observed noisy conditional probability is $P(\tilde{Y} = k \mid f(X) = \mathbf{p}) = [T^\top \mathbf{p}]_k$. Consequently, the uncorrected model's Expected Calibration Error on corrupted data $\tilde{\mathcal{D}}$ is lower-bounded by:
$$\text{ECE}_{\tilde{\mathcal{D}}}(f) \ge \mathbb{E}_{\mathbf{p}} \left[ \left| \max_{k} p_k - \max_k [T^\top \mathbf{p}]_k \right| \right] > 0$$
even when the underlying representation $f(X)$ perfectly preserves clean class discriminability.
