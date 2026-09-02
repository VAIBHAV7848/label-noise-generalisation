# Formal Theoretical Propositions

Here we state the core theoretical propositions governing generalisation, excess risk bounds, and calibration error under noisy labels.

---

## Proposition 1 (Unbiasedness of Backward Loss Correction)
Let $T \in [0, 1]^{K \times K}$ be an invertible class-conditional noise transition matrix. Define the backward corrected loss vector $\vec{\tilde{\ell}}(f(x)) = T^{-1} \vec{\ell}(f(x))$. Then for any measurable hypothesis $f: \mathcal{X} \to \Delta^{K-1}$ and any bounded surrogate loss $\ell$:
$$\mathbb{E}_{(X, \tilde{Y}) \sim \tilde{\mathcal{D}}} [\tilde{\ell}(f(X), \tilde{Y})] = \mathbb{E}_{(X, Y) \sim \mathcal{D}} [\ell(f(X), Y)]$$
*(Proof provided in `proofs/forward_backward_unbiasedness.md`)*

---

## Proposition 2 (Excess Risk under Imperfect Transition Matrix $\hat{T}$)
Let $\hat{T}$ be an estimated transition matrix with estimation error $\Delta T = \hat{T} - T$ such that $\|\Delta T\|_F \le \epsilon$. Let $\hat{\ell}_{\text{backward}}(f(x), \tilde{y}) = [\hat{T}^{-1} \vec{\ell}(f(x))]_{\tilde{y}}$. 
Assume $\ell$ is $M$-bounded and the condition number of $T$ is $\kappa(T) = \|T\|_2 \|T^{-1}\|_2$.
Then the expected risk estimation bias is bounded by:
$$\left| \mathbb{E}_{\tilde{\mathcal{D}}} [\hat{\ell}_{\text{backward}}(f(X), \tilde{Y})] - R_{\mathcal{D}}(f) \right| \le M \sqrt{K} \|T^{-1}\|_2^2 \epsilon + \mathcal{O}(\epsilon^2)$$
Furthermore, the excess risk $\mathcal{E}(\hat{f}) = R(\hat{f}) - \min_{f \in \mathcal{F}} R(f)$ of the empirical minimizer $\hat{f} = \arg\min_{f \in \mathcal{F}} \hat{R}_{\tilde{S}, \hat{T}}(f)$ satisfies:
$$\mathcal{E}(\hat{f}) \le 2 M \sqrt{K} \|T^{-1}\|_2^2 \epsilon + \frac{4 L_{\ell} \|T^{-1}\|_2}{\sqrt{n}} \mathcal{R}_n(\mathcal{F}) + \mathcal{O}\left( \frac{M \|T^{-1}\|_2 \sqrt{\ln(1/\delta)}}{\sqrt{n}} \right)$$
with probability at least $1 - \delta$.

---

## Proposition 3 (Noise Tolerance Barrier of Convex Multi-Class Losses)
Let $K \ge 3$. There does not exist any strictly convex, classification-calibrated surrogate loss function $\ell: \Delta^{K-1} \times \{1, \dots, K\} \to \mathbb{R}_+$ that is symmetric in the sense of $\sum_{k=1}^K \ell(\mathbf{p}, k) = C$ for all $\mathbf{p} \in \Delta^{K-1}$.
- **Implication**: Any loss function that is inherently noise-tolerant to symmetric multi-class noise must either be non-convex (like MAE / 0-1 surrogates) or require non-zero optimization trade-offs (like GCE / SCE).

---

## Proposition 4 (Posterior Calibration Distortion under Asymmetric Noise)
Let the clean classifier be perfectly calibrated on $\mathcal{D}$ such that $P(Y = k \mid f(X) = \mathbf{p}) = p_k$. Under class-conditional noise with transition matrix $T \ne I$, the observed noisy conditional probability is:
$$P(\tilde{Y} = k \mid f(X) = \mathbf{p}) = [T^\top \mathbf{p}]_k$$
Consequently, the uncorrected model's Expected Calibration Error on corrupted data $\tilde{\mathcal{D}}$ is lower-bounded by:
$$\text{ECE}_{\tilde{\mathcal{D}}}(f) \ge \mathbb{E}_{\mathbf{p}} \left[ \left| \max_{k} p_k - \max_k [T^\top \mathbf{p}]_k \right| \right] > 0$$
even when the underlying representation $f(X)$ perfectly preserves clean class discriminability.
