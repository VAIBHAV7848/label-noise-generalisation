# Formal Theoretical Propositions (Audited & Verified)

Here we state the mathematically verified theoretical propositions governing excess risk bounds, loss convexity constraints, and calibration distortion under label noise.

---

## Proposition 1 (Unbiasedness of Backward Loss Correction)
Let $T \in [0, 1]^{K \times K}$ be an invertible class-conditional noise transition matrix ($T_{ij} = P(\tilde{Y}=j \mid Y=i)$). Define the backward corrected loss vector $\vec{\tilde{\ell}}(f(x)) = T^{-1} \vec{\ell}(f(x))$. Then for any measurable hypothesis $f: \mathcal{X} \to \Delta^{K-1}$ and any surrogate loss $\ell$ with finite expectation:
$$\mathbb{E}_{(X, \tilde{Y}) \sim \tilde{\mathcal{D}}} [\tilde{\ell}(f(X), \tilde{Y})] = \mathbb{E}_{(X, Y) \sim \mathcal{D}} [\ell(f(X), Y)]$$
*(Proof provided in `proofs/forward_backward_unbiasedness.md`)*

> **Remark on Negative Per-Sample Loss Values**: Because the inverse transition matrix $T^{-1}$ possesses negative off-diagonal entries for any non-identity matrix $T \ne I$, individual per-example backward loss values $\hat{\ell}_{\text{backward}}(f(x_i), \tilde{y}_i) = [T^{-1} \vec{\ell}(f(x_i))]_{\tilde{y}_i}$ can legitimately be negative on specific sample instances. The expectation over the noisy conditional distribution $\mathbb{E}_{\tilde{Y} \mid x}[\hat{\ell}_{\text{backward}}] = \mathbf{p}(x)^\top \vec{\ell}(f(x)) \ge 0$ is strictly non-negative for non-negative base losses. In contrast, the empirical pilot pipeline utilizes Forward Loss Correction $\hat{\ell}_{\text{forward}}(f(x), \tilde{y}) = -\log([T^\top f(x)]_{\tilde{y}}) \ge 0$, which is strictly non-negative per sample.

---

## Proposition 2A (Pointwise Expected Risk Bias under Matrix Perturbation)
Let $T \in [0, 1]^{K \times K}$ be an invertible class-conditional transition matrix, and let $\hat{T}$ be any fixed or conditionally given transition matrix with estimation error $\|E\|_F = \|\hat{T} - T\|_F \le \epsilon < \frac{1}{\|T^{-1}\|_2}$. Let $\hat{\ell}_{\text{backward}}(f(x), \tilde{y}) = [\hat{T}^{-1} \vec{\ell}(f(x))]_{\tilde{y}}$. 

Assume the base surrogate loss $\ell: \Delta^{K-1} \times \{1, \dots, K\} \to \mathbb{R}_+$ is $M$-bounded ($0 \le \ell \le M$). Then for any hypothesis $f: \mathcal{X} \to \Delta^{K-1}$, the pointwise expected risk estimation bias satisfies:
$$\left| \mathbb{E}_{\tilde{\mathcal{D}}} [\hat{\ell}_{\text{backward}}(f(X), \tilde{Y})] - R_{\mathcal{D}}(f) \right| \le \frac{\sqrt{K} M \|T^{-1}\|_2^2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon}$$

---

## Proposition 2B (Finite-Sample Clean Excess Risk under Sample-Split / Independent Estimation)
Assume the base surrogate loss $\ell$ is $M$-bounded ($0 \le \ell \le M$) and $L_{\ell}$-Lipschitz with respect to the $\ell_2$ norm. Let $\hat{T}$ be estimated from an independent training partition $S_T$ (or conditional on $\hat{T}$ fixed with respect to $S_R$) such that $\|\hat{T} - T\|_F \le \epsilon < \frac{1}{\|T^{-1}\|_2}$. Let $\hat{f} = \arg\min_{f \in \mathcal{F}} \hat{R}_{S_R, \hat{T}}(f)$ be the empirical risk minimizer over an independent noisy sample $S_R = \{(x_i, \tilde{y}_i)\}_{i=1}^{n_R} \overset{\text{i.i.d.}}{\sim} \tilde{\mathcal{D}}$.

Then with probability at least $1 - \delta$ over the random draw of $S_R$:
$$\mathcal{E}(\hat{f}) = R_{\mathcal{D}}(\hat{f}) - \min_{f \in \mathcal{F}} R_{\mathcal{D}}(f) \le \frac{2 \sqrt{K} M \|T^{-1}\|_2^2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon} + 4 \sqrt{2} L_{\ell} \|\hat{T}^{-1}\|_2 \mathcal{R}_{n_R}(\mathcal{F}) + 2 \sqrt{K} M \|\hat{T}^{-1}\|_2 \sqrt{\frac{\ln(2/\delta)}{2n_R}}$$

---

### Remark on Loss Boundedness & Cross-Entropy Applicability
1. **Bounded Surrogate Losses**: Proposition 2 applies directly to naturally bounded surrogate losses, such as Generalized Cross Entropy (GCE, where $M = 1/q$ and $L_\ell = 1$), 0-1 surrogate losses, and Symmetric Cross Entropy.
2. **Standard Cross-Entropy Applicability**: Standard multiclass Cross-Entropy $\ell(\mathbf{p}, y) = -\log p_y$ is unbounded on the open simplex as $p_y \to 0$. Proposition 2B applies to Cross-Entropy under the standard **probability-clamping condition** $p_k(x) \ge \epsilon_{\text{clamp}} > 0$ (enforced in deep learning implementations, yielding $M = -\log \epsilon_{\text{clamp}}$ and $L_\ell = 1/\epsilon_{\text{clamp}}$) or under bounded logit domains $\|f(x)\|_\infty \le B$.
3. **Unbounded CE Formulation**: Unclipped theoretical Cross-Entropy on the open simplex violates McDiarmid's uniform bounded difference condition and requires sub-exponential / Bernstein concentration bounds.

---

### Remark on Same-Sample Data-Dependent Estimation $\hat{T}(S)$
When $\hat{T} = \hat{T}(S)$ is estimated from the **exact same noisy sample** $S$ used for empirical risk minimization without sample splitting:
1. Standard single-function McDiarmid concentration and Rademacher symmetrization do not apply directly because the loss function $g_f(z) = [\hat{T}(S)^{-1} \vec{\ell}(f(x))]_{\tilde{y}}$ is coupled to all instances in $S$.
2. To bound this same-sample case rigorously, one must take a uniform supremum over the compact perturbation ball $\mathcal{T}_\epsilon = \{ A \in \mathbb{R}^{K \times K} : \|A - T\|_F \le \epsilon \}$, adding a metric-entropy / covering net term $O\left(\frac{K^2 \ln(n)}{n}\right)$, or establish uniform algorithmic stability of $\hat{T}(S)$.
3. Therefore, Proposition 2B is formally stated under the standard sample-splitting / conditional independence model.

---

## Proposition 3 (Noise Tolerance Barrier of Convex Multi-Class Losses)
Let $K \ge 3$. There does not exist any strictly convex, classification-calibrated surrogate loss function $\ell: \Delta^{K-1} \times \{1, \dots, K\} \to \mathbb{R}_+$ that is symmetric in the sense of $\sum_{k=1}^K \ell(\mathbf{p}, k) = C$ for all $\mathbf{p} \in \Delta^{K-1}$. (Charoenphakdee et al., ICML 2019).

---

## Proposition 4 (Posterior Calibration Distortion under Noise)

### Theorem 4A (Exact Population Vector Calibration Distortion)
Let $f: \mathcal{X} \to \Delta^{K-1}$ be perfectly calibrated on clean distribution $\mathcal{D}$ ($\mathbb{E}[\mathbf{e}_Y \mid f(X) = \mathbf{p}] = \mathbf{p}$). Under class-conditional noise with transition matrix $T$, the Vector Calibration Error on corrupted data $\tilde{\mathcal{D}}$ satisfies the exact identity:
$$\text{VCE}_{\tilde{\mathcal{D}}}(f) = \mathbb{E}_{\mathbf{p}} [ \| (T^\top - I) \mathbf{p} \|_1 ]$$
For any distribution whose posterior support is not confined to the stationary eigenspace of $T^\top$, $\text{VCE}_{\tilde{\mathcal{D}}}(f) > 0$ strictly holds whenever $T \ne I$.

### Theorem 4B (Top-Label ECE Upper Bound & Symmetric Exact Distortion)
1. For general transition matrices $T$, top-label confidence calibration is bounded by:
   $$\text{ECE}_{\tilde{\mathcal{D}}}(f) \le \mathbb{E}_{\mathbf{p}} [ | [T^\top \mathbf{p}]_{\hat{Y}} - \max_k p_k | ]$$
2. Under symmetric label noise with flip probability $\eta \in (0, \frac{K-1}{K})$, for every instance with confidence $\hat{P} > 1/K$, the calibration distortion sign is uniformly negative, yielding the exact equality:
   $$\text{ECE}_{\tilde{\mathcal{D}}}(f) = \frac{K \eta}{K-1} \mathbb{E}_{\hat{P}} \left[ \hat{P} - \frac{1}{K} \right] > 0$$
*(Counterexample and proof documented in `07_REVIEW/proposition4_calibration_audit.md`)*
