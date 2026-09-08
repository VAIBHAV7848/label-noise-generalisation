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
Assume the base surrogate loss $\ell$ is $M$-bounded ($0 \le \ell \le M$) and $L_{\ell, 2}$-Lipschitz with respect to the Euclidean $\ell_2$ norm on $\Delta^{K-1}$. Let $\hat{T}$ be estimated from an independent training partition $S_T$ (or conditional on $\hat{T}$ fixed with respect to $S_R$) such that $\|\hat{T} - T\|_F \le \epsilon < \frac{1}{\|T^{-1}\|_2}$. Let $\hat{f} = \arg\min_{f \in \mathcal{F}} \hat{R}_{S_R, \hat{T}}(f)$ be the empirical risk minimizer over an independent noisy sample $S_R = \{(x_i, \tilde{y}_i)\}_{i=1}^{n_R} \overset{\text{i.i.d.}}{\sim} \tilde{\mathcal{D}}$.

Then with probability at least $1 - \delta$ over the random draw of $S_R$:
$$\mathcal{E}(\hat{f}) = R_{\mathcal{D}}(\hat{f}) - \min_{f \in \mathcal{F}} R_{\mathcal{D}}(f) \le \frac{2 \sqrt{K} M \|T^{-1}\|_2^2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon} + 4 \sqrt{2} L_{\ell, 2} \|\hat{T}^{-1}\|_2 \mathcal{R}_{n_R}(\mathcal{F}) + 2 \sqrt{K} M \|\hat{T}^{-1}\|_2 \sqrt{\frac{\ln(2/\delta)}{2n_R}}$$

---

### Loss Taxonomy & Applicability Matrix for Proposition 2B

The uniform concentration in Proposition 2B strictly requires $0 \le \ell \le M$ and $L_{\ell, 2}$-Lipschitz continuity with respect to the $\ell_2$ norm on the prediction domain. Below is the first-principles audit of all project-relevant losses:

| Loss Function | Definition $\ell(\mathbf{p}, y)$ | $M$ (Bound) | Globally $\ell_2$-Lipschitz? | Exact $L_{\ell, 2}$ Constant | Mathematical Domain Conditions | Proposition 2B Scope |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **Mean Absolute Error (MAE / $L_1$)** | $\|\mathbf{p} - \mathbf{e}_y\|_1 = 2 - 2 p_y$ | $2$ | **YES** | $2$ (or $2\sqrt{\frac{K-1}{K}}$ on simplex) | None (unconditionally holds globally on $\Delta^{K-1}$). | **Category A (Directly Applicable)** |
| **Forward Loss Correction (Dense $T$, $T_{\min} > 0$)** | $-\log([T^\top \mathbf{p}]_y) \equiv -\log([\mathbf{p} T]_y)$ | $-\log(T_{\min})$ | **YES** | $\frac{\|T_{:, y}\|_2}{T_{\min}} \le \frac{1}{T_{\min}}$ | Dense transition matrix with strictly positive entries $T_{\min} = \min_{i, j} T_{ij} > 0$ (e.g. Symmetric noise $\eta=0.2, 0.5$). | **Category A (Directly Applicable without Clamping)** |
| **Forward Loss Correction (Sparse $T$, $T_{\min} = 0$)** | $-\log([T^\top \mathbf{p}]_y) \equiv -\log([\mathbf{p} T]_y)$ | $+\infty$ (unbounded unclipped) | **NO** | $\frac{1}{\epsilon_{\text{clamp}}} = 10^7$ (clamped) | Sparse matrix ($T_{\min} = 0$, e.g. Asymmetric pair-flip or Clean $T=I$); requires implementation clamp $\epsilon_{\text{clamp}} = 10^{-7}$ or support floor. | **Category C (Conditional on Numerical Clamping)** |
| **Backward Loss Correction ($\ell_{\text{backward}}$)** | $[T^{-1} \vec{\ell}(\mathbf{p})]_y$ | $\sqrt{K} M_{\text{base}} \|T^{-1}\|_2$ | **Depends on base** | $\|T^{-1}\|_2 L_{\text{base}, 2}$ | Base surrogate loss $\ell$ is $M_{\text{base}}$-bounded & $L_{\text{base}, 2}$-Lipschitz. | **Category A (for MAE base) / Category C (for CE base)** |
| **Generalized Cross Entropy (GCE, $q=0.7$)** | $\frac{1 - p_y^q}{q}$ | $\frac{1}{q} \approx 1.43$ | **NO** ($\lim_{p \to 0} L_q' = -\infty$) | $\epsilon_{\text{clamp}}^{q-1} \approx 125.89$ (on clamped domain) | Globally bounded; requires probability floor $p_y \ge \epsilon_{\text{clamp}} > 0$ for Lipschitz condition ($L_q' \to -\infty$ as $p_y \to 0$). | **Category C (Conditional on Clamping)** |
| **Reverse Cross Entropy (RCE)** | $-\sum_k p_k \log(\bar{\mathbf{e}}_{y, k})$ | $-\log(\epsilon_{\text{clamp}}) \approx 16.12$ | **YES** | $-\log(\epsilon_{\text{clamp}}) \approx 16.12$ | One-hot target vector clamped to $\bar{\mathbf{e}}_{y, k} \ge \epsilon_{\text{clamp}} = 10^{-7}$. | **Category A (under Clamped RCE)** |
| **Categorical Cross-Entropy (CE)** | $-\log p_y$ | $+\infty$ (unbounded) | **NO** ($\lim_{p \to 0} -1/p = -\infty$) | $\frac{1}{\epsilon_{\text{clamp}}} = 10^7$ (on clamped domain) | Unbounded on open simplex; requires $p_y \ge \epsilon_{\text{clamp}} > 0$ or bounded logits. | **Category B (Empirical Baseline) / Category C (Clamped)** |
| **Symmetric Cross Entropy (SCE)** | $\alpha \ell_{\text{CE}} + \beta \ell_{\text{RCE}}$ | $+\infty$ (unbounded) | **NO** (due to CE term) | $\frac{\alpha}{\epsilon_{\text{clamp}}} + \beta \ln(\frac{1}{\epsilon_{\text{clamp}}})$ (clamped) | Unbounded on open simplex; requires $p_y \ge \epsilon_{\text{clamp}} > 0$ for CE component. | **Category B (Empirical Baseline) / Category C (Clamped)** |

---

### Four-Tier Scope Categorization for Proposition 2B

1. **Category A — Genuinely Covered by Proposition 2B Unconditionally**:
   - Multi-Class MAE ($M = 2, L_{\ell, 2} = 2$).
   - Forward Loss Correction under dense transition matrices with non-zero noise floor $T_{\min} = \min_{i, j} T_{ij} > 0$ ($M = -\log T_{\min}, L_{\ell, 2} \le 1/T_{\min}$).
   - Backward Loss Correction using MAE as base loss ($M = 2\sqrt{K}\|T^{-1}\|_2, L_{\ell, 2} = 2\|T^{-1}\|_2$).
2. **Category B — Empirical Baselines & Diagnostic Controls**:
   - Standard Uncorrected Categorical Cross-Entropy (CE).
   - Label Smoothing Cross-Entropy.
   *(Evaluated empirically in the 84-run pilot to benchmark deep learning baselines; not claimed as covered by Proposition 2B without domain clamping).*
3. **Category C — Covered Under Explicit Probability Clamping / Domain Restrictions**:
   - Forward Loss Correction under sparse asymmetric pair-flip noise ($T_{\min} = 0$) or clean data ($T=I$), which is bounded only due to the implementation's numerical clamp $p_{\text{corrupted}} \ge \epsilon_{\text{clamp}} = 10^{-7}$ ($M = -\log(10^{-7}) \approx 16.12, L_{\ell, 2} = 10^7$).
   - Generalized Cross Entropy (GCE, $q=0.7$) with probability floor $p_y \ge \epsilon_{\text{clamp}} > 0$ ($M \approx 1.43, L_{\ell, 2} \approx 125.89$).
   - Symmetric Cross Entropy (SCE) with probability floor $p_y \ge \epsilon_{\text{clamp}} > 0$.
   - Backward Loss Correction with clamped CE base loss.
4. **Category D — Requiring Sub-Exponential / Bernstein Concentration Arguments**:
   - Unclipped continuous Cross-Entropy on the open probability simplex $(0, 1]^K$.

---

### Remark on Matrix Orientation and Equivalences
- Let $\mathbf{p}_{\text{row}} \in \Delta^{K-1}$ denote a row probability vector and $\mathbf{p}_{\text{col}} \in \Delta^{K-1}$ denote a column probability vector.
- Let $T \in [0, 1]^{K \times K}$ be a row-stochastic transition matrix where $T_{ij} = P(\tilde{Y}=j \mid Y=i)$ ($\sum_{j=1}^K T_{ij} = 1$).
- The corrupted class probability vector is identically expressed as:
  $$\tilde{\mathbf{p}}_{\text{row}} = \mathbf{p}_{\text{row}} T \quad \Longleftrightarrow \quad \tilde{\mathbf{p}}_{\text{col}} = T^\top \mathbf{p}_{\text{col}}$$
- For observed noisy label $\tilde{y}$, the $j$-th corrupted component is $[\mathbf{p}_{\text{row}} T]_{\tilde{y}} = [T^\top \mathbf{p}_{\text{col}}]_{\tilde{y}} = \sum_{i=1}^K p_i T_{i, \tilde{y}}$.
- In PyTorch code (`src/losses/loss_correction.py`), `torch.matmul(probs, self.T)` operates on batch row vectors, exactly matching $[T^\top \mathbf{p}]_{\tilde{y}}$.

---

### Remark on Same-Sample Data-Dependent Estimation $\hat{T}(S)$ & Heuristic Classification
When $\hat{T} = \hat{T}(S)$ is estimated from the **exact same noisy sample** $S$ used for empirical risk minimization without sample splitting:
1. Standard single-function McDiarmid concentration and Rademacher symmetrization do not apply directly because the loss function $g_f(z) = [\hat{T}(S)^{-1} \vec{\ell}(f(x))]_{\tilde{y}}$ is coupled to all instances in $S$. Changing a single instance $(x_i, \tilde{y}_i)$ perturbs $\hat{T}(S)$, shifting all $n$ loss terms in the empirical risk summation.
2. To bound this same-sample case rigorously, one must either:
   - take a uniform supremum over the compact perturbation ball $\mathcal{B}_\epsilon(T) = \{ A \in \mathbb{R}^{K \times K} : \|A - T\|_F \le \epsilon, A\mathbf{1}=\mathbf{1} \}$, introducing a metric-entropy / covering net complexity term $\mathcal{O}\left(\sqrt{\frac{K^2 \ln(n)}{n}}\right)$, or
   - establish uniform algorithmic stability of the transition estimator $\beta \le \mathcal{O}(1/n)$.
3. **Formal Classification of Pilot Tracks**:
   - `Forward (True T)` and `Forward (Perturbed T)` are fixed/oracle deterministic operators that strictly satisfy $S_T \perp S_R$ and are **fully covered** by Proposition 2B.
   - `Forward (Anchor T)` and `Forward (ConfLearning T)` as executed in standard practice reuse the training set ($S_T = S_R$) and are classified as **empirical heuristics outside the strict coverage of Proposition 2B**.
   - A dedicated sample-split condition ($S_T \cap S_R = \emptyset$, $S_T \perp S_R$) is required to achieve direct empirical alignment with Proposition 2B for data-driven transition estimation.

---

### Analysis of Finite-Sample Generalization vs. Population Unbiasedness: The True-$T$ Oracle vs. Robust Losses (GCE)
An apparent paradox in empirical benchmarks is that under Symmetric 50% noise, bounded robust losses (GCE) outperform the unbiased True-$T$ Forward Correction oracle ($82.40\%$ vs $79.17\%$). This outcome does not contradict Proposition 1 or Proposition 2B, but illustrates fundamental principles of statistical learning and deep network optimization:

1. **Population Unbiasedness vs. Finite-Sample Risk Minimization**:
   - Proposition 1 proves *population-level* unbiasedness ($\mathbb{E}_{\tilde{\mathcal{D}}}[\tilde{\ell}] = R_{\mathcal{D}}(f)$). Population unbiasedness guarantees consistency as $n \to \infty$ under exact global empirical risk minimization.
   - It does *not* imply that on a finite sample of size $n$ optimized via stochastic gradient descent (SGD), the unbiased estimator minimizes clean test error. In non-convex optimization with overparameterized models, the bias-variance tradeoff of the gradient estimator strongly governs generalization.
2. **Gradient Variance and Truncation**:
   - In Forward Correction ($-\log([T^\top f(x)]_{\tilde{y}})$), under 50% noise, half of the training examples have incorrect labels. The loss remains steep at low probabilities, so every corrupted sample exerts a persistent gradient pull on parameters.
   - In GCE ($\ell_q(p, y) = \frac{1 - p_y^q}{q}$, $q=0.7$), the gradient magnitude is $|\nabla_z \ell_q| \propto p_y^q (1 - p_y)$. As the network rapidly learns simple clean patterns in early epochs, corrupted labels receive low clean model probability ($p_{\tilde{y}} \to 0$). The factor $p_{\tilde{y}}^{0.7}$ dynamically attenuates the gradient of corrupted instances toward zero, acting as an implicit continuous noise trimmer and reducing stochastic gradient variance on the clean data manifold.
3. **Invariance of the Bayes Boundary under Symmetric Noise**:
   - Under symmetric noise and balanced priors, the corrupted posterior satisfies $\arg\max_k [T^\top \mathbf{p}(x)]_k = \arg\max_k \mathbf{p}(x)$. The Bayes decision boundary is invariant to symmetric noise. Consequently, directional matrix inversion is unnecessary; variance reduction (achieved by GCE) dominates.
   - Under asymmetric noise (e.g., 40% pair-flip), the Bayes decision boundary is severely displaced. GCE cannot distinguish systematic corruption from minority clean instances and memorizes the shifted boundary ($78.73\%$, $-2.88\%$ vs CE). In contrast, True $T$ correctly inverts the directional boundary shift ($87.80\%$, $+6.19\%$ vs CE).
4. **Regularization Evidence via Perturbed $T$**:
   - `Forward (Perturbed T)` ($T_{\text{perturbed}} = 0.5 T + 0.5 U$) achieves $82.00\%$ on Symmetric 50%, matching GCE and surpassing unregularized True $T$ ($79.17\%$). Mixing with the uniform matrix $U$ injects label smoothing, regularizing gradient variance and demonstrating that regularization outperforms exact unbiasedness on symmetric noise.

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
