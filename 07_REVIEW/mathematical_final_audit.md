# Mathematical Final Audit: First-Principles Proofs & Calibration Theorems

**Document Type**: Independent Mathematical Verification & Symbolic Derivation Audit  
**Framework**: Academic Research Skills (ARS) Mathematical Rigor Gate  
**Single Source of Truth**: [`VAIBHAV7848/label-noise-generalisation`](https://github.com/VAIBHAV7848/label-noise-generalisation)

---

## 1. Propositions 2A & 2B: Complete First-Principles Derivation

### 1.1 Statements

#### Proposition 2A (Pointwise Risk Estimation Bias)
Let $T \in [0, 1]^{K \times K}$ be an invertible class-conditional noise transition matrix ($T_{ij} = P(\tilde{Y}=j \mid Y=i)$), and let $\hat{T} = T + E$ be an estimated transition matrix with Frobenius perturbation $\|E\|_F \le \epsilon < \frac{1}{\|T^{-1}\|_2}$. Let $\hat{\ell}_{\text{backward}}(f(x), \tilde{y}) = [\hat{T}^{-1} \vec{\ell}(f(x))]_{\tilde{y}}$. Assume the base surrogate loss $\ell: \Delta^{K-1} \times \{1, \dots, K\} \to \mathbb{R}_+$ is $M$-bounded ($0 \le \ell \le M$).

Then for any hypothesis $f: \mathcal{X} \to \Delta^{K-1}$:
$$\left| \mathbb{E}_{\tilde{\mathcal{D}}} [\hat{\ell}_{\text{backward}}(f(X), \tilde{Y})] - R_{\mathcal{D}}(f) \right| \le \frac{\sqrt{K} M \|T^{-1}\|_2^2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon}$$

#### Proposition 2B (Finite-Sample Clean Excess Risk under Sample Splitting)
Assume the base surrogate loss $\ell$ is $M$-bounded ($0 \le \ell \le M$) and $L_{\ell, 2}$-Lipschitz with respect to the $\ell_2$ norm. Let $\hat{T}$ be estimated from an independent training partition $S_T$ (or conditional on $\hat{T}$ fixed with respect to $S_R$) such that $\|\hat{T} - T\|_F \le \epsilon < \frac{1}{\|T^{-1}\|_2}$. Let $\hat{f} = \arg\min_{f \in \mathcal{F}} \hat{R}_{S_R, \hat{T}}(f)$ be the empirical risk minimizer over an independent noisy sample $S_R = \{(x_i, \tilde{y}_i)\}_{i=1}^{n_R} \overset{\text{i.i.d.}}{\sim} \tilde{\mathcal{D}}$.

Then with probability at least $1 - \delta$ over the random draw of $S_R$:
$$\mathcal{E}(\hat{f}) = R_{\mathcal{D}}(\hat{f}) - \min_{f \in \mathcal{F}} R_{\mathcal{D}}(f) \le \frac{2 \sqrt{K} M \|T^{-1}\|_2^2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon} + 4 \sqrt{2} L_{\ell, 2} \|\hat{T}^{-1}\|_2 \mathcal{R}_{n_R}(\mathcal{F}) + 2 \sqrt{K} M \|\hat{T}^{-1}\|_2 \sqrt{\frac{\ln(2/\delta)}{2n_R}}$$

---

### 1.2 Step-by-Step Symbolic Proof

#### Step 1: Matrix Inversion Perturbation
Using the fundamental resolvent identity:
$$\hat{T}^{-1} - T^{-1} = -\hat{T}^{-1} (\hat{T} - T) T^{-1} = -T^{-1} E \hat{T}^{-1}$$
Taking the operator 2-norm:
$$\|\hat{T}^{-1} - T^{-1}\|_2 \le \|T^{-1}\|_2 \|E\|_2 \|\hat{T}^{-1}\|_2$$
Using the submultiplicative Neumann series bound:
$$\|\hat{T}^{-1}\|_2 = \|(T + E)^{-1}\|_2 \le \frac{\|T^{-1}\|_2}{1 - \|T^{-1}\|_2 \|E\|_2} \le \frac{\|T^{-1}\|_2}{1 - \|T^{-1}\|_2 \epsilon}$$
Substituting back gives:
$$\|\hat{T}^{-1} - T^{-1}\|_2 \le \|T^{-1}\|_2 \epsilon \cdot \frac{\|T^{-1}\|_2}{1 - \|T^{-1}\|_2 \epsilon} = \frac{\|T^{-1}\|_2^2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon}$$

#### Step 2: Conditional Expectation Expansion
Let $\mathbf{p}(x) = (P(Y=1 \mid X=x), \dots, P(Y=K \mid X=x))^\top \in \Delta^{K-1}$.  
Under class-conditional noise, the noisy conditional distribution is $\tilde{\mathbf{p}}(x) = T^\top \mathbf{p}(x)$.  
The conditional expectation of the backward loss under noisy labels is:
$$\mathbb{E}_{\tilde{Y} \mid X=x} [\hat{\ell}_{\text{backward}}(f(x), \tilde{Y})] = \sum_{j=1}^K [T^\top \mathbf{p}(x)]_j [\hat{T}^{-1} \vec{\ell}(f(x))]_j = (T^\top \mathbf{p}(x))^\top \hat{T}^{-1} \vec{\ell}(f(x)) = \mathbf{p}(x)^\top T \hat{T}^{-1} \vec{\ell}(f(x))$$
The clean conditional expectation is:
$$R_{\mathcal{D}}(f \mid X=x) = \mathbf{p}(x)^\top \vec{\ell}(f(x)) = \mathbf{p}(x)^\top T T^{-1} \vec{\ell}(f(x))$$
Subtracting yields:
$$\mathbb{E}_{\tilde{Y} \mid X=x} [\hat{\ell}_{\text{backward}}(f(x), \tilde{Y})] - R_{\mathcal{D}}(f \mid X=x) = \mathbf{p}(x)^\top T (\hat{T}^{-1} - T^{-1}) \vec{\ell}(f(x))$$

#### Step 3: Norm Bounding
By Cauchy-Schwarz and matrix operator norms:
$$\left| \mathbf{p}(x)^\top T (\hat{T}^{-1} - T^{-1}) \vec{\ell}(f(x)) \right| \le \|\mathbf{p}(x)^\top T\|_2 \cdot \|\hat{T}^{-1} - T^{-1}\|_2 \cdot \|\vec{\ell}(f(x))\|_2$$
- For any probability vector $\mathbf{p} \in \Delta^{K-1}$, $\tilde{\mathbf{p}} = \mathbf{p}^\top T$ is also a probability vector in $\Delta^{K-1}$, satisfying $\|\tilde{\mathbf{p}}\|_2 \le \|\tilde{\mathbf{p}}\|_1 = 1$.
- For bounded loss $0 \le \ell \le M$, the $K$-dimensional loss vector satisfies $\|\vec{\ell}(f(x))\|_2 = \sqrt{\sum_{k=1}^K \ell(f(x), k)^2} \le \sqrt{K} M$.
- Combining with Step 1 yields Proposition 2A:
$$\left| \mathbb{E}_{\tilde{\mathcal{D}}} [\hat{\ell}_{\text{backward}}(f(X), \tilde{Y})] - R_{\mathcal{D}}(f) \right| \le 1 \cdot \frac{\|T^{-1}\|_2^2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon} \cdot \sqrt{K} M = \frac{\sqrt{K} M \|T^{-1}\|_2^2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon}$$

#### Step 4: Excess Risk Decomposition under Sample Splitting
Conditioning on $S_T$ fixes $\hat{T}$ with respect to $S_R$. Decomposing:
$$R_{\mathcal{D}}(\hat{f}) - R_{\mathcal{D}}(f^*) \le 2 \Delta_{\text{bias}} + 2 \sup_{f \in \mathcal{F}} | R_{\tilde{\mathcal{D}}, \hat{T}}(f) - \hat{R}_{S_R, \hat{T}}(f) |$$
The sum of the two bias terms is $2 \Delta_{\text{bias}} = \frac{2 \sqrt{K} M \|T^{-1}\|_2^2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon}$.  
Applying the vector contraction inequality (Maurer, 2016) for Lipschitz loss composition with operator norm $\|\hat{T}^{-1}\|_2$ gives the Rademacher complexity bound $4 \sqrt{2} L_{\ell, 2} \|\hat{T}^{-1}\|_2 \mathcal{R}_{n_R}(\mathcal{F})$.  
Applying McDiarmid's concentration with loss bound $\sqrt{K} M \|\hat{T}^{-1}\|_2$ yields $2 \sqrt{K} M \|\hat{T}^{-1}\|_2 \sqrt{\frac{\ln(2/\delta)}{2n_R}}$. $\blacksquare$

---

## 2. Loss Function Boundedness & Euclidean Lipschitz Taxonomy

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

## 3. Calibration Theorems 4A & 4B

### Theorem 4A (Exact Population Vector Calibration Distortion)
Let $f: \mathcal{X} \to \Delta^{K-1}$ be perfectly calibrated on clean distribution $\mathcal{D}$ ($\mathbb{E}[\mathbf{e}_Y \mid f(X) = \mathbf{p}] = \mathbf{p}$). Under class-conditional noise with transition matrix $T$, the Vector Calibration Error on corrupted data $\tilde{\mathcal{D}}$ satisfies the exact identity:
$$\text{VCE}_{\tilde{\mathcal{D}}}(f) = \mathbb{E}_{\mathbf{p}} [ \| (T^\top - I) \mathbf{p} \|_1 ]$$
For any distribution whose posterior support is not confined to the stationary eigenspace of $T^\top$, $\text{VCE}_{\tilde{\mathcal{D}}}(f) > 0$ strictly holds whenever $T \ne I$.

### Theorem 4B (Top-Label ECE Upper Bound & Symmetric Exact Distortion)
1. For general transition matrices $T$, top-label confidence calibration is bounded by:
   $$\text{ECE}_{\tilde{\mathcal{D}}}(f) \le \mathbb{E}_{\mathbf{p}} [ | [T^\top \mathbf{p}]_{\hat{Y}} - \max_k p_k | ]$$
2. Under symmetric label noise with flip probability $\eta \in (0, \frac{K-1}{K})$, for every instance with confidence $\hat{P} > 1/K$, the calibration distortion sign is uniformly negative, yielding the exact equality:
   $$\text{ECE}_{\tilde{\mathcal{D}}}(f) = \frac{K \eta}{K-1} \mathbb{E}_{\hat{P}} \left[ \hat{P} - \frac{1}{K} \right] > 0$$
