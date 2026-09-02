# Mathematical Final Audit: First-Principles Proofs & Calibration Theorems

**Document Type**: Independent Mathematical Verification & Symbolic Derivation Audit  
**Framework**: Academic Research Skills (ARS) Mathematical Rigor Gate  
**Single Source of Truth**: [`VAIBHAV7848/label-noise-generalisation`](https://github.com/VAIBHAV7848/label-noise-generalisation)

---

## 1. Proposition 2: Complete First-Principles Derivation

### 1.1 Statement
Let $T \in [0, 1]^{K \times K}$ be an invertible class-conditional noise transition matrix ($T_{ij} = P(\tilde{Y}=j \mid Y=i)$), and let $\hat{T} = T + E$ be an estimated transition matrix with Frobenius perturbation $\|E\|_F \le \epsilon < \frac{1}{\|T^{-1}\|_2}$. Let $\hat{\ell}_{\text{backward}}(f(x), \tilde{y}) = [\hat{T}^{-1} \vec{\ell}(f(x))]_{\tilde{y}}$. Assume the base surrogate loss $\ell: \Delta^{K-1} \times \{1, \dots, K\} \to \mathbb{R}_+$ is $M$-bounded ($0 \le \ell \le M$) and $L_\ell$-Lipschitz with respect to the $\ell_2$ norm.

Then:
1. **Pointwise Risk Bias**:
   $$\left| \mathbb{E}_{\tilde{\mathcal{D}}} [\hat{\ell}_{\text{backward}}(f(X), \tilde{Y})] - R_{\mathcal{D}}(f) \right| \le \frac{\sqrt{K} M \|T^{-1}\|_2^2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon}$$
2. **Excess Clean Risk**: With probability at least $1 - \delta$ over the draw of noisy sample $\tilde{S}$ of size $n$, the empirical risk minimizer $\hat{f} = \arg\min_{f \in \mathcal{F}} \hat{R}_{\tilde{S}, \hat{T}}(f)$ satisfies:
   $$\mathcal{E}(\hat{f}) \le \frac{2 \sqrt{K} M \|T^{-1}\|_2^2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon} + 4 \sqrt{2} L_{\ell} \|\hat{T}^{-1}\|_2 \mathcal{R}_n(\mathcal{F}) + 2 \sqrt{K} M \|\hat{T}^{-1}\|_2 \sqrt{\frac{\ln(2/\delta)}{2n}}$$

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
- Combining with Step 1:
$$\left| \mathbb{E}_{\tilde{\mathcal{D}}} [\hat{\ell}_{\text{backward}}(f(X), \tilde{Y})] - R_{\mathcal{D}}(f) \right| \le 1 \cdot \frac{\|T^{-1}\|_2^2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon} \cdot \sqrt{K} M = \frac{\sqrt{K} M \|T^{-1}\|_2^2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon}$$

#### Step 4: Excess Risk Decomposition
$$R_{\mathcal{D}}(\hat{f}) - R_{\mathcal{D}}(f^*) \le \underbrace{\left| R_{\mathcal{D}}(\hat{f}) - R_{\tilde{\mathcal{D}}, \hat{T}}(\hat{f}) \right|}_{\le \Delta_{\text{bias}}} + \underbrace{\left( R_{\tilde{\mathcal{D}}, \hat{T}}(\hat{f}) - \hat{R}_{\tilde{S}, \hat{T}}(\hat{f}) \right)}_{\le \text{Gen Gap}} + \underbrace{\left( \hat{R}_{\tilde{S}, \hat{T}}(\hat{f}) - \hat{R}_{\tilde{S}, \hat{T}}(f^*) \right)}_{\le 0} + \underbrace{\left( \hat{R}_{\tilde{S}, \hat{T}}(f^*) - R_{\tilde{\mathcal{D}}, \hat{T}}(f^*) \right)}_{\le \text{Gen Gap}} + \underbrace{\left| R_{\tilde{\mathcal{D}}, \hat{T}}(f^*) - R_{\mathcal{D}}(f^*) \right|}_{\le \Delta_{\text{bias}}}$$
The sum of the two bias terms is $2 \Delta_{\text{bias}} = \frac{2 \sqrt{K} M \|T^{-1}\|_2^2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon}$.  
Applying the vector contraction inequality (Maurer, 2016) for Lipschitz loss composition with operator norm $\|\hat{T}^{-1}\|_2$ gives the Rademacher complexity bound $4 \sqrt{2} L_\ell \|\hat{T}^{-1}\|_2 \mathcal{R}_n(\mathcal{F})$.  
Applying McDiarmid's concentration with loss bound $\sqrt{K} M \|\hat{T}^{-1}\|_2$ yields $2 \sqrt{K} M \|\hat{T}^{-1}\|_2 \sqrt{\frac{\ln(2/\delta)}{2n}}$. $\blacksquare$

---

## 2. Theorems 4A & 4B: Population Identities vs. Empirical Metrics

### 2.1 Theorem 4A (Vector Calibration Distortion)
$$\text{VCE}_{\tilde{\mathcal{D}}}(f) = \mathbb{E}_{\mathbf{p}} [ \| (T^\top - I) \mathbf{p} \|_1 ]$$
- **Orientation**: $T_{ij} = P(\tilde{Y}=j \mid Y=i)$, so $T^\top \mathbf{p}$ maps clean posterior vector $\mathbf{p} \in \Delta^{K-1}$ to noisy posterior vector $\tilde{\mathbf{p}} \in \Delta^{K-1}$.
- **Generality**: Holds for all transition matrices without requiring symmetry.

### 2.2 Theorem 4B (Symmetric Top-Label ECE)
Under symmetric noise with flip rate $\eta \in (0, \frac{K-1}{K})$:
$$\text{ECE}_{\tilde{\mathcal{D}}}(f) = \frac{K \eta}{K-1} \mathbb{E}_{\hat{P}} \left[ \hat{P} - \frac{1}{K} \right]$$
- **Sign Uniformity**: For any instance with confidence $\hat{P} > 1/K$, the calibration gap is $[T^\top \mathbf{p}]_{\hat{Y}} - \hat{P} = -\frac{K \eta}{K-1} (\hat{P} - 1/K) < 0$. Because the sign is uniformly negative across all confidence levels $> 1/K$, no sign-cancellation occurs under expectation.

### 2.3 Critical Distinction: Population Identity vs. Finite-Sample Binning
- **Population ECE**: Theoretical expectation under exact continuous posterior conditioning $\mathbb{E}[|\mathbb{E}[\mathbf{1}_{\tilde{Y}=\hat{Y}} \mid \hat{P}] - \hat{P}|]$.
- **Empirical $\widehat{\text{ECE}}$**: Computed via 15-bin equal-width discretization $\sum_{m=1}^{15} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$.
- **Scientific Clarification**: Empirical $\widehat{\text{ECE}}$ estimates the population quantity with $O(1/\sqrt{N})$ sampling error and $O(1/M)$ binning discretization bias. Empirical measurements validate the directional prediction of Theorem 4B, but finite-sample binning must not be conflated with the exact analytical identity.
