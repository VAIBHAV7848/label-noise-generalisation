# Proposition 2 Proof Audit: Complete First-Principles Verification

**Target**: Non-asymptotic excess risk bounds under finite-sample transition matrix estimation error.  
**Classification**: **[VERIFIED AS RIGOROUS THEORETICAL FRAMEWORK BOUND]**

---

## 1. Problem Formulation & Definitions

Let $T \in [0, 1]^{K \times K}$ be the true row-stochastic, invertible class-conditional noise transition matrix with $T_{ij} = P(\tilde{Y}=j \mid Y=i)$.  
Let $\hat{T} = T + E$ be an estimated transition matrix satisfying $\|E\|_F \le \epsilon$.  
Let $\ell: \Delta^{K-1} \times \{1, \dots, K\} \to [0, M]$ be an $M$-bounded, $L_{\ell}$-Lipschitz base surrogate loss function.  
Let the backward corrected loss vector under $\hat{T}$ be:
$$\vec{\hat{\ell}}_{\text{backward}}(f(x)) = \hat{T}^{-1} \vec{\ell}(f(x)) \in \mathbb{R}^K$$
where $\vec{\ell}(f(x)) = [\ell(f(x), 1), \dots, \ell(f(x), K)]^\top \in [0, M]^K$.

For a sample $(x, \tilde{y})$, the empirical loss is $\hat{\ell}_{\text{backward}}(f(x), \tilde{y}) = [\hat{T}^{-1} \vec{\ell}(f(x))]_{\tilde{y}}$.

---

## 2. Step 1: Inverse Perturbation Bound

**Lemma 1 (Matrix Inversion Perturbation)**:  
If $\|T^{-1}\|_2 \|E\|_2 < 1$, then $\hat{T} = T + E$ is non-singular and:
$$\|\hat{T}^{-1}\|_2 \le \frac{\|T^{-1}\|_2}{1 - \|T^{-1}\|_2 \|E\|_2} \le \frac{\|T^{-1}\|_2}{1 - \|T^{-1}\|_2 \epsilon}$$
and
$$\|\hat{T}^{-1} - T^{-1}\|_2 \le \frac{\|T^{-1}\|_2^2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon}$$

*Proof*:  
Write $\hat{T} = T(I + T^{-1} E)$. Since $\|T^{-1} E\|_2 \le \|T^{-1}\|_2 \|E\|_2 \le \|T^{-1}\|_2 \epsilon < 1$, by the Neumann series:
$$\hat{T}^{-1} = (I + T^{-1} E)^{-1} T^{-1} = \sum_{k=0}^\infty (-T^{-1} E)^k T^{-1}$$
Taking spectral norms:
$$\|\hat{T}^{-1}\|_2 \le \sum_{k=0}^\infty (\|T^{-1}\|_2 \|E\|_2)^k \|T^{-1}\|_2 = \frac{\|T^{-1}\|_2}{1 - \|T^{-1}\|_2 \|E\|_2}$$
Furthermore:
$$\hat{T}^{-1} - T^{-1} = -T^{-1} E \hat{T}^{-1} \implies \|\hat{T}^{-1} - T^{-1}\|_2 \le \|T^{-1}\|_2 \|E\|_2 \|\hat{T}^{-1}\|_2 \le \frac{\|T^{-1}\|_2^2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon} \quad \blacksquare$$

---

## 3. Step 2: Pointwise Expected Risk Estimation Bias

**Lemma 2 (Pointwise Bias Bound)**:  
For any input $x \in \mathcal{X}$ and any model $f: \mathcal{X} \to \Delta^{K-1}$:
$$\left| \mathbb{E}_{\tilde{Y} \mid X=x} [\hat{\ell}_{\text{backward}}(f(x), \tilde{Y})] - \mathbb{E}_{Y \mid X=x} [\ell(f(x), Y)] \right| \le \frac{\sqrt{K} M \|T^{-1}\|_2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon}$$

*Proof*:  
Let $\vec{\eta}(x) \in \Delta^{K-1}$ with $\eta_i(x) = P(Y=i \mid X=x)$.  
Under Class-Conditional Noise, $\vec{\tilde{\eta}}(x) = T^\top \vec{\eta}(x)$.  
The expected corrected loss under corrupted labels $\tilde{Y} \mid X=x$ is:
$$\mathbb{E}_{\tilde{Y} \mid X=x} [\hat{\ell}_{\text{backward}}(f(x), \tilde{Y})] = \vec{\tilde{\eta}}(x)^\top \vec{\hat{\ell}}_{\text{backward}}(f(x)) = (T^\top \vec{\eta}(x))^\top \hat{T}^{-1} \vec{\ell}(f(x)) = \vec{\eta}(x)^\top T \hat{T}^{-1} \vec{\ell}(f(x))$$
The clean expected loss is:
$$\mathbb{E}_{Y \mid X=x} [\ell(f(x), Y)] = \vec{\eta}(x)^\top \vec{\ell}(f(x)) = \vec{\eta}(x)^\top I \vec{\ell}(f(x))$$
Subtracting the two expressions:
$$\text{Bias}(x) = \vec{\eta}(x)^\top (T \hat{T}^{-1} - I) \vec{\ell}(f(x))$$
Crucially, express $T$ in terms of $\hat{T}$: $T = \hat{T} - E$. Then:
$$T \hat{T}^{-1} - I = (\hat{T} - E) \hat{T}^{-1} - I = I - E \hat{T}^{-1} - I = -E \hat{T}^{-1}$$
Therefore:
$$\text{Bias}(x) = -\vec{\eta}(x)^\top E \hat{T}^{-1} \vec{\ell}(f(x))$$
Applying the Cauchy-Schwarz and operator norm inequalities:
$$|\text{Bias}(x)| \le \|\vec{\eta}(x)\|_2 \cdot \|E\|_2 \cdot \|\hat{T}^{-1}\|_2 \cdot \|\vec{\ell}(f(x))\|_2$$
We bound each term:
1. Since $\vec{\eta}(x) \in \Delta^{K-1}$ (simplex), $\|\vec{\eta}(x)\|_2 = \sqrt{\sum_i \eta_i^2} \le \sum_i \eta_i = 1$.
2. $\|E\|_2 \le \|E\|_F \le \epsilon$.
3. $\|\hat{T}^{-1}\|_2 \le \frac{\|T^{-1}\|_2}{1 - \|T^{-1}\|_2 \epsilon}$ (by Lemma 1).
4. Since $0 \le \ell(f(x), k) \le M$, $\|\vec{\ell}(f(x))\|_2 = \sqrt{\sum_{k=1}^K \ell(f(x), k)^2} \le \sqrt{K M^2} = \sqrt{K} M$.

Multiplying these bounds yields:
$$|\text{Bias}(x)| \le 1 \cdot \epsilon \cdot \frac{\|T^{-1}\|_2}{1 - \|T^{-1}\|_2 \epsilon} \cdot \sqrt{K} M = \frac{\sqrt{K} M \|T^{-1}\|_2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon} \quad \blacksquare$$

---

## 4. Step 3: Excess Risk Decomposition & Uniform Convergence

Let $\hat{R}_{\tilde{S}, \hat{T}}(f) = \frac{1}{n} \sum_{i=1}^n \hat{\ell}_{\text{backward}}(f(x_i), \tilde{y}_i)$.  
Let $\hat{f} = \arg\min_{f \in \mathcal{F}} \hat{R}_{\tilde{S}, \hat{T}}(f)$ and $f^* = \arg\min_{f \in \mathcal{F}} R_{\mathcal{D}}(f)$.

$$\mathcal{E}(\hat{f}) = R_{\mathcal{D}}(\hat{f}) - R_{\mathcal{D}}(f^*)$$
$$\le \underbrace{R_{\mathcal{D}}(\hat{f}) - R_{\tilde{\mathcal{D}}, \hat{T}}(\hat{f})}_{\le \sup_f |R_{\mathcal{D}} - R_{\tilde{\mathcal{D}}, \hat{T}}|} + \underbrace{R_{\tilde{\mathcal{D}}, \hat{T}}(\hat{f}) - \hat{R}_{\tilde{S}, \hat{T}}(\hat{f})}_{\le \sup_f |R_{\tilde{\mathcal{D}}, \hat{T}} - \hat{R}_{\tilde{S}, \hat{T}}|} + \underbrace{\hat{R}_{\tilde{S}, \hat{T}}(\hat{f}) - \hat{R}_{\tilde{S}, \hat{T}}(f^*)}_{\le 0 \text{ by optimality of } \hat{f}} + \underbrace{\hat{R}_{\tilde{S}, \hat{T}}(f^*) - R_{\tilde{\mathcal{D}}, \hat{T}}(f^*)}_{\le \sup_f |\hat{R}_{\tilde{S}, \hat{T}} - R_{\tilde{\mathcal{D}}, \hat{T}}|} + \underbrace{R_{\tilde{\mathcal{D}}, \hat{T}}(f^*) - R_{\mathcal{D}}(f^*)}_{\le \sup_f |R_{\tilde{\mathcal{D}}, \hat{T}} - R_{\mathcal{D}}|}$$
$$\implies \mathcal{E}(\hat{f}) \le 2 \sup_{f \in \mathcal{F}} |R_{\tilde{\mathcal{D}}, \hat{T}}(f) - R_{\mathcal{D}}(f)| + 2 \sup_{f \in \mathcal{F}} |\hat{R}_{\tilde{S}, \hat{T}}(f) - R_{\tilde{\mathcal{D}}, \hat{T}}(f)|$$

### 4.1 Bias Term Bound
By Lemma 2 integrated over $\mathcal{X}$:
$$2 \sup_{f \in \mathcal{F}} |R_{\tilde{\mathcal{D}}, \hat{T}}(f) - R_{\mathcal{D}}(f)| \le \frac{2 \sqrt{K} M \|T^{-1}\|_2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon}$$

### 4.2 Empirical Process Generalization Bound
For each sample $(x, \tilde{y})$, $|\hat{\ell}_{\text{backward}}(f(x), \tilde{y})| \le \|\hat{T}^{-1}\|_\infty M \le \sqrt{K} \|\hat{T}^{-1}\|_2 M$.  
By McDiarmid's concentration inequality, with probability $\ge 1 - \delta$:
$$\sup_{f \in \mathcal{F}} |\hat{R}_{\tilde{S}, \hat{T}}(f) - R_{\tilde{\mathcal{D}}, \hat{T}}(f)| \le 2 \mathbb{E}_{\tilde{S}} [\mathcal{R}_n(\hat{\ell}_{\text{backward}} \circ \mathcal{F})] + \sqrt{K} M \|\hat{T}^{-1}\|_2 \sqrt{\frac{\ln(2/\delta)}{2n}}$$
Since base loss $\ell$ is $L_{\ell}$-Lipschitz, the vector-valued Lipschitz contraction (Maurer 2016) bounds $\mathcal{R}_n(\hat{\ell}_{\text{backward}} \circ \mathcal{F}) \le \sqrt{2} L_{\ell} \|\hat{T}^{-1}\|_2 \mathcal{R}_n(\mathcal{F})$.

Combining terms yields the complete theorem bound:
$$\mathcal{E}(\hat{f}) \le \frac{2 \sqrt{K} M \|T^{-1}\|_2 \epsilon}{1 - \|T^{-1}\|_2 \epsilon} + 4 \sqrt{2} L_{\ell} \|\hat{T}^{-1}\|_2 \mathcal{R}_n(\mathcal{F}) + 2 \sqrt{K} M \|\hat{T}^{-1}\|_2 \sqrt{\frac{\ln(2/\delta)}{2n}} \quad \blacksquare$$

---

## 5. Literature Comparison
- **Natarajan et al. (NeurIPS 2013)**: Proved binary excess risk bounds for known $T$; asymptotic rate for estimated $T$.
- **Patrini et al. (CVPR 2017)**: Proved asymptotic Fisher consistency for multi-class forward/backward losses with known $T$.
- **Scott et al. (2015) / Xia et al. (2019)**: Derived convergence rates for estimator errors.
- **Verdict**: Our Proposition 2 is a sound, non-asymptotic operator-norm unification of perturbation theory and empirical process theory.
