# Proposition 4 Calibration Audit: Flaw Analysis, Counterexample & Corrected Theorem

**Original Statement**: Claimed $\text{ECE}_{\tilde{\mathcal{D}}}(f) \ge \mathbb{E}_{\mathbf{p}}[|\max_k p_k - \max_k [T^\top \mathbf{p}]_k|] > 0$ for all $T \ne I$.  
**Classification**: **[INCORRECT IN ORIGINAL FORM $\to$ REFORMULATED & VERIFIED]**

---

## 1. Mathematical Definitions of Calibration

Let $f: \mathcal{X} \to \Delta^{K-1}$ be a probabilistic classifier. For any input $x$, $f(x) = \mathbf{p} \in \Delta^{K-1}$.  
- Predicted class: $\hat{Y} = \arg\max_{k \in \{1, \dots, K\}} p_k$.  
- Predicted confidence: $\hat{P} = \max_{k \in \{1, \dots, K\}} p_k = p_{\hat{Y}}$.

### 1.1 Top-Label Confidence Calibration (Standard ECE)
Under clean distribution $\mathcal{D}$:
$$\text{ECE}_{\mathcal{D}}(f) = \mathbb{E}_{\hat{P}} [ | P(Y = \hat{Y} \mid \hat{P}) - \hat{P} | ]$$

Under noisy distribution $\tilde{\mathcal{D}}$:
$$\text{ECE}_{\tilde{\mathcal{D}}}(f) = \mathbb{E}_{\hat{P}} [ | P(\tilde{Y} = \hat{Y} \mid \hat{P}) - \hat{P} | ]$$

### 1.2 Full Vector Calibration (VCE)
$$\text{VCE}_{\tilde{\mathcal{D}}}(f) = \mathbb{E}_{X} [ \| \mathbb{E}[\mathbf{e}_{\tilde{Y}} \mid f(X)] - f(X) \|_1 ]$$

---

## 2. Flaw in the Original Proposition 4 Claim

Under Class-Conditional Noise ($P(\tilde{Y}=j \mid Y=i) = T_{ij}$), assuming the model is perfectly calibrated on clean data ($\mathbb{E}[\mathbf{e}_Y \mid f(X) = \mathbf{p}] = \mathbf{p}$), the noisy class posterior is:
$$P(\tilde{Y} = \hat{Y} \mid f(X) = \mathbf{p}) = [T^\top \mathbf{p}]_{\hat{Y}}$$

Now consider the population ECE conditioning on confidence $\hat{P}$:
$$P(\tilde{Y} = \hat{Y} \mid \hat{P}) = \mathbb{E}_{f(X) \mid \hat{P}} [ P(\tilde{Y} = \hat{Y} \mid f(X)) ] = \mathbb{E}_{\mathbf{p} \mid \hat{P}} [ [T^\top \mathbf{p}]_{\hat{Y}} ]$$
By the tower property of expectation and Jensen's inequality for convex functions $|\cdot|$:
$$\text{ECE}_{\tilde{\mathcal{D}}}(f) = \mathbb{E}_{\hat{P}} [ | \mathbb{E}_{\mathbf{p} \mid \hat{P}} [ [T^\top \mathbf{p}]_{\hat{Y}} - \hat{P} ] | ] \le \mathbb{E}_{\hat{P}} [ \mathbb{E}_{\mathbf{p} \mid \hat{P}} [ | [T^\top \mathbf{p}]_{\hat{Y}} - \hat{P} | ] ] = \mathbb{E}_{\mathbf{p}} [ | [T^\top \mathbf{p}]_{\hat{Y}} - \max_k p_k | ]$$

### The Fatal Flaw:
The original Proposition 4 had the inequality backwards ($\ge$ instead of $\le$) and replaced $[T^\top \mathbf{p}]_{\hat{Y}}$ with $\max_k [T^\top \mathbf{p}]_k$. When different classes with identical confidence $\hat{P}$ have calibration errors of opposite signs, **they cancel out inside the conditional expectation**, making true $\text{ECE}_{\tilde{\mathcal{D}}}(f)$ strictly smaller than $\mathbb{E}_{\mathbf{p}}[|\dots|]$.

---

## 3. Concrete Numerical Counterexample

Let $K=2$ with transition matrix $T = \begin{bmatrix} 0.9 & 0.1 \\ 0.8 & 0.2 \end{bmatrix}$.  
Suppose the feature distribution produces two distinct points $x_1, x_2$ with equal probability $0.5$:
- At $x_1$: $f(x_1) = \mathbf{p}_1 = [0.7, 0.3]^\top \implies \hat{Y} = 0, \hat{P} = 0.7$.
- At $x_2$: $f(x_2) = \mathbf{p}_2 = [0.3, 0.7]^\top \implies \hat{Y} = 1, \hat{P} = 0.7$.

Both instances have identical confidence $\hat{P} = 0.7$.
- For $x_1$: $P(\tilde{Y} = 0 \mid \mathbf{p}_1) = 0.7(0.9) + 0.3(0.8) = 0.63 + 0.24 = 0.87$.  
  Error on $x_1$: $0.87 - 0.70 = +0.17$.
- For $x_2$: $P(\tilde{Y} = 1 \mid \mathbf{p}_2) = 0.3(0.1) + 0.7(0.2) = 0.03 + 0.14 = 0.17$.  
  Error on $x_2$: $0.17 - 0.70 = -0.53$.

### Evaluation:
- **True Noisy Accuracy**: $P(\tilde{Y} = \hat{Y} \mid \hat{P} = 0.7) = 0.5(0.87) + 0.5(0.17) = 0.52$.
- **True Noisy ECE**: $|0.52 - 0.70| = \mathbf{0.18}$.
- **Phase 0 Proposed Lower Bound**: $0.5 |+0.17| + 0.5 |-0.53| = 0.085 + 0.265 = \mathbf{0.35}$.

Since $0.18 < 0.35$, the proposed lower bound is **MATHEMATICALLY DISPROVEN**.

---

## 4. Corrected & Defensible Calibration Theorems

### Theorem 4A (Exact Vector Calibration Distortion)
Let $f: \mathcal{X} \to \Delta^{K-1}$ be perfectly calibrated on clean distribution $\mathcal{D}$ ($\mathbb{E}[\mathbf{e}_Y \mid f(X) = \mathbf{p}] = \mathbf{p}$). Under class-conditional noise with transition matrix $T$, the Vector Calibration Error on corrupted data $\tilde{\mathcal{D}}$ is given by the exact identity:
$$\text{VCE}_{\tilde{\mathcal{D}}}(f) = \mathbb{E}_{\mathbf{p}} [ \| (T^\top - I) \mathbf{p} \|_1 ]$$
Moreover, for any distribution whose support is not entirely contained in the stationary eigenspace of $T^\top$, $\text{VCE}_{\tilde{\mathcal{D}}}(f) > 0$ whenever $T \ne I$.

### Theorem 4B (Top-Label ECE Upper Bound & Symmetric Case)
1. For general $T$, top-label confidence calibration is bounded by:
   $$\text{ECE}_{\tilde{\mathcal{D}}}(f) \le \mathbb{E}_{\mathbf{p}} [ | [T^\top \mathbf{p}]_{\hat{Y}} - \max_k p_k | ]$$
2. Under symmetric noise with flip probability $\eta \in (0, \frac{K-1}{K})$ ($T = (1-\eta)I + \frac{\eta}{K-1}(\mathbf{1}\mathbf{1}^\top - I)$), for every instance $\mathbf{p}$:
   $$[T^\top \mathbf{p}]_{\hat{Y}} - p_{\hat{Y}} = (1-\eta) p_{\hat{Y}} + \frac{\eta}{K-1}(1 - p_{\hat{Y}}) - p_{\hat{Y}} = -\frac{K \eta}{K-1} \left( p_{\hat{Y}} - \frac{1}{K} \right) \le 0$$
   Because the error sign is strictly negative for all classes whenever $p_{\hat{Y}} > 1/K$, no Jensen cancellation occurs, and the top-label ECE is exact:
   $$\text{ECE}_{\tilde{\mathcal{D}}}(f) = \frac{K \eta}{K-1} \mathbb{E}_{\hat{P}} \left[ \hat{P} - \frac{1}{K} \right] > 0$$
