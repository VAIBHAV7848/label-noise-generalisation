# Source Note: Natarajan et al. (NeurIPS 2013)

- **Title**: Learning with Noisy Labels
- **Authors**: Nagarajan Natarajan, Inderjit S. Dhillon, Pradeep K. Ravikumar, Ambuj Tewari
- **Venue**: Advances in Neural Information Processing Systems (NeurIPS 2013), Vol. 26
- **DOI / URL**: https://proceedings.neurips.cc/paper_files/paper/2013/file/3871bd64012152bfb53fdf04b401193f-Paper.pdf

---

## 1. Problem Studied
Binary and multi-class classification under class-conditional label noise where the true label $Y \in \{-1, +1\}$ is flipped to $\tilde{Y}$ with class-dependent probabilities $\rho_+ = P(\tilde{Y}=-1 \mid Y=+1)$ and $\rho_- = P(\tilde{Y}=+1 \mid Y=-1)$, assuming $\rho_+ + \rho_- < 1$.

---

## 2. Key Theoretical Contributions

### 2.1 Unbiased Loss Construction (Backward Correction)
For any surrogate loss $\ell(f(x), y)$, Natarajan et al. construct the corrected loss $\tilde{\ell}(f(x), \tilde{y})$ such that:
$$\mathbb{E}_{\tilde{Y} \mid Y=y}[\tilde{\ell}(f(x), \tilde{Y})] = \ell(f(x), y)$$

For binary classification:
$$\tilde{\ell}(f(x), +1) = \frac{(1 - \rho_-) \ell(f(x), +1) - \rho_+ \ell(f(x), -1)}{1 - \rho_+ - \rho_-}$$
$$\tilde{\ell}(f(x), -1) = \frac{(1 - \rho_+) \ell(f(x), -1) - \rho_- \ell(f(x), +1)}{1 - \rho_+ - \rho_-}$$

In matrix notation for $K$-class classification with transition matrix $T \in \mathbb{R}^{K \times K}$ where $T_{ij} = P(\tilde{Y}=j \mid Y=i)$:
$$\vec{\tilde{\ell}}(f(x)) = T^{-1} \vec{\ell}(f(x))$$
where $\vec{\ell}(f(x)) = [\ell(f(x), 1), \dots, \ell(f(x), K)]^\top$.

### 2.2 Finite-Sample Excess Risk Bounds
Let $\hat{f}$ be the minimizer of empirical risk $\hat{R}_{\tilde{\ell}}(f)$ over hypothesis class $\mathcal{F}$ with Rademacher complexity $\mathcal{R}_n(\mathcal{F})$. Then with probability at least $1 - \delta$:
$$R(\hat{f}) - \min_{f \in \mathcal{F}} R(f) \le \frac{2 L_{\ell}}{1 - \rho_+ - \rho_-} \mathcal{R}_n(\mathcal{F}) + \mathcal{O}\left(\frac{L_{\ell}}{1 - \rho_+ - \rho_-} \sqrt{\frac{\ln(1/\delta)}{n}}\right)$$
where $L_{\ell}$ is the Lipschitz constant of loss $\ell$.

---

## 3. Assumptions
1. Noise is strictly class-conditional: $P(\tilde{Y} \mid X, Y) = P(\tilde{Y} \mid Y)$.
2. The noise rates $\rho_+, \rho_-$ (or transition matrix $T$) are known or accurately estimable.
3. Invertibility: $1 - \rho_+ - \rho_- \ne 0$ (i.e., $\det(T) \ne 0$).

---

## 4. Limitations & Failure Modes
- **Negative Risk / Non-Convexity**: Even if the base loss $\ell$ is convex, $\tilde{\ell}$ can take negative values, breaking convexity and causing gradient descent instability.
- **Noise Matrix Estimation**: Does not provide a scalable method for deep network transition matrix estimation on complex datasets.

---

## 5. Direct Relevance to Our Project
This paper provides the foundational theoretical baseline for unbiased risk estimation. In our project, we extend this by analyzing the excess risk penalty when $\hat{T} \approx T$ under estimation error $\epsilon = \| \hat{T} - T \|_F$.
