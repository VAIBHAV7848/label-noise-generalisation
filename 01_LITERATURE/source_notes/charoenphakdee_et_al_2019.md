# Source Note: Charoenphakdee et al. (ICML 2019)

- **Title**: On Symmetric Losses for Learning from Corrupted Labels
- **Authors**: Nontawat Charoenphakdee, Jonghyeok Lee, Masashi Sugiyama
- **Venue**: International Conference on Machine Learning (ICML 2019), PMLR 97:961–970
- **DOI / URL**: https://proceedings.mlr.press/v97/charoenphakdee19a.html

---

## 1. Problem Studied
General conditions under which multi-class surrogate loss functions are inherently robust to label noise without requiring explicit estimation of the noise transition matrix $T$.

---

## 2. Key Theoretical Results

### 2.1 Definition of Symmetric Multi-Class Loss
A multi-class loss function $\ell: \Delta^{K-1} \times \{1, \dots, K\} \to \mathbb{R}$ is **symmetric** if there exists a constant $C$ such that:
$$\sum_{k=1}^K \ell(f(x), k) = C, \quad \forall f(x) \in \Delta^{K-1}$$

### 2.2 Noise Tolerance under Symmetric Noise
Under symmetric noise with flip rate $\eta < \frac{K-1}{K}$:
$$R_{\tilde{\mathcal{D}}}(f) = (1 - \eta) R_{\mathcal{D}}(f) + \frac{\eta}{K-1} \sum_{k=1}^K \mathbb{E}_{X}[\ell(f(X), k)] - \frac{\eta}{K-1} R_{\mathcal{D}}(f)$$
Substituting $\sum_{k=1}^K \ell(f(X), k) = C$:
$$R_{\tilde{\mathcal{D}}}(f) = \left( 1 - \frac{\eta K}{K-1} \right) R_{\mathcal{D}}(f) + \frac{\eta C}{K-1}$$
Since $\left( 1 - \frac{\eta K}{K-1} \right) > 0$ and $\frac{\eta C}{K-1}$ is a constant independent of $f$, **any minimizer of noisy risk $R_{\tilde{\mathcal{D}}}(f)$ is an exact minimizer of clean risk $R_{\mathcal{D}}(f)$**.

### 2.3 Failure under Asymmetric / Class-Conditional Noise
Charoenphakdee et al. proved that symmetric losses are **NOT** inherently noise-tolerant under asymmetric noise unless the noise transition matrix satisfies strict symmetry conditions. Furthermore, they proved that common convex losses cannot satisfy the multi-class symmetric loss condition, explaining why MAE is non-convex or difficult to optimize.

---

## 3. Relevance to Our Project
This theorem provides the mathematical bedrock for evaluating robust losses vs. loss correction in our theoretical framework (`02_THEORY/`).
