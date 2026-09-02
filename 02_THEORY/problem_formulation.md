# Mathematical Problem Formulation & Unified Notation Convention

This document establishes the universal mathematical conventions, dimensions, and operator orientations used throughout the theoretical proofs, methods, and implementations.

---

## 1. Universal Mathematical Convention

| Symbol | Space / Domain | Definition / Role |
| :--- | :--- | :--- |
| $K$ | $\mathbb{N}_{\ge 2}$ | Number of distinct classification categories. |
| $\mathcal{X}$ | $\mathbb{R}^d$ | Continuous input feature space. |
| $\mathcal{Y}$ | $\{1, 2, \dots, K\}$ | Label space with 1-based indexing (or $\{0, \dots, K-1\}$ in code). |
| $X$ | $\mathcal{X}$ | Observable input feature random variable. |
| $Y$ | $\mathcal{Y}$ | Latent true uncorrupted class label. |
| $\tilde{Y}$ | $\mathcal{Y}$ | Observable corrupted (noisy) class label. |
| $\mathcal{D}$ | $\mathcal{P}(\mathcal{X} \times \mathcal{Y})$ | True clean joint data distribution. |
| $\tilde{\mathcal{D}}$ | $\mathcal{P}(\mathcal{X} \times \mathcal{Y})$ | Corrupted observable data distribution. |
| $f(x)$ | $\Delta^{K-1}$ | Probabilistic classifier outputting probability simplex vector. |
| $\vec{\eta}(x)$ | $\Delta^{K-1}$ | Clean posterior column vector: $[\vec{\eta}(x)]_i = P(Y=i \mid X=x)$. |
| $\vec{\tilde{\eta}}(x)$ | $\Delta^{K-1}$ | Corrupted posterior column vector: $[\vec{\tilde{\eta}}(x)]_j = P(\tilde{Y}=j \mid X=x)$. |
| $T$ | $[0, 1]^{K \times K}$ | Row-stochastic transition matrix: $T_{ij} = P(\tilde{Y}=j \mid Y=i)$, $\sum_j T_{ij} = 1$. |
| $\vec{\ell}(f(x))$ | $[0, M]^K$ | Loss column vector: $[\vec{\ell}(f(x))]_k = \ell(f(x), k)$. |

---

## 2. Derivations of Fundamental Relations

### 2.1 Corrupted Posterior Vector Relation
Under the Class-Conditional Noise (CCN) assumption ($P(\tilde{Y} \mid X, Y) = P(\tilde{Y} \mid Y)$):
$$P(\tilde{Y}=j \mid X=x) = \sum_{i=1}^K P(Y=i \mid X=x) P(\tilde{Y}=j \mid Y=i, X=x) = \sum_{i=1}^K \eta_i(x) T_{ij}$$
In vector-matrix form:
$$\vec{\tilde{\eta}}(x) = T^\top \vec{\eta}(x)$$
In row-vector form ($\mathbf{p} \in \mathbb{R}^{1 \times K}$):
$$\tilde{\mathbf{p}} = \mathbf{p} T$$

### 2.2 Expected Clean and Noisy Risk
- **Clean Expected Risk**:
  $$R_{\mathcal{D}}(f) = \mathbb{E}_{(X, Y) \sim \mathcal{D}} [\ell(f(X), Y)] = \mathbb{E}_X [ \vec{\eta}(X)^\top \vec{\ell}(f(X)) ]$$
- **Noisy Expected Risk**:
  $$R_{\tilde{\mathcal{D}}}(f) = \mathbb{E}_{(X, \tilde{Y}) \sim \tilde{\mathcal{D}}} [\ell(f(X), \tilde{Y})] = \mathbb{E}_X [ \vec{\tilde{\eta}}(X)^\top \vec{\ell}(f(X)) ] = \mathbb{E}_X [ \vec{\eta}(X)^\top T \vec{\ell}(f(X)) ]$$

### 2.3 Backward Loss Correction Matrix Orientation
We seek corrected loss vector $\vec{\tilde{\ell}}(f(x))$ such that $\mathbb{E}_{\tilde{Y} \mid X} [\tilde{\ell}(f(X), \tilde{Y})] = \mathbb{E}_{Y \mid X} [\ell(f(X), Y)]$.
$$\vec{\tilde{\eta}}(x)^\top \vec{\tilde{\ell}}(f(x)) = (T^\top \vec{\eta}(x))^\top \vec{\tilde{\ell}}(f(x)) = \vec{\eta}(x)^\top T \vec{\tilde{\ell}}(f(x)) \equiv \vec{\eta}(x)^\top \vec{\ell}(f(x))$$
Therefore:
$$\vec{\tilde{\ell}}(f(x)) = T^{-1} \vec{\ell}(f(x))$$
In batch tensor notation with loss matrix $\mathbf{L} \in \mathbb{R}^{B \times K}$:
$$\tilde{\mathbf{L}} = \mathbf{L} (T^{-1})^\top$$
The sample loss for noisy label $\tilde{y}$ is extracted as $\tilde{L}_{b, \tilde{y}} = \sum_{k=1}^K [T^{-1}]_{\tilde{y}, k} L_{b, k}$.
