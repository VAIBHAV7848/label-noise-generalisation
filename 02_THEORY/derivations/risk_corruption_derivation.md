# Derivation: Risk Corruption under Transition Matrix

Here we derive the exact relationship between clean expected risk and noisy expected risk under a general class-conditional noise transition matrix $T \in [0, 1]^{K \times K}$.

---

## 1. Clean Risk Definition

Let $(X, Y) \sim \mathcal{D}$ with input feature $X \in \mathcal{X}$, clean label $Y \in \{1, \dots, K\}$, and class posterior $\eta_i(x) = P(Y=i \mid X=x)$. 
For a vector of surrogate losses $\vec{\ell}(f(x)) = [\ell(f(x), 1), \ell(f(x), 2), \dots, \ell(f(x), K)]^\top \in \mathbb{R}^K$, the clean expected risk is:

$$R_{\mathcal{D}}(f) = \mathbb{E}_{(X, Y) \sim \mathcal{D}} [\ell(f(X), Y)] = \mathbb{E}_X \left[ \sum_{i=1}^K P(Y=i \mid X) \ell(f(X), i) \right] = \mathbb{E}_X \left[ \vec{\eta}(X)^\top \vec{\ell}(f(X)) \right]$$

---

## 2. Noisy Risk Expansion

Under the class-conditional corruption model, the noisy label $\tilde{Y} \in \{1, \dots, K\}$ satisfies:
$$P(\tilde{Y} = j \mid Y = i, X = x) = T_{ij}$$

The noisy expected risk under $(X, \tilde{Y}) \sim \tilde{\mathcal{D}}$ is:

$$R_{\tilde{\mathcal{D}}}(f) = \mathbb{E}_{(X, \tilde{Y}) \sim \tilde{\mathcal{D}}} [\ell(f(X), \tilde{Y})] = \mathbb{E}_X \left[ \sum_{j=1}^K P(\tilde{Y}=j \mid X) \ell(f(X), j) \right]$$

Using the law of total probability:
$$P(\tilde{Y}=j \mid X) = \sum_{i=1}^K P(\tilde{Y}=j \mid Y=i, X) P(Y=i \mid X) = \sum_{i=1}^K T_{ij} \eta_i(X) = [T^\top \vec{\eta}(X)]_j$$

Substituting this into the noisy risk:

$$R_{\tilde{\mathcal{D}}}(f) = \mathbb{E}_X \left[ \sum_{j=1}^K [T^\top \vec{\eta}(X)]_j \ell(f(X), j) \right] = \mathbb{E}_X \left[ (T^\top \vec{\eta}(X))^\top \vec{\ell}(f(X)) \right] = \mathbb{E}_X \left[ \vec{\eta}(X)^\top T \vec{\ell}(f(X)) \right]$$

---

## 3. Special Case: Symmetric (Uniform) Label Noise

Under symmetric noise with flip rate $\eta \in [0, 1)$:
$$T_{ij} = \begin{cases} 1 - \eta & \text{if } i = j \\ \frac{\eta}{K-1} & \text{if } i \ne j \end{cases}$$
In matrix form:
$$T = \left( 1 - \eta - \frac{\eta}{K-1} \right) I_K + \frac{\eta}{K-1} \mathbf{1}_K \mathbf{1}_K^\top = \left( 1 - \frac{\eta K}{K-1} \right) I_K + \frac{\eta}{K-1} \mathbf{1}_K \mathbf{1}_K^\top$$

Substituting $T$ into the noisy risk expression:

$$R_{\tilde{\mathcal{D}}}(f) = \mathbb{E}_X \left[ \vec{\eta}(X)^\top \left( \left( 1 - \frac{\eta K}{K-1} \right) I_K + \frac{\eta}{K-1} \mathbf{1}_K \mathbf{1}_K^\top \right) \vec{\ell}(f(X)) \right]$$

$$R_{\tilde{\mathcal{D}}}(f) = \left( 1 - \frac{\eta K}{K-1} \right) \mathbb{E}_X [\vec{\eta}(X)^\top \vec{\ell}(f(X))] + \frac{\eta}{K-1} \mathbb{E}_X \left[ \vec{\eta}(X)^\top \mathbf{1}_K \mathbf{1}_K^\top \vec{\ell}(f(X)) \right]$$

Since $\vec{\eta}(X)^\top \mathbf{1}_K = \sum_{i=1}^K \eta_i(X) = 1$:

$$R_{\tilde{\mathcal{D}}}(f) = \left( 1 - \frac{\eta K}{K-1} \right) R_{\mathcal{D}}(f) + \frac{\eta}{K-1} \mathbb{E}_X \left[ \sum_{k=1}^K \ell(f(X), k) \right]$$

### Key Observation
If the loss function satisfies the symmetric condition $\sum_{k=1}^K \ell(f(x), k) = C$ for all $x$, the second term becomes a constant $\frac{\eta C}{K-1}$, proving that:
$$\arg\min_f R_{\tilde{\mathcal{D}}}(f) = \arg\min_f R_{\mathcal{D}}(f)$$
when $\eta < \frac{K-1}{K}$.
