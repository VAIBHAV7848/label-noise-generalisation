# Theoretical Assumptions & Identifiability Conditions

To establish mathematically sound bounds and algorithms, we explicitly catalog all required theoretical assumptions and their validity in real-world contexts.

---

## Assumption 1: Class-Conditional Noise (CCN)
> **Statement**: The noisy label $\tilde{Y}$ is conditionally independent of input features $X$ given the true label $Y$:
> $$P(\tilde{Y} = j \mid X = x, Y = i) = P(\tilde{Y} = j \mid Y = i) = T_{ij}$$
- **Role in Theory**: Enables representation of label corruption via a global $K \times K$ transition matrix $T$ independent of $x$.
- **Realism Audit**: Holds strictly in synthetic benchmarks (uniform/pair flips); violated in human crowdsourced datasets (e.g. CIFAR-10N) where ambiguous images have higher flipping rates.

---

## Assumption 2: Invertibility & Dominant Diagonal of Transition Matrix
> **Statement**: The transition matrix $T \in [0, 1]^{K \times K}$ is non-singular ($\det(T) \ne 0$), and for every true class $i$, the probability of remaining clean exceeds the probability of flipping to any other individual class:
> $$T_{ii} > T_{ij}, \quad \forall j \ne i$$
- **Role in Theory**: Required for backward loss correction ($\vec{\tilde{\ell}} = T^{-1} \vec{\ell}$) to exist and prevent degenerate label swaps where classes become unidentifiable.
- **Realism Audit**: Valid whenever overall noise rate $\eta < 0.5$ (binary) or $\eta < \frac{K-1}{K}$ (multi-class).

---

## Assumption 3: Anchor Point Existence (Identifiability Condition)
> **Statement**: For each class $i \in \{1, \dots, K\}$, there exists at least one observable feature instance $x^i \in \mathcal{X}$ (an "anchor point") such that:
> $$P(Y = i \mid X = x^i) = 1, \quad \text{and} \quad P(Y = k \mid X = x^i) = 0 \quad (\forall k \ne i)$$
- **Role in Theory**: Guarantees non-parametric identifiability of $T$. Since $\vec{\tilde{\eta}}(x^i) = T^\top \vec{\eta}(x^i) = T^\top \mathbf{e}_i = \text{Row}_i(T)$, observing the posterior at anchor points allows exact extraction of row $i$ of $T$.
- **Realism Audit**: Plausible in image domains with prototypical, unambiguous samples; difficult to strictly verify without ground truth labels.

---

## Assumption 4: Bounded & Lipschitz Loss Functions
> **Statement**: The loss function $\ell(\cdot, y)$ is $M$-bounded ($0 \le \ell(f(x), y) \le M$) and $L_{\ell}$-Lipschitz continuous with respect to the $\ell_{\infty}$ or $\ell_2$ norm on $\Delta^{K-1}$:
> $$|\ell(\mathbf{p}, y) - \ell(\mathbf{q}, y)| \le L_{\ell} \|\mathbf{p} - \mathbf{q}\|_2, \quad \forall \mathbf{p}, \mathbf{q} \in \Delta^{K-1}$$
- **Role in Theory**: Enables finite-sample Rademacher complexity generalization bounds and excess risk concentration via McDiarmid's inequality.
- **Realism Audit**: Standard for bounded losses (GCE, SCE, MAE); for Cross-Entropy, bounded by clipping softmax predictions $\epsilon \le f_k(x) \le 1-\epsilon$.
